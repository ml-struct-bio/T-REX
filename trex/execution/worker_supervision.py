"""Worker completion, timeout salvage, and campaign-shutdown draining."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..campaign.runtime.timeouts import (
    BINDCRAFT_FLOOR_S,
    BINDCRAFT_YIELD_WINDOW_S,
    worker_timeout_reason,
)
from ..campaign.runtime.worker import WorkerSlot


@dataclass(frozen=True)
class WorkerSupervisionDependencies:
    """Controller-owned operations required by worker lifecycle supervision."""

    parse_worker_output: Callable[..., int]
    record_parse_failure: Callable[..., None]
    parse_incremental_bindcraft_output: Callable[..., int]
    count_bindcraft_scoreable_results: Callable[[Path | None], int | None]
    resolve_hard_ceiling_seconds: Callable[[str], float]
    terminate_worker_process: Callable[[Any], None]


@dataclass(frozen=True)
class WorkerSupervisionReport:
    """Inspectable outcome of one non-blocking worker-pool observation."""

    completed_workers: int = 0
    timed_out_workers: int = 0
    workers_with_new_results: int = 0
    archived_results: int = 0
    incremental_results: int = 0
    parse_failures: int = 0


@dataclass(frozen=True)
class WorkerDrainReport:
    """Inspectable outcome of the campaign-shutdown drain."""

    drained_workers: int = 0
    timed_out_workers: int = 0
    archived_results: int = 0
    parse_failures: int = 0


def _method_family(worker_slot: WorkerSlot) -> str:
    return (
        worker_slot.cand.method_family
        if worker_slot.cand is not None
        else "?"
    )


def _target_id(target: Any) -> str | None:
    value = getattr(target, "target_id", None)
    return str(value) if value is not None else None


def supervise_worker_slots(
    worker_slots: Sequence[WorkerSlot],
    *,
    archive: Any,
    target: Any,
    auto_chain_sequence: list[int],
    dependencies: WorkerSupervisionDependencies,
    clock: Callable[[], float] = time.time,
) -> WorkerSupervisionReport:
    """Observe busy slots once and handle completion or timeout.

    This function never plans or dispatches new candidates. Completed workers
    are parsed and freed. Running BindCraft workers may be parsed incrementally;
    timed-out workers are salvage-parsed before their process group is stopped.
    A failure in parsing one worker is recorded and isolated from the rest of
    the pool.
    """

    completed_workers = 0
    timed_out_workers = 0
    workers_with_new_results = 0
    archived_results = 0
    incremental_results = 0
    parse_failures = 0

    for worker_slot in worker_slots:
        if not worker_slot.busy:
            continue

        method_family = _method_family(worker_slot)
        elapsed_worker_seconds = clock() - worker_slot.launched_at
        if worker_slot.is_done():
            completed_workers += 1
            return_code = worker_slot.proc.returncode  # type: ignore[union-attr]
            try:
                archived_result_count = dependencies.parse_worker_output(
                    worker_slot,
                    return_code=return_code,
                    archive=archive,
                    target=target,
                    auto_chain_sequence=auto_chain_sequence,
                    elapsed_gpu_hours=elapsed_worker_seconds / 3600.0,
                )
                archived_results += archived_result_count
                if archived_result_count > 0:
                    workers_with_new_results += 1
            except Exception as exc:  # noqa: BLE001
                parse_failures += 1
                dependencies.record_parse_failure(
                    archive,
                    worker_slot,
                    (
                        "clean-completion parse/archive error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    target_id=_target_id(target),
                )
                print(
                    f"  [reap_parse_err][gpu={worker_slot.gpu_id}] "
                    f"{method_family} rc={return_code}: "
                    f"{type(exc).__name__}: {exc} — slot freed, continuing",
                    flush=True,
                )
            worker_slot.free()
            continue

        observed_at = clock()
        scoreable_result_count: int | None = None
        progress_count: int | None = None
        if method_family == "bindcraft":
            try:
                incremental_result_count = (
                    dependencies.parse_incremental_bindcraft_output(
                        worker_slot,
                        archive=archive,
                        target=target,
                        auto_chain_sequence=auto_chain_sequence,
                        observed_at=observed_at,
                    )
                )
                incremental_results += incremental_result_count
            except Exception as exc:  # noqa: BLE001
                parse_failures += 1
                print(
                    f"  [bindcraft_incremental_parse_err]"
                    f"[gpu={worker_slot.gpu_id}] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        if (
            method_family == "bindcraft"
            and elapsed_worker_seconds > BINDCRAFT_FLOOR_S
        ):
            scoreable_result_count = (
                dependencies.count_bindcraft_scoreable_results(
                    worker_slot.out_dir
                )
            )
            progress_count = scoreable_result_count

        timeout_ceiling_seconds = (
            dependencies.resolve_hard_ceiling_seconds(method_family)
        )
        (
            timeout_reason,
            worker_slot.prev_yield_n,
            worker_slot.last_yield_at,
        ) = worker_timeout_reason(
            method_family,
            elapsed_slot_s=elapsed_worker_seconds,
            ceiling_s=timeout_ceiling_seconds,
            now_t=observed_at,
            accepted_count=scoreable_result_count,
            progress_count=progress_count,
            prev_yield_n=worker_slot.prev_yield_n,
            last_yield_at=worker_slot.last_yield_at,
        )
        if timeout_reason is None:
            continue

        timed_out_workers += 1
        timeout_detail = (
            f"ceiling={timeout_ceiling_seconds}s"
            if timeout_reason == "hard-ceiling"
            else (
                f"floor={BINDCRAFT_FLOOR_S}s "
                f"yield_window={BINDCRAFT_YIELD_WINDOW_S}s "
                f"scoreable_progress={worker_slot.prev_yield_n} "
                f"scoreable={scoreable_result_count}"
            )
        )
        print(
            f"  [worker_timeout][gpu={worker_slot.gpu_id}] "
            f"{method_family} killed "
            f"({timeout_reason}, elapsed={elapsed_worker_seconds:.0f}s "
            f"{timeout_detail}) -- salvage parse before terminate",
            flush=True,
        )
        try:
            archived_result_count = dependencies.parse_worker_output(
                worker_slot,
                return_code=-1,
                archive=archive,
                target=target,
                auto_chain_sequence=auto_chain_sequence,
                elapsed_gpu_hours=elapsed_worker_seconds / 3600.0,
            )
            archived_results += archived_result_count
            if archived_result_count > 0:
                workers_with_new_results += 1
                print(
                    f"  [worker_timeout_salvaged][gpu={worker_slot.gpu_id}] "
                    f"recovered {archived_result_count} records before kill",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            parse_failures += 1
            dependencies.record_parse_failure(
                archive,
                worker_slot,
                (
                    "timeout-salvage parse/archive error: "
                    f"{type(exc).__name__}: {exc}"
                ),
                target_id=_target_id(target),
            )
            print(
                f"  [worker_timeout_parse_err][gpu={worker_slot.gpu_id}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        dependencies.terminate_worker_process(worker_slot.proc)
        worker_slot.free()

    return WorkerSupervisionReport(
        completed_workers=completed_workers,
        timed_out_workers=timed_out_workers,
        workers_with_new_results=workers_with_new_results,
        archived_results=archived_results,
        incremental_results=incremental_results,
        parse_failures=parse_failures,
    )


def drain_worker_slots(
    worker_slots: Sequence[WorkerSlot],
    *,
    grace_seconds_per_slot: float,
    archive: Any,
    target: Any,
    auto_chain_sequence: list[int],
    dependencies: WorkerSupervisionDependencies,
    clock: Callable[[], float] = time.time,
) -> WorkerDrainReport:
    """Preserve the original sequential, per-slot shutdown grace.

    Each busy slot receives its own full wait timeout. There is no shared
    completion deadline. Timeout handling salvages results before terminating
    the process, and retains the slot reference until controller exit, as in
    the original shutdown loop. Normal completion frees the slot.
    """

    drained_workers = 0
    timed_out_workers = 0
    archived_results = 0
    parse_failures = 0

    for worker_slot in worker_slots:
        if not worker_slot.busy:
            continue
        drained_workers += 1
        method_family = _method_family(worker_slot)
        print(
            f"  [drain][gpu={worker_slot.gpu_id}] waiting up to "
            f"{grace_seconds_per_slot:.0f}s for {method_family} to finish",
            flush=True,
        )
        try:
            return_code = worker_slot.proc.wait(  # type: ignore[union-attr]
                timeout=grace_seconds_per_slot
            )
        except subprocess.TimeoutExpired:
            timed_out_workers += 1
            elapsed_worker_seconds = clock() - worker_slot.launched_at
            print(
                f"  [drain_timeout][gpu={worker_slot.gpu_id}] "
                "attempt salvage parse before terminate",
                flush=True,
            )
            try:
                archived_results += dependencies.parse_worker_output(
                    worker_slot,
                    return_code=-1,
                    archive=archive,
                    target=target,
                    auto_chain_sequence=auto_chain_sequence,
                    elapsed_gpu_hours=elapsed_worker_seconds / 3600.0,
                )
            except Exception as exc:  # noqa: BLE001
                parse_failures += 1
                dependencies.record_parse_failure(
                    archive,
                    worker_slot,
                    (
                        "drain-timeout parse/archive error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    target_id=_target_id(target),
                )
                print(
                    f"  [drain_timeout_parse_err][gpu={worker_slot.gpu_id}] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            dependencies.terminate_worker_process(worker_slot.proc)
            # Match the original final drain: no further dispatch follows.
            continue

        try:
            elapsed_worker_seconds = clock() - worker_slot.launched_at
            archived_results += dependencies.parse_worker_output(
                worker_slot,
                return_code=return_code,
                archive=archive,
                target=target,
                auto_chain_sequence=auto_chain_sequence,
                elapsed_gpu_hours=elapsed_worker_seconds / 3600.0,
            )
        except Exception as exc:  # noqa: BLE001
            parse_failures += 1
            dependencies.record_parse_failure(
                archive,
                worker_slot,
                (
                    f"drain parse/archive error: {type(exc).__name__}: {exc}"
                ),
                target_id=_target_id(target),
            )
            print(
                f"  [drain_parse_err][gpu={worker_slot.gpu_id}] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        worker_slot.free()

    return WorkerDrainReport(
        drained_workers=drained_workers,
        timed_out_workers=timed_out_workers,
        archived_results=archived_results,
        parse_failures=parse_failures,
    )


__all__ = [
    "WorkerDrainReport",
    "WorkerSupervisionDependencies",
    "WorkerSupervisionReport",
    "drain_worker_slots",
    "supervise_worker_slots",
]
