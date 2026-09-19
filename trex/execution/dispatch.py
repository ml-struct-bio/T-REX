"""Pending-candidate dispatch and worker-slot admission.

This module owns the state transition from a queued candidate identifier to a
running worker slot.  Scientific prioritization and backend launching remain
injected dependencies so this queue machinery can be tested without an LLM,
GPU, or subprocess.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..schemas import ActionCandidate, DispatchRecord, LaunchDecision


MAX_DISPATCH_RETRIES = 3


class PermanentDispatchSkip(RuntimeError):
    """Deterministic dispatch precondition failure that must not be retried."""


class DispatchableWorkerSlot(Protocol):
    """Mutable worker-slot surface required by the queue dispatcher."""

    slot_id: int
    gpu_id: str
    proc: Any
    cand: ActionCandidate | None
    out_dir: Path | None
    parent_pdb_str: str
    parent_result_id: str
    target_chains_csv: str
    binder_chain: str
    tick_id: str
    launched_at: float

    @property
    def busy(self) -> bool: ...


DispatchOutcome = (
    tuple[Any, Path, str, str]
    | tuple[Any, Path, str, str, str, str]
)
DispatchCandidate = Callable[..., DispatchOutcome | None]


@dataclass(frozen=True)
class DispatchQueueDependencies:
    """Controller policies used by the generic queue state machine.

    Keeping these functions explicit avoids an import cycle back into the
    production controller and makes the orchestration independently testable.
    """

    prioritize_pending_queue: Callable[
        [Any, list[str], Mapping[str, ActionCandidate]], None
    ]
    high_cost_defer_reason: Callable[
        [Sequence[DispatchableWorkerSlot], ActionCandidate, Any], str | None
    ]
    latest_launch_decision: Callable[[Any, str], LaunchDecision | None]
    dispatch_record_metadata: Callable[..., dict[str, Any]]
    short_id: Callable[[str], str]
    process_start_ticks: Callable[[int | None], int | None]


def _append_dispatch_record(
    archive: Any,
    record_factory: Callable[[], DispatchRecord],
    *,
    gpu_id: str,
    candidate_id: str,
    worker_is_running: bool = False,
) -> None:
    """Best-effort audit write that never loses an already-running worker."""

    if archive is None:
        return
    try:
        archive.append(record_factory())
    except Exception as exc:  # noqa: BLE001
        suffix = " — worker remains tracked" if worker_is_running else ""
        print(
            f"  [dispatch_archive_err][gpu={gpu_id}] {candidate_id}: "
            f"{type(exc).__name__}: {exc}{suffix}",
            flush=True,
        )


def dispatch_pending_to_free_slots(
    worker_slots: Sequence[DispatchableWorkerSlot],
    pending_candidate_ids: list[str],
    candidates_by_id: Mapping[str, ActionCandidate],
    *,
    archive: Any,
    target: Any,
    target_pdb: str | None,
    archive_root: Path,
    round_id: int,
    dispatch_candidate: DispatchCandidate,
    dependencies: DispatchQueueDependencies,
    retry_counts: dict[str, int] | None = None,
    deferred_audits: set[str] | None = None,
    max_dispatch_retries: int = MAX_DISPATCH_RETRIES,
) -> int:
    """Start queued candidates immediately on free worker slots.

    Candidates that fail for a transient reason are requeued once, after the
    current slot scan, so one bad candidate cannot spin or block other work.
    Permanent precondition failures and retries beyond ``max_dispatch_retries``
    are removed from the queue and recorded in the campaign audit trail.

    Returns the number of workers started.
    """

    dispatched_count = 0
    retry_later: list[str] = []
    capacity_deferred_ids: set[str] = set()
    dependencies.prioritize_pending_queue(
        archive, pending_candidate_ids, candidates_by_id
    )
    active_candidate_ids = {
        slot.cand.candidate_id
        for slot in worker_slots
        if slot.busy and slot.cand is not None
    }

    for worker_slot in worker_slots:
        if worker_slot.busy:
            continue
        while pending_candidate_ids:
            candidate_id = pending_candidate_ids.pop(0)
            if (
                candidate_id in active_candidate_ids
                or candidate_id in capacity_deferred_ids
            ):
                continue
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None:
                continue

            defer_reason = dependencies.high_cost_defer_reason(
                worker_slots, candidate, archive
            )
            if defer_reason:
                # Capacity pressure is temporary, so retain the selected card
                # behind other work and audit the admission decision once.
                capacity_deferred_ids.add(candidate_id)
                retry_later.append(candidate_id)
                defer_key = f"{candidate_id}\0{defer_reason}"
                should_audit = (
                    deferred_audits is None or defer_key not in deferred_audits
                )
                if deferred_audits is not None:
                    deferred_audits.add(defer_key)
                launch_decision = dependencies.latest_launch_decision(
                    archive, candidate_id
                )
                if should_audit and archive is not None:
                    _append_dispatch_record(
                        archive,
                        lambda: DispatchRecord(
                            dispatch_id=(
                                f"dispatch_deferred_v7r{round_id:03d}_"
                                f"{dependencies.short_id(candidate_id + ':' + defer_reason)}"
                            ),
                            launch_id=(
                                launch_decision.launch_id
                                if launch_decision is not None
                                else None
                            ),
                            tick_id=f"v7r{round_id:03d}",
                            candidate_id=candidate_id,
                            status="dispatch_failed",
                            worker_slot=str(worker_slot.slot_id),
                            gpu_id=str(worker_slot.gpu_id),
                            **dependencies.dispatch_record_metadata(
                                cand=candidate,
                                launch_decision=launch_decision,
                            ),
                            why=defer_reason,
                        ),
                        gpu_id=str(worker_slot.gpu_id),
                        candidate_id=candidate_id,
                    )
                if should_audit:
                    print(
                        f"  [dispatch_defer][gpu={worker_slot.gpu_id}] "
                        f"{candidate_id}: {defer_reason}",
                        flush=True,
                    )
                continue

            launch_decision = dependencies.latest_launch_decision(
                archive, candidate_id
            )
            attempt = (
                retry_counts.get(candidate_id, 0) + 1
                if retry_counts is not None
                else 1
            )
            dispatch_error: str | None = None
            permanent_skip = False
            try:
                outcome = dispatch_candidate(
                    candidate,
                    gpu_id=worker_slot.gpu_id,
                    archive=archive,
                    target=target,
                    target_pdb=target_pdb,
                    round_id=round_id,
                    archive_root=archive_root,
                )
            except PermanentDispatchSkip as exc:
                dispatch_error = str(exc)
                permanent_skip = True
                outcome = None
                print(
                    f"  [dispatch_skip][gpu={worker_slot.gpu_id}] "
                    f"{candidate_id}: {dispatch_error}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                dispatch_error = f"{type(exc).__name__}: {exc}"
                outcome = None
                print(
                    f"  [dispatch_err][gpu={worker_slot.gpu_id}] "
                    f"{candidate_id}: {dispatch_error}",
                    flush=True,
                )

            if outcome is None:
                if permanent_skip:
                    if archive is not None:
                        _append_dispatch_record(
                            archive,
                            lambda: DispatchRecord(
                                dispatch_id=(
                                    f"dispatch_permanent_skip_v7r{round_id:03d}_"
                                    f"{dependencies.short_id(candidate_id)}_{attempt:02d}"
                                ),
                                launch_id=(
                                    launch_decision.launch_id
                                    if launch_decision is not None
                                    else None
                                ),
                                tick_id=f"v7r{round_id:03d}",
                                candidate_id=candidate_id,
                                status="dispatch_failed",
                                worker_slot=str(worker_slot.slot_id),
                                gpu_id=str(worker_slot.gpu_id),
                                **dependencies.dispatch_record_metadata(
                                    cand=candidate,
                                    launch_decision=launch_decision,
                                ),
                                attempt=attempt,
                                why=dispatch_error or "permanent dispatch skip",
                            ),
                            gpu_id=str(worker_slot.gpu_id),
                            candidate_id=candidate_id,
                        )
                    if retry_counts is not None:
                        retry_counts.pop(candidate_id, None)
                    continue

                if retry_counts is not None:
                    next_retry_count = retry_counts.get(candidate_id, 0) + 1
                    if next_retry_count <= max_dispatch_retries:
                        retry_counts[candidate_id] = next_retry_count
                        retry_later.append(candidate_id)
                    elif archive is not None:
                        _append_dispatch_record(
                            archive,
                            lambda: DispatchRecord(
                                dispatch_id=(
                                    f"dispatch_failed_v7r{round_id:03d}_"
                                    f"{dependencies.short_id(candidate_id)}_"
                                    f"{next_retry_count:02d}"
                                ),
                                launch_id=(
                                    launch_decision.launch_id
                                    if launch_decision is not None
                                    else None
                                ),
                                tick_id=f"v7r{round_id:03d}",
                                candidate_id=candidate_id,
                                status="dispatch_failed",
                                worker_slot=str(worker_slot.slot_id),
                                gpu_id=str(worker_slot.gpu_id),
                                **dependencies.dispatch_record_metadata(
                                    cand=candidate,
                                    launch_decision=launch_decision,
                                ),
                                attempt=next_retry_count,
                                why=(
                                    dispatch_error
                                    or "dispatch_fn returned None after bounded retries"
                                ),
                            ),
                            gpu_id=str(worker_slot.gpu_id),
                            candidate_id=candidate_id,
                        )
                continue

            if len(outcome) == 4:
                process, output_dir, parent_pdb_path, parent_result_id = outcome
                target_chains_csv = ""
                binder_chain = ""
            else:
                (
                    process,
                    output_dir,
                    parent_pdb_path,
                    parent_result_id,
                    target_chains_csv,
                    binder_chain,
                ) = outcome

            worker_slot.proc = process
            worker_slot.cand = candidate
            worker_slot.out_dir = output_dir
            worker_slot.parent_pdb_str = parent_pdb_path
            worker_slot.parent_result_id = parent_result_id
            worker_slot.target_chains_csv = target_chains_csv
            worker_slot.binder_chain = binder_chain
            worker_slot.tick_id = f"v7r{round_id:03d}"
            worker_slot.launched_at = time.time()

            if archive is not None:
                def create_started_dispatch_record() -> DispatchRecord:
                    process_id = (
                        int(getattr(process, "pid"))
                        if getattr(process, "pid", None) is not None
                        else None
                    )
                    return DispatchRecord(
                        dispatch_id=(
                            f"dispatch_started_v7r{round_id:03d}_"
                            f"{dependencies.short_id(candidate_id)}_{attempt:02d}"
                        ),
                        launch_id=(
                            launch_decision.launch_id
                            if launch_decision is not None
                            else None
                        ),
                        tick_id=f"v7r{round_id:03d}",
                        candidate_id=candidate_id,
                        status="started",
                        worker_slot=str(worker_slot.slot_id),
                        gpu_id=str(worker_slot.gpu_id),
                        output_dir=str(output_dir),
                        parent_result_id=parent_result_id,
                        parent_pdb_path=parent_pdb_path,
                        **dependencies.dispatch_record_metadata(
                            cand=candidate,
                            launch_decision=launch_decision,
                        ),
                        worker_pid=process_id,
                        worker_pgid=process_id,
                        worker_pid_start_ticks=dependencies.process_start_ticks(
                            process_id
                        ),
                        worker_host=socket.gethostname(),
                        slurm_job_id=os.environ.get("SLURM_JOB_ID") or None,
                        attempt=attempt,
                        why="worker subprocess started from controller pending queue",
                    )

                _append_dispatch_record(
                    archive,
                    create_started_dispatch_record,
                    gpu_id=str(worker_slot.gpu_id),
                    candidate_id=candidate_id,
                    worker_is_running=True,
                )
            if retry_counts is not None:
                retry_counts.pop(candidate_id, None)
            active_candidate_ids.add(candidate_id)
            dispatched_count += 1
            break

    # Requeue only after scanning every slot; this prevents an immediate retry
    # loop while preserving transient failures for the next controller round.
    pending_candidate_ids.extend(retry_later)
    return dispatched_count


__all__ = [
    "DispatchCandidate",
    "DispatchOutcome",
    "DispatchQueueDependencies",
    "MAX_DISPATCH_RETRIES",
    "PermanentDispatchSkip",
    "dispatch_pending_to_free_slots",
]
