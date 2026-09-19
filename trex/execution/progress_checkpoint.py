"""Durable controller progress checkpoints between scientific planning ticks."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from ..archive import Archive
from ..campaign.runtime import WorkerSlot
from ..schemas import ActionCandidate, EvidenceSummary, TargetConstraint

if TYPE_CHECKING:
    from ..tick.config import LiveTickConfig
else:
    LiveTickConfig = Any


CONTROLLER_CHECKPOINT_FILE = "controller_checkpoint.json"


def read_controller_checkpoint(archive: Archive) -> dict[str, Any] | None:
    """Read a valid atomic progress checkpoint, tolerating legacy archives."""

    path = archive.root / CONTROLLER_CHECKPOINT_FILE
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        elapsed_wall_hours = float(payload.get("elapsed_wall_h", -1.0))
        round_id = int(payload.get("round_id", 0))
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(elapsed_wall_hours)
        or elapsed_wall_hours < 0.0
        or round_id < 0
    ):
        return None
    return payload


def write_controller_checkpoint(
    archive: Archive,
    *,
    target_id: str,
    round_id: int,
    elapsed_wall_h: float,
    remaining_wall_h: float,
    evidence: dict[str, Any],
) -> Path:
    """Atomically persist wall-budget and evidence progress outside JSONL history."""

    payload = {
        "schema_version": "v7_controller_checkpoint_v1",
        "target_id": str(target_id),
        "round_id": max(0, int(round_id)),
        "tick_id": f"v7checkpoint{max(0, int(round_id)):03d}",
        "elapsed_wall_h": max(0.0, float(elapsed_wall_h)),
        "remaining_wall_h": max(0.0, float(remaining_wall_h)),
        "updated_at_unix": time.time(),
        "evidence": dict(evidence or {}),
    }
    path = archive.root / CONTROLLER_CHECKPOINT_FILE
    temporary_path = archive.root / (
        f".{CONTROLLER_CHECKPOINT_FILE}.{os.getpid()}.tmp"
    )
    try:
        with open(temporary_path, "w") as checkpoint_file:
            json.dump(payload, checkpoint_file, sort_keys=True)
            checkpoint_file.write("\n")
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return path


def resolve_charged_gpu_count(default: float = 4.0) -> float:
    """Return the billable GPU count used for cheap checkpoint accounting."""

    raw_count = os.environ.get("TREX_CHARGED_GPUS")
    if raw_count:
        try:
            parsed_count = float(raw_count)
            if math.isfinite(parsed_count) and parsed_count > 0.0:
                return parsed_count
        except (TypeError, ValueError):
            pass
    return max(0.0, float(default))


def build_cheap_checkpoint_evidence(
    latest_summary: EvidenceSummary | None,
    *,
    latest_state: str | None,
    elapsed_wall_h: float,
    remaining_wall_h: float,
    charged_gpu_count: float,
    worker_wall_gpu_count: float,
) -> dict[str, Any]:
    """Build self-contained checkpoint evidence without a reducer pass."""

    elapsed_wall_hours = max(0.0, float(elapsed_wall_h))
    remaining_wall_hours = max(0.0, float(remaining_wall_h))
    charged_gpu_count = max(0.0, float(charged_gpu_count))
    worker_wall_gpu_count = max(0.0, float(worker_wall_gpu_count))
    charged_gpu_hours = (
        elapsed_wall_hours * charged_gpu_count if charged_gpu_count else None
    )
    worker_wall_gpu_hours = (
        elapsed_wall_hours * worker_wall_gpu_count
        if worker_wall_gpu_count
        else None
    )
    run_su_count = int(getattr(latest_summary, "run_su_count", 0) or 0)

    evidence: dict[str, Any] = {
        "state_label": (
            getattr(latest_summary, "state_label", None)
            if latest_summary is not None
            else latest_state
        ),
        "run_su_count": run_su_count,
        "elapsed_wall_h": elapsed_wall_hours,
        "remaining_wall_h": remaining_wall_hours,
        "charged_gpu_count": charged_gpu_count or None,
        "charged_gpu_h_total": charged_gpu_hours,
        "worker_wall_gpu_count": worker_wall_gpu_count or None,
        "worker_wall_gpu_h_total": worker_wall_gpu_hours,
    }
    if charged_gpu_hours and charged_gpu_hours > 0.0:
        evidence["run_su_per_charged_gpu_h_total"] = (
            run_su_count / charged_gpu_hours
        )
    if worker_wall_gpu_hours and worker_wall_gpu_hours > 0.0:
        evidence["run_su_per_worker_wall_gpu_h_total"] = (
            run_su_count / worker_wall_gpu_hours
        )
    return evidence


@dataclass(frozen=True)
class ProgressCheckpointRequest:
    """Runtime state needed to decide and write one progress checkpoint."""

    archive: Archive
    target: TargetConstraint
    worker_slots: Sequence[WorkerSlot]
    pending_candidate_ids: Sequence[str]
    live_tick_config: LiveTickConfig
    round_id: int
    elapsed_wall_hours: float
    maximum_wall_hours: float
    elapsed_offset_hours: float
    controller_started_monotonic: float
    worker_gpu_count: int
    latest_state: str | None
    last_checkpoint_at: float
    interval_seconds: float = 60.0


@dataclass(frozen=True)
class ProgressCheckpointDependencies:
    """Controller callbacks needed for checkpoint evidence refresh."""

    score_conversion_feedback_inflight: Callable[
        [Archive, Sequence[WorkerSlot]], bool
    ]
    summarize_pending_family_load: Callable[
        [Sequence[WorkerSlot], Sequence[str], dict[str, ActionCandidate]],
        dict[str, Any],
    ]
    run_live_tick: Callable[..., dict[str, Any]]
    charged_gpu_count: Callable[[], float] = resolve_charged_gpu_count
    build_cheap_evidence: Callable[..., dict[str, Any]] = (
        build_cheap_checkpoint_evidence
    )
    write_checkpoint: Callable[..., Path] = write_controller_checkpoint


@dataclass(frozen=True)
class ProgressCheckpointResult:
    """Updated checkpoint cadence and monitoring state."""

    last_checkpoint_at: float
    latest_state: str | None
    wrote_checkpoint: bool
    evidence: dict[str, Any] | None = None
    error: str | None = None


def refresh_progress_checkpoint(
    request: ProgressCheckpointRequest,
    dependencies: ProgressCheckpointDependencies,
    *,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> ProgressCheckpointResult:
    """Write a due progress checkpoint without adding a scientific tick.

    Canonical score conversion requires a current evidence refresh.  During
    other long-running work, the latest summary is reused and only inexpensive
    wall-time accounting is refreshed.  Errors are reported in the typed result
    so the controller can keep running and retry on its next poll.
    """

    checkpoint_started_at = wall_clock()
    if (
        checkpoint_started_at - request.last_checkpoint_at
        < request.interval_seconds
    ):
        return ProgressCheckpointResult(
            last_checkpoint_at=request.last_checkpoint_at,
            latest_state=request.latest_state,
            wrote_checkpoint=False,
        )

    try:
        conversion_feedback_inflight = (
            dependencies.score_conversion_feedback_inflight(
                request.archive, request.worker_slots
            )
        )
        if conversion_feedback_inflight:
            inflight_gpu_hours = sum(
                max(0.0, (checkpoint_started_at - slot.launched_at) / 3600.0)
                for slot in request.worker_slots
                if slot.busy and slot.launched_at > 0
            )
            candidates_by_id = {
                candidate.candidate_id: candidate
                for candidate in request.archive.iter_records(ActionCandidate)
            }
            pending_family_load = dependencies.summarize_pending_family_load(
                request.worker_slots,
                request.pending_candidate_ids,
                candidates_by_id,
            )
            checkpoint_summary = dependencies.run_live_tick(
                request.archive,
                request.target,
                tick_id=f"v7checkpoint{max(request.round_id, 1):03d}",
                tick_id_int=max(request.round_id, 1),
                elapsed_wall_h=request.elapsed_wall_hours,
                remaining_wall_h=max(
                    0.0,
                    request.maximum_wall_hours - request.elapsed_wall_hours,
                ),
                pending_children=sum(
                    1 for slot in request.worker_slots if slot.busy
                ),
                cfg=request.live_tick_config,
                available_slots_override=max(
                    1,
                    sum(1 for slot in request.worker_slots if not slot.busy),
                ),
                inflight_gpu_h=inflight_gpu_hours,
                pending_family_load=pending_family_load,
                evidence_only=True,
            )
            evidence = dict(checkpoint_summary.get("evidence", {}))
        else:
            summaries = list(request.archive.iter_records(EvidenceSummary))
            latest_summary = summaries[-1] if summaries else None
            evidence = dependencies.build_cheap_evidence(
                latest_summary,
                latest_state=request.latest_state,
                elapsed_wall_h=request.elapsed_wall_hours,
                remaining_wall_h=max(
                    0.0,
                    request.maximum_wall_hours - request.elapsed_wall_hours,
                ),
                charged_gpu_count=dependencies.charged_gpu_count(),
                worker_wall_gpu_count=request.worker_gpu_count,
            )

        checkpoint_elapsed_hours = request.elapsed_offset_hours + (
            monotonic_clock() - request.controller_started_monotonic
        ) / 3600.0
        dependencies.write_checkpoint(
            request.archive,
            target_id=request.target.target_id,
            round_id=request.round_id,
            elapsed_wall_h=checkpoint_elapsed_hours,
            remaining_wall_h=max(
                0.0,
                request.maximum_wall_hours - checkpoint_elapsed_hours,
            ),
            evidence=evidence,
        )
        state_label = evidence.get("state_label")
        latest_state = str(state_label) if state_label else request.latest_state
        checkpoint_completed_at = wall_clock()
        print(
            "  [controller_checkpoint] "
            f"elapsed={checkpoint_elapsed_hours:.3f}h "
            f"SU={evidence.get('run_su_count')} "
            f"state={evidence.get('state_label')}",
            flush=True,
        )
        return ProgressCheckpointResult(
            last_checkpoint_at=checkpoint_completed_at,
            latest_state=latest_state,
            wrote_checkpoint=True,
            evidence=evidence,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print(f"  [controller_checkpoint_err] {error}", flush=True)
        return ProgressCheckpointResult(
            last_checkpoint_at=request.last_checkpoint_at,
            latest_state=request.latest_state,
            wrote_checkpoint=False,
            error=error,
        )


__all__ = [
    "CONTROLLER_CHECKPOINT_FILE",
    "ProgressCheckpointDependencies",
    "ProgressCheckpointRequest",
    "ProgressCheckpointResult",
    "build_cheap_checkpoint_evidence",
    "read_controller_checkpoint",
    "resolve_charged_gpu_count",
    "refresh_progress_checkpoint",
    "write_controller_checkpoint",
]
