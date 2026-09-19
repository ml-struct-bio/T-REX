"""Independent contracts for the worker dispatch state machine."""

from __future__ import annotations

from pathlib import Path

from trex.campaign.runtime import WorkerSlot
from trex.execution import (
    DispatchQueueDependencies,
    MAX_DISPATCH_RETRIES,
    PermanentDispatchSkip,
    dispatch_pending_to_free_slots,
)
from trex.schemas import ActionCandidate, FeasibilityCheck


def _candidate(candidate_id: str = "candidate-1") -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        hypothesis_ids=["hypothesis-1"],
        parent_result_id=None,
        method_family="bindcraft",
        operator_id="bindcraft_default",
        lane_id="bindcraft",
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="Generate a binder candidate.",
        evidence_refs=["evidence-1"],
        feasibility=FeasibilityCheck(
            backend_healthy=True,
            runtime_bucket_id="runtime-v1",
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
        ),
    )


def _dependencies() -> DispatchQueueDependencies:
    return DispatchQueueDependencies(
        prioritize_pending_queue=lambda archive, pending, candidates: None,
        high_cost_defer_reason=lambda slots, candidate, archive: None,
        latest_launch_decision=lambda archive, candidate_id: None,
        dispatch_record_metadata=lambda **kwargs: {},
        short_id=lambda value: value[:10],
        process_start_ticks=lambda process_id: 123 if process_id else None,
    )


def test_dispatch_api_is_exported_from_execution_package() -> None:
    assert MAX_DISPATCH_RETRIES == 3
    assert issubclass(PermanentDispatchSkip, RuntimeError)
    assert callable(dispatch_pending_to_free_slots)


def test_audit_failure_does_not_lose_a_running_worker(tmp_path: Path) -> None:
    """Process metadata is best-effort after the worker has already started."""

    class ProcessWithUnreadablePid:
        @property
        def pid(self) -> int:
            raise OSError("process metadata temporarily unavailable")

        def poll(self) -> None:
            return None

    class Archive:
        def append(self, record: object) -> None:
            raise AssertionError("unreachable when record construction fails")

    candidate = _candidate()
    worker_slot = WorkerSlot(slot_id=0, gpu_id="1")

    started = dispatch_pending_to_free_slots(
        worker_slots=[worker_slot],
        pending_candidate_ids=[candidate.candidate_id],
        candidates_by_id={candidate.candidate_id: candidate},
        archive=Archive(),
        target=None,
        target_pdb=None,
        archive_root=tmp_path,
        round_id=1,
        dispatch_candidate=lambda candidate, **kwargs: (
            ProcessWithUnreadablePid(),
            tmp_path / "worker-output",
            "",
            "",
        ),
        dependencies=_dependencies(),
    )

    assert started == 1
    assert worker_slot.busy
    assert worker_slot.cand is candidate
