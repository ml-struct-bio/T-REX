"""Independent worker completion, timeout, and drain contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trex.campaign.runtime import WorkerSlot
from trex.execution import (
    WorkerSupervisionDependencies,
    drain_worker_slots,
    supervise_worker_slots,
)
from trex.schemas import ActionCandidate, FeasibilityCheck


def _candidate(method_family: str) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=f"candidate-{method_family}",
        hypothesis_ids=["hypothesis-1"],
        parent_result_id=None,
        method_family=method_family,
        operator_id=f"{method_family}_default",
        lane_id=method_family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="Generate evidence.",
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


class FakeProcess:
    def __init__(
        self,
        *,
        return_code: int | None,
        wait_times_out: bool = False,
    ) -> None:
        self.returncode = return_code
        self.wait_times_out = wait_times_out

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        if self.wait_times_out:
            raise subprocess.TimeoutExpired("worker", timeout)
        return int(self.returncode or 0)


def _worker_slot(
    method_family: str,
    process: FakeProcess,
    *,
    launched_at: float = 100.0,
) -> WorkerSlot:
    worker_slot = WorkerSlot(slot_id=0, gpu_id="1")
    worker_slot.proc = process
    worker_slot.cand = _candidate(method_family)
    worker_slot.out_dir = Path("/tmp/worker-output")
    worker_slot.launched_at = launched_at
    return worker_slot


def _dependencies(
    *,
    parse_worker_output: MagicMock | None = None,
    record_parse_failure: MagicMock | None = None,
    parse_incremental: MagicMock | None = None,
    terminate: MagicMock | None = None,
    hard_ceiling_seconds: float = 10_000.0,
) -> WorkerSupervisionDependencies:
    return WorkerSupervisionDependencies(
        parse_worker_output=parse_worker_output or MagicMock(return_value=0),
        record_parse_failure=record_parse_failure or MagicMock(),
        parse_incremental_bindcraft_output=(
            parse_incremental or MagicMock(return_value=0)
        ),
        count_bindcraft_scoreable_results=MagicMock(return_value=0),
        resolve_hard_ceiling_seconds=lambda method_family: hard_ceiling_seconds,
        terminate_worker_process=terminate or MagicMock(),
    )


def test_completed_worker_is_parsed_and_freed() -> None:
    parse_worker_output = MagicMock(return_value=2)
    worker_slot = _worker_slot("complexa_beam", FakeProcess(return_code=0))

    report = supervise_worker_slots(
        [worker_slot],
        archive=object(),
        target=MagicMock(target_id="target-1"),
        auto_chain_sequence=[0],
        dependencies=_dependencies(parse_worker_output=parse_worker_output),
        clock=lambda: 3_700.0,
    )

    assert report.completed_workers == 1
    assert report.workers_with_new_results == 1
    assert report.archived_results == 2
    assert not worker_slot.busy
    assert parse_worker_output.call_args.kwargs["return_code"] == 0
    assert parse_worker_output.call_args.kwargs["elapsed_gpu_hours"] == pytest.approx(1.0)


def test_parse_failure_is_recorded_and_does_not_hold_slot() -> None:
    parse_worker_output = MagicMock(side_effect=ValueError("malformed output"))
    record_parse_failure = MagicMock()
    worker_slot = _worker_slot("complexa_beam", FakeProcess(return_code=1))

    report = supervise_worker_slots(
        [worker_slot],
        archive=object(),
        target=MagicMock(target_id="target-1"),
        auto_chain_sequence=[0],
        dependencies=_dependencies(
            parse_worker_output=parse_worker_output,
            record_parse_failure=record_parse_failure,
        ),
        clock=lambda: 200.0,
    )

    assert report.completed_workers == 1
    assert report.parse_failures == 1
    assert not worker_slot.busy
    assert "clean-completion" in record_parse_failure.call_args.args[2]


def test_running_bindcraft_is_incrementally_parsed_without_being_freed() -> None:
    parse_incremental = MagicMock(return_value=3)
    worker_slot = _worker_slot("bindcraft", FakeProcess(return_code=None))

    report = supervise_worker_slots(
        [worker_slot],
        archive=object(),
        target=MagicMock(target_id="target-1"),
        auto_chain_sequence=[0],
        dependencies=_dependencies(parse_incremental=parse_incremental),
        clock=lambda: 200.0,
    )

    assert report.incremental_results == 3
    assert report.timed_out_workers == 0
    assert worker_slot.busy
    parse_incremental.assert_called_once()


def test_hard_ceiling_salvages_before_terminating_worker() -> None:
    call_order: list[str] = []
    parse_worker_output = MagicMock(
        side_effect=lambda *args, **kwargs: call_order.append("parse") or 1
    )
    terminate = MagicMock(
        side_effect=lambda process: call_order.append("terminate")
    )
    worker_slot = _worker_slot(
        "complexa_beam",
        FakeProcess(return_code=None),
        launched_at=0.0,
    )

    report = supervise_worker_slots(
        [worker_slot],
        archive=object(),
        target=MagicMock(target_id="target-1"),
        auto_chain_sequence=[0],
        dependencies=_dependencies(
            parse_worker_output=parse_worker_output,
            terminate=terminate,
            hard_ceiling_seconds=100.0,
        ),
        clock=lambda: 200.0,
    )

    assert report.timed_out_workers == 1
    assert report.archived_results == 1
    assert call_order == ["parse", "terminate"]
    assert not worker_slot.busy
    assert parse_worker_output.call_args.kwargs["return_code"] == -1


def test_drain_timeout_preserves_original_exit_slot_reference() -> None:
    call_order: list[str] = []
    parse_worker_output = MagicMock(
        side_effect=lambda *args, **kwargs: call_order.append("parse") or 1
    )
    terminate = MagicMock(
        side_effect=lambda process: call_order.append("terminate")
    )
    worker_slot = _worker_slot(
        "boltzgen",
        FakeProcess(return_code=None, wait_times_out=True),
        launched_at=0.0,
    )

    report = drain_worker_slots(
        [worker_slot],
        grace_seconds_per_slot=60.0,
        archive=object(),
        target=MagicMock(target_id="target-1"),
        auto_chain_sequence=[0],
        dependencies=_dependencies(
            parse_worker_output=parse_worker_output,
            terminate=terminate,
        ),
        clock=lambda: 3_600.0,
    )

    assert report.drained_workers == 1
    assert report.timed_out_workers == 1
    assert report.archived_results == 1
    assert call_order == ["parse", "terminate"]
    assert worker_slot.busy  # Original final drain retains this until process exit.
    assert parse_worker_output.call_args.kwargs["elapsed_gpu_hours"] == pytest.approx(1.0)
