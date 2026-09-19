"""Independent contracts for durable between-tick progress reporting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from trex.execution.progress_checkpoint import (
    ProgressCheckpointDependencies,
    ProgressCheckpointRequest,
    build_cheap_checkpoint_evidence,
    refresh_progress_checkpoint,
)
from trex.schemas import ActionCandidate, EvidenceSummary


class FakeArchive:
    root = Path("/unused")

    def iter_records(self, record_type: type):
        if record_type in (ActionCandidate, EvidenceSummary):
            return iter(())
        raise AssertionError(f"unexpected archive record type: {record_type}")


def _request(**overrides) -> ProgressCheckpointRequest:
    values = {
        "archive": FakeArchive(),
        "target": SimpleNamespace(target_id="target-1"),
        "worker_slots": (),
        "pending_candidate_ids": (),
        "live_tick_config": object(),
        "round_id": 2,
        "elapsed_wall_hours": 1.5,
        "maximum_wall_hours": 4.0,
        "elapsed_offset_hours": 1.0,
        "controller_started_monotonic": 100.0,
        "worker_gpu_count": 3,
        "latest_state": "stalled",
        "last_checkpoint_at": 0.0,
        "interval_seconds": 60.0,
    }
    values.update(overrides)
    return ProgressCheckpointRequest(**values)


def _dependencies(**overrides) -> ProgressCheckpointDependencies:
    values = {
        "score_conversion_feedback_inflight": MagicMock(return_value=False),
        "summarize_pending_family_load": MagicMock(return_value={}),
        "run_live_tick": MagicMock(return_value={"evidence": {}}),
        "charged_gpu_count": MagicMock(return_value=4.0),
        "build_cheap_evidence": build_cheap_checkpoint_evidence,
        "write_checkpoint": MagicMock(return_value=Path("checkpoint.json")),
    }
    values.update(overrides)
    return ProgressCheckpointDependencies(**values)


def test_checkpoint_skips_work_until_interval_is_due() -> None:
    feedback_inflight = MagicMock()
    dependencies = _dependencies(
        score_conversion_feedback_inflight=feedback_inflight
    )

    result = refresh_progress_checkpoint(
        _request(last_checkpoint_at=10.0),
        dependencies,
        wall_clock=lambda: 69.0,
    )

    assert not result.wrote_checkpoint
    assert result.last_checkpoint_at == 10.0
    assert result.latest_state == "stalled"
    feedback_inflight.assert_not_called()


def test_cheap_checkpoint_updates_wall_accounting_and_completion_cadence() -> None:
    write_checkpoint = MagicMock(return_value=Path("checkpoint.json"))
    dependencies = _dependencies(write_checkpoint=write_checkpoint)
    wall_times = iter((100.0, 160.0))

    result = refresh_progress_checkpoint(
        _request(),
        dependencies,
        wall_clock=lambda: next(wall_times),
        monotonic_clock=lambda: 3_700.0,
    )

    assert result.wrote_checkpoint
    assert result.last_checkpoint_at == 160.0
    assert result.latest_state == "stalled"
    assert result.evidence is not None
    assert result.evidence["charged_gpu_h_total"] == 6.0
    assert result.evidence["worker_wall_gpu_h_total"] == 4.5
    assert write_checkpoint.call_args.kwargs["elapsed_wall_h"] == 2.0
    assert write_checkpoint.call_args.kwargs["remaining_wall_h"] == 2.0


def test_score_conversion_checkpoint_refreshes_evidence_without_llm_tick() -> None:
    worker_slot = SimpleNamespace(busy=True, launched_at=100.0)
    summarize_pending_load = MagicMock(return_value={"by_family": {}})
    run_live_tick = MagicMock(return_value={
        "evidence": {"state_label": "productive", "run_su_count": 3}
    })
    dependencies = _dependencies(
        score_conversion_feedback_inflight=MagicMock(return_value=True),
        summarize_pending_family_load=summarize_pending_load,
        run_live_tick=run_live_tick,
    )
    wall_times = iter((3_700.0, 3_760.0))

    result = refresh_progress_checkpoint(
        _request(worker_slots=(worker_slot,), pending_candidate_ids=("queued-1",)),
        dependencies,
        wall_clock=lambda: next(wall_times),
        monotonic_clock=lambda: 3_700.0,
    )

    assert result.wrote_checkpoint
    assert result.latest_state == "productive"
    summarize_pending_load.assert_called_once()
    tick_arguments = run_live_tick.call_args.kwargs
    assert tick_arguments["tick_id"] == "v7checkpoint002"
    assert tick_arguments["evidence_only"] is True
    assert tick_arguments["pending_children"] == 1
    assert tick_arguments["available_slots_override"] == 1
    assert tick_arguments["inflight_gpu_h"] == 1.0


def test_checkpoint_failure_is_retryable_and_keeps_prior_state() -> None:
    dependencies = _dependencies(
        score_conversion_feedback_inflight=MagicMock(
            side_effect=RuntimeError("archive temporarily unavailable")
        )
    )

    result = refresh_progress_checkpoint(
        _request(last_checkpoint_at=12.0),
        dependencies,
        wall_clock=lambda: 100.0,
    )

    assert not result.wrote_checkpoint
    assert result.last_checkpoint_at == 12.0
    assert result.latest_state == "stalled"
    assert result.error == "RuntimeError: archive temporarily unavailable"
