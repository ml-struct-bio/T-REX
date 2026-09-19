"""Contracts for pre-tick archive reads and bounded trajectory context."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from trex.archive import Archive
from trex.schemas import (
    ActionCandidate,
    DispatchRecord,
    FeasibilityCheck,
    HypothesisCard,
    LLMCallRecord,
    LaunchDecision,
    PredictedChange,
    PreserveConstraint,
    ResultRecord,
    SupervisorDecision,
)
from trex.tick import (
    build_recent_tick_trajectory,
    load_tick_archive_snapshot,
    recent_fallback_rate,
    resolve_warmstart_coverage,
    summarize_llm_health,
)


def _action(
    candidate_id: str,
    family: str,
    *,
    config_delta: dict | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        hypothesis_ids=["hypothesis-1"],
        parent_result_id=None,
        method_family=family,
        operator_id=f"{family}_default",
        lane_id=family,
        config_delta=config_delta or {},
        downstream_route_plan=[],
        estimated_cost_class="standard",
        expected_signal="Generate evidence.",
        evidence_refs=["baseline"],
        feasibility=FeasibilityCheck(
            backend_healthy=True,
            runtime_bucket_id="runtime-v1",
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
        ),
    )


def _result(
    result_id: str,
    *,
    target_id: str = "target-1",
    parent_ids: list[str] | None = None,
) -> ResultRecord:
    return ResultRecord(
        result_id=result_id,
        parent_ids=parent_ids or [],
        target_id=target_id,
        backend_family="complexa_beam",
        runtime_bucket_id="runtime-v1",
        metrics={},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.5,
        exit_status="ok",
    )


def _hypothesis(
    hypothesis_id: str,
    *,
    target_id: str = "target-1",
    tick_created: int = 1,
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        target_id=target_id,
        tick_created=tick_created,
        claim="Canonical scoring improves interface confidence.",
        mode_affinity={"exploit": 0.2, "rescue": 0.6, "explore": 0.2},
        evidence_refs=["baseline"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["baseline"], 0.2, None)
        ],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["complexa_beam"],
    )


def _launch(
    candidate_id: str,
    *,
    status: str = "launched",
) -> LaunchDecision:
    return LaunchDecision(
        launch_id=f"launch-{candidate_id}-{status}",
        tick_id="tick-1",
        candidate_id=candidate_id,
        status=status,  # type: ignore[arg-type]
        resource_class_concrete={},
        why="test",
    )


def _llm_call(
    call_id: str,
    status: str,
    latency_s: float,
    *,
    confidence: float | None = None,
    abstain: bool = False,
    fallback: bool = False,
) -> LLMCallRecord:
    return LLMCallRecord(
        call_id=call_id,
        tick_id="tick-1",
        role="planner",
        model="model-1",
        model_digest=None,
        prompt_hash=f"hash-{call_id}",
        schema_version="v1",
        latency_s=latency_s,
        tokens_in=10,
        tokens_out=5,
        parse_status=status,  # type: ignore[arg-type]
        confidence=confidence,
        abstain=abstain,
        fallback_triggered=fallback,
    )


def test_archive_snapshot_preserves_target_and_latest_row_views(
    tmp_path: Path,
) -> None:
    archive = Archive(tmp_path / "archive")
    warmstart = _action("warmstart_complexa", "complexa_beam")
    rejected = _action("candidate-rejected", "bindcraft")
    archive.append(warmstart)
    archive.append(rejected)
    archive.append(
        _result("result-target", parent_ids=["unknown", warmstart.candidate_id])
    )
    archive.append(
        _result(
            "result-other-target",
            target_id="target-2",
            parent_ids=[warmstart.candidate_id],
        )
    )

    first_revision = _hypothesis("hypothesis-revised")
    active = _hypothesis("hypothesis-active", tick_created=2)
    archive.append(first_revision)
    archive.append(_hypothesis("hypothesis-other", target_id="target-2"))
    archive.append(replace(first_revision, status="supported"))
    archive.append(active)
    archive.append(_launch(warmstart.candidate_id))
    archive.append(_launch(rejected.candidate_id, status="rejected"))
    archive.append(DispatchRecord(
        dispatch_id="dispatch-started",
        tick_id="tick-1",
        candidate_id=warmstart.candidate_id,
        status="started",
    ))
    archive.append(DispatchRecord(
        dispatch_id="dispatch-failed",
        tick_id="tick-1",
        candidate_id=rejected.candidate_id,
        status="dispatch_failed",
    ))
    archive.append(SupervisorDecision(
        tick_id="tick-1",
        mode_mixture={"exploit": 1.0},
        candidate_decisions=[],
        clamps_applied=[],
        fallback_used=False,
        rationale="test",
    ))
    archive.append(_llm_call("call-1", "ok", 1.0))

    snapshot = load_tick_archive_snapshot(archive, "target-1")

    assert [result.result_id for result in snapshot.target_results] == [
        "result-target"
    ]
    assert [action.candidate_id for action in snapshot.actions] == [
        "warmstart_complexa",
        "candidate-rejected",
    ]
    assert [
        (hypothesis.hypothesis_id, hypothesis.status)
        for hypothesis in snapshot.latest_target_hypotheses
    ] == [
        ("hypothesis-revised", "supported"),
        ("hypothesis-active", "active"),
    ]
    assert snapshot.lifecycle_hypotheses == tuple(
        archive.retrieve_active_hypotheses(
            target_id="target-1",
            limit=None,
            include_terminal=False,
        )
    )
    assert snapshot.active_hypotheses == tuple(
        archive.retrieve_active_hypotheses(
            target_id="target-1",
            limit=8,
            include_terminal=True,
        )
    )
    assert snapshot.launched_decisions == (_launch(warmstart.candidate_id),)
    assert len(snapshot.dispatch_records) == 2
    assert snapshot.evidence_records == ()
    assert snapshot.prior_target_evidence == ()
    assert len(snapshot.supervisor_decisions) == 1
    assert len(snapshot.llm_call_records) == 1
    assert snapshot.spawning_action_by_result_id == {
        "result-target": warmstart
    }


def test_warmstart_coverage_reports_completed_missing_and_seen() -> None:
    actions = (
        _action("warmstart_complexa", "complexa_beam"),
        _action("candidate-bindcraft", "bindcraft"),
    )
    coverage = resolve_warmstart_coverage(
        actions=actions,
        launched_decisions=(_launch("warmstart_complexa"),),
        required_families=("complexa_beam", "bindcraft", "boltzgen"),
        unavailable_families=("boltzgen",),
    )

    assert coverage.completed_families == frozenset({"complexa_beam"})
    assert coverage.missing_families == frozenset({"bindcraft"})
    assert not coverage.seen

    complete = resolve_warmstart_coverage(
        actions=actions,
        launched_decisions=(_launch("warmstart_complexa"),),
        required_families=("complexa_beam", "boltzgen"),
        unavailable_families=("boltzgen",),
    )
    assert complete.seen
    assert not complete.missing_families


def test_trajectory_uses_started_dispatches_and_bounded_sorted_knobs() -> None:
    actions = (
        _action(
            "candidate-a",
            "bindcraft",
            config_delta={
                "zeta": 6,
                "beta": 2,
                "alpha": 1,
                "gamma": 3,
                "epsilon": 5,
                "delta": 4,
            },
        ),
        _action("candidate-b", "bindcraft", config_delta={"temperature": 0.2}),
        _action("candidate-failed", "complexa_beam"),
    )
    dispatches = (
        DispatchRecord("d-a", "tick-2", "candidate-a", "started"),
        DispatchRecord("d-b", "tick-2", "candidate-b", "started"),
        DispatchRecord(
            "d-failed",
            "tick-2",
            "candidate-failed",
            "dispatch_failed",
        ),
    )
    evidence = [
        SimpleNamespace(
            tick_id=f"tick-{index}",
            state_label="productive" if index == 2 else "low_evidence",
            strict_count=index,
            run_su_count=index + 1,
            run_su_count_delta=1,
            near_miss_count=index + 2,
        )
        for index in range(3)
    ]

    trajectory = build_recent_tick_trajectory(
        prior_evidence=evidence,  # type: ignore[arg-type]
        actions=actions,
        dispatch_records=dispatches,
        max_ticks=2,
    )

    assert [entry["tick_id"] for entry in trajectory] == ["tick-1", "tick-2"]
    assert trajectory[0]["launches"] == []
    assert trajectory[1]["n_launches"] == 2
    assert trajectory[1]["dominant_family"] == "bindcraft"
    assert trajectory[1]["launches"][0] == {
        "family": "bindcraft",
        "key_knobs": ["alpha", "beta", "delta", "epsilon", "gamma"],
        "config_delta": {
            "alpha": 1,
            "beta": 2,
            "delta": 4,
            "epsilon": 5,
            "gamma": 3,
        },
    }


def test_llm_history_helpers_preserve_fallback_and_health_semantics() -> None:
    calls = (
        _llm_call("ok", "ok", 1.0),
        _llm_call("parse", "parse_fail", 2.0, fallback=True),
        _llm_call("schema", "schema_fail", 3.0),
        _llm_call("low", "ok", 4.0, confidence=0.2, fallback=True),
        _llm_call("abstain", "ok", 5.0, abstain=True),
    )

    assert recent_fallback_rate(calls, window=4) == 0.75
    health = summarize_llm_health(calls, "model-1", window=5)
    assert health.last_calls_window == [
        "ok",
        "parse_fail",
        "schema_fail",
        "low_conf",
        "abstain",
    ]
    assert health.parse_fail_rate == 0.2
    assert health.schema_fail_rate == 0.2
    assert health.timeout_count == 0
    assert health.median_latency_s == 3.0
