"""Sanity: every dataclass roundtrips to JSON."""

from __future__ import annotations

import json

from trex.schemas import (
    ActionCandidate,
    AxisStat,
    CandidateDecision,
    EvidenceSummary,
    Example,
    FeasibilityCheck,
    HypothesisCard,
    JointPatternCount,
    LLMHealthSummary,
    LaunchDecision,
    MetricCalibration,
    MethodHealthSummary,
    PanelSelection,
    PredictedChange,
    PreserveConstraint,
    ReasoningTrace,
    ResultRecord,
    RouteHealthSummary,
    RouteRecord,
    RouteStageRecord,
    RuntimeBucket,
    SupervisorDecision,
    TargetConstraint,
    to_jsonable,
)


def _roundtrip(record):
    blob = json.dumps(to_jsonable(record))
    parsed = json.loads(blob)
    assert isinstance(parsed, dict)


def test_target_constraint_roundtrip():
    _roundtrip(TargetConstraint(target_id="cd45", target_class="receptor"))


def test_result_record_roundtrip():
    _roundtrip(
        ResultRecord(
            result_id="r1",
            parent_ids=["p1"],
            target_id="cd45",
            backend_family="complexa_beam",
            runtime_bucket_id="rb1",
            metrics={"pLDDT": 85.0, "iPAE": 0.3},
            metrics_calibrated={"pLDDT": 85.0, "iPAE": 0.3},
            route_lineage=[],
            gpu_h=1.2,
            exit_status="ok",
        )
    )


def test_hypothesis_card_roundtrip():
    h = HypothesisCard(
        hypothesis_id="h1",
        target_id="cd45",
        tick_created=1,
        claim="iPAE improves with MPNN redesign",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["e1"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["r1"], 0.2, None)
        ],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
        reasoning_trace=ReasoningTrace(
            observed_signal="near-miss passes pLDDT and scRMSD but fails iPAE",
            inference="interface placement is the blocker",
            action_implication="run a rescue family aimed at lowering iPAE",
        ),
    )
    _roundtrip(h)


def test_candidate_decision_reasoning_trace_roundtrip():
    _roundtrip(
        CandidateDecision(
            candidate_id="c1",
            mode="rescue",
            rank_in_mode=1,
            resource_class="standard",
            what="run refilter",
            why="near-miss is close",
            evidence_refs=["r1"],
            expected_signal="iPAE decreases",
            stop_or_downgrade_if="iPAE does not improve",
            reasoning_trace=ReasoningTrace(
                observed_signal="r1 is an iPAE-only near-miss",
                inference="canonical refilter can test the route",
                action_implication="rank this candidate in rescue",
            ),
        )
    )


def test_runtime_bucket_roundtrip():
    _roundtrip(
        RuntimeBucket(
            bucket_id="rb1",
            container_digest="sha256:abc",
            ckpt_digests={"complexa": "sha256:def"},
            scoring_script_digest="sha256:ghi",
            calibration_version="v1",
            created_at="2026-05-24T00:00:00Z",
        )
    )


def test_route_record_roundtrip():
    _roundtrip(
        RouteRecord(
            route_id="route_1",
            origin_family="BoltzGen",
            origin_artifact_id="art_1",
            parent_quality_stratum="raw_only_external",
            runtime_bucket_id="rb1",
            stages=[
                RouteStageRecord(
                    stage="refilter", input_count=10, converted_count=8, failed_count=2
                )
            ],
            matched_control_route_ids=[],
            credit_status="provisional_until_control",
            stage_credit={"generator": 0.2, "refilter": 0.3},
        )
    )


def test_feasibility_check_all_ok():
    fc = FeasibilityCheck(
        backend_healthy=True,
        runtime_bucket_id="rb1",
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )
    assert fc.all_ok()


def test_feasibility_check_not_ok_without_bucket():
    fc = FeasibilityCheck(
        backend_healthy=True,
        runtime_bucket_id=None,
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )
    assert not fc.all_ok()
