"""Contracts for deterministic canonical score-conversion scheduling."""

from __future__ import annotations

from pathlib import Path

from trex.capability_registry import default_registry
from trex.execution import (
    ScoreConversionSchedulingRequest,
    build_score_conversion_schedule,
    select_score_conversion_parents,
)
from trex.refilter_roles import CANONICAL_SCORE_CONVERSION
from trex.schemas import ActionCandidate, FeasibilityCheck, ResultRecord


def _source_candidate(
    method_family: str = "boltzgen",
    *,
    downstream_method_families: list[str] | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id="source-candidate",
        hypothesis_ids=["hypothesis-1"],
        parent_result_id="source-parent",
        method_family=method_family,
        operator_id="source-operator",
        lane_id="source-lane",
        config_delta={},
        downstream_route_plan=(
            downstream_method_families
            if downstream_method_families is not None
            else ["structure_refilter"]
        ),
        estimated_cost_class="standard",
        expected_signal="generate diagnostic structures",
        evidence_refs=["source-parent"],
        feasibility=FeasibilityCheck(
            backend_healthy=True,
            runtime_bucket_id="runtime-1",
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
        ),
        baseline_result_id="baseline-1",
    )


def _result(
    result_id: str,
    structure_path: Path,
    *,
    method_family: str = "boltzgen",
    metrics: dict[str, float] | None = None,
    bins: dict[str, str] | None = None,
) -> ResultRecord:
    return ResultRecord(
        result_id=result_id,
        parent_ids=["source-candidate"],
        target_id="target-1",
        backend_family=method_family,
        runtime_bucket_id="runtime-1",
        metrics=metrics or {},
        metrics_calibrated={},
        route_lineage=[method_family],
        gpu_h=0.1,
        exit_status="ok",
        bins=bins or {},
        artifacts={"pdb_path": str(structure_path)},
    )


def test_schedule_exposes_candidates_counts_and_sequence_state(
    tmp_path: Path,
) -> None:
    lower_scoring_result = _result("result-low", tmp_path / "low.pdb")
    higher_scoring_result = _result("result-high", tmp_path / "high.pdb")
    scores_by_result_id = {"result-low": 1.0, "result-high": 5.0}

    schedule = build_score_conversion_schedule(
        ScoreConversionSchedulingRequest(
            source_candidate=_source_candidate(),
            result_records=(lower_scoring_result, higher_scoring_result),
            tick_id="tick-7",
            starting_sequence_number=12,
        ),
        score_parent_result=lambda result: scores_by_result_id[result.result_id],
    )

    assert schedule.skip_reason is None
    assert schedule.eligible_parent_result_count == 2
    assert schedule.selected_parent_result_count == 2
    assert schedule.scheduled_candidate_count == 2
    assert schedule.final_sequence_number == 14
    assert [
        candidate.candidate_id for candidate in schedule.scheduled_candidates
    ] == [
        "chain_tick-7_boltzgen_to_structure_refilter_013",
        "chain_tick-7_boltzgen_to_structure_refilter_014",
    ]
    assert [
        candidate.parent_result_id
        for candidate in schedule.scheduled_candidates
    ] == ["result-high", "result-low"]
    for candidate in schedule.scheduled_candidates:
        assert candidate.method_family == "structure_refilter"
        assert candidate.refilter_role == CANONICAL_SCORE_CONVERSION
        assert candidate.baseline_result_id == "baseline-1"
        assert candidate.evidence_refs == [candidate.parent_result_id]


def test_direct_complexa_with_canonical_metrics_is_not_rescheduled(
    tmp_path: Path,
) -> None:
    canonical_result = _result(
        "complexa-result",
        tmp_path / "complexa.pdb",
        method_family="complexa_beam",
        metrics={"pLDDT": 94.0, "iPAE": 0.12, "binder_scRMSD": 0.8},
    )

    schedule = build_score_conversion_schedule(
        ScoreConversionSchedulingRequest(
            source_candidate=_source_candidate("complexa_beam"),
            result_records=(canonical_result,),
            tick_id="tick-8",
            starting_sequence_number=3,
        ),
        score_parent_result=lambda result: (_ for _ in ()).throw(
            AssertionError(f"canonical result was unexpectedly scored: {result}")
        ),
    )

    assert schedule.scheduled_candidates == ()
    assert schedule.final_sequence_number == 3
    assert schedule.skip_reason == "canonical_score_conversion_not_required"


def test_boltzgen_aggregate_result_reports_no_eligible_parent(
    tmp_path: Path,
) -> None:
    aggregate_result = _result(
        "aggregate-result",
        tmp_path / "design.cif",
        bins={"boltzgen_design_id": "design"},
    )

    schedule = build_score_conversion_schedule(
        ScoreConversionSchedulingRequest(
            source_candidate=_source_candidate(),
            result_records=(aggregate_result,),
            tick_id="tick-9",
            starting_sequence_number=0,
        ),
        score_parent_result=lambda result: 1.0,
    )

    assert schedule.scheduled_candidate_count == 0
    assert schedule.eligible_parent_result_count == 0
    assert schedule.skip_reason == "no_eligible_parent_results"


def test_unavailable_downstream_family_is_visible_in_schedule(
    tmp_path: Path,
) -> None:
    registry = default_registry().with_override(
        "structure_refilter",
        availability="unavailable",
    )
    result_record = _result("result-1", tmp_path / "result.pdb")

    schedule = build_score_conversion_schedule(
        ScoreConversionSchedulingRequest(
            source_candidate=_source_candidate(),
            result_records=(result_record,),
            tick_id="tick-10",
            starting_sequence_number=4,
        ),
        score_parent_result=lambda result: 1.0,
        capability_registry=registry,
    )

    assert schedule.scheduled_candidate_count == 0
    assert schedule.final_sequence_number == 4
    assert schedule.eligible_parent_result_count == 1
    assert schedule.selected_parent_result_count == 1
    assert schedule.unavailable_downstream_method_families == (
        "structure_refilter",
    )
    assert schedule.skip_reason == "no_available_downstream_method_family"


def test_parent_selection_deduplicates_the_same_structure_path(
    tmp_path: Path,
) -> None:
    shared_path = tmp_path / "shared.pdb"
    lower_scoring_result = _result("duplicate-low", shared_path)
    higher_scoring_result = _result("duplicate-high", shared_path)

    selected_results = select_score_conversion_parents(
        [(1.0, lower_scoring_result), (5.0, higher_scoring_result)],
        candidate_limit=2,
    )

    assert [result.result_id for _, result in selected_results] == [
        "duplicate-high"
    ]
