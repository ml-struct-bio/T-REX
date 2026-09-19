"""Direct tests for the focused route-evidence component boundary."""

from __future__ import annotations

from trex.evidence.attribution import _su_key
from trex.evidence.route_identity import canonical_config_signature
from trex.evidence.route_status import (
    classify_marginal_route_status,
    classify_route_status,
)
from trex.evidence.route_values import (
    RouteAccumulator,
    build_route_value_summaries,
)
from trex.evidence_reducer import (
    StateClassifierConfig,
    _route_diagnostic_improvement,
    build_route_values,
)
from trex.schemas import ResultRecord, RouteValueSummary


def _strict_result(result_id: str = "strict_1") -> ResultRecord:
    return ResultRecord(
        result_id=result_id,
        parent_ids=[],
        target_id="target",
        backend_family="complexa_beam",
        runtime_bucket_id="bucket",
        metrics={"pLDDT": 94.0, "iPAE": 0.15, "binder_scRMSD": 0.8},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.25,
        exit_status="ok",
        bins={"foldseek_su": "cluster_1"},
    )


def test_focused_builder_matches_compatibility_entry_point() -> None:
    result = _strict_result()
    config = StateClassifierConfig()

    focused_rows = build_route_value_summaries(
        results=[result],
        window_results=[result],
        spawning_actions={},
        config=config,
        route_diagnostic_improvement=_route_diagnostic_improvement,
    )
    compatibility_rows = build_route_values(
        [result], [result], {}, cfg=config,
    )

    assert focused_rows == compatibility_rows
    assert focused_rows
    assert all(isinstance(row, RouteValueSummary) for row in focused_rows)
    assert _su_key(result) == "cluster_1"
    assert canonical_config_signature({"beam_width": 4}).startswith("beam_width=4#")


def test_route_accumulator_has_isolated_mutable_defaults() -> None:
    identity = dict(
        strategy_key="route::complexa",
        root_family="complexa_beam",
        action_family="complexa_beam",
        scoring_family=None,
        operator_id="complexa_beam",
        config_signature="default",
        refilter_role=None,
        config_delta={},
        parent_strategy_key=None,
    )
    first = RouteAccumulator(**identity)
    second = RouteAccumulator(**identity)

    first.su_bins.add("cluster_1")
    first.evidence_refs.append("result_1")

    assert second.su_bins == set()
    assert second.evidence_refs == []


def test_route_status_policy_is_directly_reviewable() -> None:
    config = StateClassifierConfig()
    common = dict(
        config=config,
        route_gpu_h=1.0,
        new_su=0,
        recent_su=0,
        medium_recent_su=0,
        near_recent=0,
        recent_route_gpu_h=1.0,
        rate=None,
        best_rate=0.0,
        strict_per_su=None,
        duplicate_bin_fraction=None,
    )

    assert classify_route_status(
        **common,
        family="complexa_beam",
        pending_promising_score_conversion_count=1,
    ) == "awaiting_score_conversion"
    assert classify_route_status(
        **common,
        family="structure_refilter",
    ) == "plumbing"


def test_marginal_route_status_preserves_under_tested_gate() -> None:
    config = StateClassifierConfig(route_zero_su_defer_gpu_h=3.0)

    status = classify_marginal_route_status(
        config=config,
        route_gpu_h=0.25,
        new_su=0,
        strict_per_su=None,
        duplicate_bin_fraction=None,
        lifetime_rate=None,
        record_recent_su=0,
        record_recent_rate=None,
        gpu_recent_su=0,
        gpu_recent_gpu_h=0.25,
        gpu_recent_rate=None,
        medium_recent_su=0,
        near_recent=0,
        family="complexa_beam",
    )

    assert status == "under_tested"
