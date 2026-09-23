"""Synthetic campaign evidence used by deterministic regression tests."""
from __future__ import annotations
from trex import SCHEMA_VERSION
from trex.schemas import (AxisStat, Example, Recipe, EvidenceSummary, JointPatternCount, LLMHealthSummary, MethodHealthSummary, RouteHealthSummary)

def _llm_health_default(model: str) -> LLMHealthSummary:
    return LLMHealthSummary(
        model=model,
        last_calls_window=[],
        parse_fail_rate=0.0,
        schema_fail_rate=0.0,
        timeout_count=0,
        median_latency_s=0.0,
    )


def _route_health_default() -> RouteHealthSummary:
    return RouteHealthSummary(
        raw_routed=0,
        score_files_completed=0,
        backlog_used=0,
        backlog_cap=96,
        near_miss_conversion=None,
        strict_conversion=None,
        panel_ready_conversion=None,
    )


def _axis(name: str, *, p: int, np: int, f: int, md: float | None) -> AxisStat:
    return AxisStat(
        pass_count=p,
        near_pass_count=np,
        fail_count=f,
        median_raw=None,
        median_calibrated=None,
        median_deficit=md,
        calibration_status="provisional",
        n=p + np + f,
    )


def _example(rid: str, fam: str, *, pLDDT: float, iPAE: float) -> Example:
    # deficits computed against Complexa strict-success thresholds
    # (single source: success_criteria.STRICT_SUCCESS)
    from trex.success_criteria import STRICT_SUCCESS
    return Example(
        result_id=rid,
        family=fam,
        parent_id=None,
        axis_values={"pLDDT": pLDDT, "iPAE": iPAE},
        axis_deficits={
            "pLDDT": max(0.0, STRICT_SUCCESS["pLDDT"][0] - pLDDT),
            "iPAE": max(0.0, iPAE - STRICT_SUCCESS["iPAE"][0]),
        },
        joint_pattern_label=None,
    )


def case_stalled(model: str) -> tuple[str, EvidenceSummary]:
    e = EvidenceSummary(
        tick_id="tick_015",
        target_id="sc2rbd_like",
        target_class="viral_rbd",
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=24.0,
        remaining_wall_h=24.0,
        completed_children=22,
        pending_children=3,
        worker_gpu_h_total=60.0,
        worker_gpu_h_last_3_ticks=14.0,
        strict_count=0,
        global_new_strict=0,
        run_su_count=0,
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
        duplicate_fraction=0.65,
        top_bin_share=0.80,
        axis_stats={
            "pLDDT": _axis("pLDDT", p=4, np=4, f=14, md=8.0),
            "iPAE": _axis("iPAE", p=4, np=4, f=14, md=0.30),
            "binder_scRMSD": _axis("binder_scRMSD", p=10, np=6, f=6, md=0.5),
        },
        joint_patterns=[
            JointPatternCount(("pLDDT", "iPAE"), "both_pass", 2),
            JointPatternCount(("pLDDT", "iPAE"), "both_fail", 12),
            JointPatternCount(("pLDDT", "iPAE"), "A_pass_B_fail", 2),
            JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 2),
            JointPatternCount(("pLDDT", "iPAE"), "both_near_pass", 4),
        ],
        near_miss_count=1,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        method_health={
            "complexa_beam": MethodHealthSummary(
                family="complexa_beam",
                attempts=18, completions=18, timeouts=0, nonzero_exits=0,
                raw_artifacts=18, accepted_artifacts=18, score_files=18,
                strict_yield=0, near_miss_yield=1, routed_proxy=None,
            ),
            "bindcraft": MethodHealthSummary(
                family="bindcraft",
                attempts=2, completions=1, timeouts=1, nonzero_exits=0,
                raw_artifacts=1, accepted_artifacts=1, score_files=0,
                strict_yield=0, near_miss_yield=0, routed_proxy=None,
            ),
        },
        route_health=_route_health_default(),
        llm_health=_llm_health_default(model),
        state_label="stalled",
        # Stalled: both pLDDT (<<90, deficit >5) and iPAE (>>0.226, deficit >0.05) fail
        examples=[
            _example("r_sc2_021", "complexa_beam", pLDDT=70.0, iPAE=0.65),
            _example("r_sc2_022", "complexa_beam", pLDDT=72.0, iPAE=0.62),
        ],
        metric_availability={
            "complexa_beam": {"pLDDT": True, "iPAE": True, "binder_scRMSD": True},
            "bindcraft": {"pLDDT": False, "iPAE": False, "binder_scRMSD": False},
        },
    )
    from dataclasses import replace as _replace
    e = _replace(
        e,
        recent_fallback_high=False,
        recipes=[
            Recipe(
                recipe_hash="rcp_fail_x1",
                operator_id="complexa_beam_default",
                method_family="complexa_beam",
                config_delta={"beam_width": 4, "n_branch": 4},
                recipe_class="joint_fail",
                target_id=e.target_id,
                target_class=e.target_class,
                descendant_count=12,
                median_metrics={"pLDDT": 70.0, "iPAE": 0.65},
                representative_result_ids=["r_sc2_021"],
                recency_tick=14,
            ),
        ],
    )
    return "stalled_sc2rbd_like", e
