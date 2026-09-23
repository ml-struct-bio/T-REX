"""Planner output validation on five synthetic campaign-evidence cases.

Requires an existing LLM endpoint. No molecular backend is run.
Use python -m benchmarks.llm_validation.planner_validation --help for options.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from trex import SCHEMA_VERSION
from trex.planner import (
    PlannerCallConfig,
    VALID_ACTION_FAMILIES,
    call_planner,
)
from trex.schemas import (
    AxisStat,
    Example,
    Recipe,
    EvidenceSummary,
    JointPatternCount,
    LLMHealthSummary,
    MethodHealthSummary,
    RouteHealthSummary,
)


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


def case_productive(model: str) -> tuple[str, EvidenceSummary]:
    e = EvidenceSummary(
        tick_id="tick_010",
        target_id="cd45_like",
        target_class="receptor_extracellular",
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=18.0,
        remaining_wall_h=30.0,
        completed_children=42,
        pending_children=4,
        worker_gpu_h_total=85.0,
        worker_gpu_h_last_3_ticks=12.0,
        strict_count=8,
        global_new_strict=8,
        run_su_count=6,
        run_su_count_delta=2,
        su_per_gpu_h_recent=0.17,
        duplicate_fraction=0.30,
        top_bin_share=0.40,
        axis_stats={
            "pLDDT": _axis("pLDDT", p=20, np=4, f=2, md=1.0),
            "iPAE": _axis("iPAE", p=18, np=4, f=4, md=0.05),
            "binder_scRMSD": _axis("binder_scRMSD", p=15, np=8, f=3, md=0.3),
        },
        joint_patterns=[
            JointPatternCount(("pLDDT", "iPAE"), "both_pass", 18),
            JointPatternCount(("pLDDT", "iPAE"), "both_fail", 2),
            JointPatternCount(("pLDDT", "iPAE"), "A_pass_B_fail", 4),
            JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 0),
            JointPatternCount(("pLDDT", "iPAE"), "both_near_pass", 2),
        ],
        near_miss_count=6,
        panel_ready_count=4,
        panel_ready_bins_covered=3,
        method_health={
            "complexa_beam": MethodHealthSummary(
                family="complexa_beam",
                attempts=20, completions=20, timeouts=0, nonzero_exits=0,
                raw_artifacts=20, accepted_artifacts=20, score_files=20,
                strict_yield=6, near_miss_yield=2, routed_proxy=None,
            ),
        },
        route_health=_route_health_default(),
        llm_health=_llm_health_default(model),
        state_label="productive",
        # Complexa thresholds: pLDDT>=90, iPAE<=7/31 (~0.226), scRMSD<1.5
        # productive example clearly passes strict gate
        examples=[_example("r_001", "complexa_beam", pLDDT=92.0, iPAE=0.18)],
        metric_availability={"complexa_beam": {"pLDDT": True, "iPAE": True, "binder_scRMSD": True}},
    )
    # Provide successful recipes to test evidence-based proposal reuse.
    from dataclasses import replace as _replace
    e = _replace(e, recipes=[
        Recipe(
            recipe_hash="rcp_proven_a1",
            operator_id="complexa_beam_default",
            method_family="complexa_beam",
            config_delta={"beam_width": 4, "n_branch": 4, "sampling_temp": 0.2},
            recipe_class="strict_success",
            target_id=e.target_id,
            target_class=e.target_class,
            descendant_count=4,
            # Median metrics consistent with strict_success: pLDDT 91, iPAE 0.21, scRMSD 1.3
            median_metrics={"pLDDT": 91.0, "iPAE": 0.21, "binder_scRMSD": 1.3},
            representative_result_ids=["r_001", "r_007"],
            recency_tick=9,
        ),
        Recipe(
            recipe_hash="rcp_near_b1",
            operator_id="interface_redesign",
            method_family="proteinmpnn_redesign",
            config_delta={"num_seq_per_target": 4, "sampling_temp": 0.1},
            recipe_class="near_miss",
            target_id=e.target_id,
            target_class=e.target_class,
            descendant_count=2,
            # Near-miss on iPAE only: pLDDT passes, iPAE in near-pass region
            # (>0.226 but <0.276), scRMSD passes
            median_metrics={"pLDDT": 91.0, "iPAE": 0.26, "binder_scRMSD": 1.4},
            representative_result_ids=["r_088"],
            recency_tick=9,
        ),
    ])
    return "productive_cd45_like", e


def case_rescue_rich(model: str) -> tuple[str, EvidenceSummary]:
    e = EvidenceSummary(
        tick_id="tick_012",
        target_id="betv1_like",
        target_class="allergen",
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=20.0,
        remaining_wall_h=28.0,
        completed_children=30,
        pending_children=4,
        worker_gpu_h_total=70.0,
        worker_gpu_h_last_3_ticks=10.0,
        strict_count=2,
        global_new_strict=2,
        run_su_count=2,
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
        duplicate_fraction=0.45,
        top_bin_share=0.50,
        axis_stats={
            "pLDDT": _axis("pLDDT", p=20, np=5, f=5, md=0.5),  # passes mostly
            "iPAE": _axis("iPAE", p=4, np=15, f=11, md=0.25),   # dominant fail
            "binder_scRMSD": _axis("binder_scRMSD", p=20, np=8, f=2, md=0.2),
        },
        joint_patterns=[
            JointPatternCount(("pLDDT", "iPAE"), "both_pass", 4),
            JointPatternCount(("pLDDT", "iPAE"), "both_fail", 3),
            JointPatternCount(("pLDDT", "iPAE"), "A_pass_B_fail", 13),
            JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 0),
            JointPatternCount(("pLDDT", "iPAE"), "both_near_pass", 5),
        ],
        near_miss_count=18,
        panel_ready_count=2,
        panel_ready_bins_covered=2,
        method_health={
            "complexa_beam": MethodHealthSummary(
                family="complexa_beam",
                attempts=22, completions=22, timeouts=0, nonzero_exits=0,
                raw_artifacts=22, accepted_artifacts=22, score_files=22,
                strict_yield=2, near_miss_yield=18, routed_proxy=None,
            ),
        },
        route_health=_route_health_default(),
        llm_health=_llm_health_default(model),
        state_label="rescue_rich",
        # Rescue-rich: pLDDT passes (>=90), iPAE in near-pass region (0.226-0.276)
        # — iPAE-only failure pattern, sequence redesign expected to help
        examples=[_example("r_betv_010", "complexa_beam", pLDDT=91.0, iPAE=0.26)],
        metric_availability={"complexa_beam": {"pLDDT": True, "iPAE": True, "binder_scRMSD": True}},
    )
    return "rescue_rich_betv1_like", e


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


def case_ambiguous(model: str) -> tuple[str, EvidenceSummary]:
    """All three axes uncalibrated for the most-launched family."""
    e = EvidenceSummary(
        tick_id="tick_005",
        target_id="cbago_like",
        target_class="enzyme_pocket",
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=10.0,
        remaining_wall_h=38.0,
        completed_children=15,
        pending_children=3,
        worker_gpu_h_total=40.0,
        worker_gpu_h_last_3_ticks=10.0,
        strict_count=0,
        global_new_strict=0,
        run_su_count=0,
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
        duplicate_fraction=None,
        top_bin_share=None,
        axis_stats={
            "pLDDT": _axis("pLDDT", p=0, np=0, f=0, md=None),
            "iPAE": _axis("iPAE", p=0, np=0, f=0, md=None),
            "binder_scRMSD": _axis("binder_scRMSD", p=0, np=0, f=0, md=None),
        },
        joint_patterns=[],
        near_miss_count=0,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        method_health={
        },
        route_health=_route_health_default(),
        llm_health=_llm_health_default(model),
        state_label="low_evidence",
        examples=[],
        metric_availability={
        },
    )
    return "ambiguous_cbago_like", e


def case_panel_ready_diversity_short(model: str) -> tuple[str, EvidenceSummary]:
    """Hits but redundant — diversity-improvement is the relevant axis."""
    e = EvidenceSummary(
        tick_id="tick_018",
        target_id="pdl1_like",
        target_class="checkpoint",
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=22.0,
        remaining_wall_h=26.0,
        completed_children=35,
        pending_children=2,
        worker_gpu_h_total=80.0,
        worker_gpu_h_last_3_ticks=12.0,
        strict_count=12,
        global_new_strict=4,  # most are dupes
        run_su_count=4,
        run_su_count_delta=1,
        su_per_gpu_h_recent=0.08,
        duplicate_fraction=0.55,
        top_bin_share=0.78,  # single Foldseek bin dominates
        axis_stats={
            "pLDDT": _axis("pLDDT", p=24, np=3, f=2, md=0.5),
            "iPAE": _axis("iPAE", p=20, np=5, f=4, md=0.05),
            "binder_scRMSD": _axis("binder_scRMSD", p=22, np=4, f=3, md=0.2),
        },
        joint_patterns=[
            JointPatternCount(("pLDDT", "iPAE"), "both_pass", 20),
            JointPatternCount(("pLDDT", "iPAE"), "both_fail", 2),
            JointPatternCount(("pLDDT", "iPAE"), "A_pass_B_fail", 4),
            JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 0),
            JointPatternCount(("pLDDT", "iPAE"), "both_near_pass", 1),
        ],
        near_miss_count=5,
        panel_ready_count=8,
        panel_ready_bins_covered=2,  # only 2 of 4 bin types covered
        method_health={
            "complexa_beam": MethodHealthSummary(
                family="complexa_beam",
                attempts=25, completions=25, timeouts=0, nonzero_exits=0,
                raw_artifacts=25, accepted_artifacts=25, score_files=25,
                strict_yield=10, near_miss_yield=3, routed_proxy=None,
            ),
        },
        route_health=_route_health_default(),
        llm_health=_llm_health_default(model),
        state_label="stalled",  # top_bin_share dominates
        # Productive evidence but mode-collapsed in Foldseek space:
        # strict-passing example (pLDDT>=90, iPAE<=0.226) but all in one cluster
        examples=[_example("r_pdl_080", "complexa_beam", pLDDT=91.0, iPAE=0.20)],
        metric_availability={"complexa_beam": {"pLDDT": True, "iPAE": True, "binder_scRMSD": True}},
    )
    return "stalled_redundant_pdl1", e


def case_stress_max_context(model: str) -> tuple[str, EvidenceSummary]:
    """Maximum realistic EvidenceSummary: 8 method families, dense axis data,
    6 examples covering all joint patterns. Validates that Qwen still produces
    valid JSON when the context fills closer to the LLM budget."""
    method_families = [
        ("complexa_beam", 30, 28, 0, 0, 28, 28, 28, 5, 3),
        ("complexa_best_of_n", 20, 18, 0, 0, 18, 18, 18, 2, 4),
        ("proteinmpnn_redesign", 15, 14, 0, 0, 14, 14, 14, 1, 3),
        ("structure_refilter", 12, 11, 0, 0, 11, 11, 11, 1, 2),
        ("bindcraft", 5, 3, 1, 1, 3, 3, 1, 0, 0),
        ("boltzgen", 8, 6, 1, 1, 6, 4, 4, 1, 1),
    ]
    mh = {
        f: MethodHealthSummary(
            family=f, attempts=a, completions=cm, timeouts=to, nonzero_exits=ne,
            raw_artifacts=ra, accepted_artifacts=aa, score_files=sf,
            strict_yield=sy, near_miss_yield=nm, routed_proxy=None,
        )
        for (f, a, cm, to, ne, ra, aa, sf, sy, nm) in method_families
    }
    e = EvidenceSummary(
        tick_id="tick_stress",
        target_id="multi_target_stress",
        target_class="mixed",
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=30.0,
        remaining_wall_h=18.0,
        completed_children=104,
        pending_children=8,
        worker_gpu_h_total=250.0,
        worker_gpu_h_last_3_ticks=18.0,
        strict_count=15,
        global_new_strict=11,
        run_su_count=8,
        run_su_count_delta=1,
        su_per_gpu_h_recent=0.055,
        duplicate_fraction=0.42,
        top_bin_share=0.55,
        axis_stats={
            "pLDDT": _axis("pLDDT", p=50, np=12, f=20, md=2.5),
            "iPAE": _axis("iPAE", p=40, np=15, f=27, md=0.12),
            "binder_scRMSD": _axis("binder_scRMSD", p=45, np=20, f=17, md=0.4),
        },
        joint_patterns=[
            JointPatternCount(("pLDDT", "iPAE"), "both_pass", 35),
            JointPatternCount(("pLDDT", "iPAE"), "both_fail", 14),
            JointPatternCount(("pLDDT", "iPAE"), "A_pass_B_fail", 13),
            JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 5),
            JointPatternCount(("pLDDT", "iPAE"), "both_near_pass", 8),
        ],
        near_miss_count=21,
        panel_ready_count=6,
        panel_ready_bins_covered=3,
        method_health=mh,
        route_health=RouteHealthSummary(
            raw_routed=12, score_files_completed=10, backlog_used=44, backlog_cap=96,
            near_miss_conversion=0.18, strict_conversion=0.08, panel_ready_conversion=0.03,
        ),
        llm_health=_llm_health_default(model),
        state_label="rescue_rich",
        examples=[
            # stress case: mix of strict/near-miss/fail under Complexa thresholds
            _example("r_str_001", "complexa_beam", pLDDT=92.0, iPAE=0.20),     # strict pass
            _example("r_str_002", "complexa_beam", pLDDT=75.0, iPAE=0.65),     # both fail
            _example("r_str_003", "proteinmpnn_redesign", pLDDT=91.0, iPAE=0.26),  # iPAE near-pass
            _example("r_str_004", "bindcraft", pLDDT=68.0, iPAE=0.70),         # both fail
            _example("r_str_005", "boltzgen", pLDDT=88.0, iPAE=0.27),          # near-pass both
        ],
        metric_availability={
            f: {"pLDDT": True, "iPAE": True, "binder_scRMSD": True}
            for f, *_ in method_families
        },
    )
    return "stress_max_context", e


CASES = [
    case_productive,
    case_rescue_rich,
    case_stalled,
    case_ambiguous,
    case_panel_ready_diversity_short,
]


STRESS_CASES = [
    case_stress_max_context,
]


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cfg = PlannerCallConfig(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        confidence_threshold=0.0,  # we want all valid outputs through for analysis
        prompt_variant=args.prompt_variant,
        temperature=args.temperature,
    )

    if args.stress_only:
        case_list = STRESS_CASES
    elif args.include_stress:
        case_list = CASES + STRESS_CASES
    else:
        case_list = CASES

    records: list[dict[str, Any]] = []
    started = time.time()
    for repeat in range(args.repeats):
        for build in case_list:
            name, evidence = build(cfg.model)
            tick_id_int = repeat * 100 + case_list.index(build)
            t0 = time.time()
            out = call_planner(
                evidence,
                active_hypotheses=[],
                seed_action_families=list(VALID_ACTION_FAMILIES),
                tick_id_int=tick_id_int,
                cfg=cfg,
            )
            elapsed = time.time() - t0
            rec = {
                "case": name,
                "repeat": repeat,
                "tick_id_int": tick_id_int,
                "elapsed_s": round(elapsed, 2),
                "valid": out.valid,
                "abstain": out.abstain,
                "confidence": out.confidence,
                "fail_reason": out.fail_reason,
                "n_cards": len(out.cards),
                "card_action_families": [
                    list(c.recommended_action_families) for c in out.cards
                ],
                "card_axes": [
                    [pc.axis for pc in c.predicted_metric_changes] for c in out.cards
                ],
                "card_mode_affinities": [c.mode_affinity for c in out.cards],
                "card_config_delta_suggestions": [
                    c.config_delta_suggestions for c in out.cards
                ],
                "temperature": cfg.temperature,
                "prompt_variant": cfg.prompt_variant,
                "usage": out.usage,
            }
            if not out.valid:
                rec["raw_excerpt"] = (out.raw_text or "")[:1200]
            records.append(rec)
            print(json.dumps({k: rec[k] for k in ("case", "valid", "confidence", "n_cards", "fail_reason", "elapsed_s")}))

    valid_records = [r for r in records if r["valid"]]
    confs = [r["confidence"] for r in valid_records]

    summary = {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "prompt_variant": cfg.prompt_variant,
        "total_calls": len(records),
        "valid_calls": len(valid_records),
        "valid_rate": (len(valid_records) / len(records)) if records else 0.0,
        "confidence_min": min(confs) if confs else None,
        "confidence_max": max(confs) if confs else None,
        "confidence_median": statistics.median(confs) if confs else None,
        "confidence_stdev": statistics.pstdev(confs) if len(confs) > 1 else 0.0,
        "median_latency_s": (
            statistics.median([r["elapsed_s"] for r in records]) if records else None
        ),
        "wall_clock_s": round(time.time() - started, 1),
        "schema_version": SCHEMA_VERSION,
    }

    return {"summary": summary, "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vllm/Qwen/Qwen3.6-27B-FP8")
    p.add_argument("--base-url", default="http://127.0.0.1:8500/v1")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature override. None = vLLM server default.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--min-valid-rate",
        type=float,
        default=0.80,
        help="S3 acceptance threshold (default 80%%).",
    )
    p.add_argument("--fail-under", action="store_true")
    p.add_argument(
        "--prompt-variant",
        choices=["default", "terse", "fewshot", "axis_first"],
        default="default",
    )
    p.add_argument(
        "--include-stress",
        action="store_true",
        help="Include the stress max-context case after the standard 5.",
    )
    p.add_argument(
        "--stress-only",
        action="store_true",
        help="Run only the stress max-context case.",
    )
    args = p.parse_args()

    report = run_smoke(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))

    if args.fail_under and report["summary"]["valid_rate"] < args.min_valid_rate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
