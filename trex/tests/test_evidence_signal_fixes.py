"""Tests for near-miss yield, normalized deficits, and marginal structural progress."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from trex.archive import Archive
from trex.evidence_reducer import (
    ReducerConfig,
    build_strategy_feedback,
    method_health,
    reduce_evidence,
    rescue_axis_concentration,
)
from trex.refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from trex.live_tick import (
    FoldseekConfig,
    LiveTickConfig,
    _charged_gpu_count_from_env,
    run_live_tick,
)
from trex.schemas import (
    ActionCandidate,
    AxisStat,
    EvidenceSummary,
    FeasibilityCheck,
    PlannerOutput,
    ResultRecord,
    SupervisorOutput,
    TargetConstraint,
)


def _rec(rid, fam, metrics, bins=None, parent_ids=None):
    return ResultRecord(
        result_id=rid, parent_ids=list(parent_ids or []), target_id="t1",
        backend_family=fam, runtime_bucket_id="rb1",
        metrics=metrics, metrics_calibrated=dict(metrics), route_lineage=[],
        gpu_h=0.5, exit_status="ok", bins=bins or {},
    )


def _action(cid, fam, *, parent=None, op=None, config=None, refilter_role=None):
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id=cid,
        hypothesis_ids=["h1"],
        parent_result_id=parent,
        method_family=fam,
        operator_id=op or f"{fam}_default",
        lane_id=fam,
        config_delta=dict(config or {}),
        downstream_route_plan=[],
        estimated_cost_class="standard",
        expected_signal="test",
        evidence_refs=[],
        feasibility=feas,
        refilter_role=refilter_role,
    )


# --- 1. near_miss_yield ------------------------------------------------------

def test_near_miss_yield_is_populated_per_family():
    # near-miss: pLDDT passes, scRMSD passes, iPAE fails by one band only
    near = [
        _rec(f"n{i}", "bindcraft", {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0})
        for i in range(3)
    ]
    mh = method_health(near)
    assert "bindcraft" in mh
    assert mh["bindcraft"].near_miss_yield == 3, "near_miss_yield must count near-misses"
    assert mh["bindcraft"].strict_yield == 0


def test_near_miss_dedups_by_refilter_source_on_degraded_foldseek(tmp_path):
    """Deduplicate repeated evaluations of one near-miss source when structural clustering
    is unavailable. This diagnostic fallback does not grant SU credit.
    """
    near = [
        _rec(f"r{i}", "structure_refilter",
             {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0},
             bins={"refilter_source": "backboneX.pdb"})  # same basin, NO foldseek bin
        for i in range(3)
    ]
    mh = method_health(near)
    assert mh["structure_refilter"].near_miss_yield == 1, (
        "3 refolds of one near-miss basin (same refilter_source) must = 1"
    )


def test_near_miss_yield_uses_exact_near_miss_cluster_map():
    """Use the full near-miss cluster map for cumulative family evidence."""
    near = [
        _rec(f"n{i}", "complexa_beam",
             {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0})
        for i in range(3)
    ]
    nm_map = {r.result_id: "foldseek_nm:shared" for r in near}

    mh = method_health(near, near_miss_cluster_by_result_id=nm_map)

    assert mh["complexa_beam"].near_miss_yield == 1


def test_near_miss_only_family_not_misread_as_explore_trigger():
    """The planner's explore trigger is strict_yield==0 AND near_miss_yield==0.
    A near-miss-only family must NOT satisfy it."""
    near = [_rec("n0", "complexa_fk_steering",
                 {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0})]
    mh = method_health(near)["complexa_fk_steering"]
    explore_trigger = mh.strict_yield == 0 and mh.near_miss_yield == 0
    assert not explore_trigger


def test_strategy_feedback_includes_near_miss_and_failure_counts():
    recs = [
        _rec("near", "complexa_beam",
             {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}),
        _rec("fail", "complexa_beam",
             {"pLDDT": 60.0, "iPAE": 0.8, "binder_scRMSD": 3.0}),
    ]
    fb = build_strategy_feedback(recs, spawning_actions={}, cfg=ReducerConfig())
    row = next(x for x in fb if x["family"] == "complexa_beam")
    assert row["feedback_scope"] == "exact_route_operator_config"
    assert row["repeat_policy"] == "not_a_ban_single_failure_is_insufficient"
    assert row["strategy_key"].startswith("route::complexa_beam:")
    assert row["near_miss_count"] == 1
    assert row["failure_count"] == 1
    assert row["joint_fail_count"] == 1
    assert row["representative_near_miss_ids"] == ["near"]
    assert row["representative_failure_ids"] == ["fail"]
    assert row["dominant_failure_axes"]


def test_strategy_feedback_dedups_near_miss_by_structure_bin():
    recs = [
        _rec(f"near{i}", "structure_refilter",
             {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0},
             bins={"refilter_source": "same_parent"})
        for i in range(3)
    ]
    fb = build_strategy_feedback(recs, spawning_actions={}, cfg=ReducerConfig())
    row = next(x for x in fb if x["family"] == "structure_refilter")
    assert row["near_miss_count"] == 1
    assert len(row["representative_near_miss_ids"]) == 1
    assert row["result_count"] == 3


def test_strategy_feedback_refilter_success_has_no_direct_rate():
    recs = [
        _rec("strict", "structure_refilter",
             {"pLDDT": 95.0, "iPAE": 0.2, "binder_scRMSD": 1.0},
             bins={"foldseek_su": "cluster_a"})
    ]
    fb = build_strategy_feedback(recs, spawning_actions={}, cfg=ReducerConfig())
    row = next(x for x in fb if x["family"] == "structure_refilter")
    assert row["strict_su"] == 1
    assert row["su_per_gpu_h"] is None


def test_strategy_feedback_attempts_are_unique_candidate_launches():
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    action = ActionCandidate(
        candidate_id="cand_multi",
        hypothesis_ids=["h"],
        parent_result_id=None,
        method_family="complexa_beam",
        operator_id="complexa_beam_default",
        lane_id="complexa_beam",
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",
        expected_signal="multi output",
        evidence_refs=[],
        feasibility=feas,
    )
    recs = [
        _rec("out1", "complexa_beam",
             {"pLDDT": 70.0, "iPAE": 0.8, "binder_scRMSD": 3.0},
             parent_ids=["cand_multi"]),
        _rec("out2", "complexa_beam",
             {"pLDDT": 71.0, "iPAE": 0.7, "binder_scRMSD": 2.8},
             parent_ids=["cand_multi"]),
    ]
    fb = build_strategy_feedback(
        recs,
        spawning_actions={"out1": action, "out2": action},
        cfg=ReducerConfig(),
    )
    row = next(x for x in fb if x["family"] == "complexa_beam")
    assert row["attempts"] == 1
    assert row["result_count"] == 2


def test_advisory_refold_gpu_not_charged_to_chained_su_rate():
    parent = __import__("dataclasses").replace(
        _rec("bc_parent", "bindcraft", {}),
        gpu_h=2.0,
    )
    canonical = __import__("dataclasses").replace(
        _rec(
            "af2_child",
            "structure_refilter",
            {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
            bins={"foldseek_su": "su_1", "refilter_role": CANONICAL_SCORE_CONVERSION},
            parent_ids=["bc_parent"],
        ),
        gpu_h=0.1,
    )
    advisory = __import__("dataclasses").replace(
        _rec(
            "refold_child",
            "structure_refilter",
            {},
            bins={"refilter_role": PARENT_MODEL_REFOLD},
            parent_ids=["bc_parent"],
        ),
        gpu_h=0.4,
    )
    mh = method_health(
        [parent, canonical, advisory],
        spawning_actions={
            "bc_parent": _action("bc_parent", "bindcraft"),
            "af2_child": _action("af2_child", "structure_refilter", parent="bc_parent", refilter_role=CANONICAL_SCORE_CONVERSION),
            "refold_child": _action("refold_child", "structure_refilter", parent="bc_parent", refilter_role=PARENT_MODEL_REFOLD),
        },
    )
    assert mh["bindcraft"].chained_strict_yield_su == 1
    assert abs(mh["bindcraft"].chained_su_per_gpu_h - (1.0 / 2.1)) < 1e-9


def test_strategy_feedback_diagnostic_lane_has_no_direct_su_rate():
    """strategy_feedback.su_per_gpu_h must mean direct generator SU rate.

    Diagnostic-only lanes receive their validated credit through
    method_health.chained_* fields, so exposing a direct su_per_gpu_h here
    would conflict with the planner's diagnostic-lane instructions.
    """
    parent = __import__("dataclasses").replace(
        _rec("bc_parent", "bindcraft", {}),
        gpu_h=2.0,
    )
    canonical = __import__("dataclasses").replace(
        _rec(
            "af2_child",
            "structure_refilter",
            {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
            bins={"foldseek_su": "su_1", "refilter_role": CANONICAL_SCORE_CONVERSION},
            parent_ids=["bc_parent"],
        ),
        gpu_h=0.1,
    )
    fb = build_strategy_feedback(
        [parent, canonical],
        spawning_actions={
            "bc_parent": _action("bc_parent", "bindcraft"),
            "af2_child": _action("af2_child", "structure_refilter", parent="bc_parent", refilter_role=CANONICAL_SCORE_CONVERSION),
        },
        cfg=ReducerConfig(),
    )
    row = next(x for x in fb if x["family"] == "bindcraft")
    assert row["strict_su"] == 1
    assert row["su_per_gpu_h"] is None


def test_strategy_feedback_complexa_su_comes_from_direct_af2folding_scores():
    parent = __import__("dataclasses").replace(
        _rec(
            "cx_parent",
            "complexa_beam",
            {
                "pLDDT": 95.0,
                "iPAE": 0.1,
                "binder_scRMSD": 1.0,
                "complexa_native_pLDDT": 95.0,
                "complexa_native_iPAE": 0.1,
                "complexa_native_binder_scRMSD": 1.0,
            },
            bins={"foldseek_su": "su_cx", "strict_score_source": "complexa_af2folding_canonical"},
            parent_ids=["cx_cand"],
        ),
        gpu_h=0.5,
    )
    fb = build_strategy_feedback(
        [parent],
        spawning_actions={"cx_parent": _action("cx_cand", "complexa_beam")},
        cfg=ReducerConfig(),
    )
    row = next(x for x in fb if x["family"] == "complexa_beam")
    assert row["strict_su"] == 1
    assert row["diagnostic_only_count"] == 0
    assert row["su_per_gpu_h"] == 2.0
    assert row.get("chained_su_per_gpu_h") is None

def test_strategy_feedback_keeps_diagnostic_lane_with_refilter_failures():
    """A diagnostic generator must remain visible even when canonical refilter
    failures make it a mixed diagnostic+failed row and Complexa failures crowd
    the default Top-K failure slice.
    """
    recs = []
    spawning = {}
    for i in range(4):
        action = _action(
            f"cx_cand_{i}",
            "complexa_beam",
            config={"beam_width": 4 + i},
        )
        for j in range(8):
            rid = f"cx_{i}_{j}"
            recs.append(_rec(
                rid,
                "complexa_beam",
                {"pLDDT": 60.0, "iPAE": 0.8, "binder_scRMSD": 3.0},
                parent_ids=[action.candidate_id],
            ))
            spawning[rid] = action

    bg_action = _action("bg_cand", "boltzgen", config={"num_designs": 32})
    for j in range(4):
        rid = f"bg_raw_{j}"
        recs.append(_rec(
            rid,
            "boltzgen",
            {},
            bins={"boltzgen_design_iptm": str(0.10 + 0.01 * j)},
            parent_ids=["bg_cand"],
        ))
        spawning[rid] = bg_action
    for j in range(2):
        rid = f"bg_refilter_fail_{j}"
        recs.append(_rec(
            rid,
            "structure_refilter",
            {"pLDDT": 70.0, "iPAE": 0.7, "binder_scRMSD": 2.5},
            bins={"refilter_role": CANONICAL_SCORE_CONVERSION},
            parent_ids=[f"chain_bg_{j}", f"bg_raw_{j}"],
        ))
        spawning[rid] = _action(
            f"chain_bg_{j}",
            "structure_refilter",
            parent=f"bg_raw_{j}",
            refilter_role=CANONICAL_SCORE_CONVERSION,
        )

    fb = build_strategy_feedback(recs, spawning, ReducerConfig())
    row = next(x for x in fb if x["family"] == "boltzgen")
    assert row["diagnostic_only_count"] == 4
    assert row["failure_count"] == 2
    assert row["advisory_scores"]["design_iptm"]["direction"] == "increase"


def test_strategy_feedback_does_not_count_diagnostic_only_raw_record_as_failure():
    recs = [
        _rec("bc_raw", "bindcraft",
             {"bindcraft_native_pLDDT": 92.0, "bindcraft_native_iPAE": 0.2}),
    ]
    fb = build_strategy_feedback(recs, spawning_actions={}, cfg=ReducerConfig())
    row = next(x for x in fb if x["family"] == "bindcraft")
    assert row["diagnostic_only_count"] == 1
    assert row["failure_count"] == 0
    assert row["near_miss_count"] == 0


# --- 2. rescue_axis_concentration -------------------------------------------

def _axis(deficit):
    return AxisStat(
        pass_count=0, near_pass_count=0, fail_count=5,
        median_raw=0.0, median_calibrated=0.0, median_deficit=deficit,
        calibration_status="provisional", n=5,
    )


def test_rescue_axis_concentration_uses_margin_units():
    # pLDDT deficit 8 pts (margin 5 → 1.6), iPAE deficit 0.3 (margin 0.05 → 6.0)
    stats = {"pLDDT": _axis(8.0), "iPAE": _axis(0.3), "binder_scRMSD": _axis(0.0)}
    conc = rescue_axis_concentration(stats)
    # margin-normalized: iPAE (6.0) dominates → 6.0 / (1.6 + 6.0) ≈ 0.789
    assert abs(conc - (6.0 / 7.6)) < 1e-6
    # Compare normalized deficits rather than summing measurements in unlike units.
    assert conc < 0.90


# --- 3. run_su_count_delta marginal -----------------------------------------

def _planner_empty():
    return PlannerOutput(valid=True, abstain=False, confidence=0.7, fail_reason=None,
                         cards=[], rationale="x", raw_text="{}", usage={})


def _sup_ok():
    return SupervisorOutput(valid=True, abstain=False, confidence=0.7, fail_reason=None,
                            mode_mixture={"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
                            candidate_decisions=[], rationale="y", raw_text="{}", usage={})


def _strict(rid, bin_id):
    # foldseek_su = strict-only SU bin (SU dedup reads this); foldseek =
    # whole-archive bin (duplicate_fraction). Equal here since the test
    # pre-clusters directly (foldseek disabled in these tick configs).
    return _rec(rid, "complexa_beam",
                {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
                bins={"foldseek": bin_id, "foldseek_su": bin_id})


def _run_tick(arc, tmp_path):
    target = TargetConstraint(target_id="t1", target_class="test")
    cfg = LiveTickConfig(window_size=3, foldseek=FoldseekConfig(enabled=False))
    with patch("trex.live_tick.call_planner", return_value=_planner_empty()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(arc, target, tick_id="t_001", tick_id_int=1,
                      elapsed_wall_h=0.0, remaining_wall_h=48.0, cfg=cfg)
    evs = list(arc.iter_records(EvidenceSummary))
    return evs[-1]


def test_run_su_count_delta_zero_when_only_rediscovering_clusters(tmp_path: Path):
    arc = Archive(tmp_path / "arc_dup")
    # window_size=3: pre-window {A,B,C}, window re-discovers {A,B,C} → 0 NEW SU
    for rid, b in [("r0", "A"), ("r1", "B"), ("r2", "C"),
                   ("r3", "A"), ("r4", "B"), ("r5", "C")]:
        arc.append(_strict(rid, b))
    ev = _run_tick(arc, tmp_path)
    assert ev.run_su_count == 3                 # 3 unique clusters run-level
    assert ev.run_su_count_delta == 0, "re-discovering existing clusters is NOT new SU"


def test_run_su_count_delta_counts_new_cluster_in_window(tmp_path: Path):
    arc = Archive(tmp_path / "arc_new")
    # window_size=3: pre-window {A,B,C}, window {A,B,D} → D is new → delta 1
    for rid, b in [("r0", "A"), ("r1", "B"), ("r2", "C"),
                   ("r3", "A"), ("r4", "B"), ("r5", "D")]:
        arc.append(_strict(rid, b))
    ev = _run_tick(arc, tmp_path)
    assert ev.run_su_count == 4
    assert ev.run_su_count_delta == 1, "a cluster first seen in the window IS new SU"


def test_reduce_evidence_preserves_historical_charged_gpu_audit_rates():
    ev = reduce_evidence(
        tick_id="v7r001",
        target_id="t1",
        target_class="test",
        elapsed_wall_h=2.0,
        remaining_wall_h=46.0,
        pending_children=0,
        worker_gpu_h_total=1.0,
        all_results=[],
        window_results=[],
        run_su_count=4,
        run_su_count_delta=1,
        duplicate_fraction=None,
        near_miss_count=0,
        top_bin_share=None,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="test",
        charged_gpu_count=4.0,
        charged_gpu_h_total=8.0,
        charged_gpu_h_recent=2.0,
        charged_gpu_h_scope="env:TREX_CHARGED_GPUS",
    )
    assert ev.charged_gpu_count == 4.0
    assert ev.charged_gpu_h_total == 8.0
    assert ev.charged_gpu_h_recent == 2.0
    assert ev.charged_gpu_h_scope == "env:TREX_CHARGED_GPUS"
    # Worker-compute productivity is separate from reserved-allocation metadata.
    assert ev.run_su_per_worker_gpu_h_total == 4.0
    assert ev.run_su_per_charged_gpu_h_total == 0.5
    assert ev.run_su_per_charged_gpu_h_recent == 0.5


def test_method_health_recent_near_miss_is_marginal_not_rediscovered():
    old = _rec(
        "old_nm",
        "complexa_beam",
        {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0},
    )
    rediscovered = _rec(
        "new_nm_same_basin",
        "complexa_beam",
        {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0},
    )
    fresh = _rec(
        "new_nm_fresh_basin",
        "complexa_fk_steering",
        {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0},
    )

    ev = reduce_evidence(
        tick_id="v7r002",
        target_id="t1",
        target_class="test",
        elapsed_wall_h=2.0,
        remaining_wall_h=46.0,
        pending_children=0,
        worker_gpu_h_total=2.0,
        all_results=[old, rediscovered, fresh],
        window_results=[rediscovered, fresh],
        run_su_count=0,
        run_su_count_delta=0,
        duplicate_fraction=None,
        near_miss_count=2,
        top_bin_share=None,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="test",
        near_miss_cluster_by_result_id={
            "old_nm": "nm_shared",
            "new_nm_same_basin": "nm_shared",
            "new_nm_fresh_basin": "nm_fresh",
        },
    )

    assert ev.method_health["complexa_beam"].near_miss_yield == 1
    assert ev.method_health["complexa_beam"].near_miss_yield_recent == 0
    assert ev.method_health["complexa_fk_steering"].near_miss_yield_recent == 1


def test_degraded_near_miss_dedup_does_not_drive_feedback_state_or_routes():
    recs = [
        _rec(f"nm{i}", "complexa_beam",
             {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0})
        for i in range(3)
    ]

    ev = reduce_evidence(
        tick_id="v7r003",
        target_id="t1",
        target_class="test",
        elapsed_wall_h=2.0,
        remaining_wall_h=46.0,
        pending_children=0,
        worker_gpu_h_total=4.0,
        all_results=recs,
        window_results=recs,
        run_su_count=0,
        run_su_count_delta=0,
        duplicate_fraction=0.0,
        near_miss_count=3,
        top_bin_share=0.0,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="test",
        near_miss_dedup_status="no_binary",
        near_miss_dedup_coverage=0.0,
    )

    assert ev.near_miss_count == 3  # raw evidence remains visible with provenance
    assert ev.near_miss_dedup_status == "no_binary"
    assert ev.state_label == "stalled", "untrusted near-miss count must not create rescue_rich"
    assert ev.method_health["complexa_beam"].near_miss_yield == 0
    assert ev.method_health["complexa_beam"].near_miss_yield_recent == 0
    family_row = next(r for r in ev.route_values if r.scope == "family" and r.family == "complexa_beam")
    assert family_row.near_miss_count == 0
    assert all(row.get("near_miss_count", 0) == 0 for row in ev.strategy_feedback)


def test_disabled_near_miss_dedup_does_not_drive_feedback_state_or_routes():
    recs = [
        _rec(f"nm_disabled{i}", "complexa_beam",
             {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0})
        for i in range(3)
    ]

    ev = reduce_evidence(
        tick_id="v7r003",
        target_id="t1",
        target_class="test",
        elapsed_wall_h=2.0,
        remaining_wall_h=46.0,
        pending_children=0,
        worker_gpu_h_total=4.0,
        all_results=recs,
        window_results=recs,
        run_su_count=0,
        run_su_count_delta=0,
        duplicate_fraction=0.0,
        near_miss_count=3,
        top_bin_share=0.0,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="test",
        near_miss_dedup_status="disabled",
        near_miss_dedup_coverage=None,
    )

    assert ev.near_miss_count == 3
    assert ev.state_label == "stalled"
    assert ev.method_health["complexa_beam"].near_miss_yield == 0
    family_row = next(r for r in ev.route_values if r.scope == "family" and r.family == "complexa_beam")
    assert family_row.near_miss_count == 0
    assert all(row.get("near_miss_count", 0) == 0 for row in ev.strategy_feedback)


def test_charged_gpu_count_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("TREX_CHARGED_GPUS", "4")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")
    n, scope = _charged_gpu_count_from_env()
    assert n == 4.0
    assert scope == "env:TREX_CHARGED_GPUS"


def test_su_uses_strict_only_bin_not_whole_archive_hub(tmp_path: Path):
    """Count strict-only clusters without merging them through nonqualified structures."""
    arc = Archive(tmp_path / "arc_hub")
    for rid in ("s0", "s1"):
        arc.append(_rec(rid, "complexa_beam",
                        {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
                        # whole-archive merged both into "hub" (non-strict
                        # bridge); strict-only keeps them distinct.
                        bins={"foldseek": "hub", "foldseek_su": f"su_{rid}"}))
    ev = _run_tick(arc, tmp_path)
    assert ev.run_su_count == 2, (
        "SU must use strict-only foldseek_su (2 distinct), not the "
        "whole-archive foldseek hub (would wrongly give 1)"
    )


def test_method_health_strict_yield_su_uses_strict_only_bin(tmp_path: Path):
    """Family and campaign SU counts use the same strict-only clusters."""
    from trex.evidence_reducer import method_health
    recs = [
        _rec("s0", "complexa_beam", {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
             bins={"foldseek": "hub", "foldseek_su": "su_a"}),
        _rec("s1", "complexa_beam", {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
             bins={"foldseek": "hub", "foldseek_su": "su_b"}),
    ]
    mh = method_health(recs)
    assert mh["complexa_beam"].strict_yield_su == 2, (
        "per-family strict_yield_su must use foldseek_su (SSOT with run_su_count)"
    )


def test_refilter_refolds_of_same_backbone_dedup_to_one_su():
    """Two structure_refilter strict records in the SAME official Foldseek SU
    cluster must count as 1 SU, not 2."""
    recs = [
        _rec("r1", "structure_refilter",
             {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
             bins={"refilter_source": "backboneX.pdb", "foldseek_su": "su_backboneX"}),
        _rec("r2", "structure_refilter",
             {"pLDDT": 96.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
             bins={"refilter_source": "backboneX.pdb", "foldseek_su": "su_backboneX"}),  # same backbone
    ]
    mh = method_health(recs)["structure_refilter"]
    assert mh.strict_yield_su == 1, "refolds of one backbone are 1 SU, not 2"


def test_refilter_excluded_from_su_per_gpuh_exploit_ranking():
    """A refilter that catches an SU at ~0 gpu_h must NOT show an inflated
    su_per_gpu_h (it would mis-steer the planner's exploit ranking); it is a
    scoring pass, not a generator. strict_yield_su is still reported."""
    recs = [
        _rec("r1", "structure_refilter",
             {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
             bins={"refilter_source": "bb.pdb", "foldseek_su": "su_bb"}),
    ]
    recs[0] = __import__("dataclasses").replace(recs[0], gpu_h=0.01)
    mh = method_health(recs)["structure_refilter"]
    assert mh.su_per_gpu_h is None, "refilter excluded from su_per_gpu_h ranking"
    assert mh.strict_yield_su == 1, "but its SU is still reported"


def test_su_per_gpu_h_recent_decays_while_lifetime_does_not():
    """Recent productivity can be zero while cumulative productivity remains positive."""
    import dataclasses
    strict = {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0}
    fail = {"pLDDT": 70.0, "iPAE": 0.6, "binder_scRMSD": 3.0}
    old = [dataclasses.replace(_rec(f"o{i}", "experimental_generator", strict,
                                    bins={"foldseek_su": f"c{i}"}), gpu_h=0.5)
           for i in range(2)]                              # 2 early strict SU
    recent = [dataclasses.replace(_rec(f"r{i}", "experimental_generator", fail), gpu_h=0.5)
              for i in range(3)]                            # recent: only failures
    life = method_health(old + recent)["experimental_generator"]
    win = method_health(recent)["experimental_generator"]   # the recent window
    assert life.su_per_gpu_h is not None and life.su_per_gpu_h > 0  # lifetime stays high
    assert (win.su_per_gpu_h or 0) == 0          # recent dropped to 0 (no new SU)


def test_unregistered_generator_su_per_gpuh_still_computed():
    """Rate-rankable generators (non-refilter/non-diagnostic) still get su_per_gpu_h."""
    import dataclasses
    r = dataclasses.replace(
        _rec("g1", "experimental_generator",
             {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0},
             bins={"foldseek_su": "clu1"}),
        gpu_h=0.5,
    )
    mh = method_health([r])["experimental_generator"]
    assert mh.su_per_gpu_h is not None and mh.su_per_gpu_h > 0


def test_cheap_exact_route_many_failures_cools_down_config_not_family():
    """A cheap direct-scored exact config can fail many scored samples before
    accumulating 3 GPU-h. Cool down that exact route, but do not mark the whole
    Complexa family dry: the next action should be a different beam/noise config,
    not a family ban.
    """
    recs = [
        ResultRecord(
            result_id=f"bad_reward_{i}",
            parent_ids=[],
            target_id="t1",
            backend_family="complexa_beam",
            runtime_bucket_id="rb1",
            metrics={"pLDDT": 70.0, "iPAE": 0.8, "binder_scRMSD": 40.0},
            metrics_calibrated={"pLDDT": 70.0, "iPAE": 0.8, "binder_scRMSD": 40.0},
            route_lineage=[],
            gpu_h=0.01,
            exit_status="ok",
            bins={},
        )
        for i in range(32)
    ]
    ev = reduce_evidence(
        tick_id="v7r010",
        target_id="t1",
        target_class="test",
        elapsed_wall_h=1.0,
        remaining_wall_h=47.0,
        pending_children=0,
        worker_gpu_h_total=sum(float(r.gpu_h or 0.0) for r in recs),
        all_results=recs,
        window_results=recs,
        run_su_count=0,
        run_su_count_delta=0,
        duplicate_fraction=None,
        near_miss_count=0,
        top_bin_share=None,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="test",
    )
    route = next(r for r in ev.route_values if r.scope == "route" and r.family == "complexa_beam")
    family = next(r for r in ev.route_values if r.scope == "family" and r.family == "complexa_beam")
    assert route.route_gpu_h < 3.0
    assert route.completions == 32
    assert route.status == "defer"
    assert route.marginal_status == "dry_low_quality"
    assert family.marginal_status == "under_tested"
