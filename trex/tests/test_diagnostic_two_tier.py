"""v7_3 two-tier diagnostic-axis behavior (§4.1 redesign, 2026-06-10).

Pins the new semantics: pass/near/fail are classified against the QUALITY
threshold; below_accept_count is the subset failing the tool's ACCEPT floor;
the two previously-dead axes (buried_sasa, binder_pTM) now discriminate; and
BoltzGen axes are read from r.bins. These are advisory only — none of this
touches strict_success or SU.
"""
from __future__ import annotations

from trex.evidence_reducer import (
    CORROBORATION_ONLY_AXES,
    DIAGNOSTIC_AXIS_REMEDIATION,
    DIAGNOSTIC_AXIS_THRESHOLDS,
    build_diagnostic_axis_stats,
    label_axis,
    worst_actionable_diagnostic_axis,
)
from trex.live_tick import FoldseekConfig
from trex.schemas import ResultRecord


def _bc_rec(rid: str, **diag: float) -> ResultRecord:
    """A BindCraft record carrying diagnostic axes in metrics."""
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0, **diag},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok",
        bins={},
    )


def _bg_rec(rid: str, **diag: float) -> ResultRecord:
    """A BoltzGen record: metrics={} (unscored for the strict path), diagnostic
    values live in bins as boltzgen_<axis> strings (as the parser emits them)."""
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok",
        bins={f"boltzgen_{k}": f"{v:.4f}" for k, v in diag.items()},
    )


def test_tuple_is_four_field_two_tier():
    # (pass_threshold|None, quality_threshold, direction, near_margin)
    for axis, tup in DIAGNOSTIC_AXIS_THRESHOLDS.items():
        assert len(tup) == 4, axis
        pass_thr, quality_thr, direction, margin = tup
        assert pass_thr is None or isinstance(pass_thr, float)
        assert isinstance(quality_thr, float)
        assert direction in ("increase", "decrease")
        assert isinstance(margin, float) and margin > 0


def test_interface_dG_quality_classification_and_below_accept():
    # pass=0.0 (dG<0 accept), quality=-56.0, decrease, margin=9.0
    recs = [
        _bc_rec("a", interface_dG=-60.0),  # better than quality -> pass
        _bc_rec("b", interface_dG=-50.0),  # within 9 of -56 -> near
        _bc_rec("c", interface_dG=-40.0),  # fails quality, but dG<0 -> passes accept
        _bc_rec("d", interface_dG=5.0),    # positive dG -> fails accept floor
    ]
    s = build_diagnostic_axis_stats(recs)["interface_dG"]
    assert (s.pass_count, s.near_pass_count, s.fail_count) == (1, 1, 2)
    assert s.below_accept_count == 1            # only the +5 design clears no floor
    assert s.pass_threshold == 0.0 and s.quality_threshold == -56.0


def test_buried_sasa_no_longer_dead():
    # old (1200,'increase',200) passed everything >=1177; new pass=None,
    # quality=1650, margin=330 -> the 1200-1650 tail now discriminates.
    recs = [
        _bc_rec("a", buried_sasa=1300.0),  # fail (350 below quality > 330)
        _bc_rec("b", buried_sasa=1400.0),  # near (250 <= 330)
        _bc_rec("c", buried_sasa=1700.0),  # pass
    ]
    s = build_diagnostic_axis_stats(recs)["buried_sasa"]
    assert (s.pass_count, s.near_pass_count, s.fail_count) == (1, 1, 1)
    assert s.pass_threshold is None            # BindCraft dSASA>=1 is a no-op
    assert s.below_accept_count == 0           # quality-only axis: no accept floor


def test_binder_pTM_quality_only_no_floor():
    # old invented (0.60,'increase',0.05) sat below the whole range -> 100% pass.
    # new pass=None, quality=0.79, margin=0.05 flags the low-confidence tail.
    recs = [
        _bc_rec("a", binder_pTM_avg=0.61),  # fail (0.18 below quality)
        _bc_rec("b", binder_pTM_avg=0.75),  # near (0.04 <= 0.05)
        _bc_rec("c", binder_pTM_avg=0.85),  # pass
    ]
    s = build_diagnostic_axis_stats(recs)["binder_pTM_avg"]
    assert (s.pass_count, s.near_pass_count, s.fail_count) == (1, 1, 1)
    assert s.pass_threshold is None and s.below_accept_count == 0


def test_boltzgen_axes_read_from_bins():
    # BoltzGen previously had ZERO diagnostic axes. design_to_target_iptm:
    # pass=0.50, quality=0.60, increase, margin=0.09.
    recs = [
        _bg_rec("a", design_to_target_iptm=0.72, design_ptm=0.85),  # pass
        _bg_rec("b", design_to_target_iptm=0.55, design_ptm=0.82),  # near
        _bg_rec("c", design_to_target_iptm=0.45, design_ptm=0.60),  # fail+below accept
        _bg_rec("d", design_to_target_iptm=0.65, design_ptm=0.81),  # pass
    ]
    stats = build_diagnostic_axis_stats(recs)
    assert "design_to_target_iptm" in stats     # sourced from bins, not metrics
    s = stats["design_to_target_iptm"]
    assert s.n == 4
    assert s.below_accept_count == 1            # only the 0.45 design clears no floor
    assert s.pass_threshold == 0.50 and s.quality_threshold == 0.60


def test_boltzgen_bins_do_not_pollute_strict_path():
    # BoltzGen records stay metrics={} so they are NOT strict successes.
    from trex.success_criteria import is_strict_success
    rec = _bg_rec("a", design_to_target_iptm=0.99, design_ptm=0.99)
    assert not is_strict_success(rec.metrics)


def _cpx_rec(rid: str, min_ipae: float) -> ResultRecord:
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.2,
                 "min_ipae": min_ipae},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.5, exit_status="ok",
        bins={},
    )


def test_near_band_never_overflows_accept_floor():
    # min_ipae: pass=0.2258, quality=0.07, decrease, margin=0.21. The naive
    # quality near-band reaches 0.07+0.21=0.28 > accept 0.2258, so a design at
    # 0.25 (worse than the accept floor) must be FAIL + below_accept, NOT 'near'.
    recs = [_cpx_rec("a", 0.05), _cpx_rec("b", 0.15), _cpx_rec("c", 0.25)]
    s = build_diagnostic_axis_stats(recs)["min_ipae"]
    assert (s.pass_count, s.near_pass_count, s.fail_count) == (1, 1, 1)
    assert s.below_accept_count == 1            # the 0.25 design, forced to fail
    assert s.pass_threshold is not None


def test_coverage_is_per_source_not_global():
    # A BoltzGen-heavy window must NOT dilute the BindCraft axes below the 25%
    # gate (the C-6 bug). 4 BindCraft + 20 BoltzGen: a global denominator (24)
    # would drop interface_dG (4 < 0.25*24=6); per-source (denom=4) keeps it.
    bc = [_bc_rec(f"bc{i}", interface_dG=-55.0) for i in range(4)]
    bg = [_bg_rec(f"bg{i}", design_to_target_iptm=0.62) for i in range(20)]
    stats = build_diagnostic_axis_stats(bc + bg)
    assert "interface_dG" in stats and stats["interface_dG"].n == 4
    assert "design_to_target_iptm" in stats and stats["design_to_target_iptm"].n == 20


def test_label_axis_rejects_nonfinite():
    for bad in (float("nan"), float("inf"), float("-inf"), None):
        assert label_axis(bad, 0.65, "increase", 0.05)[0] == "missing"
    # and a NaN metric is NOT silently counted as pass in the aggregator
    recs = [_bc_rec(f"a{i}", shape_complementarity=0.70) for i in range(3)]
    recs.append(_bc_rec("nan", shape_complementarity=float("nan")))
    s = build_diagnostic_axis_stats(recs)["shape_complementarity"]
    assert s.n == 3 and s.pass_count == 3   # the NaN record is dropped, not a pass


def test_foldseek_decouples_collapse_from_su_objective():
    # The live SU objective uses TM 0.60 (easy-cluster
    # --tmscore-threshold 0.6 on binder chains); the collapse / near-miss diversity
    # T-ReX live control keeps strict-SU, recent collapse, and near-miss dedup
    # on the same TM0.60 structural novelty threshold. TM0.5/TM0.8 are
    # post-hoc reporting or explicit ablation thresholds.
    cfg = FoldseekConfig()
    assert cfg.min_tm_score == 0.60          # live strict-SU objective
    assert cfg.collapse_tm_score == 0.60     # duplicate_fraction / top_bin_share
    assert cfg.min_tm_score == cfg.collapse_tm_score

def test_minority_family_axes_not_suppressed():
    # Complexa-heavy window (CD45-like): 30 Complexa + 4 BindCraft. A shared
    # n_metrics denominator (round(0.25*34)=9) would drop ALL BindCraft interface
    # axes (4 < 9) — suppressing the minority family on exactly the specialized
    # targets. Per-axis carrier gating keeps them (4 >= DIAGNOSTIC_MIN_N=3).
    cpx = [_cpx_rec(f"c{i}", 0.05) for i in range(30)]
    bc = [_bc_rec(f"b{i}", interface_dG=-55.0) for i in range(4)]
    stats = build_diagnostic_axis_stats(cpx + bc)
    assert "interface_dG" in stats and stats["interface_dG"].n == 4   # minority kept
    assert "min_ipae" in stats and stats["min_ipae"].n == 30          # majority kept
    # symmetric: BindCraft-heavy window must keep the lone Complexa axis
    stats2 = build_diagnostic_axis_stats(
        [_bc_rec(f"b{i}", interface_dG=-55.0) for i in range(30)]
        + [_cpx_rec(f"c{i}", 0.05) for i in range(4)]
    )
    assert "min_ipae" in stats2 and stats2["min_ipae"].n == 4
    assert "interface_dG" in stats2 and stats2["interface_dG"].n == 30


def test_remediation_map_is_consistent_with_thresholds():
    # every diagnostic axis has a remediation entry (lever or corroboration-only)
    assert set(DIAGNOSTIC_AXIS_REMEDIATION) == set(DIAGNOSTIC_AXIS_THRESHOLDS)
    assert CORROBORATION_ONLY_AXES == frozenset(
        a for a, v in DIAGNOSTIC_AXIS_REMEDIATION.items() if v is None
    )
    for a in ("ipTM", "min_ipae", "binder_pLDDT_avg", "avg_ipsae"):
        assert DIAGNOSTIC_AXIS_REMEDIATION[a] is not None
    for a in ("shape_complementarity", "interface_unsat_hbonds"):
        assert a in CORROBORATION_ONLY_AXES               # no direct lever


def test_worst_actionable_diagnostic_skips_corroboration_only():
    # ipTM=0.65 (levered, below quality 0.75) + interface_unsat_hbonds=4
    # (corroboration-only, below quality 2): the blocker must be the LEVERED axis.
    rec = ResultRecord(
        result_id="x", parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.20, "binder_scRMSD": 1.1,
                 "ipTM": 0.65, "interface_unsat_hbonds": 4.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok", bins={},
    )
    assert worst_actionable_diagnostic_axis(rec) == "ipTM"
    # when only a corroboration-only axis fails, there is no actionable blocker
    rec2 = ResultRecord(
        result_id="y", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.20, "binder_scRMSD": 1.1,
                 "shape_complementarity": 0.50, "ipTM": 0.90},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok", bins={},
    )
    assert worst_actionable_diagnostic_axis(rec2) is None


def test_planner_prompt_surfaces_lever_map():
    from trex.planner import PLANNER_SYSTEM
    assert "DIAGNOSTIC REMEDIATION LEVERS" in PLANNER_SYSTEM
    assert "weights_iptm" in PLANNER_SYSTEM            # the real lever, now surfaced
    assert "CORROBORATION-ONLY" in PLANNER_SYSTEM
    assert "shape_complementarity" in PLANNER_SYSTEM   # listed as corroboration-only


def test_dropped_axes_absent():
    # contact_density / interface_contact_density (inverted) and design_iptm
    # (duplicate) / structure_confidence / native_rmsd (dead) were intentionally
    # not carried as diagnostic axes.
    for dropped in (
        "contact_density", "interface_contact_density",
        "design_iptm", "structure_confidence", "native_rmsd",
    ):
        assert dropped not in DIAGNOSTIC_AXIS_THRESHOLDS
