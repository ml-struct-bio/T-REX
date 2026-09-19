"""#10 (2026-06-13): the parent_model_refold loop must close.

An advisory refold writes refold_* (never the canonical strict keys) and links
to its parent via bins["refilter_source"]. build_refold_probe_outcomes joins the
two so the LLM can act on the answer:
  - structure_limited: refold would pass the FIXED gate while the parent's
    canonical score did not  -> regenerate/redesign that parent to mint SU.
  - confirmed_limited: refold also fails -> the design itself is the problem.
A probe on a parent that ALREADY passes canonical is redundant (ignored), and a
record with no refold_* keys is not a probe.
"""

from __future__ import annotations

from trex.evidence_reducer import build_refold_probe_outcomes
from trex.schemas import ResultRecord


def _rec(rid, *, metrics, bins=None, fam="complexa_beam") -> ResultRecord:
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1", backend_family=fam,
        runtime_bucket_id="rb1", metrics=metrics, metrics_calibrated=dict(metrics),
        route_lineage=[], gpu_h=0.3, exit_status="ok",  # type: ignore[arg-type]
        bins=bins or {},
    )


_PASS = {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0}        # strict pass
_FAIL_SCRMSD = {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 2.5}  # only scRMSD fails


def _refold(rid, parent_rid, *, refold_pass: bool) -> ResultRecord:
    r = _PASS if refold_pass else _FAIL_SCRMSD
    return _rec(
        rid,
        metrics={
            "refold_pLDDT": r["pLDDT"],
            "refold_iPAE": r["iPAE"],
            "refold_binder_scRMSD": r["binder_scRMSD"],
        },
        bins={"refilter_source": parent_rid, "af2_strict_basis": "advisory_refold"},
        fam="structure_refilter",
    )


def test_empty_when_no_probes():
    res = [_rec("p1", metrics=_FAIL_SCRMSD)]
    assert build_refold_probe_outcomes(res, res) == {}


def test_structure_limited_parent_surfaced():
    parent = _rec("p1", metrics=_FAIL_SCRMSD)        # canonical FAILS
    refold = _refold("rf1", "p1", refold_pass=True)  # friendlier fold PASSES
    res = [parent, refold]
    out = build_refold_probe_outcomes(res, res)
    assert out["probed"] == 1
    assert out["structure_limited"] == 1
    assert out["confirmed_limited"] == 0
    assert out["example_parent_ids"] == ["p1"]


def test_confirmed_limited_when_refold_also_fails():
    parent = _rec("p1", metrics=_FAIL_SCRMSD)
    refold = _refold("rf1", "p1", refold_pass=False)  # still fails
    res = [parent, refold]
    out = build_refold_probe_outcomes(res, res)
    assert out["probed"] == 1
    assert out["structure_limited"] == 0
    assert out["confirmed_limited"] == 1
    assert "example_parent_ids" not in out


def test_redundant_probe_on_passing_parent_ignored():
    parent = _rec("p1", metrics=_PASS)               # parent ALREADY mints SU
    refold = _refold("rf1", "p1", refold_pass=True)
    res = [parent, refold]
    out = build_refold_probe_outcomes(res, res)
    # probed counts the refold record, but it is neither structure_limited nor
    # confirmed_limited because the parent already passes on its own.
    assert out["probed"] == 1
    assert out["structure_limited"] == 0
    assert out["confirmed_limited"] == 0
    assert out.get("parent_already_passes") == 1     # R2: counted so buckets sum


def test_unjoined_when_parent_unresolvable():
    """R1: refilter_source fell back to a basename/path (no result_id match) ->
    unjoined, NOT structure_limited — we cannot prove the parent failed canonical."""
    refold = _refold("rf1", "some_basename.pdb", refold_pass=True)  # parent absent
    out = build_refold_probe_outcomes([refold], [refold])
    assert out["probed"] == 1
    assert out["structure_limited"] == 0
    assert out.get("unjoined") == 1
    assert "example_parent_ids" not in out


def test_buckets_sum_to_probed():
    """R2: probed == structure_limited + confirmed_limited + parent_already_passes
    + unjoined (legible accounting for the LLM)."""
    recs = [
        _rec("pass_parent", metrics=_PASS),
        _rec("fail_parent", metrics=_FAIL_SCRMSD),
        _refold("rf_struct", "fail_parent", refold_pass=True),   # structure_limited
        _refold("rf_conf", "fail_parent", refold_pass=False),    # confirmed_limited
        _refold("rf_redund", "pass_parent", refold_pass=True),   # parent_already_passes
        _refold("rf_unjoined", "missing.pdb", refold_pass=True),  # unjoined
    ]
    out = build_refold_probe_outcomes(recs, recs)
    total = (out["structure_limited"] + out["confirmed_limited"]
             + out.get("parent_already_passes", 0) + out.get("unjoined", 0))
    assert out["probed"] == total == 4


def test_non_refold_records_not_counted():
    res = [_rec("p1", metrics=_PASS), _rec("p2", metrics=_FAIL_SCRMSD)]
    assert build_refold_probe_outcomes(res, res) == {}


def test_parent_lookup_uses_all_results_window_counts():
    """Parent may be OLD (outside the window); the refold (in window) must still
    join to it via all_results."""
    parent = _rec("p_old", metrics=_FAIL_SCRMSD)
    refold = _refold("rf1", "p_old", refold_pass=True)
    out = build_refold_probe_outcomes([refold], [parent, refold])
    assert out["structure_limited"] == 1
    assert out["example_parent_ids"] == ["p_old"]


def _stub_planner():
    from trex.schemas import PlannerOutput
    return PlannerOutput(valid=True, abstain=False, confidence=0.7,
                         fail_reason=None, cards=[], rationale="x",
                         raw_text='{"cards": []}', usage={})


def _stub_sup():
    from trex.schemas import SupervisorOutput
    return SupervisorOutput(valid=True, abstain=False, confidence=0.7,
                            fail_reason=None,
                            mode_mixture={"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
                            candidate_decisions=[], rationale="y",
                            raw_text='{}', usage={})


def test_signal_reaches_prompt_view_via_live_tick_and_dropped_when_empty(tmp_path):
    """The whole point of #10: the refold-probe block must reach the LLM-facing
    prompt view through the real reduce_evidence wiring, and the empty case must
    NOT add noise. Uses run_live_tick to build a real EvidenceSummary."""
    from unittest.mock import patch
    from trex.archive import Archive
    from trex.live_tick import LiveTickConfig, run_live_tick
    from trex.planner import build_evidence_for_prompt
    from trex.schemas import EvidenceSummary, TargetConstraint

    def _rid(name, metrics, bins, target="t1"):
        return ResultRecord(
            result_id=name, parent_ids=[], target_id=target,
            backend_family="complexa_beam", runtime_bucket_id="rb1",
            metrics=metrics, metrics_calibrated=dict(metrics), route_lineage=[],
            gpu_h=0.3, exit_status="ok", bins=bins,  # type: ignore[arg-type]
        )

    target = TargetConstraint(target_id="t1", target_class="c")

    # (a) populated: a canonical-FAILING parent + an advisory refold that passes
    arc = Archive(tmp_path / "pop")
    arc.append(_rid("p1", _FAIL_SCRMSD, {"foldseek": "B1", "foldseek_su": "B1"}))
    arc.append(_rid("rf1", {"refold_pLDDT": 95.0, "refold_iPAE": 0.10,
                            "refold_binder_scRMSD": 1.0},
                    {"refilter_source": "p1", "af2_strict_basis": "advisory_refold"}))
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.refold_probe_outcomes.get("structure_limited") == 1
    view = build_evidence_for_prompt(ev)
    assert view.get("refold_probe_outcomes", {}).get("structure_limited") == 1

    # (b) empty: no refold records -> block absent from the prompt view
    arc2 = Archive(tmp_path / "empty")
    arc2.append(_rid("q1", _PASS, {"foldseek": "C1", "foldseek_su": "C1"}))
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc2, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    ev2 = list(arc2.iter_records(EvidenceSummary))[-1]
    assert ev2.refold_probe_outcomes == {}
    assert "refold_probe_outcomes" not in build_evidence_for_prompt(ev2)


def test_negative_evidence_block_surfaces_zero_su_lane(tmp_path):
    """Review #5: a family with >=1.5 GPU-h and 0 SU must appear in the TL;DR
    'NEGATIVE EVIDENCE' block so the LLM cannot miss the wasted lane."""
    from unittest.mock import patch
    from trex.archive import Archive
    from trex.live_tick import LiveTickConfig, run_live_tick
    from trex.planner import evidence_tldr
    from trex.schemas import EvidenceSummary, TargetConstraint

    def _r(rid, fam, m, gpu, bins, exit_status="ok"):
        return ResultRecord(
            result_id=rid, parent_ids=[], target_id="t1", backend_family=fam,
            runtime_bucket_id="rb1", metrics=m, metrics_calibrated=dict(m),
            route_lineage=[], gpu_h=gpu, exit_status=exit_status, bins=bins,  # type: ignore[arg-type]
        )

    arc = Archive(tmp_path / "neg")
    arc.append(_r("bc1", "bindcraft", {"bindcraft_native_pLDDT": 80.0}, 2.0, {}))
    arc.append(_r("bc2", "bindcraft", {}, 0.5, {}, exit_status="timeout"))
    arc.append(_r("cx1", "complexa_beam", _PASS, 0.4,
                  {"foldseek": "B1", "foldseek_su": "B1"}))
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    tl = evidence_tldr(ev)
    assert "NEGATIVE EVIDENCE" in tl
    assert "bindcraft" in tl.split("NEGATIVE EVIDENCE")[1]


def test_f1_strict_su_top_bin_share_needs_min_population(tmp_path):
    """F1: a tiny strict-SU window (1-2 SU in one cluster) must NOT signal collapse
    (share stays None); only at >= MIN_COUNT does the top-bin share fire."""
    from unittest.mock import patch
    from trex.archive import Archive
    from trex.live_tick import LiveTickConfig, run_live_tick
    from trex.schemas import EvidenceSummary, TargetConstraint

    def _strict(rid, fb):
        m = {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0}
        return ResultRecord(
            result_id=rid, parent_ids=[], target_id="t1", backend_family="complexa_beam",
            runtime_bucket_id="rb1", metrics=m, metrics_calibrated=dict(m), route_lineage=[],
            gpu_h=0.3, exit_status="ok", bins={"foldseek": fb, "foldseek_su": fb})

    target = TargetConstraint(target_id="t1", target_class="c")
    for n, expect_none in ((2, True), (5, False)):
        arc = Archive(tmp_path / f"f1_{n}")
        for i in range(n):
            arc.append(_strict(f"s{i}", "C1"))  # all in ONE strict-SU cluster
        with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
             patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
            run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                          elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
        ev = list(arc.iter_records(EvidenceSummary))[-1]
        if expect_none:
            assert ev.strict_su_top_bin_share is None, f"n={n} should not signal collapse"
        else:
            assert ev.strict_su_top_bin_share == 1.0, f"n={n} should signal collapse"


def test_evidence2_strict_su_top_bin_share_surfaced_in_tldr(tmp_path):
    """evidence-2: when strict_su_top_bin_share fires (>=4 unique SU in one basin),
    the TL;DR must name it and flag the diversify trigger so the LLM's PROPOSAL
    reasoning sees the SU-objective collapse read (not only the deterministic clamp)."""
    from unittest.mock import patch
    from trex.archive import Archive
    from trex.live_tick import LiveTickConfig, run_live_tick
    from trex.planner import evidence_tldr
    from trex.schemas import EvidenceSummary, TargetConstraint

    def _strict(rid, fb):
        m = {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0}
        return ResultRecord(
            result_id=rid, parent_ids=[], target_id="t1", backend_family="complexa_beam",
            runtime_bucket_id="rb1", metrics=m, metrics_calibrated=dict(m), route_lineage=[],
            gpu_h=0.3, exit_status="ok", bins={"foldseek": fb, "foldseek_su": fb})

    arc = Archive(tmp_path / "ev2")
    for i in range(5):
        arc.append(_strict(f"s{i}", "C1"))   # 5 unique SU all in ONE basin → share 1.0
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.strict_su_top_bin_share == 1.0
    tl = evidence_tldr(ev)
    assert "strict_su_top_bin_share" in tl
    assert "COLLAPSE>=0.5:diversify" in tl
