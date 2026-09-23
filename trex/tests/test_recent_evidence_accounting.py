"""Tests for recent-evidence reuse, route compute accounting, and normalized LLM-call
status.
"""

from __future__ import annotations

from types import SimpleNamespace

from trex.evidence_reducer import method_health
from trex.live_tick import _decision_signature, _llm_call_record
from trex.controller import _route_row_current_rate, _route_row_recent_su
from trex.schemas import AxisStat, MethodHealthSummary, ResultRecord
from trex.tests.test_recipe_extraction import _ac


def _mh(**kw) -> MethodHealthSummary:
    base = dict(
        family="bindcraft", attempts=1, completions=1, timeouts=0, nonzero_exits=0,
        raw_artifacts=1, accepted_artifacts=1, score_files=1, strict_yield=0,
        near_miss_yield=0, routed_proxy=None,
    )
    base.update(kw)
    return MethodHealthSummary(**base)


def _ev(**kw):
    base = dict(
        state_label="productive", run_su_count=5, run_su_count_delta=1,
        su_per_gpu_h_recent=0.4, global_new_strict=5, near_miss_count=2,
        top_bin_share=0.3, duplicate_fraction=0.2, sequence_dedup_status="ok",
        sequence_dedup_coverage=1.0, seq_unique_strict_count=5,
        joint_struct_seq_unique_count=5, seq_duplicate_fraction=0.1,
        top_seq_bin_share=0.2, production_panel_status="ok",
        production_panel_selected_ids=["a", "b"], production_panel_value=0.7,
        remaining_wall_h=40.0, method_health={"complexa_beam": _mh(family="complexa_beam")},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_signature_changes_when_top_level_recent_signal_changes():
    a = _ev(su_per_gpu_h_recent=0.4, run_su_count_delta=1)
    b = _ev(su_per_gpu_h_recent=0.0, run_su_count_delta=0)  # recent rate dried up
    assert _decision_signature(a) != _decision_signature(b)


def test_signature_changes_when_per_family_recent_rate_changes():
    a = _ev(method_health={"complexa_beam": _mh(family="complexa_beam", su_per_gpu_h_recent=0.5)})
    b = _ev(method_health={"complexa_beam": _mh(family="complexa_beam", su_per_gpu_h_recent=0.0)})
    assert _decision_signature(a) != _decision_signature(b)


def test_signature_changes_when_diagnostic_driver_changes():
    def stat(deficit: float) -> AxisStat:
        return AxisStat(
            pass_count=0, near_pass_count=2, fail_count=1,
            median_raw=0.60, median_calibrated=0.60, median_deficit=deficit,
            calibration_status="provisional", n=3,
        )

    a = _ev(
        diagnostic_axis_stats={"ipTM": stat(0.05)},
        diagnostic_driver_tldr="ipTM:increase,fail=1/3,near=2,def=0.05,levered",
    )
    b = _ev(
        diagnostic_axis_stats={"ipTM": stat(0.30)},
        diagnostic_driver_tldr="ipTM:increase,fail=1/3,near=2,def=0.30,levered",
    )
    assert _decision_signature(a) != _decision_signature(b)


def test_signature_changes_when_route_diagnostic_improvement_changes():
    base_row = {
        "strategy_key": "route::boltzgen:default:{}",
        "scope": "route",
        "status": "observed",
        "route_role": "generator_with_af2_score_conversion",
        "record_recent_new_su": 0,
        "new_su_per_route_gpu_h": 0.0,
        "route_gpu_h": 5.0,
        "strict_per_su": 0.0,
        "marginal_status": "dry_low_quality",
        "pending_score_conversion_count": 0,
    }
    a = _ev(route_values=[dict(base_row, diagnostic_improvement_score=0.0)])
    b = _ev(route_values=[dict(
        base_row,
        diagnostic_improvement_score=0.62,
        diagnostic_improvement_axes=["design_to_target_iptm:levered:score=0.62"],
    )])
    assert _decision_signature(a) != _decision_signature(b)


def test_signature_changes_when_family_diagnostic_improvement_changes():
    base_row = {
        "strategy_key": "family::boltzgen",
        "scope": "family",
        "family": "boltzgen",
        "status": "observed",
        "marginal_status": "dry_low_quality",
        "diagnostic_improvement_score": 0.0,
    }
    a = _ev(route_values=[dict(base_row)])
    b = _ev(route_values=[dict(
        base_row,
        diagnostic_improvement_score=0.48,
        diagnostic_improvement_axes=["design_to_target_iptm:levered:score=0.48"],
    )])
    assert _decision_signature(a) != _decision_signature(b)


def test_route_recent_signal_ignores_unsafe_score_conversion_record_window():
    score_conversion_row = {
        "route_role": "generator_with_af2_score_conversion",
        "canonical_refilter_gpu_h": 0.08,
        "record_recent_new_su": 2,
        "record_recent_new_su_per_route_gpu_h": 25.0,
        "new_su_recent_gpu": 0,
        "medium_recent_new_su": 1,
        "medium_recent_new_su_per_route_gpu_h": 0.2,
        "new_su_per_route_gpu_h": 0.05,
    }
    assert _route_row_recent_su(score_conversion_row) == 1
    assert _route_row_current_rate(score_conversion_row) == 0.2

    direct_row = dict(
        score_conversion_row,
        route_role="direct_scored_generator",
        canonical_refilter_gpu_h=0.0,
        medium_recent_new_su=0,
        medium_recent_new_su_per_route_gpu_h=None,
    )
    assert _route_row_recent_su(direct_row) == 2
    assert _route_row_current_rate(direct_row) == 25.0


def test_signature_stable_when_only_lifetime_changes_but_recent_same():
    # lifetime su_per_gpu_h differs but everything decision-relevant is identical
    a = _ev(method_health={"complexa_beam": _mh(family="complexa_beam", su_per_gpu_h=9.9, su_per_gpu_h_recent=0.5)})
    b = _ev(method_health={"complexa_beam": _mh(family="complexa_beam", su_per_gpu_h=1.1, su_per_gpu_h_recent=0.5)})
    # Hold cumulative evidence equal when testing the recent-evidence signature.
    c = _ev(method_health={"complexa_beam": _mh(family="complexa_beam", su_per_gpu_h=1.1, su_per_gpu_h_recent=0.5)})
    assert _decision_signature(b) == _decision_signature(c)


def test_chained_rate_includes_failed_refold_route_cost():
    """A FAILED downstream refold still consumed GPU and is real route cost."""
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=2.0,
        exit_status="ok", panel_ready=False,
    )
    strict_child = ResultRecord(
        result_id="refilter_ok", parent_ids=["chain_a", "bc_parent"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", panel_ready=False, bins={"foldseek_su": "FS_new"},
    )
    failed_child = ResultRecord(  # refold that did NOT pass strict
        result_id="refilter_fail", parent_ids=["chain_b", "bc_parent"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 70.0, "iPAE": 0.5, "binder_scRMSD": 3.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", panel_ready=False, bins={"foldseek_su": "FS_bad"},
    )
    spawning = {
        "bc_parent": _ac("bc_cand", family="bindcraft", op="bindcraft_default", config={}),
        "refilter_ok": _ac("chain_a", family="structure_refilter", op="af2_multimer", parent_result_id="bc_parent"),
        "refilter_fail": _ac("chain_b", family="structure_refilter", op="af2_multimer", parent_result_id="bc_parent"),
    }
    mh = method_health([parent, strict_child, failed_child], spawning)
    assert mh["bindcraft"].chained_strict_yield_su == 1
    # route cost = 2.0 (upstream) + 0.05 (ok refold) + 0.05 (failed refold) = 2.10
    assert mh["bindcraft"].chained_su_per_gpu_h == 1.0 / 2.10


def _call_record(fail_reason):
    return _llm_call_record(
        "planner", tick_id="t1", cfg_model="m", prompt_text="p", raw_text="r",
        valid=False, fail_reason=fail_reason, confidence=None, abstain=False,
        fallback=True, latency_s=0.1, usage={},
    )


# The parse_status values _llm_call_record can actually store on a valid=False
# path (= fail_reason.split(":")[0]) plus the critic's direct "timeout" and the
# valid=True "ok". MUST stay in sync with schemas.LLMCallRecord.parse_status.
_PARSE_STATUS_CONTRACT = {
    "ok", "parse_fail", "schema_fail", "timeout",
    "call_error", "no_hypotheses_or_candidates",
}


def test_parse_status_producer_emits_clean_contract_tokens():
    """Each valid=False fail_reason yields a clean token in the declared set —
    split(':')[0] is sufficient (no parametrized tokens reach this path: abstain/
    low_confidence/empty_cards are returned with valid=True → 'ok')."""
    for fr, expect in [
        ("call_error:TimeoutError:boom", "call_error"),
        ("schema_fail:missing_field", "schema_fail"),
        ("parse_fail", "parse_fail"),
        ("no_hypotheses_or_candidates", "no_hypotheses_or_candidates"),
        (None, "parse_fail"),
    ]:
        rec = _call_record(fr)
        got = rec.parse_status
        assert got == expect
        assert rec.fail_reason == fr
        assert got in _PARSE_STATUS_CONTRACT, f"{got!r} drifted from schema Literal"


def test_parse_status_literal_matches_contract():
    """The schema Literal source text must enumerate exactly the contract set —
    catches drift if a producer/Literal is edited without the other."""
    import inspect

    from trex import schemas

    src = inspect.getsource(schemas.LLMCallRecord)
    decl = src.split("parse_status: Literal[", 1)[1].split("]", 1)[0]
    declared = {tok.strip().strip('"').strip("'") for tok in decl.split(",") if tok.strip()}
    assert declared == _PARSE_STATUS_CONTRACT, (declared, _PARSE_STATUS_CONTRACT)


def _refilter_strict(rid, parent_rid, fs, gpu_h=0.05):
    return ResultRecord(
        result_id=rid, parent_ids=["chain", parent_rid], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.8},
        metrics_calibrated={}, route_lineage=["structure_refilter"], gpu_h=gpu_h,
        exit_status="ok", panel_ready=False, bins={"foldseek_su": fs},
    )


def _bc_gen(rid, gpu_h=2.0):
    return ResultRecord(
        result_id=rid, parent_ids=["c"], target_id="t", backend_family="bindcraft",
        runtime_bucket_id="rb", metrics={}, metrics_calibrated={},
        route_lineage=["bindcraft"], gpu_h=gpu_h, exit_status="ok", panel_ready=False,
    )


def _reduce(all_results, window_results, spawning):
    from trex.evidence_reducer import reduce_evidence
    return reduce_evidence(
        tick_id="v7r005", target_id="t", target_class="tc",
        elapsed_wall_h=1.0, remaining_wall_h=40.0, pending_children=0,
        worker_gpu_h_total=sum(r.gpu_h for r in all_results),
        all_results=all_results, window_results=window_results,
        run_su_count=1, run_su_count_delta=1, duplicate_fraction=0.0,
        near_miss_count=0, top_bin_share=0.0, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="m", spawning_actions=spawning,
    )


def test_chained_recent_populated_when_diagnostic_lane_productive_this_window():
    parent = _bc_gen("bc_parent", gpu_h=2.0)
    child = _refilter_strict("refilter_child", "bc_parent", "FS_new", gpu_h=0.05)
    spawning = {
        "bc_parent": _ac("c", family="bindcraft", op="bindcraft_default", config={}),
        "refilter_child": _ac("chain_a", family="structure_refilter",
                              op="af2_multimer", parent_result_id="bc_parent"),
    }
    ev = _reduce([parent, child], [parent, child], spawning)
    mh = ev.method_health["bindcraft"]
    assert mh.chained_strict_yield_su == 1            # cumulative
    assert mh.chained_strict_yield_su_recent == 1     # marginal, first-seen this window
    # route-level recent rate = 1 / (2.0 upstream + 0.05 refilter)
    assert mh.chained_su_per_gpu_h_recent == 1.0 / 2.05


def test_chained_recent_is_none_when_lane_went_dry_but_lifetime_high():
    """Past chained successes need not imply recent productivity."""
    old_parent = _bc_gen("bc_old", gpu_h=2.0)
    old_chain = _refilter_strict("refilter_old", "bc_old", "FS_x", gpu_h=0.05)
    new_parent = _bc_gen("bc_new", gpu_h=2.0)   # still launching bindcraft, but no NEW chained SU
    spawning = {
        "bc_old": _ac("c0", family="bindcraft", op="bindcraft_default", config={}),
        "refilter_old": _ac("chain_old", family="structure_refilter",
                            op="af2_multimer", parent_result_id="bc_old"),
        "bc_new": _ac("c1", family="bindcraft", op="bindcraft_default", config={}),
    }
    all_results = [old_parent, old_chain, new_parent]
    window_results = [new_parent]                # the chained SU is PRE-window
    ev = _reduce(all_results, window_results, spawning)
    mh = ev.method_health["bindcraft"]
    assert mh.chained_strict_yield_su == 1                  # lifetime still credits it
    assert mh.chained_su_per_gpu_h is not None              # lifetime rate > 0
    assert mh.chained_strict_yield_su_recent == 0           # nothing new this window
    assert mh.chained_su_per_gpu_h_recent is None  # Lifetime credit is retained, but a window without new SUs has zero recent productivity.


def test_bindcraft_accepted_pdb_emitted_when_native_metrics_missing(tmp_path):
    """An Accepted PDB whose native CSV metrics are missing must still emit a
    (non-strict) record with pdb_path so the auto-chain can refilter it."""
    import csv as _csv

    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext
    from trex.success_criteria import is_strict_success

    designs = tmp_path / "designs"
    accepted = designs / "Accepted"
    accepted.mkdir(parents=True)
    (accepted / "d1_model1.pdb").write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    # CSV row with the Design name but NO metric columns at all.
    with (designs / "final_design_stats.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["Design"])
        w.writeheader()
        w.writerow({"Design": "d1"})

    ctx = ParserContext(
        target_id="t", runtime_bucket_id="rb", candidate_id="cand",
        parent_ids=["cand"], method_family="bindcraft",
    )
    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 1
    r = recs[0]
    assert r.artifacts.get("pdb_path", "").endswith("d1_model1.pdb")
    assert not is_strict_success(r.metrics)          # never strict (no strict keys)
    # no native strict-axis keys present (all were missing)
    assert "bindcraft_native_pLDDT" not in r.metrics
