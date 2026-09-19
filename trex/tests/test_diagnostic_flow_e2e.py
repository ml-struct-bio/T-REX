"""End-to-end smoke: every intermediate (diagnostic) metric reaches BOTH the
Planner prompt and the Supervisor/critic payload, while the strict success
criterion (pLDDT/iPAE/scRMSD) is untouched.

Deterministic (no LLM): exercises the real reduce_evidence → build_evidence_for_prompt
(Planner) and build_critic_payload (Supervisor) path on a mixed BindCraft/Complexa/
BoltzGen archive. The qwen_* smokes exercise the live LLM citation behavior; this
one proves the plumbing.
"""
from __future__ import annotations

import types

from trex.critic import build_critic_payload
from trex.evidence_reducer import reduce_evidence
from trex.planner import (
    PLANNER_SYSTEM,
    build_evidence_for_prompt,
    build_user_prompt,
)
from trex.schemas import ResultRecord, to_jsonable
from trex.success_criteria import is_strict_success


def _complexa(rid: str, *, iptm: float, strict: bool, su_bin: str,
              min_ipae: float = 0.05, ipae: float | None = None) -> ResultRecord:
    # min_ipae=0.05 PASSES its 0.07 quality band; avg/max_ipsae pass theirs — so a
    # "clean" record's only possible levered blocker is ipTM.
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1", backend_family="complexa_beam",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 93.0,
                 "iPAE": ipae if ipae is not None else (0.18 if strict else 0.26),
                 "binder_scRMSD": 1.1,
                 "ipTM": iptm, "min_ipae": min_ipae, "avg_ipsae": 0.58, "max_ipsae": 0.62},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.5, exit_status="ok",
        bins={"foldseek_su": su_bin},
    )


def _bindcraft(rid: str, *, dG: float = -60.0, strict: bool = True) -> ResultRecord:
    # dG=-60 PASSES the -56 quality band; binder_pLDDT_avg/buried_sasa pass too —
    # so a clean BindCraft record has no levered diagnostic blocker.
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1", backend_family="bindcraft",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 93.0 if strict else 80.0,
                 "iPAE": 0.18 if strict else 0.40, "binder_scRMSD": 1.0,
                 "interface_dG": dG, "shape_complementarity": 0.63,
                 "interface_hbonds": 6.0, "interface_unsat_hbonds": 2.0,
                 "buried_sasa": 1700.0, "binder_pLDDT_avg": 0.92,
                 "binder_pTM_avg": 0.80, "hotspot_rmsd": 1.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.3, exit_status="ok",
        bins={"foldseek_su": rid} if strict else {},
    )


def _boltzgen(rid: str, *, iptm: float, ptm: float) -> ResultRecord:
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1", backend_family="boltzgen",
        runtime_bucket_id="rb1", metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.2, exit_status="ok",
        bins={"boltzgen_design_to_target_iptm": f"{iptm:.4f}",
              "boltzgen_design_iiptm": f"{iptm:.4f}",
              "boltzgen_design_ptm": f"{ptm:.4f}",
              "boltzgen_min_design_to_target_pae": "12.0"},
    )


def _archive() -> list[ResultRecord]:
    recs: list[ResultRecord] = []
    # Strict success blocked ONLY by a levered diagnostic (ipTM 0.65 < 0.75 band).
    # Best strict quality (lowest iPAE) so it ranks into the top-K best exemplars.
    recs.append(_complexa("cx_blocked", iptm=0.65, strict=True, su_bin="b1", ipae=0.15))
    # Clean Complexa strict successes (ipTM 0.85 passes; no levered blocker)
    for i in range(3):
        recs.append(_complexa(f"cx{i}", iptm=0.85, strict=True, su_bin=f"b{i+2}"))
    # Complexa near-misses (fail iPAE within margin)
    for i in range(2):
        recs.append(_complexa(f"cxnm{i}", iptm=0.70, strict=False, su_bin=f"nm{i}"))
    # BindCraft (interface-physics axes) — clean; mix of strict + non-strict
    for i in range(5):
        recs.append(_bindcraft(f"bc{i}", strict=(i % 2 == 0)))
    # BoltzGen advisory (bin axes only; metrics={} → never strict)
    for i in range(5):
        recs.append(_boltzgen(f"bg{i}", iptm=0.55, ptm=0.82))
    return recs


def _reduce(recs):
    return reduce_evidence(
        tick_id="t1", target_id="t1", target_class="test",
        elapsed_wall_h=12.0, remaining_wall_h=36.0, pending_children=0,
        worker_gpu_h_total=10.0, all_results=recs, window_results=recs,
        run_su_count=5, run_su_count_delta=2, duplicate_fraction=0.1,
        near_miss_count=3, top_bin_share=0.4, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="test",
        near_miss_cluster_by_result_id={},
    )


def test_strict_criterion_unchanged_by_diagnostics():
    # A BindCraft record with PERFECT diagnostics but failing strict is NOT strict.
    great_diag_bad_strict = _bindcraft("x", dG=-79.0, strict=False)
    assert not is_strict_success(great_diag_bad_strict.metrics)
    # A clean strict success stays strict regardless of its diagnostics.
    assert is_strict_success(_complexa("y", iptm=0.10, strict=True, su_bin="z").metrics)


def test_all_diagnostic_axes_reach_the_reducer():
    ev = _reduce(_archive())
    das = ev.diagnostic_axis_stats
    # BindCraft physics + Complexa confidence + BoltzGen bin axes all present
    for axis in ("interface_dG", "shape_complementarity", "ipTM", "min_ipae",
                 "design_to_target_iptm", "design_ptm"):
        assert axis in das, f"{axis} missing from diagnostic_axis_stats"
    # two-tier fields populated (not the old single-tier shape)
    iptm = das["ipTM"]
    assert iptm.quality_threshold is not None
    assert iptm.pass_threshold is not None
    assert hasattr(iptm, "below_accept_count")
    # alt-model advisory medians present for BoltzGen
    assert "boltzgen" in (ev.diagnostic_alt_model_scores or {})


def test_per_design_diagnostic_blocker_on_exemplars():
    from trex.evidence_reducer import CORROBORATION_ONLY_AXES
    ev = _reduce(_archive())
    blockers = {e.result_id: e.diagnostic_blocking_axis for e in ev.exemplars}
    # the strict-but-diagnostically-blocked design is flagged with its LEVERED axis
    assert blockers.get("cx_blocked") == "ipTM"
    # a corroboration-only axis is NEVER reported as a per-design blocker
    assert not (set(blockers.values()) & CORROBORATION_ONLY_AXES)
    # at least one clean strict success carries no actionable blocker
    assert any(v is None for v in blockers.values())


def test_planner_prompt_sees_all_diagnostics():
    ev = _reduce(_archive())
    view = build_evidence_for_prompt(ev)
    assert "diagnostic_axis_stats" in view and view["diagnostic_axis_stats"]
    assert "diagnostic_alt_model_scores" in view and view["diagnostic_alt_model_scores"]
    assert view.get("diagnostic_driver_tldr") and view["diagnostic_driver_tldr"] != "none"
    # exemplars carry the per-design diagnostic blocker into the prompt
    exemplars = to_jsonable(ev.exemplars)
    assert any(e.get("diagnostic_blocking_axis") == "ipTM" for e in exemplars)
    # the prompt string + system prompt expose the lever map
    prompt = build_user_prompt(ev, [], [])
    assert "diagnostic_axis_stats" in prompt
    assert "diagnostic_driver_tldr=" in prompt
    assert "norm_def=" in prompt  # TL;DR severity is margin-normalized, not raw-unit ranked.
    tldr = prompt.split("diagnostic_driver_tldr=", 1)[1].split(" | ", 1)[0].split("\n", 1)[0]
    assert view["diagnostic_driver_tldr"] == tldr
    drivers = tldr.split(";")
    assert len(drivers) <= 4
    assert "levered" in drivers[0]
    # Near-pass-only axes should rank by margin-normalized rescue signal, not by
    # string-order ties from fail_count == 0. Keep this property-level so future
    # diagnostic threshold recalibration does not break on a harmless axis rename.
    first = drivers[0]
    assert "fail=0/" in first and "near=0" not in first and "levered" in first, drivers
    assert sum("levered" in d for d in drivers) >= 2, drivers
    assert any("corroborates" in d for d in drivers), drivers
    assert "DIAGNOSTIC REMEDIATION LEVERS" in PLANNER_SYSTEM


def test_diagnostics_are_family_balanced_not_just_recent():
    # Recent 60 records are ALL BoltzGen, but Complexa + BindCraft were active
    # earlier. With a flat last-60 diagnostic window their axes would VANISH;
    # the family-balanced window keeps every active family's interface read.
    recs = []
    for i in range(80):
        recs.append(_complexa(f"cx{i}", iptm=0.85, strict=(i % 2 == 0), su_bin=f"b{i%20}"))
    for i in range(80):
        recs.append(_bindcraft(f"bc{i}", strict=(i % 2 == 0)))
    for i in range(80):
        recs.append(_boltzgen(f"bg{i}", iptm=0.55, ptm=0.82))
    ev = reduce_evidence(
        tick_id="t1", target_id="t1", target_class="test", elapsed_wall_h=24.0,
        remaining_wall_h=24.0, pending_children=0, worker_gpu_h_total=100.0,
        all_results=recs, window_results=recs[-60:], run_su_count=20,
        run_su_count_delta=1, duplicate_fraction=0.2, near_miss_count=4,
        top_bin_share=0.3, panel_ready_count=0, panel_ready_bins_covered=0,
        llm_model="test", near_miss_cluster_by_result_id={},
    )
    das = set(ev.diagnostic_axis_stats)
    assert "design_ptm" in das       # BoltzGen — the recent window
    assert "ipTM" in das             # Complexa — NOT in the recent 60, kept anyway
    assert "interface_dG" in das     # BindCraft — NOT in the recent 60, kept anyway


def test_dedup_provenance_collapsed_to_one_key():
    ev = _reduce(_archive())
    view = build_evidence_for_prompt(ev)
    # the ~14 dedup/provenance status scalars collapse to a single dedup_trust
    assert "dedup_trust" in view
    for k in ("foldseek_su_status", "structure_dedup_scope",
              "near_miss_dedup_status", "sequence_dedup_status",
              "whole_archive_structure_dedup_fallback_count"):
        assert k not in view


def test_near_miss_dedup_disabled_is_visible_to_planner_when_raw_near_miss_exists():
    ev = reduce_evidence(
        tick_id="t1", target_id="t1", target_class="test",
        elapsed_wall_h=12.0, remaining_wall_h=36.0, pending_children=0,
        worker_gpu_h_total=10.0, all_results=_archive(), window_results=_archive(),
        run_su_count=5, run_su_count_delta=2, duplicate_fraction=0.1,
        near_miss_count=3, top_bin_share=0.4, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="test",
        near_miss_cluster_by_result_id={},
        near_miss_dedup_status="disabled",
        near_miss_dedup_coverage=None,
    )

    view = build_evidence_for_prompt(ev)

    assert isinstance(view["dedup_trust"], dict)
    assert view["dedup_trust"]["near_miss_dedup_status"] == "disabled"


def test_diagnosis_outcomes_wired_into_evidence():
    # parent ipTM-blocked, child improves ipTM + strict → diagnosis_outcomes
    recs = [
        ResultRecord(result_id="p", parent_ids=[], target_id="t1",
                     backend_family="complexa_beam", runtime_bucket_id="rb1",
                     metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.2, "ipTM": 0.65},
                     metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok", bins={}),
        ResultRecord(result_id="c", parent_ids=["p"], target_id="t1",
                     backend_family="complexa_beam", runtime_bucket_id="rb1",
                     metrics={"pLDDT": 93.0, "iPAE": 0.18, "binder_scRMSD": 1.1, "ipTM": 0.85},
                     metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok", bins={}),
    ]
    ev = reduce_evidence(
        tick_id="t", target_id="t1", target_class="c", elapsed_wall_h=1.0,
        remaining_wall_h=47.0, pending_children=0, worker_gpu_h_total=1.0,
        all_results=recs, window_results=recs, run_su_count=1, run_su_count_delta=1,
        duplicate_fraction=0.0, near_miss_count=0, top_bin_share=0.0,
        panel_ready_count=0, panel_ready_bins_covered=0, llm_model="t",
        near_miss_cluster_by_result_id={})
    assert "ipTM" in ev.diagnosis_outcomes
    assert ev.diagnosis_outcomes["ipTM"]["improved"] == 1
    assert "remediation outcomes" in build_user_prompt(ev, [], [])


def test_unified_reasoner_flag_default_off():
    from trex.live_tick import LiveTickConfig
    assert LiveTickConfig().unified_reasoner is False   # needs LLM re-validation


def test_prompt_form_tldr_units_and_denoising():
    ev = _reduce(_archive())
    prompt = build_user_prompt(ev, [], [])
    head = prompt.split("\n", 1)[0]
    assert head.startswith("TLDR")               # decision 5-tuple is FIRST
    assert "state=" in head and "run_SU=" in head
    assert "pLDDT 0-100" in head                  # units stated up top
    view = build_evidence_for_prompt(ev)
    # name-lie fixed in the LLM view; route-attribution window now lives
    # inside objective_summary instead of as another top-level denominator.
    assert "worker_gpu_h_last_3_ticks" not in view
    assert "worker_gpu_h_window" not in view
    assert "completed_worker_gpu_h_window" in view["objective_summary"]
    # strict axis_stats no longer carry inert two-tier fields
    for s in view["axis_stats"].values():
        assert "below_accept_count" not in s and "quality_threshold" not in s
    # but DIAGNOSTIC axes MUST retain the two-tier fields (regression guard: a
    # future _trim that stripped them from diagnostics would be silent otherwise)
    for s in view["diagnostic_axis_stats"].values():
        assert "quality_threshold" in s and "below_accept_count" in s
    # floats are rounded (no 0.36363636... tails)
    import json as _json
    assert "3636363636" not in _json.dumps(view)


def test_supervisor_prompt_guidance_is_first():
    from trex.supervisor import build_user_prompt as sup_prompt
    ev = _reduce(_archive())
    prompt = sup_prompt(ev, [], [])
    head = prompt.split("\n{", 1)[0]              # everything before the JSON body
    assert "TLDR" in head and "MIXTURE GUIDANCE" in head   # evidence guidance up front
    assert "mode_mixture" in prompt


def test_supervisor_payload_sees_all_diagnostics():
    ev = _reduce(_archive())
    planner_stub = types.SimpleNamespace(
        valid=True, abstain=False, rationale="r", cards=[]
    )
    payload = build_critic_payload(ev, planner_stub)
    das = payload["diagnostic_axis_stats"]
    # Supervisor now sees the FULL two-tier view, not just {pass, median}
    assert "ipTM" in das
    for field in ("near_pass", "fail", "below_accept", "quality_threshold"):
        assert field in das["ipTM"], f"critic missing diagnostic field {field}"
    assert "boltzgen" in payload["diagnostic_alt_model_scores"]
    # and the per-design diagnostic blockers
    blocked = {b["result_id"]: b["diagnostic_blocking_axis"]
               for b in payload["exemplar_diagnostic_blockers"]}
    assert blocked.get("cx_blocked") == "ipTM"
