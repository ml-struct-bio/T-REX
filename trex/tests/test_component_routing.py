"""Tests for measurement-specific hypotheses, route accounting, refinement defaults, and
exhausted lineages.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from trex.candidate_builder import BuilderConfig, build_candidates
from trex.evidence_reducer import (
    ReducerConfig,
    build_exemplars,
    build_route_values,
    dominant_deficit_axis,
    extract_recipes,
    method_health,
    reduce_evidence,
    representative_examples,
    stuck_lineage_roots,
)
from trex.schemas import (
    ActionCandidate,
    AxisStat,
    EvidenceSummary,
    Exemplar,
    FeasibilityCheck,
    HypothesisCard,
    JointPatternCount,
    LLMHealthSummary,
    PredictedChange,
    PreserveConstraint,
    Recipe,
    ReasoningTrace,
    ResultRecord,
    RouteHealthSummary,
    RouteValueSummary,
)

STRICT = {"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 1.0}


def _rec(rid, fam, metrics, *, parent=None, gpu_h=0.5, bins=None, tick="v7r001"):
    bins = dict(bins or {})
    # Provide both scored-structure and strict-only cluster bins.
    if "foldseek" in bins and "foldseek_su" not in bins:
        bins["foldseek_su"] = bins["foldseek"]
    return ResultRecord(
        result_id=rid, parent_ids=([parent] if parent else []), target_id="t1",
        backend_family=fam, runtime_bucket_id="rb1",
        metrics=metrics, metrics_calibrated=dict(metrics), route_lineage=[],
        gpu_h=gpu_h, exit_status="ok", bins=bins, tick_id=tick,
    )


# dominant_deficit_axis

def test_dominant_deficit_axis_none_when_all_pass():
    assert dominant_deficit_axis({}) is None
    assert dominant_deficit_axis({"pLDDT": 0.0, "iPAE": 0.0, "binder_scRMSD": 0.0}) is None


def test_dominant_deficit_axis_picks_largest_raw_when_unambiguous():
    assert dominant_deficit_axis({"pLDDT": 10.0, "iPAE": 0.0}) == "pLDDT"
    assert dominant_deficit_axis({"binder_scRMSD": 1.0}) == "binder_scRMSD"


def test_dominant_deficit_axis_is_margin_normalized():
    # RAW argmax would pick pLDDT (2.0 > 0.174). Margin-normalized: pLDDT 2/5=0.4
    # vs iPAE 0.174/0.05=3.48 -> iPAE is the true blocker. This normalization is
    # the whole point (incommensurate units).
    assert dominant_deficit_axis({"pLDDT": 2.0, "iPAE": 0.174}) == "iPAE"


def test_representative_examples_sets_dominant_axis():
    cfg = ReducerConfig()
    # interface blocker: fold + sequence pass, iPAE fails
    r = _rec("r1", "complexa_beam", {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0})
    ex = representative_examples([r], cfg)
    assert len(ex) == 1
    assert ex[0].dominant_deficit_axis == "iPAE"


# Per-recipe SU per GPU-hour

def _su_recipe(records):
    recs = extract_recipes(records, current_tick=1)
    strict = [r for r in recs if r.recipe_class == "strict_success"]
    assert strict, "expected a strict_success recipe"
    return strict[0]


def test_recipe_su_per_gpu_h_strict_only():
    # two distinct Foldseek clusters -> 2 SU, total gpu_h 1.0 -> 2.0 SU/gpu-h
    recs = [
        _rec("s1", "complexa_beam", STRICT, gpu_h=0.5, bins={"foldseek": "A"}),
        _rec("s2", "complexa_beam", STRICT, gpu_h=0.5, bins={"foldseek": "B"}),
    ]
    rcp = _su_recipe(recs)
    assert rcp.su_per_gpu_h == 2.0


def test_recipe_su_per_gpu_h_dedups_same_cluster():
    # same Foldseek cluster -> 1 SU even though 2 strict records, gpu_h 1.0 -> 1.0
    recs = [
        _rec("s1", "complexa_beam", STRICT, gpu_h=0.5, bins={"foldseek": "A"}),
        _rec("s2", "complexa_beam", STRICT, gpu_h=0.5, bins={"foldseek": "A"}),
    ]
    assert _su_recipe(recs).su_per_gpu_h == 1.0


def test_recipe_faster_route_earns_higher_su_per_gpu_h():
    slow = [_rec("s1", "complexa_beam", STRICT, gpu_h=1.0, bins={"foldseek": "A"})]
    fast = [_rec("s1", "complexa_beam", STRICT, gpu_h=0.25, bins={"foldseek": "A"})]
    assert _su_recipe(fast).su_per_gpu_h > _su_recipe(slow).su_per_gpu_h


def test_recipe_su_per_gpu_h_none_for_non_strict():
    # near-miss (iPAE fails by a band, fold+seq pass) -> NOT strict -> no SU credit
    near = {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}
    recs = [_rec(f"n{i}", "complexa_beam", near, gpu_h=0.5) for i in range(3)]
    out = extract_recipes(recs, current_tick=1)
    for r in out:
        if r.recipe_class != "strict_success":
            assert r.su_per_gpu_h is None


# stuck_lineage_roots

def test_stuck_lineage_flags_parent_after_k_non_improving():
    cfg = ReducerConfig()
    iface_fail = {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}  # iPAE dominant
    parent = _rec("P", "complexa_beam", iface_fail)
    kids = [_rec(f"c{i}", "structure_refilter", iface_fail, parent="P") for i in range(3)]
    stuck = stuck_lineage_roots([parent, *kids], cfg)
    roots = {e["root_result_id"] for e in stuck}
    assert "P" in roots
    e = next(x for x in stuck if x["root_result_id"] == "P")
    assert e["dominant_axis"] == "iPAE" and e["attempts"] == 3


def test_complexa_ipae_noise_pair_adds_sequence_hallucination():
    ev = dataclasses.replace(_evidence(), state_label="stalled", near_miss_count=1)
    hyp = _hyp(
        axis="iPAE",
        family="complexa_beam",
        suggestions={"complexa_beam": {"sc_scale_noise": 0.30}},
    )
    cand = _only(build_candidates([hyp], ev))
    assert cand.config_delta["sc_scale_noise"] == 0.30
    assert cand.config_delta["refinement_algorithm"] == "sequence_hallucination"
    assert any("auto_pair:refinement_algorithm=sequence_hallucination" in r for r in cand.feasibility.reasons)


def test_v73_sparse_complexa_reward_retry_defers_reward_to_material_search():
    ev = dataclasses.replace(_evidence(), state_label="stalled")
    hyp = _hyp(
        axis="iPAE",
        family="complexa_beam",
        suggestions={"complexa_beam": {"reward_i_ptm_weight": 1.5, "reward_plddt_weight": 0.5}},
    )
    cand = _only(build_candidates([hyp], ev))
    assert "reward_i_ptm_weight" not in cand.config_delta
    assert "reward_plddt_weight" not in cand.config_delta
    assert cand.config_delta["beam_width"] == 8
    assert cand.config_delta["n_branch"] == 4
    assert any("reward_deferred:evidence_sparse" in r for r in cand.feasibility.reasons)


def test_v73_evidence_rich_complexa_reward_keeps_reward_and_uses_material_search():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        axis_stats={"iPAE": AxisStat(0, 0, 32, 0.65, 0.65, 0.42, "calibrated", 32)},
    )
    hyp = _hyp(
        axis="iPAE",
        family="complexa_beam",
        suggestions={"complexa_beam": {"reward_i_ptm_weight": 1.5, "reward_plddt_weight": 0.5}},
    )
    cand = _only(build_candidates([hyp], ev))
    assert cand.config_delta["reward_i_ptm_weight"] == 1.5
    assert cand.config_delta["reward_plddt_weight"] == 0.5
    # Reward is allowed only after axis evidence is present, and then paired
    # with a material search change; for beam this is breadth, not hard-coded
    # sc_scale_noise.
    assert cand.config_delta["beam_width"] == 8
    assert cand.config_delta["n_branch"] == 4
    assert "sc_scale_noise" not in cand.config_delta
    assert not any("reward_deferred:evidence_sparse" in r for r in cand.feasibility.reasons)


def test_v73_plddt_reward_retry_uses_breadth_not_forced_sc_noise():
    ev = dataclasses.replace(
        _evidence(),
        state_label="productive",
        axis_stats={"pLDDT": AxisStat(0, 0, 32, 72.0, 72.0, 18.0, "calibrated", 32)},
    )
    hyp = _hyp(
        axis="pLDDT",
        family="complexa_beam",
        suggestions={"complexa_beam": {"reward_plddt_weight": 1.5}},
    )
    cand = _only(build_candidates([hyp], ev))
    assert cand.config_delta["reward_plddt_weight"] == 1.5
    assert cand.config_delta["beam_width"] == 7
    assert cand.config_delta["n_branch"] == 4
    assert "sc_scale_noise" not in cand.config_delta
    assert any("axis_matched_material_search_for_reward_retry" in r for r in cand.feasibility.reasons)


def test_v73_reward_with_existing_material_knob_is_not_overpaired():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        axis_stats={"iPAE": AxisStat(0, 0, 32, 0.65, 0.65, 0.42, "calibrated", 32)},
    )
    hyp = _hyp(
        axis="iPAE",
        family="complexa_beam",
        suggestions={"complexa_beam": {"reward_i_ptm_weight": 1.5, "beam_width": 8}},
    )
    cand = _only(build_candidates([hyp], ev))
    assert cand.config_delta["reward_i_ptm_weight"] == 1.5
    assert cand.config_delta["beam_width"] == 8
    assert "n_branch" not in cand.config_delta
    assert not any("axis_matched_material_search_for_reward_retry" in r for r in cand.feasibility.reasons)


def test_untrusted_near_miss_count_does_not_auto_pair_sequence_hallucination():
    ev = dataclasses.replace(
        _evidence(),
        state_label="productive",
        near_miss_count=1,
        near_miss_dedup_status="disabled",
        near_miss_dedup_coverage=None,
    )
    hyp = _hyp(
        axis="iPAE",
        family="complexa_beam",
        suggestions={"complexa_beam": {"sc_scale_noise": 0.30}},
    )
    cand = _only(build_candidates([hyp], ev))
    assert cand.config_delta["sc_scale_noise"] == 0.30
    assert "refinement_algorithm" not in cand.config_delta
    assert not any("auto_pair:refinement_algorithm=sequence_hallucination" in r for r in cand.feasibility.reasons)


def test_stuck_lineage_not_flagged_when_a_child_improves():
    cfg = ReducerConfig()
    iface_fail = {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}
    parent = _rec("P", "complexa_beam", iface_fail)
    # one child passes iPAE (closes the dominant deficit) -> not stuck
    kids = [
        _rec("c0", "structure_refilter", iface_fail, parent="P"),
        _rec("c1", "structure_refilter", STRICT, parent="P"),
        _rec("c2", "structure_refilter", iface_fail, parent="P"),
    ]
    stuck = stuck_lineage_roots([parent, *kids], cfg)
    assert "P" not in {e["root_result_id"] for e in stuck}


def test_stuck_lineage_pLDDT_gives_up_after_one():
    cfg = ReducerConfig()
    fold_fail = {"pLDDT": 80.0, "iPAE": 0.1, "binder_scRMSD": 1.0}  # pLDDT dominant
    parent = _rec("P", "complexa_beam", fold_fail)
    kids = [_rec("c0", "structure_refilter", fold_fail, parent="P")]  # only 1 attempt
    stuck = stuck_lineage_roots([parent, *kids], cfg)
    e = next(x for x in stuck if x["root_result_id"] == "P")
    assert e["dominant_axis"] == "pLDDT" and e["reason"] == "structural_pLDDT"


def test_stuck_lineage_below_k_not_flagged():
    cfg = ReducerConfig()
    iface_fail = {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}
    parent = _rec("P", "complexa_beam", iface_fail)
    kids = [_rec(f"c{i}", "structure_refilter", iface_fail, parent="P") for i in range(2)]
    assert stuck_lineage_roots([parent, *kids], cfg) == []  # iPAE needs K=3


# Candidate construction with axis-specific settings and exhausted parent lineages

def _evidence(*, stuck=None, recipes=None, exemplars=None,
              production_panel_selected_ids=None,
              production_near_miss_ids=None,
              parent_artifact_result_ids=None,
              route_values=None) -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="t1", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=1, remaining_wall_h=10, completed_children=0, pending_children=0,
        worker_gpu_h_total=1, worker_gpu_h_last_3_ticks=1,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="productive", examples=[], metric_availability={},
        recipes=recipes or [],
        exemplars=exemplars or [],
        stuck_lineage_roots=stuck or [],
        production_panel_selected_ids=production_panel_selected_ids or [],
        production_near_miss_ids=production_near_miss_ids or [],
        parent_artifact_result_ids=parent_artifact_result_ids or [],
        route_values=route_values or [],
    )


def _hyp(axis="iPAE", family="complexa_beam", suggestions=None, baseline=("b",)):
    return HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim="x", mode_affinity={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange(axis, "decrease", list(baseline), 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=[family],
        config_delta_suggestions=suggestions or {},
    )


def _only(cands):
    cs = [c for c in cands if c.hypothesis_ids == ["h1"]]
    assert len(cs) == 1, [c.candidate_id for c in cands]
    return cs[0]


def _exemplar(rid: str) -> Exemplar:
    return Exemplar(
        kind="near_miss",
        result_id=rid,
        family="complexa_beam",
        operator_id="complexa_beam_default",
        config_delta={},
        metrics={"pLDDT": 95.0, "iPAE": 0.4, "binder_scRMSD": 1.0},
        axis_deficits={"iPAE": 0.3},
        dominant_deficit_axis="iPAE",
        parent_result_id=None,
    )


def test_rec3_seq_blocker_seeds_hallucination():
    c = _only(build_candidates([_hyp(axis="binder_scRMSD")], _evidence(),
                               include_warmstart=False))
    assert c.config_delta == {"refinement_algorithm": "sequence_hallucination"}


def test_rec3_interface_blocker_seeds_metric_adaptive_noise():
    c = _only(build_candidates([_hyp(axis="iPAE")], _evidence(),
                               include_warmstart=False))
    assert c.config_delta == {"sc_scale_noise": 0.2975}


def test_rec3_structural_blocker_seeds_nothing():
    # pLDDT (a bad fold) has no fixed-backbone tweak -> no soft-default
    c = _only(build_candidates([_hyp(axis="pLDDT")], _evidence(),
                               include_warmstart=False))
    assert c.config_delta == {}


def test_rec3_does_not_override_llm_config():
    c = _only(build_candidates(
        [_hyp(axis="iPAE", suggestions={"complexa_beam": {"beam_width": 8}})],
        _evidence(), include_warmstart=False))
    assert c.config_delta == {"beam_width": 8}
    assert "sc_scale_noise" not in c.config_delta


_STUCK = [{"root_result_id": "P", "family": "complexa_beam",
           "dominant_axis": "iPAE", "attempts": 3, "reason": "no_improvement_after_3"}]


def test_rec4_blocks_refinement_of_stuck_parent():
    # structure_refilter REFINES the parent backbone (requires_parent_pdb=True) ->
    # refining a stuck parent is blocked.
    ev = _evidence(stuck=_STUCK, exemplars=[_exemplar("P")], parent_artifact_result_ids=["P"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter",
                                     baseline=("P",))], ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert any("lineage_stuck_regenerate" in r for r in c.feasibility.reasons)


def test_rec4_does_not_block_denovo_generator_citing_stuck_parent():
    # A generator that does not consume the parent structure remains eligible when
    # citing a parent whose refinement lineage is exhausted.
    ev = _evidence(stuck=_STUCK)
    c = _only(build_candidates([_hyp(axis="iPAE", family="complexa_beam",
                                     baseline=("P",))], ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert not any("lineage_stuck_regenerate" in r for r in c.feasibility.reasons)
    assert c.parent_result_id is None


def test_rec4_does_not_block_non_stuck_parent():
    ev = _evidence(stuck=_STUCK, exemplars=[_exemplar("Q")], parent_artifact_result_ids=["Q"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter",
                                     baseline=("Q",))], ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id == "Q"


_STUCK_PARTIAL = [{"root_result_id": "P", "family": "complexa_beam",
                   "dominant_axis": "iPAE", "attempts": 3,
                   "reason": "no_improvement_after_3",
                   "exhausted_families": ["proteinmpnn_redesign"]}]


def test_gap2_blocks_only_the_exhausted_rescue_family():
    ev = _evidence(stuck=_STUCK_PARTIAL, exemplars=[_exemplar("P")], parent_artifact_result_ids=["P"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=("P",))], ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert any("rescue_exhausted" in r for r in c.feasibility.reasons)


def test_gap2_allows_untried_rescue_family_on_same_parent():
    # An unexhausted family can still refine this parent.
    ev = _evidence(stuck=_STUCK_PARTIAL, exemplars=[_exemplar("P")], parent_artifact_result_ids=["P"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter",
                                     baseline=("P",))], ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert not any(("rescue_exhausted" in r or "lineage_stuck" in r)
                   for r in c.feasibility.reasons)


def test_gap2_empty_exhausted_set_blocks_all_refinement():
    # mined-out / pLDDT arms emit exhausted_families=[] -> block ALL refinement.
    stuck = [{"root_result_id": "P", "family": "complexa_beam", "dominant_axis": None,
              "attempts": 6, "reason": "strict_duplicate_2of6", "exhausted_families": []}]
    ev = _evidence(stuck=stuck, exemplars=[_exemplar("P")], parent_artifact_result_ids=["P"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter",
                                     baseline=("P",))], ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert any("lineage_stuck_regenerate" in r for r in c.feasibility.reasons)


# Deep-stall fallback is evidence/capability driven, not a fixed FK/MCTS ladder.

def test_gap4_deep_stall_fallback_not_gated_by_warmstart():
    from dataclasses import replace
    ev = replace(_evidence(), state_label="deep_stall")
    cands = build_candidates([], ev, include_warmstart=False)  # cold-start already used
    fallback = [c for c in cands if c.candidate_id.startswith("evidence_fallback")]
    assert fallback
    assert not any(c.candidate_id.startswith("deepstall_ladder_") for c in cands)
    assert not {"structure_refilter", "proteinmpnn_redesign"} & {c.method_family for c in fallback}


def test_gap4_no_fallback_outside_stalled_states():
    cands = build_candidates([], _evidence(), include_warmstart=True)  # productive
    assert not any(c.candidate_id.startswith("evidence_fallback") for c in cands)


def test_deep_stall_complexa_only_cards_get_two_cross_family_probes():
    route_values = [{
        "strategy_key": "family::complexa_beam",
        "scope": "family",
        "family": "complexa_beam",
        "action_family": "complexa_beam",
        "route_gpu_h": 24.0,
        "new_su": 1,
        "new_su_per_route_gpu_h": 0.04,
        "gpu_recent_new_su_per_route_gpu_h": 0.0,
        "record_recent_new_su_per_route_gpu_h": 0.0,
        "marginal_status": "dry",
        "status": "observed",
    }]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_beam")], ev, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    fams = {c.method_family for c in cross if c.feasibility.all_ok()}
    assert {"bindcraft", "boltzgen"} <= fams
    assert any(c.hypothesis_ids == ["h1"] and c.method_family == "complexa_beam" for c in cands)


def test_diagnostic_i4_mcts_seed_from_plddt_fail_ipae_pass():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates(
        [_hyp(axis="pLDDT", family="complexa_beam")],
        ev,
        cfg=BuilderConfig(i4_mcts_seed_enabled=True),
        include_warmstart=False,
    )
    i4 = [c for c in cands if c.candidate_id.startswith("diagnostic_i4_mcts_")]
    assert len(i4) == 1
    cand = i4[0]
    assert cand.method_family == "complexa_mcts"
    assert cand.config_delta["n_simulations"] == 20
    assert cand.config_delta["exploration_prob"] == 0.5
    assert cand.config_delta["exploration_constant"] == 1.0
    assert cand.config_delta["filter_samples_limit"] == 100
    assert cand.config_delta["reward_i_pae_weight"] == -1.0
    assert cand.config_delta["reward_plddt_weight"] == 1.0
    assert cand.config_delta["refinement_algorithm"] == "sequence_hallucination"
    assert cand.config_delta["greedy_percentage"] == 5.0
    assert "hard_target_signal" in cand.evidence_refs


def test_diagnostic_i4_mcts_seed_from_axis_stats_plddt_fail_ipae_near():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        axis_stats={
            "pLDDT": AxisStat(1, 1, 8, 82.0, 82.0, 8.0, "calibrated", 10),
            "iPAE": AxisStat(2, 4, 4, 0.23, 0.23, 0.01, "calibrated", 10),
        },
    )
    cands = build_candidates(
        [_hyp(axis="pLDDT", family="complexa_beam")],
        ev,
        cfg=BuilderConfig(i4_mcts_seed_enabled=True),
        include_warmstart=False,
    )
    i4 = [c for c in cands if c.candidate_id.startswith("diagnostic_i4_mcts_")]
    assert len(i4) == 1


def test_diagnostic_i4_mcts_not_for_joint_interface_failure():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        axis_stats={
            "pLDDT": AxisStat(1, 1, 8, 82.0, 82.0, 8.0, "calibrated", 10),
            "iPAE": AxisStat(0, 0, 10, 0.62, 0.62, 0.39, "calibrated", 10),
        },
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "both_fail", 10)],
    )
    cands = build_candidates(
        [_hyp(axis="pLDDT", family="complexa_beam")],
        ev,
        cfg=BuilderConfig(i4_mcts_seed_enabled=True),
        include_warmstart=False,
    )
    assert not any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_uses_dry_floor_not_productive_label():
    ev = dataclasses.replace(
        _evidence(),
        state_label="productive",
        gpu_h_since_last_su=7.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates(
        [_hyp(axis="pLDDT", family="complexa_beam")],
        ev,
        cfg=BuilderConfig(i4_mcts_seed_enabled=True),
        include_warmstart=False,
    )
    assert any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_not_exposed_in_low_evidence():
    ev = dataclasses.replace(
        _evidence(),
        state_label="low_evidence",
        gpu_h_since_last_su=7.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates(
        [_hyp(axis="pLDDT", family="complexa_beam")],
        ev,
        cfg=BuilderConfig(i4_mcts_seed_enabled=True),
        include_warmstart=False,
    )
    assert not any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_is_not_automatic_before_warning_dry_floor():
    ev = dataclasses.replace(
        _evidence(),
        state_label="stalled",
        gpu_h_since_last_su=5.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates([_hyp(axis="pLDDT", family="complexa_beam")], ev, include_warmstart=False)
    assert not any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_is_available_at_warning_when_untried():
    ev = dataclasses.replace(
        _evidence(),
        state_label="stalled",
        gpu_h_since_last_su=7.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates([_hyp(axis="pLDDT", family="complexa_beam")], ev, include_warmstart=False)
    assert any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_is_default_on_at_deep_stall_when_untried():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates([_hyp(axis="pLDDT", family="complexa_beam")], ev, include_warmstart=False)
    assert any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_suppressed_after_sufficient_dry_probe():
    route_values = [{
        "strategy_key": "family::complexa_mcts",
        "scope": "family",
        "family": "complexa_mcts",
        "action_family": "complexa_mcts",
        "route_gpu_h": 3.5,
        "completions": 2,
        "new_su": 0,
        "near_miss_recent": 0,
        "pending_score_conversion_count": 0,
        "diagnostic_improvement_score": 0.0,
        "marginal_status": "dry_low_quality",
        "status": "observed",
    }]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates([_hyp(axis="pLDDT", family="complexa_beam")], ev, include_warmstart=False)
    assert not any(c.candidate_id.startswith("diagnostic_i4_mcts_") for c in cands)


def test_i4_mcts_seed_not_suppressed_by_generic_llm_mcts():
    ev = dataclasses.replace(
        _evidence(),
        state_label="deep_stall",
        gpu_h_since_last_su=14.0,
        joint_patterns=[JointPatternCount(("pLDDT", "iPAE"), "A_fail_B_pass", 3)],
    )
    cands = build_candidates([_hyp(axis="pLDDT", family="complexa_mcts")], ev, include_warmstart=False)
    i4 = [c for c in cands if c.candidate_id.startswith("diagnostic_i4_mcts_")]
    llm = [c for c in cands if c.hypothesis_ids == ["h1"] and c.method_family == "complexa_mcts"]
    assert llm
    assert i4
    assert i4[0].config_delta["n_simulations"] == 20
    assert i4[0].config_delta["refinement_algorithm"] == "sequence_hallucination"


def test_llm_can_propose_i4_mcts_config_without_auto_seed():
    suggestions = {
        "complexa_mcts": {
            "nsteps": 400,
            "nsamples": 4,
            "batch_size": 16,
            "n_simulations": 20,
            "exploration_prob": 0.5,
            "exploration_constant": 1.0,
            "filter_samples_limit": 100,
            "reward_i_pae_weight": -1.0,
            "reward_plddt_weight": 1.0,
            "refinement_algorithm": "sequence_hallucination",
            "enable_greedy_optimization": True,
            "n_greedy_iters": 15,
            "greedy_percentage": 5.0,
        }
    }
    cand = _only(build_candidates(
        [_hyp(axis="pLDDT", family="complexa_mcts", suggestions=suggestions)],
        _evidence(),
        include_warmstart=False,
    ))
    assert cand.method_family == "complexa_mcts"
    assert cand.config_delta["n_simulations"] == 20
    assert cand.config_delta["exploration_prob"] == 0.5
    assert cand.config_delta["exploration_constant"] == 1.0
    # MCTS is already material search. Sparse evidence defers scalar-only reward
    # retries, but does not strip rewards from an explicit I.4-style search card.
    assert cand.config_delta["reward_i_pae_weight"] == -1.0
    assert cand.config_delta["reward_plddt_weight"] == 1.0
    assert cand.config_delta["refinement_algorithm"] == "sequence_hallucination"
    assert cand.config_delta["filter_samples_limit"] == 100
    assert not any("reward_deferred:evidence_sparse" in r for r in cand.feasibility.reasons)
    assert cand.feasibility.all_ok(), cand.feasibility.reasons


def test_stalled_productive_recent_su_does_not_force_cross_family_probe():
    route_values = [{
        "strategy_key": "family::complexa_beam",
        "scope": "family",
        "family": "complexa_beam",
        "action_family": "complexa_beam",
        "route_gpu_h": 12.0,
        "new_su": 12,
        "new_su_recent_gpu": 2,
        "gpu_recent_new_su_per_route_gpu_h": 1.5,
        "marginal_status": "productive",
        "status": "promote",
    }]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="stalled",
        gpu_h_since_last_su=8.0,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_beam")], ev, include_warmstart=False)
    assert not any(c.candidate_id.startswith("evidence_fallback_cross_family") for c in cands)

def test_route_values_surface_diagnostic_improvement_score_for_no_su_route():
    recs = []
    for i, val in enumerate([0.42, 0.45, 0.46, 0.48, 0.61, 0.63, 0.64]):
        recs.append(_rec(
            f"bg{i}",
            "boltzgen",
            {},
            gpu_h=1.0,
            bins={"boltzgen_design_to_target_iptm": str(val)},
            tick=f"v7r{i:03d}",
        ))
    rows = build_route_values(recs, recs[-3:], spawning_actions=None)
    route = next(r for r in rows if r.scope == "route" and r.family == "boltzgen")
    fam = next(r for r in rows if r.scope == "family" and r.family == "boltzgen")
    assert route.new_su == 0
    assert route.diagnostic_improvement_score >= 0.35
    assert any("design_to_target_iptm" in x for x in route.diagnostic_improvement_axes)
    assert fam.diagnostic_improvement_score == route.diagnostic_improvement_score


def test_deep_stall_cross_family_skips_known_dry_low_quality_roots():
    route_values = [
        {
            "strategy_key": "family::complexa_beam",
            "scope": "family",
            "family": "complexa_beam",
            "action_family": "complexa_beam",
            "route_gpu_h": 30.0,
            "new_su": 1,
            "gpu_recent_new_su_per_route_gpu_h": 0.0,
            "record_recent_new_su_per_route_gpu_h": 0.0,
            "marginal_status": "dry",
            "status": "observed",
        },
        {
            "strategy_key": "family::boltzgen",
            "scope": "family",
            "family": "boltzgen",
            "action_family": "boltzgen",
            "route_gpu_h": 8.0,
            "new_su": 0,
            "near_miss_recent": 0,
            "gpu_recent_new_su_per_route_gpu_h": 0.0,
            "marginal_status": "dry_low_quality",
            "status": "observed",
        },
    ]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="deep_stall",
        gpu_h_since_last_su=20.0,
        worker_gpu_h_total=40.0,
        run_su_count=1,
        run_su_count_delta=0,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_mcts")], ev, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    fams = [c.method_family for c in cross if c.feasibility.all_ok()]
    assert "bindcraft" in fams
    assert "boltzgen" not in fams


def test_deep_stall_cross_family_keeps_diagnostic_improving_dry_root():
    route_values = [
        {
            "strategy_key": "family::complexa_beam",
            "scope": "family",
            "family": "complexa_beam",
            "action_family": "complexa_beam",
            "route_gpu_h": 30.0,
            "new_su": 1,
            "gpu_recent_new_su_per_route_gpu_h": 0.0,
            "marginal_status": "dry",
            "status": "observed",
        },
        {
            "strategy_key": "family::boltzgen",
            "scope": "family",
            "family": "boltzgen",
            "action_family": "boltzgen",
            "route_gpu_h": 8.0,
            "new_su": 0,
            "near_miss_recent": 0,
            "gpu_recent_new_su_per_route_gpu_h": 0.0,
            "marginal_status": "dry_low_quality",
            "status": "observed",
            "diagnostic_improvement_score": 0.55,
            "diagnostic_improvement_axes": ["design_to_target_iptm:levered:score=0.55"],
        },
    ]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="deep_stall",
        gpu_h_since_last_su=20.0,
        worker_gpu_h_total=40.0,
        run_su_count=1,
        run_su_count_delta=0,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_mcts")], ev, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    fams = {c.method_family for c in cross if c.feasibility.all_ok()}
    assert {"bindcraft", "boltzgen"} <= fams
    boltz = next(c for c in cross if c.method_family == "boltzgen")
    assert "diagnostic_score=0.55" in boltz.expected_signal


def test_cross_family_ranks_su_value_above_diagnostic_support():
    route_values = [
        {
            "strategy_key": "family::complexa_beam",
            "scope": "family",
            "family": "complexa_beam",
            "action_family": "complexa_beam",
            "route_gpu_h": 30.0,
            "new_su": 1,
            "marginal_status": "dry",
            "status": "observed",
        },
        {
            "strategy_key": "family::bindcraft",
            "scope": "family",
            "family": "bindcraft",
            "action_family": "bindcraft",
            "route_gpu_h": 5.0,
            "new_su": 1,
            "new_su_per_route_gpu_h": 0.2,
            "gpu_recent_new_su_per_route_gpu_h": 0.2,
            "marginal_status": "productive",
            "status": "promote",
        },
        {
            "strategy_key": "family::boltzgen",
            "scope": "family",
            "family": "boltzgen",
            "action_family": "boltzgen",
            "route_gpu_h": 5.0,
            "new_su": 0,
            "near_miss_recent": 0,
            "marginal_status": "dry_low_quality",
            "status": "observed",
            "diagnostic_improvement_score": 1.0,
            "diagnostic_improvement_axes": ["design_to_target_iptm:levered:score=1.00"],
        },
    ]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="deep_stall",
        gpu_h_since_last_su=20.0,
        worker_gpu_h_total=40.0,
        run_su_count=1,
        run_su_count_delta=0,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_mcts")], ev, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    assert [c.method_family for c in cross[:2]] == ["bindcraft", "boltzgen"]


def test_deep_stall_recovery_keeps_one_cross_family_probe_without_dropping_winner():
    route_values = [{
        "strategy_key": "family::complexa_mcts",
        "scope": "family",
        "family": "complexa_mcts",
        "action_family": "complexa_mcts",
        "route_gpu_h": 5.0,
        "new_su": 2,
        "new_su_recent_gpu": 1,
        "gpu_recent_new_su_per_route_gpu_h": 0.7,
        "record_recent_new_su_per_route_gpu_h": 0.7,
        "marginal_status": "productive",
        "status": "promote",
    }]
    history = [
        {"tick_id": f"v7r{40+i:03d}", "state_label": "deep_stall", "run_su_count_delta": 0}
        for i in range(5)
    ] + [
        {"tick_id": "v7r053", "state_label": "productive_duplicate", "run_su_count_delta": 1}
    ]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="productive",
        run_su_count=6,
        run_su_count_delta=1,
        worker_gpu_h_total=44.0,
        run_su_per_worker_gpu_h_total=0.14,
        recent_ticks_history=history,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_mcts")], ev, include_warmstart=False)
    assert any(c.hypothesis_ids == ["h1"] and c.method_family == "complexa_mcts" for c in cands)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    assert len([c for c in cross if c.feasibility.all_ok()]) >= 1
    assert all(not c.method_family.startswith("complexa_") for c in cross)


def test_deep_stall_recovery_does_not_force_probe_when_total_rate_is_healthy():
    route_values = [{
        "strategy_key": "family::complexa_mcts",
        "scope": "family",
        "family": "complexa_mcts",
        "action_family": "complexa_mcts",
        "route_gpu_h": 5.0,
        "new_su": 8,
        "new_su_recent_gpu": 1,
        "gpu_recent_new_su_per_route_gpu_h": 0.7,
        "marginal_status": "productive",
        "status": "promote",
    }]
    history = [
        {"tick_id": f"v7r{40+i:03d}", "state_label": "deep_stall", "run_su_count_delta": 0}
        for i in range(5)
    ]
    ev = dataclasses.replace(
        _evidence(route_values=route_values),
        state_label="productive",
        run_su_count=20,
        run_su_count_delta=1,
        worker_gpu_h_total=20.0,
        run_su_per_worker_gpu_h_total=1.0,
        recent_ticks_history=history,
    )
    cands = build_candidates([_hyp(axis="iPAE", family="complexa_mcts")], ev, include_warmstart=False)
    assert not any(c.candidate_id.startswith("evidence_fallback_cross_family") for c in cands)

def test_parent_required_rejects_aggregate_ref_as_parent():
    ev = _evidence()
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter",
                                     baseline=("diagnostic_chain_backlog",))],
                               ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert c.parent_result_id is None
    assert any("no_concrete_parent_result_id" in r for r in c.feasibility.reasons)


def test_parent_required_resolves_recipe_ref_to_representative_result():
    recipe = Recipe(
        recipe_hash="abc123",
        operator_id="complexa_beam_default",
        method_family="complexa_beam",
        config_delta={},
        recipe_class="near_miss",
        target_id="t",
        target_class="c",
        descendant_count=2,
        median_metrics={"iPAE": 0.4},
        representative_result_ids=["r_rep", "r_rep2"],
        recency_tick=3,
    )
    ev = _evidence(recipes=[recipe], parent_artifact_result_ids=["r_rep", "r_rep2"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=("recipe_abc123",))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id == "r_rep"
    assert c.baseline_result_id == "r_rep"

    c2 = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                      baseline=("r_rep2",))],
                                ev, include_warmstart=False))
    assert c2.feasibility.all_ok()
    assert c2.parent_result_id == "r_rep2"
    assert c2.baseline_result_id == "r_rep2"


def test_parent_required_resolves_exemplar_dot_alias_to_result_id():
    rid = "03d021e5fca142ac"
    ev = _evidence(exemplars=[_exemplar(rid)], parent_artifact_result_ids=[rid])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=(f"exemplars.{rid}",))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id == rid
    assert c.baseline_result_id == rid


def test_parent_required_rejects_unknown_explicit_result_id_ref():
    rid = "abcdef1234567890"
    ev = _evidence()
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=(rid,))],
                               ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert c.parent_result_id is None
    assert c.baseline_result_id is None
    assert any("unknown_result_id:abcdef1234567890" in r for r in c.feasibility.reasons)


def test_parent_required_accepts_known_explicit_result_id_ref():
    rid = "abcdef1234567890"
    ev = _evidence(exemplars=[_exemplar(rid)], parent_artifact_result_ids=[rid])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=(rid,))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id == rid
    assert c.baseline_result_id == rid


def test_reduce_evidence_records_only_artifact_backed_parent_ids(tmp_path):
    pdb = tmp_path / "parent.pdb"
    pdb.write_text("ATOM\n")
    with_artifact = dataclasses.replace(
        _rec("with_artifact", "complexa_beam", STRICT),
        artifacts={"pdb_path": str(pdb)},
    )
    missing_artifact = dataclasses.replace(
        _rec("missing_artifact", "complexa_beam", STRICT),
        artifacts={"pdb_path": str(tmp_path / "gone.pdb")},
    )
    no_artifact = _rec("no_artifact", "complexa_beam", STRICT)
    ev = reduce_evidence(
        tick_id="t1", target_id="t1", target_class="c",
        elapsed_wall_h=1.0, remaining_wall_h=10.0, pending_children=0,
        worker_gpu_h_total=1.0,
        all_results=[with_artifact, missing_artifact, no_artifact],
        window_results=[with_artifact, missing_artifact, no_artifact],
        run_su_count=1, run_su_count_delta=1, duplicate_fraction=None,
        near_miss_count=0, top_bin_share=None, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="m",
    )
    assert ev.parent_artifact_result_ids == ["with_artifact"]


def test_parent_required_rejects_known_result_without_artifact_backing():
    rid = "abcdef1234567890"
    ev = _evidence(
        exemplars=[_exemplar(rid)],
        parent_artifact_result_ids=["1111111111111111"],
    )
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=(rid,))],
                               ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert c.parent_result_id == rid
    assert any(
        "no_usable_parent_artifact:abcdef1234567890:proteinmpnn_redesign" in r
        for r in c.feasibility.reasons
    )


def test_parent_required_rejects_known_result_when_no_artifact_ids_exposed():
    rid = "abcdef1234567890"
    ev = _evidence(exemplars=[_exemplar(rid)], parent_artifact_result_ids=[])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=(rid,))],
                               ev, include_warmstart=False))
    assert not c.feasibility.all_ok()
    assert c.parent_result_id == rid
    assert any(
        "no_usable_parent_artifact:abcdef1234567890:proteinmpnn_redesign" in r
        for r in c.feasibility.reasons
    )


def test_parent_required_accepts_artifact_backed_known_result():
    rid = "abcdef1234567890"
    ev = _evidence(
        exemplars=[_exemplar(rid)],
        parent_artifact_result_ids=[rid],
    )
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign",
                                     baseline=(rid,))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id == rid
    assert not any("no_usable_parent_artifact" in r for r in c.feasibility.reasons)


def test_feasible_parent_bound_rescue_suppresses_route_replay_backstop():
    rid = "abcdef1234567890"
    route_values = [{
        "strategy_key": "route::complexa_fk_steering:op:default",
        "scope": "route",
        "family": "complexa_fk_steering",
        "root_family": "complexa_fk_steering",
        "action_family": "complexa_fk_steering",
        "operator_id": "complexa_fk_steering",
        "config_signature": "default",
        "config_delta": {},
        "status": "promote",
        "route_role": "direct_generation",
        "route_gpu_h": 1.0,
        "new_su": 1,
        "record_recent_new_su": 1,
        "record_recent_new_su_per_route_gpu_h": 1.0,
        "new_su_per_route_gpu_h": 1.0,
        "evidence_refs": ["route_values"],
    }]
    ev = _evidence(
        exemplars=[_exemplar(rid)],
        parent_artifact_result_ids=[rid],
        route_values=route_values,
    )
    cands = build_candidates(
        [_hyp(axis="iPAE", family="proteinmpnn_redesign", baseline=(rid,))],
        ev,
        include_warmstart=False,
    )
    rescue = [c for c in cands if c.method_family == "proteinmpnn_redesign"]
    assert rescue and rescue[0].feasibility.all_ok()
    assert not any(c.candidate_id.startswith("route_replay_") for c in cands)


def test_route_cap_saturation_blocks_refilter_but_not_generator_escape():
    ev = dataclasses.replace(
        _evidence(exemplars=[_exemplar("r_ex")], parent_artifact_result_ids=["r_ex"]),
        state_label="deep_stall",
        route_health=RouteHealthSummary(10, 1, 4, 4, None, None, None),
    )
    gen = _only(build_candidates([_hyp(axis="iPAE", family="bindcraft")], ev, include_warmstart=False))
    assert gen.feasibility.all_ok()
    assert not any("route_backlog_saturated" in r for r in gen.feasibility.reasons)

    ref = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter", baseline=("exemplar_r_ex",))], ev, include_warmstart=False))
    assert not ref.feasibility.all_ok()
    assert any("route_backlog_saturated" in r for r in ref.feasibility.reasons)


def test_recipe_hash_alias_prefers_non_joint_fail_representative():
    jf = Recipe(
        recipe_hash="same", operator_id="op", method_family="complexa_beam",
        config_delta={}, recipe_class="joint_fail", target_id="t", target_class="c",
        descendant_count=1, median_metrics={}, representative_result_ids=["dead"], recency_tick=1,
    )
    strict = dataclasses.replace(jf, recipe_class="strict_success", representative_result_ids=["good"])
    ev = _evidence(recipes=[jf, strict], parent_artifact_result_ids=["good"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign", baseline=("recipe_same",))], ev, include_warmstart=False))
    assert c.parent_result_id == "good"
    assert c.feasibility.all_ok()


def _parent_source_hyp(parent_ref: str, action_text: str) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim=action_text,
        mode_affinity={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        evidence_refs=["route_values"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", [parent_ref], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
        config_delta_suggestions={},
        reasoning_trace=ReasoningTrace(action_implication=action_text),
    )


def test_parent_source_guard_blocks_boltzgen_backlog_card_on_non_boltzgen_parent():
    parent = "aaaaaaaaaaaaaaaa"
    route_values = [{
        "strategy_key": "route::proteinmpnn_from_bindcraft",
        "scope": "route",
        "family": "proteinmpnn_redesign",
        "root_family": "bindcraft",
        "action_family": "proteinmpnn_redesign",
        "evidence_refs": [parent],
    }]
    ev = _evidence(
        route_values=route_values,
        production_near_miss_ids=[parent],
        parent_artifact_result_ids=[parent],
    )
    card = _parent_source_hyp(parent, "Apply ProteinMPNN rescue to BoltzGen backlog backbones")
    cand = _only(build_candidates([card], ev, include_warmstart=False))
    assert cand.parent_result_id == parent
    assert not cand.feasibility.all_ok()
    assert any("parent_source_mismatch:requested=boltzgen" in r for r in cand.feasibility.reasons)


def test_parent_source_guard_allows_matching_boltzgen_parent():
    parent = "bbbbbbbbbbbbbbbb"
    route_values = [{
        "strategy_key": "route::boltzgen_parent",
        "scope": "route",
        "family": "boltzgen",
        "root_family": "boltzgen",
        "action_family": "boltzgen",
        "evidence_refs": [parent],
    }]
    ev = _evidence(
        route_values=route_values,
        production_near_miss_ids=[parent],
        parent_artifact_result_ids=[parent],
    )
    card = _parent_source_hyp(parent, "Apply ProteinMPNN rescue to BoltzGen backlog backbones")
    cand = _only(build_candidates([card], ev, include_warmstart=False))
    assert cand.parent_result_id == parent
    assert cand.feasibility.all_ok(), cand.feasibility.reasons

def test_parent_consuming_family_cautions_but_allows_joint_fail_only_parent_probe():
    jf = Recipe(
        recipe_hash="dead", operator_id="op", method_family="complexa_beam",
        config_delta={}, recipe_class="joint_fail", target_id="t", target_class="c",
        descendant_count=4, median_metrics={}, representative_result_ids=["dead_parent"], recency_tick=1,
    )
    ev = _evidence(recipes=[jf], parent_artifact_result_ids=["dead_parent"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign", baseline=("recipe_dead",))], ev, include_warmstart=False))
    assert c.parent_result_id == "dead_parent"
    assert c.feasibility.all_ok()
    assert any("joint_fail_parent_caution_bounded_probe:dead_parent" in r for r in c.feasibility.reasons)
    assert not any("joint_fail_parent_dead_backbone" in r for r in c.feasibility.reasons)


def test_joint_fail_soft_probe_becomes_rescue_exhausted_after_non_improving_mpnn_children():
    parent = ResultRecord(
        result_id="dead_parent", parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={"pLDDT": 84.0, "iPAE": 0.50, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok",
        bins={}, artifacts={}, panel_ready=False, tick_id="v7r001",
    )
    children = []
    for i in range(3):
        mpnn_id = f"mpnn_{i}"
        children.append(ResultRecord(
            result_id=mpnn_id, parent_ids=[f"cand_{i}", "dead_parent"], target_id="t1",
            backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
            metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
            exit_status="ok", bins={}, artifacts={}, panel_ready=False, tick_id="v7r002",
        ))
        children.append(ResultRecord(
            result_id=f"rf_{i}", parent_ids=[f"chain_{i}", mpnn_id], target_id="t1",
            backend_family="structure_refilter", runtime_bucket_id="rb1",
            metrics={"pLDDT": 84.0, "iPAE": 0.50, "binder_scRMSD": 1.0},
            metrics_calibrated={}, route_lineage=[], gpu_h=0.01, exit_status="ok",
            bins={"refilter_source": mpnn_id}, artifacts={}, panel_ready=False, tick_id="v7r002",
        ))
    stuck = stuck_lineage_roots([parent, *children], ReducerConfig())
    assert stuck and stuck[0]["root_result_id"] == "dead_parent"
    assert stuck[0]["exhausted_families"] == ["proteinmpnn_redesign"]

    jf = Recipe(
        recipe_hash="dead", operator_id="op", method_family="complexa_beam",
        config_delta={}, recipe_class="joint_fail", target_id="t", target_class="c",
        descendant_count=4, median_metrics={}, representative_result_ids=["dead_parent"], recency_tick=1,
    )
    ev = _evidence(recipes=[jf], stuck=stuck, parent_artifact_result_ids=["dead_parent"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="proteinmpnn_redesign", baseline=("recipe_dead",))], ev, include_warmstart=False))
    assert c.parent_result_id == "dead_parent"
    assert not c.feasibility.all_ok()
    assert any("joint_fail_parent_caution_bounded_probe:dead_parent" in r for r in c.feasibility.reasons)
    assert any("rescue_exhausted:dead_parent:proteinmpnn_redesign" in r for r in c.feasibility.reasons)


def test_diagnostic_joint_fail_parent_score_conversion_remains_feasible_without_caution():
    jf = Recipe(
        recipe_hash="diag", operator_id="boltzgen_default", method_family="boltzgen",
        config_delta={}, recipe_class="joint_fail", target_id="t", target_class="c",
        descendant_count=1, median_metrics={}, representative_result_ids=["diag_parent"], recency_tick=1,
    )
    ev = _evidence(recipes=[jf], parent_artifact_result_ids=["diag_parent"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter", baseline=("recipe_diag",))], ev, include_warmstart=False))
    assert c.parent_result_id == "diag_parent"
    assert c.feasibility.all_ok()
    assert not any("joint_fail_parent_caution_bounded_probe" in r for r in c.feasibility.reasons)


def test_parent_required_resolves_exemplar_alias_to_result_id():
    ev = _evidence(exemplars=[_exemplar("r_ex")], parent_artifact_result_ids=["r_ex"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="structure_refilter",
                                     baseline=("exemplar_r_ex",))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id == "r_ex"
    assert c.baseline_result_id == "r_ex"


def test_denovo_generator_does_not_store_pseudo_parent_ref():
    ev = _evidence(stuck=_STUCK)
    c = _only(build_candidates([_hyp(axis="iPAE", family="bindcraft",
                                     baseline=("recipe_deadbeef",))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id is None
    assert c.baseline_result_id is None


def test_denovo_generator_stores_comparison_baseline_not_execution_parent():
    recipe = Recipe(
        recipe_hash="abc123",
        operator_id="complexa_beam_default",
        method_family="complexa_beam",
        config_delta={},
        recipe_class="near_miss",
        target_id="t",
        target_class="c",
        descendant_count=2,
        median_metrics={"iPAE": 0.4},
        representative_result_ids=["r_rep"],
        recency_tick=3,
    )
    ev = _evidence(recipes=[recipe], parent_artifact_result_ids=["r_rep", "r_rep2"])
    c = _only(build_candidates([_hyp(axis="iPAE", family="bindcraft",
                                     baseline=("recipe_abc123",))],
                               ev, include_warmstart=False))
    assert c.feasibility.all_ok()
    assert c.parent_result_id is None
    assert c.baseline_result_id == "r_rep"


def test_conflicting_per_axis_baselines_are_not_silently_collapsed():
    ev = _evidence(
        exemplars=[_exemplar("baseline_a"), _exemplar("baseline_b")],
        parent_artifact_result_ids=["baseline_a", "baseline_b"],
    )
    hyp = dataclasses.replace(
        _hyp(axis="iPAE", family="bindcraft", baseline=("baseline_a",)),
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["baseline_a"], 0.20, None),
            PredictedChange("pLDDT", "increase", ["baseline_b"], 0.20, None),
        ],
        evidence_refs=["baseline_a"],
    )

    cand = _only(build_candidates([hyp], ev, include_warmstart=False))

    assert cand.parent_result_id is None
    assert cand.baseline_result_id is None
    assert not cand.feasibility.all_ok()
    assert "inconsistent_baseline_refs" in cand.feasibility.reasons


def test_m1_strict_refilter_recipe_has_no_su_per_gpu_h():
    # structure_refilter is role=refilter: it re-scores at ~0 gpu_h, so crediting
    # it ~20 SU/gpu-h would mis-steer exploit. Mirror the method_health exclusion.
    recs = [_rec("rf1", "structure_refilter", STRICT, gpu_h=0.05, bins={"foldseek": "A"})]
    strict = [r for r in extract_recipes(recs, current_tick=1)
              if r.recipe_class == "strict_success"]
    assert strict and all(r.method_family == "structure_refilter" for r in strict)
    assert all(r.su_per_gpu_h is None for r in strict)


def test_m2_route_su_per_gpu_h_uses_route_total_gpu_h():
    # one route (complexa_beam, no config): 1 strict @0.5 + 3 near-miss @0.5 =
    # 2.0 route gpu_h for 1 SU -> 0.5, NOT the strict-only 1/0.5 = 2.0.
    near = {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}
    recs = [_rec("s1", "complexa_beam", STRICT, gpu_h=0.5, bins={"foldseek": "A"})]
    recs += [_rec(f"n{i}", "complexa_beam", near, gpu_h=0.5) for i in range(3)]
    assert _su_recipe(recs).su_per_gpu_h == 0.5


def test_m4_child_closing_a_nondominant_axis_is_progress():
    cfg = ReducerConfig()
    both_fail = {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 3.0}  # iPAE dominant + scRMSD fails
    parent = _rec("P", "complexa_beam", both_fail)
    # children close scRMSD (the sequence axis MPNN exists to fix) but leave the
    # dominant iPAE -> progress on ANY failing axis must prevent a stuck flag.
    kids = [_rec(f"c{i}", "proteinmpnn_redesign",
                 {"pLDDT": 95.0, "iPAE": 0.5, "binder_scRMSD": 1.0}, parent="P")
            for i in range(3)]
    stuck = stuck_lineage_roots([parent, *kids], cfg)
    assert "P" not in {e["root_result_id"] for e in stuck}


def test_c4_diagnostic_only_family_rate_is_none_not_zero():
    # proteinmpnn_redesign is outputs_diagnostic_only: it emits no strict metrics
    # (its rescued SU lands on the chained refilter). Its su_per_gpu_h must read
    # None ("not rate-rankable"), NOT 0.0 ("tried, produced nothing") which would
    # steer the LLM away from the productive MPNN rescue lane.
    recs = [ResultRecord(
        result_id=f"m{i}", parent_ids=["g0"], target_id="t1",
        backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.1, exit_status="ok", bins={}, tick_id="v7r001",
    ) for i in range(3)]
    mh = method_health(recs)["proteinmpnn_redesign"]
    assert mh.su_per_gpu_h is None


def test_c10_soft_default_not_self_locked_by_joint_fail_dedup():
    # the rec-3 metric-adaptive sc_scale_noise soft-default for an un-tuned iPAE card
    # must NOT be hard-blocked even if that exact signature is a recent joint_fail
    # — else every un-tuned remediation funnels into one permanently dead config.
    jf = Recipe(
        recipe_hash="h", operator_id="complexa_beam_default", method_family="complexa_beam",
        config_delta={"sc_scale_noise": 0.30}, recipe_class="joint_fail",
        target_id="t", target_class="c", descendant_count=8,
        median_metrics={"iPAE": 0.6}, representative_result_ids=["r1"], recency_tick=2,
    )
    c = _only(build_candidates([_hyp(axis="iPAE", family="complexa_beam")],
                               _evidence(recipes=[jf]), include_warmstart=False))
    assert c.config_delta == {"sc_scale_noise": 0.2975}     # soft-default applied
    assert c.feasibility.all_ok()                          # but NOT deduped-out
    assert not any("prior_joint_fail_caution_not_ban" in r for r in c.feasibility.reasons)


# --- best/near-miss exemplars (full setup + metrics per binder) ----------------

def _ac(op, cd):
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id="c_" + op, hypothesis_ids=["h"], parent_result_id=None,
        method_family="complexa_beam", operator_id=op, lane_id="l",
        config_delta=cd, downstream_route_plan=[], estimated_cost_class="low",
        expected_signal="x", evidence_refs=[], feasibility=feas)


def test_exemplars_best_ranked_deduped_and_config_joined():
    cfg = ReducerConfig()
    r_hi = _rec("hi", "complexa_beam",
                {"pLDDT": 99.0, "iPAE": 0.05, "binder_scRMSD": 0.5, "ipTM": 0.9},
                bins={"foldseek": "A"})
    r_lo = _rec("lo", "complexa_beam",
                {"pLDDT": 91.0, "iPAE": 0.20, "binder_scRMSD": 1.4},
                bins={"foldseek": "B"})
    r_dup = _rec("dup", "complexa_beam", STRICT, bins={"foldseek": "A"})  # same bin as hi
    sp = {"hi": _ac("op_hi", {"beam_width": 8}), "lo": _ac("op_lo", {}),
          "dup": _ac("op_dup", {})}
    ex = build_exemplars([r_lo, r_dup, r_hi], sp, cfg, k_best=5, k_near=5)
    best = [e for e in ex if e.kind == "best"]
    assert [e.result_id for e in best] == ["hi", "lo"]      # quality-ranked, bin-A dup dropped
    assert best[0].operator_id == "op_hi"
    assert best[0].config_delta == {"beam_width": 8}        # provenance joined
    assert best[0].metrics.get("ipTM") == 0.9               # FULL metric vector (diagnostic axis)


def test_exemplars_near_miss_closest_first_with_setup():
    cfg = ReducerConfig()
    close = _rec("close", "complexa_beam", {"pLDDT": 95.0, "iPAE": 0.26, "binder_scRMSD": 1.0})
    far = _rec("far", "complexa_beam", {"pLDDT": 95.0, "iPAE": 0.50, "binder_scRMSD": 1.0})
    sp = {"close": _ac("op_c", {"sc_scale_noise": 0.2}), "far": _ac("op_f", {})}
    ex = build_exemplars([far, close], sp, cfg, k_best=5, k_near=5)
    near = [e for e in ex if e.kind == "near_miss"]
    assert near and near[0].result_id == "close"            # closest-to-passing first
    assert near[0].dominant_deficit_axis == "iPAE"
    assert near[0].config_delta == {"sc_scale_noise": 0.2}  # provenance joined


def test_exemplars_fallback_provenance_without_spawning_action():
    cfg = ReducerConfig()
    r = _rec("s", "complexa_beam", STRICT, bins={"foldseek": "A"})
    ex = build_exemplars([r], None, cfg, k_best=5, k_near=5)
    assert ex and ex[0].operator_id == "complexa_beam_default" and ex[0].config_delta == {}


def test_exemplars_toggle_off_yields_empty():
    # the with/without A/B: enable_exemplars=False (OFF arm) must surface NO
    # exemplars to the Planner; ON (default) surfaces them.
    from trex.evidence_reducer import reduce_evidence
    recs = [_rec("s", "complexa_beam", STRICT, bins={"foldseek": "A"})]
    common = dict(
        tick_id="t1", target_id="t1", target_class="c",
        elapsed_wall_h=1.0, remaining_wall_h=10.0, pending_children=0,
        worker_gpu_h_total=1.0, all_results=recs, window_results=recs,
        run_su_count=1, run_su_count_delta=1, duplicate_fraction=None,
        near_miss_count=0, top_bin_share=None, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="m",
    )
    assert len(reduce_evidence(**common, enable_exemplars=True).exemplars) >= 1
    assert reduce_evidence(**common, enable_exemplars=False).exemplars == []


def test_c1_bindcraft_records_get_distinct_pdb_path(tmp_path):
    # Each accepted design needs its own structure path. A shared directory can
    # resolve to the same PDB for multiple records and undercount structural diversity.
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext
    acc = tmp_path / "designs" / "Accepted"
    acc.mkdir(parents=True)
    (acc / "design_001_model1.pdb").write_text("ATOM\n")
    (acc / "design_002_model1.pdb").write_text("ATOM\n")
    (tmp_path / "designs" / "final_design_stats.csv").write_text(
        "Design,1_pLDDT,1_i_pAE,1_Binder_RMSD\n"
        "design_001,0.95,0.10,1.0\n"
        "design_002,0.96,0.12,1.1\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")
    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 2
    paths = [r.artifacts.get("pdb_path") for r in recs]
    assert all(paths) and len(set(paths)) == 2            # distinct, not collapsed
    assert {p.split("/")[-1] for p in paths} == {
        "design_001_model1.pdb", "design_002_model1.pdb"}


def test_bindcraft_accepted_pdb_match_requires_design_boundary(tmp_path):
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext

    acc = tmp_path / "designs" / "Accepted"
    acc.mkdir(parents=True)
    (acc / "design_0010_model1.pdb").write_text("ATOM\n")
    (tmp_path / "designs" / "final_design_stats.csv").write_text(
        "Design,1_pLDDT,1_i_pAE,1_Binder_RMSD\n"
        "design_001,0.95,0.10,1.0\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")

    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 1
    assert recs[0].artifacts["pdb_path"].endswith("design_0010_model1.pdb")
    assert recs[0].bins["bindcraft_orphan_accepted"] == "1"


def test_bindcraft_records_diagnostic_only_for_provenance_chain(tmp_path):
    """BindCraft native scores remain diagnostic until standardized evaluation."""
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext
    from trex.success_criteria import is_strict_success
    acc = tmp_path / "designs" / "Accepted"
    acc.mkdir(parents=True)
    (acc / "d1_model1.pdb").write_text("ATOM\n")
    (tmp_path / "designs" / "final_design_stats.csv").write_text(
        "Design,1_pLDDT,1_i_pAE,1_Binder_RMSD,1_i_pTM\n"
        "d1,0.95,0.10,1.0,0.85\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")
    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 1
    m = recs[0].metrics
    # strict-gate keys ABSENT → not strict-gated (the chain produces SU instead)
    assert "pLDDT" not in m and "iPAE" not in m and "binder_scRMSD" not in m
    assert not is_strict_success(m)
    # native scores preserved as diagnostics
    assert m["bindcraft_native_pLDDT"] == 95.0
    assert m["bindcraft_native_iPAE"] == 0.10
    # rank score drives the auto-chain's "best accepts first" selection
    assert recs[0].bins.get("bindcraft_rank_iptm") == "0.8500"
    # Accepted PDB preserved → becomes the structure_refilter parent
    assert recs[0].artifacts.get("pdb_path")


def test_bindcraft_rejected_pdb_is_emitted_for_canonical_score_conversion(tmp_path):
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext
    from trex.success_criteria import is_strict_success

    rej = tmp_path / "designs" / "Rejected"
    rej.mkdir(parents=True)
    (rej / "d1_model1.pdb").write_text("ATOM\n")
    (tmp_path / "designs" / "mpnn_design_stats.csv").write_text(
        "Design,1_pLDDT,1_i_pAE,1_Binder_RMSD,1_i_pTM\n"
        "d1,0.93,0.22,1.12,0.82\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")

    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 1
    r = recs[0]
    assert r.artifacts["pdb_path"].endswith("Rejected/d1_model1.pdb")
    assert r.bins["bindcraft_filter_status"] == "rejected"
    assert r.bins["bindcraft_rejected_artifact"] == "1"
    assert r.metrics["bindcraft_native_pLDDT"] == 93.0
    assert r.metrics["bindcraft_native_iPAE"] == 0.22
    assert r.metrics["bindcraft_native_binder_RMSD"] == 1.12
    assert "pLDDT" not in r.metrics and "iPAE" not in r.metrics and "binder_scRMSD" not in r.metrics
    assert not is_strict_success(r.metrics)


def test_bindcraft_rejected_model_suffix_uses_matching_native_model_metrics(tmp_path):
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext

    rej = tmp_path / "designs" / "Rejected"
    rej.mkdir(parents=True)
    (rej / "d1_model2.pdb").write_text("ATOM\n")
    (tmp_path / "designs" / "mpnn_design_stats.csv").write_text(
        "Design,1_pLDDT,2_pLDDT,Average_pLDDT,1_i_pAE,2_i_pAE,Average_i_pAE,"
        "1_Binder_RMSD,2_Binder_RMSD,Average_Binder_RMSD,1_i_pTM,2_i_pTM,Average_i_pTM\n"
        "d1,0.50,0.93,0.70,0.90,0.22,0.56,9.0,1.12,5.0,0.10,0.82,0.46\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")

    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 1
    r = recs[0]
    assert r.bins["bindcraft_native_model_index"] == "2"
    assert r.metrics["bindcraft_native_pLDDT"] == 93.0
    assert r.metrics["bindcraft_native_iPAE"] == 0.22
    assert r.metrics["bindcraft_native_binder_RMSD"] == 1.12
    assert r.bins["bindcraft_rank_iptm"] == "0.8200"


def test_bindcraft_capability_is_diagnostic_only_so_it_chains():
    """BindCraft capability must be outputs_diagnostic_only=True so the
    controller's auto-chain re-folds its Accepted designs through the
    independent AF2 (structure_refilter), exactly like boltzgen."""
    from trex.capability_registry import default_registry
    reg = default_registry()
    assert reg.get("bindcraft").outputs_diagnostic_only is True


def test_af2_refilter_partial_strict_axes_are_all_or_none(tmp_path):
    """A missing qualification measurement prevents partial canonical scoring."""
    import json
    from trex.output_parsers.af2_refilter import parse_af2_refilter_output
    from trex.output_parsers.types import ParserContext
    from trex.success_criteria import is_strict_success, is_near_miss
    (tmp_path / "af2_refilter_result.json").write_text(json.dumps({
        "metrics": {"plddt": 0.95, "i_pae": 0.10, "iptm": 0.8},  # NO binder_scrmsd_ca
    }))
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="rf1", parent_ids=["rf1"], tick_id="v7r001")
    recs = parse_af2_refilter_output(tmp_path, ctx)
    assert len(recs) == 1
    m = recs[0].metrics
    # partial → none of the 3 strict-gate keys present → purely diagnostic
    assert "pLDDT" not in m and "iPAE" not in m and "binder_scRMSD" not in m
    assert not is_strict_success(m) and not is_near_miss(m)
    assert m.get("ipTM") == 0.8  # diagnostic axis still surfaced


def test_bugA_planner_threads_llm_timeout():
    from unittest.mock import patch
    from trex.planner import call_planner, PlannerCallConfig
    cfg = PlannerCallConfig(model="vllm/x", base_url="http://x/v1", timeout_s=90.0)
    with patch("trex.planner.create_client") as mk:
        mk.return_value.chat.side_effect = RuntimeError("hang")  # force fallback path
        try:
            call_planner(_evidence(), active_hypotheses=[], seed_action_families=None,
                         tick_id_int=1, cfg=cfg)
        except Exception:
            pass
    assert mk.called
    assert mk.call_args.kwargs.get("timeout") == 90.0
    assert mk.call_args.kwargs.get("max_retries") == 1


def test_supervisor_prompt_excludes_health_stubs():
    # Supervisor stub leak [HIGH]: route_health/llm_health are constant stubs
    # ("route wide open / LLM healthy") that must NOT anchor the resource-allocation
    # LLM. The Supervisor must see the SAME curated view as the Planner.
    from trex.supervisor import build_user_prompt as sup_prompt
    txt = sup_prompt(_evidence(), [], [])
    assert "route_health" not in txt
    assert "llm_health" not in txt


def test_planner_prompt_marks_bootstrap_seed_as_non_evidence():
    from trex.planner import build_user_prompt as planner_prompt
    txt = planner_prompt(
        _evidence(),
        active_hypotheses=[],
        seed_action_families=["complexa_beam"],
        available_families=["complexa_beam"],
    )
    assert "deterministic bootstrap seeds" in txt
    assert "not as evidence" in txt


def test_bindcraft_missing_csv_salvages_accepted_pdb_for_score_conversion(tmp_path):
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext
    acc = tmp_path / "designs" / "Accepted"
    acc.mkdir(parents=True)
    (acc / "accepted_only_model1.pdb").write_text("ATOM\n")
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")
    recs = parse_bindcraft_output(tmp_path, ctx)
    assert len(recs) == 1
    assert recs[0].metrics == {}
    assert recs[0].artifacts["pdb_path"].endswith("accepted_only_model1.pdb")
    assert recs[0].bins["bindcraft_orphan_accepted"] == "1"


def test_bindcraft_csv_orphan_accepted_pdb_is_not_dropped(tmp_path):
    from trex.output_parsers.bindcraft import parse_bindcraft_output
    from trex.output_parsers.types import ParserContext
    acc = tmp_path / "designs" / "Accepted"
    acc.mkdir(parents=True)
    (acc / "d1_model1.pdb").write_text("ATOM\n")
    (acc / "orphan_model1.pdb").write_text("ATOM\n")
    (tmp_path / "designs" / "final_design_stats.csv").write_text(
        "Design,1_pLDDT,1_i_pAE,1_Binder_RMSD\n"
        "d1,0.95,0.10,1.0\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bc1", parent_ids=["bc1"], tick_id="v7r001")
    recs = parse_bindcraft_output(tmp_path, ctx)
    assert {r.artifacts["pdb_path"].split("/")[-1] for r in recs} == {"d1_model1.pdb", "orphan_model1.pdb"}


def test_boltzgen_relative_path_resolves_against_output_dir(tmp_path):
    from trex.output_parsers.boltzgen import parse_boltzgen_output
    from trex.output_parsers.types import ParserContext
    out = tmp_path / "bg"
    final = out / "final_ranked_designs"
    final.mkdir(parents=True)
    (final / "design_1.cif").write_text("data_1\n")
    (out / "all_designs_metrics.csv").write_text(
        "id,path,final_rank\n"
        "design_1,final_ranked_designs/design_1.cif,1\n"
    )
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bg1", parent_ids=["bg1"], tick_id="v7r001",
                        method_family="boltzgen")
    recs = parse_boltzgen_output(out, ctx)
    assert len(recs) == 1
    assert recs[0].artifacts["pdb_path"] == str(final / "design_1.cif")


def test_boltzgen_missing_csv_salvages_cif_artifacts(tmp_path):
    from trex.output_parsers.boltzgen import parse_boltzgen_output
    from trex.output_parsers.types import ParserContext
    out = tmp_path / "bg"
    final = out / "final_ranked_designs" / "final_2_designs"
    final.mkdir(parents=True)
    (final / "rank01_design_1.cif").write_text("data_1\n")
    (final / "rank02_design_2.cif").write_text("data_2\n")
    ctx = ParserContext(target_id="t1", runtime_bucket_id="rb1",
                        candidate_id="bg1", parent_ids=["bg1"], tick_id="v7r001",
                        method_family="boltzgen")
    recs = parse_boltzgen_output(out, ctx)
    assert len(recs) == 2
    assert all(r.bins.get("boltzgen_orphan_cif") == "1" for r in recs)
    assert {Path(r.artifacts["pdb_path"]).name for r in recs} == {"rank01_design_1.cif", "rank02_design_2.cif"}


def test_boltzgen_parser_skips_top_level_aggregate_design_cif(tmp_path):
    """The root design.cif is BoltzGen run provenance, not a binder design."""
    from trex.output_parsers.boltzgen import parse_boltzgen_output
    from trex.output_parsers.types import ParserContext

    out = tmp_path / "bg"
    final = out / "final_ranked_designs" / "final_1_designs"
    final.mkdir(parents=True)
    (out / "design.cif").write_text("data_aggregate\n")
    (final / "rank01_design_08.cif").write_text("data_design\n")
    ctx = ParserContext(
        target_id="t1", runtime_bucket_id="rb1", candidate_id="bg1",
        parent_ids=["bg1"], tick_id="v7r001", method_family="boltzgen",
    )

    recs = parse_boltzgen_output(out, ctx)

    assert len(recs) == 1
    assert Path(recs[0].artifacts["pdb_path"]).name == "rank01_design_08.cif"
    assert recs[0].bins.get("boltzgen_design_id") != "design"


def test_boltzgen_parser_returns_empty_for_only_aggregate_design_cif(tmp_path):
    from trex.output_parsers.boltzgen import parse_boltzgen_output
    from trex.output_parsers.types import ParserContext

    out = tmp_path / "bg"
    out.mkdir()
    (out / "design.cif").write_text("data_aggregate\n")
    ctx = ParserContext(
        target_id="t1", runtime_bucket_id="rb1", candidate_id="bg1",
        parent_ids=["bg1"], tick_id="v7r001", method_family="boltzgen",
    )

    assert parse_boltzgen_output(out, ctx) == []


def test_boltzgen_parser_skips_csv_aggregate_design_row(tmp_path):
    from trex.output_parsers.boltzgen import parse_boltzgen_output
    from trex.output_parsers.types import ParserContext

    out = tmp_path / "bg"
    final = out / "final_ranked_designs"
    final.mkdir(parents=True)
    (out / "design.cif").write_text("data_aggregate\n")
    (final / "design_1.cif").write_text("data_1\n")
    (out / "all_designs_metrics.csv").write_text(
        "id,path,final_rank\n"
        "design,design.cif,\n"
        "design_1,final_ranked_designs/design_1.cif,1\n"
    )
    ctx = ParserContext(
        target_id="t1", runtime_bucket_id="rb1", candidate_id="bg1",
        parent_ids=["bg1"], tick_id="v7r001", method_family="boltzgen",
    )

    recs = parse_boltzgen_output(out, ctx)

    assert len(recs) == 1
    assert recs[0].bins.get("boltzgen_design_id") == "design_1"
    assert Path(recs[0].artifacts["pdb_path"]).name == "design_1.cif"
