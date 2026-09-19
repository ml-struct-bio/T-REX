"""Regression tests (2026-05-28):
 - Fix A: proteinmpnn_redesign must carry downstream_route_plan=['structure_refilter']
   so the controller auto-chains its (metric-less) redesigns to AF2 and they
   produce strict SU. Without it the entire RESCUE mode produced 0 SU.
 - Fix C: a heavy generator must be infeasible when it cannot finish before the
   wall deadline (don't burn GPU-h on a launch that salvages 0 scored designs).
"""

from __future__ import annotations

from dataclasses import replace

from trex.capability_registry import default_registry
from trex.candidate_builder import (
    _dominant_route_root_group,
    build_candidates,
    feasibility_for,
    BuilderConfig,
)
from trex.schemas import (
    EvidenceSummary,
    HypothesisCard,
    LLMHealthSummary,
    PredictedChange,
    PreserveConstraint,
    Recipe,
    RouteHealthSummary,
    RouteValueSummary,
)


def _evidence(remaining_wall_h: float = 48.0) -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="t1", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=0, remaining_wall_h=remaining_wall_h, completed_children=0,
        pending_children=0, worker_gpu_h_total=0, worker_gpu_h_last_3_ticks=0,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="rescue_rich", examples=[], metric_availability={}, recipes=[],
    )


def _mpnn_card() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h_rescue", target_id="t", tick_created=1,
        claim="rescue high-pLDDT/failing-iPAE near-misses via MPNN redesign",
        mode_affinity={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        evidence_refs=["r_parent"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["r_parent"], 0.2, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
    )


def _refilter_card(parent_ref: str, config: dict | None = None) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h_refilter", target_id="t", tick_created=1,
        claim="obtain canonical AF2 strict metrics for a diagnostic artifact",
        mode_affinity={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        evidence_refs=[parent_ref],
        predicted_metric_changes=[
            PredictedChange("pLDDT", "increase", [parent_ref], 0.2, None)
        ],
        preserve_constraints=[],
        recommended_action_families=["structure_refilter"],
        config_delta_suggestions={"structure_refilter": config or {}},
    )


def _recipe_with_rep(rep: str, family: str = "boltzgen") -> Recipe:
    return Recipe(
        recipe_hash="rhash", operator_id=f"{family}_default",
        method_family=family, config_delta={"num_designs": 16},
        recipe_class="joint_fail", target_id="t", target_class="c",
        descendant_count=1, median_metrics={}, representative_result_ids=[rep],
        recency_tick=1, su_per_gpu_h=None,
    )


# --- Fix A: proteinmpnn rescue chain -----------------------------------------

def test_proteinmpnn_candidate_carries_refilter_route():
    cands = build_candidates([_mpnn_card()], _evidence(), parent_pdb_available=True)
    mpnn = [c for c in cands if c.method_family == "proteinmpnn_redesign"]
    assert mpnn, "MPNN card should yield a proteinmpnn_redesign candidate"
    for c in mpnn:
        assert c.downstream_route_plan == ["structure_refilter"], (
            "proteinmpnn_redesign must auto-chain to structure_refilter (AF2) "
            "or its redesigns never become strict SU"
        )


def test_dominant_route_root_ignores_family_rollup_double_count():
    ev = replace(
        _evidence(),
        route_values=[
            RouteValueSummary(
                strategy_key="family::complexa_beam", scope="family",
                family="complexa_beam", root_family=None,
                action_family="complexa_beam", scoring_family=None,
                operator_id="complexa_beam_default", config_signature="family_rollup",
                route_gpu_h=100.0,
            ),
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default", scope="route",
                family="bindcraft", root_family="bindcraft",
                action_family="bindcraft", scoring_family=None,
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=10.0,
            ),
        ],
    )
    assert _dominant_route_root_group(ev, default_registry()) == "bindcraft"


def test_structure_refilter_parent_cannot_be_structure_refilter_result():
    ev = replace(_evidence(), recipes=[_recipe_with_rep("rf_child")], parent_artifact_result_ids=["rf_child"])
    cands = build_candidates(
        [_refilter_card("rf_child")],
        ev,
        include_warmstart=False,
        parent_pdb_available=True,
        structure_refilter_result_ids={"rf_child"},
    )
    refilters = [c for c in cands if c.method_family == "structure_refilter"]
    assert refilters
    assert not refilters[0].feasibility.all_ok()
    assert not refilters[0].feasibility.compiler_ok
    assert any(
        "refilter_of_refilter_blocked:rf_child" in r
        for r in refilters[0].feasibility.reasons
    )


def test_material_parent_model_refold_can_retry_structure_refilter_result():
    ev = replace(_evidence(), recipes=[_recipe_with_rep("rf_child")], parent_artifact_result_ids=["rf_child"])
    cands = build_candidates(
        [_refilter_card("rf_child", {
            "model_names": "model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,model_4_multimer_v3,model_5_multimer_v3",
            "num_recycles": 6,
            "use_initial_guess": 0,
        })],
        ev,
        include_warmstart=False,
        parent_pdb_available=True,
        structure_refilter_result_ids={"rf_child"},
    )
    refilters = [c for c in cands if c.method_family == "structure_refilter"]
    assert refilters
    assert refilters[0].parent_result_id == "rf_child"
    assert refilters[0].feasibility.all_ok()
    assert refilters[0].refilter_role == "parent_model_refold"


def test_structure_refilter_generator_parent_still_allowed():
    ev = replace(_evidence(), recipes=[_recipe_with_rep("bg_parent")], parent_artifact_result_ids=["bg_parent"])
    cands = build_candidates(
        [_refilter_card("bg_parent")],
        ev,
        include_warmstart=False,
        parent_pdb_available=True,
        structure_refilter_result_ids={"some_other_af2_child"},
        structure_refilter_scored_source_ids={"some_other_generator_parent"},
    )
    refilters = [c for c in cands if c.method_family == "structure_refilter"]
    assert refilters
    assert refilters[0].parent_result_id == "bg_parent"
    assert refilters[0].feasibility.all_ok()


def test_structure_refilter_parent_cannot_be_already_scored_source():
    ev = replace(_evidence(), recipes=[_recipe_with_rep("bg_parent")], parent_artifact_result_ids=["bg_parent"])
    cands = build_candidates(
        [_refilter_card("bg_parent")],
        ev,
        include_warmstart=False,
        parent_pdb_available=True,
        structure_refilter_result_ids={"rf_child"},
        structure_refilter_scored_source_ids={"bg_parent"},
    )
    refilters = [c for c in cands if c.method_family == "structure_refilter"]
    assert refilters
    assert refilters[0].parent_result_id == "bg_parent"
    assert not refilters[0].feasibility.all_ok()
    assert not refilters[0].feasibility.compiler_ok
    assert any(
        "structure_refilter_source_already_scored:bg_parent" in r
        for r in refilters[0].feasibility.reasons
    )


def test_material_parent_model_refold_can_retry_already_scored_source():
    ev = replace(_evidence(), recipes=[_recipe_with_rep("bg_parent")], parent_artifact_result_ids=["bg_parent"])
    cands = build_candidates(
        [_refilter_card("bg_parent", {"model_names": "model_1_multimer_v3,model_2_multimer_v3"})],
        ev,
        include_warmstart=False,
        parent_pdb_available=True,
        structure_refilter_result_ids={"rf_child"},
        structure_refilter_scored_source_ids={"bg_parent"},
    )
    refilters = [c for c in cands if c.method_family == "structure_refilter"]
    assert refilters
    assert refilters[0].parent_result_id == "bg_parent"
    assert refilters[0].feasibility.all_ok()


# --- Fix C: remaining-wall feasibility gate ----------------------------------

def test_heavy_generator_infeasible_near_deadline():
    # bindcraft ~2.5h: with only 2.0h left it cannot finish → infeasible
    f = feasibility_for("bindcraft", _evidence(remaining_wall_h=2.0), BuilderConfig())
    assert not f.cost_ok
    assert any("insufficient_wall" in r for r in f.reasons)


def test_heavy_generator_feasible_with_ample_wall():
    f = feasibility_for("bindcraft", _evidence(remaining_wall_h=10.0), BuilderConfig())
    assert f.cost_ok


def test_cheap_family_feasible_near_deadline():
    # complexa_beam ~0.5h fits in 2.0h remaining
    f = feasibility_for("complexa_beam", _evidence(remaining_wall_h=2.0), BuilderConfig())
    assert f.cost_ok


def test_route_value_replay_keeps_productive_generator_config_visible():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam_default:beam_width=8",
            scope="route",
            family="complexa_beam",
            root_family="complexa_beam",
            action_family="complexa_beam",
            scoring_family=None,
            operator_id="complexa_beam_default",
            config_signature="beam_width=8,n_branch=4,nsamples=4",
            config_delta={"beam_width": 8, "n_branch": 4, "nsamples": 4},
            route_gpu_h=2.0,
            attempts=8,
            completions=8,
            strict_count=4,
            new_su=3,
            new_su_recent=1,
            new_su_per_route_gpu_h=1.5,
            recent_route_gpu_h=0.5,
            recent_new_su_per_route_gpu_h=2.0,
            near_miss_count=0,
            near_miss_recent=0,
            status="promote",
            evidence_refs=["cx_good_1"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert len(replay) == 1
    assert replay[0].method_family == "complexa_beam"
    assert replay[0].config_delta == {"beam_width": 8, "n_branch": 4, "nsamples": 4}
    assert replay[0].supervisor_mode == "exploit"
    assert replay[0].feasibility.all_ok()


def test_route_value_replay_keeps_high_su_gpuh_diversify_route_visible():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_fk_steering:complexa_fk_steering_default:beam_width=8",
            scope="route",
            family="complexa_fk_steering",
            root_family="complexa_fk_steering",
            action_family="complexa_fk_steering",
            scoring_family=None,
            operator_id="complexa_fk_steering_default",
            config_signature="beam_width=8,n_branch=4,nsamples=4",
            config_delta={"beam_width": 8, "n_branch": 4, "nsamples": 4},
            route_gpu_h=1.0,
            attempts=16,
            completions=16,
            strict_count=20,
            new_su=5,
            new_su_recent=1,
            new_su_per_route_gpu_h=5.0,
            recent_route_gpu_h=0.25,
            recent_new_su_per_route_gpu_h=4.0,
            strict_per_su=4.0,
            duplicate_bin_fraction=0.60,
            status="diversify",
            evidence_refs=["cx_diverse_su"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert len(replay) == 1
    assert replay[0].method_family == "complexa_fk_steering"
    assert replay[0].supervisor_mode == "exploit"
    assert "new_su_per_gpu_h=4.000" in replay[0].expected_signal

def test_route_value_replay_uses_gpu_window_not_refilter_only_record_rate_for_delayed_routes():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::bindcraft:bindcraft_default:default",
            scope="route",
            family="bindcraft",
            root_family="bindcraft",
            action_family="bindcraft",
            scoring_family="structure_refilter",
            operator_id="bindcraft_default",
            config_signature="default",
            route_role="generator_with_af2_score_conversion",
            config_delta={},
            route_gpu_h=0.662,
            generator_gpu_h=0.638,
            canonical_refilter_gpu_h=0.024,
            attempts=3,
            completions=3,
            strict_count=2,
            new_su=2,
            record_recent_new_su=2,
            record_recent_route_gpu_h=0.024,
            record_recent_new_su_per_route_gpu_h=83.333,
            gpu_recent_route_gpu_h=0.662,
            gpu_recent_new_su_per_route_gpu_h=3.021,
            new_su_per_route_gpu_h=3.021,
            near_miss_count=0,
            near_miss_recent=0,
            status="promote",
            evidence_refs=["rf_a", "rf_b"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert len(replay) == 1
    assert replay[0].method_family == "bindcraft"
    assert "new_su_per_gpu_h=3.021" in replay[0].expected_signal
    assert "83.333" not in replay[0].expected_signal


def test_route_value_replay_keeps_bounded_diagnostic_improvement_route_visible():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::bindcraft:bindcraft_default:weights_iptm=0.2",
            scope="route",
            family="bindcraft",
            root_family="bindcraft",
            action_family="bindcraft",
            scoring_family="structure_refilter",
            operator_id="bindcraft_default",
            config_signature="weights_iptm=0.2",
            route_role="generator_with_af2_score_conversion",
            config_delta={"weights_iptm": 0.2},
            route_gpu_h=4.0,
            generator_gpu_h=3.9,
            canonical_refilter_gpu_h=0.1,
            attempts=2,
            completions=2,
            strict_count=0,
            new_su=0,
            new_su_recent=0,
            new_su_per_route_gpu_h=None,
            recent_route_gpu_h=0.5,
            recent_new_su_per_route_gpu_h=None,
            near_miss_count=0,
            near_miss_recent=0,
            status="defer",
            marginal_status="observed",
            pending_score_conversion_count=0,
            diagnostic_improvement_score=0.62,
            diagnostic_improvement_axes=["ipTM:levered:score=0.62:n=3"],
            evidence_refs=["bindcraft_native_good"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert len(replay) == 1
    assert replay[0].method_family == "bindcraft"
    assert replay[0].supervisor_mode == "explore"
    assert replay[0].config_delta == {"weights_iptm": 0.2}
    assert "diagnostic_improvement_score=0.620" in replay[0].expected_signal
    assert "new_su_per_gpu_h=0.000" in replay[0].expected_signal


def test_route_value_replay_waits_for_pending_score_conversion_before_diagnostic_replay():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::boltzgen:boltzgen_default:default",
            scope="route",
            family="boltzgen",
            root_family="boltzgen",
            action_family="boltzgen",
            scoring_family="structure_refilter",
            operator_id="boltzgen_default",
            config_signature="default",
            route_role="generator_with_af2_score_conversion",
            config_delta={},
            route_gpu_h=2.0,
            attempts=1,
            completions=1,
            strict_count=0,
            new_su=0,
            new_su_recent=0,
            near_miss_count=0,
            near_miss_recent=0,
            status="observed",
            marginal_status="awaiting_score_conversion",
            pending_score_conversion_count=6,
            diagnostic_improvement_score=0.8,
            diagnostic_improvement_axes=["design_to_target_iptm:levered:score=0.8:n=4"],
            evidence_refs=["boltzgen_native_good"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    assert [c for c in cands if c.candidate_id.startswith("route_replay_")] == []


def test_route_value_replay_ranks_su_route_above_diagnostic_only_route():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::bindcraft:bindcraft_default:diag",
            scope="route", family="bindcraft", root_family="bindcraft",
            action_family="bindcraft", scoring_family="structure_refilter",
            operator_id="bindcraft_default", config_signature="diag",
            route_role="generator_with_af2_score_conversion", config_delta={"weights_iptm": 0.2},
            route_gpu_h=4.0, attempts=2, completions=2, strict_count=0,
            new_su=0, new_su_recent=0, near_miss_count=0, near_miss_recent=0,
            status="defer", marginal_status="observed", pending_score_conversion_count=0,
            diagnostic_improvement_score=0.9, diagnostic_improvement_axes=["ipTM:levered:score=0.9:n=3"],
            evidence_refs=["bindcraft_native_good"],
        ),
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam_default:beam_width=4",
            scope="route", family="complexa_beam", root_family="complexa",
            action_family="complexa_beam", scoring_family=None,
            operator_id="complexa_beam_default", config_signature="beam_width=4",
            config_delta={"beam_width": 4}, route_gpu_h=3.0, attempts=4, completions=4,
            strict_count=2, new_su=1, new_su_recent=1, new_su_per_route_gpu_h=0.333,
            recent_route_gpu_h=1.0, recent_new_su_per_route_gpu_h=1.0,
            near_miss_count=0, near_miss_recent=0, status="healthy", evidence_refs=["cx_su"],
        ),
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert [c.method_family for c in replay[:2]] == ["complexa_beam", "bindcraft"]
    assert replay[0].supervisor_mode == "exploit"
    assert replay[1].supervisor_mode == "explore"


def test_route_value_replay_skips_lifetime_only_route_after_target_dry_plateau():
    ev = replace(_evidence(), gpu_h_since_last_su=13.0, route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam_default:beam_width=4",
            scope="route",
            family="complexa_beam",
            root_family="complexa_beam",
            action_family="complexa_beam",
            scoring_family=None,
            operator_id="complexa_beam_default",
            config_signature="beam_width=4,n_branch=4,nsamples=4",
            config_delta={"beam_width": 4, "n_branch": 4, "nsamples": 4},
            route_gpu_h=1.7,
            attempts=16,
            completions=16,
            strict_count=24,
            new_su=5,
            new_su_recent=0,
            new_su_recent_gpu=0,
            new_su_per_route_gpu_h=2.94,
            recent_route_gpu_h=0.0,
            record_recent_route_gpu_h=0.0,
            near_miss_count=0,
            near_miss_recent=0,
            strict_per_su=4.8,
            duplicate_bin_fraction=0.79,
            status="diversify",
            evidence_refs=["cx_lifetime_only"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    assert [c for c in cands if c.candidate_id.startswith("route_replay_")] == []


def test_route_value_replay_skips_record_recent_only_route_after_target_dry_plateau():
    ev = replace(_evidence(), gpu_h_since_last_su=13.0, route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam_default:beam_width=4",
            scope="route",
            family="complexa_beam",
            root_family="complexa_beam",
            action_family="complexa_beam",
            scoring_family=None,
            operator_id="complexa_beam_default",
            config_signature="beam_width=4,n_branch=4,nsamples=4",
            config_delta={"beam_width": 4, "n_branch": 4, "nsamples": 4},
            route_gpu_h=0.68,
            attempts=4,
            completions=4,
            strict_count=11,
            new_su=1,
            new_su_recent=1,
            record_recent_new_su=1,
            new_su_recent_gpu=0,
            new_su_per_route_gpu_h=1.46,
            record_recent_new_su_per_route_gpu_h=1.46,
            recent_route_gpu_h=0.68,
            record_recent_route_gpu_h=0.68,
            near_miss_count=0,
            near_miss_recent=0,
            strict_per_su=11.0,
            duplicate_bin_fraction=0.82,
            status="diversify",
            marginal_status="productive_but_duplicate",
            evidence_refs=["cx_record_recent_but_target_dry"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    assert [c for c in cands if c.candidate_id.startswith("route_replay_")] == []

def test_route_value_replay_skips_high_cost_stale_exact_route():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::bindcraft:bindcraft_default:default",
            scope="route",
            family="bindcraft",
            root_family="bindcraft",
            action_family="bindcraft",
            scoring_family="structure_refilter",
            operator_id="bindcraft_default",
            config_signature="default",
            config_delta={},
            route_role="generator_with_af2_score_conversion",
            route_gpu_h=8.0,
            generator_gpu_h=7.2,
            canonical_refilter_gpu_h=0.8,
            attempts=4,
            completions=4,
            strict_count=2,
            new_su=1,
            new_su_recent=0,
            new_su_recent_gpu=0,
            gpu_recent_route_gpu_h=6.0,
            gpu_recent_new_su_per_route_gpu_h=0.0,
            medium_recent_new_su=0,
            medium_recent_route_gpu_h=6.0,
            medium_recent_new_su_per_route_gpu_h=0.0,
            new_su_per_route_gpu_h=0.25,
            near_miss_count=0,
            near_miss_recent=0,
            status="healthy",
            marginal_status="productive",
            evidence_refs=["bc_old_su"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    assert [c for c in cands if c.candidate_id.startswith("route_replay_")] == []


def test_route_value_replay_labels_lifetime_memory_when_not_dry_enough():
    ev = replace(_evidence(), gpu_h_since_last_su=2.0, route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam_default:beam_width=4",
            scope="route",
            family="complexa_beam",
            root_family="complexa_beam",
            action_family="complexa_beam",
            scoring_family=None,
            operator_id="complexa_beam_default",
            config_signature="beam_width=4,n_branch=4,nsamples=4",
            config_delta={"beam_width": 4, "n_branch": 4, "nsamples": 4},
            route_gpu_h=1.7,
            attempts=16,
            completions=16,
            strict_count=24,
            new_su=5,
            new_su_recent=0,
            new_su_recent_gpu=0,
            new_su_per_route_gpu_h=2.94,
            recent_route_gpu_h=0.0,
            record_recent_route_gpu_h=0.0,
            near_miss_count=0,
            near_miss_recent=0,
            status="diversify",
            evidence_refs=["cx_lifetime_only"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert len(replay) == 1
    assert "lifetime_su_per_gpu_h=2.940" in replay[0].expected_signal
    assert "no_recent_signal=1" in replay[0].expected_signal
    assert "new_su_per_gpu_h=2.940" not in replay[0].expected_signal
    assert replay[0].supervisor_mode == "explore"

def test_route_value_replay_skips_recently_dry_lifetime_route():
    ev = replace(_evidence(), route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_mcts:complexa_mcts_default:default",
            scope="route",
            family="complexa_mcts",
            root_family="complexa_mcts",
            action_family="complexa_mcts",
            scoring_family=None,
            operator_id="complexa_mcts_default",
            config_signature="default",
            config_delta={},
            route_gpu_h=4.0,
            attempts=8,
            completions=8,
            strict_count=1,
            new_su=1,
            new_su_recent=0,
            new_su_per_route_gpu_h=0.25,
            recent_route_gpu_h=1.5,
            recent_new_su_per_route_gpu_h=None,
            near_miss_count=0,
            near_miss_recent=0,
            status="healthy",
            evidence_refs=["cx_old_1"],
        )
    ])

    cands = build_candidates([], ev, include_warmstart=False)

    replay = [c for c in cands if c.candidate_id.startswith("route_replay_")]
    assert replay == []
