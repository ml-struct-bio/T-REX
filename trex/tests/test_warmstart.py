"""Cold-start root coverage tests.

Cold archives should launch the deterministic target-agnostic root probes
(complexa_beam, BoltzGen, BindCraft) before accepting LLM novelty.
"""

from __future__ import annotations

from dataclasses import replace

from trex.candidate_builder import WARMSTART_FAMILIES, _diagnostic_route_feedback_reason, build_candidates
from trex.live_tick import BOOTSTRAP_SEED_ACTION_FAMILIES
from trex.schemas import (
    EvidenceSummary,
    HypothesisCard,
    LLMHealthSummary,
    PredictedChange,
    PreserveConstraint,
    ReasoningTrace,
    Recipe,
    RouteHealthSummary,
)


def _evidence(state: str = "low_evidence", recipes=None) -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="t1", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=0, remaining_wall_h=48, completed_children=0, pending_children=0,
        worker_gpu_h_total=0, worker_gpu_h_last_3_ticks=0,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label=state,  # type: ignore[arg-type]
        examples=[], metric_availability={},
        recipes=recipes or [],
    )


def test_planner_bootstrap_seed_prompt_matches_warmstart_families():
    """Planner-visible cold-start families must match actual warmstart roots."""
    assert BOOTSTRAP_SEED_ACTION_FAMILIES == [fam for fam, _ in WARMSTART_FAMILIES]


def test_warmstart_fires_on_cold_low_evidence():
    """Empty archive + no recipes + low_evidence -> emit configured warmstart seeds."""
    e = _evidence("low_evidence", recipes=[])
    cands = build_candidates([], e)
    assert len(cands) >= len(WARMSTART_FAMILIES)
    fams = {c.method_family for c in cands}
    expected = {fam for fam, _ in WARMSTART_FAMILIES}
    assert fams == expected
    assert "complexa_best_of_n" not in fams
    assert "complexa_fk_steering" not in fams
    for fam, params in WARMSTART_FAMILIES:
        matches = [c for c in cands if c.method_family == fam]
        assert len(matches) == 1
        cand = matches[0]
        for key, value in params.items():
            assert cand.config_delta.get(key) == value
        if fam in {"boltzgen", "bindcraft"}:
            assert cand.downstream_route_plan == ["structure_refilter"]
        elif fam.startswith("complexa_"):
            assert cand.downstream_route_plan == []


def test_partial_warmstart_emits_only_uncompleted_families():
    e = _evidence("low_evidence", recipes=[])
    completed = {WARMSTART_FAMILIES[0][0], WARMSTART_FAMILIES[1][0]}

    cands = build_candidates(
        [], e, warmstart_completed_families=completed,
    )

    assert {c.method_family for c in cands} == {
        family for family, _ in WARMSTART_FAMILIES if family not in completed
    }
    assert all(c.candidate_id.startswith("warmstart_") for c in cands)

def test_cold_start_ignores_llm_non_warmstart_until_evidence_exists():
    e = _evidence("low_evidence", recipes=[])
    h = HypothesisCard(
        hypothesis_id="h_cold", target_id="t", tick_created=1,
        claim="try parent-dependent redesign too early",
        mode_affinity={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        evidence_refs=["cold_start"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["cold_start"], 0.1, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
    )
    cands = build_candidates([h], e)
    assert cands
    assert all(c.candidate_id.startswith("warmstart_") for c in cands)
    fams = {c.method_family for c in cands}
    assert "bindcraft" in fams
    assert "proteinmpnn_redesign" not in fams


def test_warmstart_skipped_when_recipes_exist():
    """If the archive has any recipe (success or failure), warmstart is
    skipped — the Planner should drive from real evidence instead."""
    e = _evidence("low_evidence", recipes=[
        Recipe(
            recipe_hash="h1", operator_id="complexa_beam_default",
            method_family="complexa_beam", config_delta={"beam_width": 4},
            recipe_class="strict_success", target_id="t", target_class="c",
            descendant_count=10, median_metrics={}, representative_result_ids=[],
            recency_tick=1,
        )
    ])
    cands = build_candidates([], e)
    # only warmstart hypothesis_ids would be ["warmstart"]; check no such candidate
    assert not any("warmstart" in c.candidate_id for c in cands)


def test_warmstart_skipped_when_state_productive():
    e = _evidence("productive", recipes=[])
    cands = build_candidates([], e)
    # neither cold-start nor stalled-fallback should fire on productive
    assert not any(
        c.candidate_id.startswith(("warmstart_", "evidence_fallback")) for c in cands
    )


def _complexa_zero_su_route() -> dict:
    return {
        "strategy_key": "family::complexa_beam",
        "scope": "family",
        "family": "complexa_beam",
        "action_family": "complexa_beam",
        "root_family": "complexa",
        "operator_id": "complexa_beam_default",
        "config_signature": "family_rollup",
        "route_gpu_h": 2.2,
        "gpu_recent_route_gpu_h": 2.2,
        "attempts": 3,
        "completions": 3,
        "new_su": 0,
        "new_su_recent_gpu": 0,
        "marginal_status": "dry_low_quality",
        "status": "dry_low_quality",
    }


def _complexa_hypothesis() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h_same_root", target_id="t", tick_created=2,
        claim="try more Complexa",
        mode_affinity={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        evidence_refs=["route_values"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["route_values"], 0.1, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["complexa_beam"],
    )


def test_early_zero_su_cross_family_probe_waits_for_first_completed_evidence():
    e = replace(
        _evidence("low_evidence", recipes=[]),
        completed_children=0,
        worker_gpu_h_total=0.2,
        gpu_h_since_last_su=0.2,
        route_values=[_complexa_zero_su_route()],
    )
    cands = build_candidates([_complexa_hypothesis()], e, include_warmstart=False)
    assert not any(c.candidate_id.startswith("evidence_fallback_cross_family") for c in cands)


def test_early_zero_su_cross_family_probe_adds_two_non_dominant_roots():
    e = replace(
        _evidence("low_evidence", recipes=[]),
        completed_children=1,
        worker_gpu_h_total=0.12,
        gpu_h_since_last_su=0.12,
        route_values=[_complexa_zero_su_route()],
    )
    cands = build_candidates([_complexa_hypothesis()], e, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    assert len(cross) == 2
    assert {c.method_family for c in cross} == {"bindcraft", "boltzgen"}
    assert {c.method_family for c in cross}.isdisjoint({"complexa_beam", "complexa_fk_steering", "complexa_mcts"})


def test_early_zero_su_cross_family_probe_counts_pending_non_dominant_root():
    e = replace(
        _evidence("low_evidence", recipes=[]),
        completed_children=1,
        worker_gpu_h_total=0.12,
        gpu_h_since_last_su=0.12,
        route_values=[_complexa_zero_su_route()],
        pending_family_load={
            "by_family": {
                "boltzgen": {"running": 1, "queued": 0, "pending_total": 1, "inflight_gpu_h": 0.4},
            },
            "high_cost_families": [],
        },
    )
    cands = build_candidates([_complexa_hypothesis()], e, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    assert len(cross) == 1
    assert {c.method_family for c in cross} == {"bindcraft"}


def test_early_zero_su_cross_family_probe_satisfied_when_two_roots_pending():
    e = replace(
        _evidence("low_evidence", recipes=[]),
        completed_children=1,
        worker_gpu_h_total=0.12,
        gpu_h_since_last_su=0.12,
        route_values=[_complexa_zero_su_route()],
        pending_family_load={
            "by_family": {
                "boltzgen": {"running": 1, "queued": 0, "pending_total": 1, "inflight_gpu_h": 0.4},
                "bindcraft": {"running": 0, "queued": 1, "pending_total": 1, "inflight_gpu_h": 0.0},
            },
            "high_cost_families": [],
        },
    )
    cands = build_candidates([_complexa_hypothesis()], e, include_warmstart=False)
    assert not any(c.candidate_id.startswith("evidence_fallback_cross_family") for c in cands)


def test_cross_family_escape_ignores_unsafe_score_conversion_record_recent_signal():
    route = dict(_complexa_zero_su_route())
    route.update({
        "strategy_key": "family::boltzgen",
        "family": "boltzgen",
        "action_family": "boltzgen",
        "root_family": "boltzgen",
        "operator_id": "boltzgen_default",
        "config_signature": "family_rollup",
        "route_role": "generator_with_af2_score_conversion",
        "route_gpu_h": 2.0,
        "canonical_refilter_gpu_h": 0.025,
        "new_su": 1,
        "new_su_recent_gpu": 0,
        "gpu_recent_new_su_per_route_gpu_h": 0.0,
        "medium_recent_new_su": 0,
        "medium_recent_new_su_per_route_gpu_h": None,
        "record_recent_new_su": 1,
        "record_recent_new_su_per_route_gpu_h": 40.0,
        "record_recent_route_gpu_h": 0.025,
        "status": "healthy",
        "marginal_status": "observed",
    })
    h = HypothesisCard(
        hypothesis_id="h_boltz", target_id="t", tick_created=3,
        claim="continue BoltzGen root",
        mode_affinity={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        evidence_refs=["route_values"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["route_values"], 0.1, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["boltzgen"],
    )
    e = replace(
        _evidence("stalled", recipes=[]),
        completed_children=8,
        worker_gpu_h_total=8.0,
        gpu_h_since_last_su=12.0,
        run_su_count=1,
        route_values=[route],
    )
    cands = build_candidates([h], e, include_warmstart=False)
    cross = [c for c in cands if c.candidate_id.startswith("evidence_fallback_cross_family")]
    assert len(cross) == 2
    assert {c.method_family for c in cross} == {"bindcraft", "complexa_beam"}


def test_early_zero_su_cross_family_signal_uses_candidate_specific_diagnostic_score():
    boltz = {
        "strategy_key": "family::boltzgen",
        "scope": "family",
        "family": "boltzgen",
        "action_family": "boltzgen",
        "root_family": "boltzgen",
        "operator_id": "boltzgen_default",
        "config_signature": "family_rollup",
        "route_gpu_h": 0.8,
        "attempts": 1,
        "completions": 1,
        "new_su": 0,
        "near_miss_recent": 0,
        "marginal_status": "under_tested",
        "status": "under_tested",
        "diagnostic_improvement_score": 0.41,
    }
    bindcraft = {
        "strategy_key": "family::bindcraft",
        "scope": "family",
        "family": "bindcraft",
        "action_family": "bindcraft",
        "root_family": "bindcraft",
        "operator_id": "bindcraft_default",
        "config_signature": "family_rollup",
        "route_gpu_h": 0.7,
        "attempts": 1,
        "completions": 1,
        "new_su": 0,
        "near_miss_recent": 0,
        "marginal_status": "under_tested",
        "status": "under_tested",
        "diagnostic_improvement_score": 0.79,
    }
    e = replace(
        _evidence("low_evidence", recipes=[]),
        completed_children=3,
        worker_gpu_h_total=2.2,
        gpu_h_since_last_su=2.2,
        route_values=[_complexa_zero_su_route(), boltz, bindcraft],
    )

    cands = build_candidates([_complexa_hypothesis()], e, include_warmstart=False)

    cross = {
        c.method_family: c.expected_signal
        for c in cands
        if c.candidate_id.startswith("evidence_fallback_cross_family")
    }
    assert "bindcraft" in cross
    assert "boltzgen" in cross
    assert "diagnostic_score=0.79" in cross["bindcraft"]
    assert "diagnostic_score=0.41" in cross["boltzgen"]


def test_evidence_fallback_fires_on_stalled_no_recipes():
    """Stalled state with empty LLM output must still produce launchable,
    registry-derived fallback candidates without a fixed family ladder."""
    e = _evidence("stalled", recipes=[])
    cands = build_candidates([], e)
    fallback_cands = [c for c in cands if c.candidate_id.startswith("evidence_fallback")]
    assert len(fallback_cands) >= 2, f"got {len(fallback_cands)}"
    fams = {c.method_family for c in fallback_cands}
    assert all(c.parent_result_id is None for c in fallback_cands)
    assert not {"structure_refilter", "proteinmpnn_redesign"} & fams


def test_evidence_fallback_fires_on_strict_duplicate_collapse():
    """Strict/SU collapse is an escape state: if the LLM gives no usable card,
    deterministic fallback must still provide fresh route-light generators."""
    e = _evidence("strict_duplicate_collapse", recipes=[])
    cands = build_candidates([], e)
    fallback_cands = [c for c in cands if c.candidate_id.startswith("evidence_fallback")]
    assert len(fallback_cands) >= 2, f"got {len(fallback_cands)}"
    fams = {c.method_family for c in fallback_cands}
    assert {"complexa_beam", "bindcraft", "boltzgen"} <= fams
    assert not {"structure_refilter", "proteinmpnn_redesign"} & fams


def test_evidence_fallback_candidate_ids_are_tick_specific():
    e1 = _evidence("stalled", recipes=[])
    e2 = replace(e1, tick_id="t2")

    ids1 = {
        c.candidate_id for c in build_candidates([], e1)
        if c.candidate_id.startswith("evidence_fallback")
    }
    ids2 = {
        c.candidate_id for c in build_candidates([], e2)
        if c.candidate_id.startswith("evidence_fallback")
    }

    assert ids1 and ids2
    assert ids1.isdisjoint(ids2)


def test_evidence_fallback_is_registry_role_based():
    """Fallback should include route-light generators and exclude parent-bound
    rescue/refilter families by registry role, not a hand-maintained escape set."""
    cands = build_candidates([], _evidence("stalled", recipes=[]))
    fallback = [c for c in cands if c.candidate_id.startswith("evidence_fallback")]
    fams = {c.method_family for c in fallback}
    assert "structure_refilter" not in fams
    assert "proteinmpnn_redesign" not in fams
    assert {"complexa_beam", "bindcraft", "boltzgen"} <= fams


def _joint_fail_recipe() -> "Recipe":
    return Recipe(
        recipe_hash="x", operator_id="op", method_family="complexa_beam",
        config_delta={}, recipe_class="joint_fail", target_id="t",
        target_class="c", descendant_count=5, median_metrics={},
        representative_result_ids=[], recency_tick=1,
    )


def test_evidence_fallback_skipped_when_llm_has_feasible_generator_hypothesis():
    """When the LLM supplies a feasible cross-paradigm escape, the deterministic
    fallback must NOT pile on and out-compete it for selector quota."""
    e = _evidence("stalled", recipes=[_joint_fail_recipe()])
    h = HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim="explore a different generator", mode_affinity={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["bindcraft"],
    )
    cands = build_candidates([h], e)
    assert not any(c.candidate_id.startswith("evidence_fallback") for c in cands)


def test_evidence_fallback_fires_when_llm_escape_is_infeasible():
    """If the only LLM escape needs a concrete parent but cites only an aggregate
    or missing ref, keep the card auditable and fill the dead slot with fallback."""
    e = _evidence("stalled", recipes=[_joint_fail_recipe()])
    h = HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim="rescue iPAE", mode_affinity={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        evidence_refs=["diagnostic_chain_backlog"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["diagnostic_chain_backlog"], 0.20, None)
        ],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
    )
    cands = build_candidates([h], e)
    llm = [c for c in cands if c.hypothesis_ids == ["h1"]]
    assert llm and not llm[0].feasibility.all_ok()
    assert any(c.candidate_id.startswith("evidence_fallback") for c in cands)


def test_evidence_fallback_does_not_override_feasible_generator_card():
    """After cold-start, a feasible generator card is enough for the selector;
    fallback should not force a cross-family prior beside it."""
    e = _evidence("stalled", recipes=[_joint_fail_recipe()])
    h = HypothesisCard(
        hypothesis_id="h_complexa", target_id="t", tick_created=1,
        claim="try more Complexa", mode_affinity={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["complexa_beam"],
    )
    cands = build_candidates([h], e)
    assert any(c.hypothesis_ids == ["h_complexa"] for c in cands)
    assert not any(c.candidate_id.startswith("evidence_fallback") for c in cands)


def test_evidence_fallback_fires_with_recipes_when_llm_empty():
    """Stalled fallback remains available when archived recipes exist but no usable LLM
    hypothesis does.
    """
    e = _evidence("stalled", recipes=[_joint_fail_recipe()])
    cands = build_candidates([], e)
    assert any(c.candidate_id.startswith("evidence_fallback") for c in cands)


def test_warmstart_can_be_disabled():
    e = _evidence("low_evidence", recipes=[])
    cands = build_candidates([], e, include_warmstart=False)
    assert cands == []


def test_cold_low_evidence_warmstart_preempts_llm_hypotheses():
    """Before any evidence exists, deterministic warmstart preempts LLM cards."""
    e = _evidence("low_evidence", recipes=[])
    h = HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim="MPNN", mode_affinity={"exploit": 0.5, "rescue": 0.5, "explore": 0.0},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
    )
    cands = build_candidates([h], e)
    fams = {c.method_family for c in cands}
    assert "complexa_beam" in fams           # from warmstart
    assert "proteinmpnn_redesign" not in fams    # LLM resumes after evidence exists


def _diag_backlog(fam: str, *, accepted: int, unscored: int, completed: int) -> dict:
    return {
        "diagnostic_families": [fam],
        "total_diagnostic_artifacts": accepted,
        "total_unscored_diagnostic_artifacts": unscored,
        "by_family": {
            fam: {
                "accepted_artifacts": accepted,
                "unscored_artifacts": unscored,
                "completed_refilters": completed,
                "chain_candidates": accepted,
                "pending_chain_candidates": max(0, unscored),
                "queued_chain_candidates": 0,
                "dispatched_chain_candidates": completed,
                "native_strict_like_total": 0,
                "native_strict_like_pending_refilter": 0,
                "proxy_promising_total": 0,
                "proxy_promising_pending_refilter": 0,
            }
        },
    }


def _boltzgen_hyp(claim: str, refs=None) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h_boltz", target_id="t", tick_created=2,
        claim=claim,
        mode_affinity={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        evidence_refs=refs or ["axis_stats"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", refs or ["axis_stats"], 0.10, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["boltzgen"],
    )


def test_score_backlog_wording_does_not_launch_generator():
    """Existing backlog scoring is system chain_refilter work, not new BoltzGen."""
    e = _evidence("stalled", recipes=[])
    h = _boltzgen_hyp(
        "Score pending BoltzGen backlog via canonical structure_refilter before judging route value",
        refs=["diagnostic_chain_backlog"],
    )
    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert not cand.feasibility.all_ok()
    assert any("score_backlog_is_system_managed:boltzgen" in r for r in cand.feasibility.reasons)


def test_new_diagnostic_generation_remains_feasible_without_backlog():
    e = _evidence("stalled", recipes=[])
    h = _boltzgen_hyp("Generate new BoltzGen scaffolds with higher diversity and noise")
    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert cand.feasibility.all_ok(), cand.feasibility.reasons


def test_new_diagnostic_generation_with_backlog_context_is_not_misclassified_as_scoring():
    """Regression: backlog evidence in observed_signal must not block new generation."""
    e = replace(
        _evidence("stalled", recipes=[]),
        diagnostic_chain_backlog=_diag_backlog("boltzgen", accepted=13, unscored=13, completed=16),
    )
    h = _boltzgen_hyp(
        "BoltzGen with increased noise_scale will generate structurally distinct binding modes that escape the non-docking basin.",
        refs=["diagnostic_chain_backlog", "method_health.boltzgen"],
    )
    h = replace(
        h,
        reasoning_trace=ReasoningTrace(
            observed_signal="BoltzGen has 13 unscored artifacts but no recent SU.",
            inference="The current generators are trapped in a non-binding basin.",
            action_implication=(
                "Propose BoltzGen with higher noise_scale to explore structurally "
                "unique binding modes. This is a new generation action, not a scoring request."
            ),
        ),
    )

    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert cand.feasibility.all_ok(), cand.feasibility.reasons
    assert not any("score_backlog_is_system_managed" in r for r in cand.feasibility.reasons)


def test_generation_while_backlog_is_scored_remains_feasible():
    e = replace(
        _evidence("stalled", recipes=[]),
        diagnostic_chain_backlog=_diag_backlog("boltzgen", accepted=29, unscored=29, completed=16),
    )
    h = _boltzgen_hyp(
        "BoltzGen with noise_scale 0.6 will generate structurally diverse binding candidates.",
        refs=["diagnostic_chain_backlog"],
    )
    h = replace(
        h,
        reasoning_trace=ReasoningTrace(
            observed_signal="BoltzGen has 29 pending unscored artifacts and recent near-misses.",
            inference="A diffusion prior can explore conformational space differently.",
            action_implication=(
                "Explore BoltzGen with a different noise scale to find new SU "
                "while the backlog is scored by the system chain-refilter lane."
            ),
        ),
    )

    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert cand.feasibility.all_ok(), cand.feasibility.reasons
    assert not any("score_backlog_is_system_managed" in r for r in cand.feasibility.reasons)


def test_diagnostic_family_waits_for_first_score_conversion_feedback():
    e = replace(
        _evidence("stalled", recipes=[]),
        diagnostic_chain_backlog=_diag_backlog("boltzgen", accepted=16, unscored=13, completed=3),
    )
    h = _boltzgen_hyp("Generate another BoltzGen batch with a different stochastic seed")
    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert not cand.feasibility.all_ok()
    assert any("awaiting_first_score_conversion:boltzgen" in r for r in cand.feasibility.reasons)


def test_same_diagnostic_route_waits_for_pending_score_conversion():
    e = replace(
        _evidence("stalled", recipes=[]),
        diagnostic_chain_backlog=_diag_backlog("boltzgen", accepted=16, unscored=12, completed=4),
        route_values=[{
            "strategy_key": "route::boltzgen:boltzgen_default:budget=4,num_designs=16",
            "scope": "route",
            "family": "boltzgen",
            "action_family": "boltzgen",
            "operator_id": "boltzgen_default",
            "config_delta": {"num_designs": 16, "budget": 4},
            "pending_score_conversion_count": 12,
            "new_su": 0,
            "new_su_recent": 0,
            "near_miss_count": 0,
            "near_miss_recent": 0,
        }],
    )
    h = _boltzgen_hyp("Generate another BoltzGen batch with the default settings")
    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert not cand.feasibility.all_ok()
    assert any("awaiting_route_score_conversion:boltzgen" in r for r in cand.feasibility.reasons)


def test_different_diagnostic_route_can_explore_after_first_score_probe():
    e = replace(
        _evidence("stalled", recipes=[]),
        diagnostic_chain_backlog=_diag_backlog("boltzgen", accepted=16, unscored=12, completed=4),
        route_values=[{
            "strategy_key": "route::boltzgen:boltzgen_default:budget=4,num_designs=16",
            "scope": "route",
            "family": "boltzgen",
            "action_family": "boltzgen",
            "operator_id": "boltzgen_default",
            "config_delta": {"num_designs": 16, "budget": 4},
            "pending_score_conversion_count": 12,
            "new_su": 0,
            "new_su_recent": 0,
            "near_miss_count": 0,
            "near_miss_recent": 0,
        }],
    )
    h = replace(
        _boltzgen_hyp("Generate BoltzGen with higher noise_scale for a distinct route"),
        config_delta_suggestions={"boltzgen": {"num_designs": 16, "budget": 4, "noise_scale": 0.55}},
    )
    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert cand.feasibility.all_ok(), cand.feasibility.reasons
    assert not any("awaiting_route_score_conversion:boltzgen" in r for r in cand.feasibility.reasons)


def test_parent_bound_diagnostic_route_is_not_blocked_by_root_replay_wait():
    e = replace(
        _evidence("stalled", recipes=[]),
        route_values=[{
            "strategy_key": "route::complexa_beam:complexa_beam_default:default->proteinmpnn_redesign:proteinmpnn_redesign_default:num_seq_per_target=8",
            "scope": "route",
            "family": "proteinmpnn_redesign",
            "action_family": "proteinmpnn_redesign",
            "operator_id": "proteinmpnn_redesign_default",
            "config_delta": {"num_seq_per_target": 8},
            "pending_score_conversion_count": 12,
            "new_su": 0,
            "new_su_recent": 0,
            "near_miss_count": 0,
            "near_miss_recent": 0,
        }],
    )
    for family, config in [
        ("proteinmpnn_redesign", {"num_seq_per_target": 8}),
    ]:
        assert _diagnostic_route_feedback_reason(family, config, e) is None


def test_diagnostic_family_reopens_after_first_feedback_microbatch():
    e = replace(
        _evidence("stalled", recipes=[]),
        diagnostic_chain_backlog=_diag_backlog("boltzgen", accepted=32, unscored=16, completed=16),
    )
    h = _boltzgen_hyp("Generate another BoltzGen batch with a different stochastic seed")
    cands = build_candidates([h], e, include_warmstart=False)
    cand = next(c for c in cands if c.hypothesis_ids == ["h_boltz"])
    assert cand.feasibility.all_ok(), cand.feasibility.reasons
