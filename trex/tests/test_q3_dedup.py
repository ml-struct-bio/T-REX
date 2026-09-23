"""Tests for failed-route annotations, which provide scoped feedback without banning
stochastic retries.
"""

from __future__ import annotations

import pytest

from trex.candidate_builder import build_candidates
from trex.schemas import (
    EvidenceSummary,
    HypothesisCard,
    LLMHealthSummary,
    PredictedChange,
    PreserveConstraint,
    Recipe,
    RouteHealthSummary,
)


def _evidence_with_failed_recipe(
    failed: list[Recipe] | None = None,
) -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="t1", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=1, remaining_wall_h=10, completed_children=10, pending_children=0,
        worker_gpu_h_total=5, worker_gpu_h_last_3_ticks=3,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="stalled",
        examples=[], metric_availability={},
        recipes=failed or [],
    )


def _hyp(suggestions: dict | None = None, family: str = "complexa_beam",
         axis: str = "iPAE") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim="x", mode_affinity={"exploit": 0.5, "rescue": 0.5, "explore": 0.0},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange(axis, "decrease", ["b"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=[family],
        config_delta_suggestions=suggestions or {},
    )


def _joint_fail_recipe(operator_id: str, config: dict) -> Recipe:
    return Recipe(
        recipe_hash="h_fail",
        operator_id=operator_id,
        method_family="complexa_beam",
        config_delta=config,
        recipe_class="joint_fail",
        target_id="t",
        target_class="c",
        descendant_count=8,
        median_metrics={"pLDDT": 70.0, "iPAE": 0.65},
        representative_result_ids=["r1"],
        recency_tick=2,
    )


def _strict_recipe(operator_id: str, config: dict) -> Recipe:
    return Recipe(
        recipe_hash="h_ok",
        operator_id=operator_id,
        method_family="complexa_beam",
        config_delta=config,
        recipe_class="strict_success",
        target_id="t",
        target_class="c",
        descendant_count=2,
        median_metrics={"pLDDT": 95.0, "iPAE": 0.20},
        representative_result_ids=["r_ok"],
        recency_tick=2,
    )


def test_stochastic_config_not_dropped_when_also_strict_success():
    """Keep a configuration that produced both failed and qualified designs eligible."""
    cfg = {"beam_width": 4, "n_branch": 4}
    e = _evidence_with_failed_recipe([
        _joint_fail_recipe("complexa_beam_default", cfg),
        _strict_recipe("complexa_beam_default", cfg),
    ])
    h = _hyp({"complexa_beam": cfg})
    cands = build_candidates([h], e)
    llm_cands = [c for c in cands if c.candidate_id.startswith("cand_")]
    assert len(llm_cands) == 1
    assert llm_cands[0].feasibility.all_ok()
    assert not any(
        "prior_joint_fail_caution_not_ban" in r
        for r in llm_cands[0].feasibility.reasons
    )


def test_candidate_kept_feasible_with_scoped_caution_when_matches_recent_joint_fail():
    """LLM suggests {beam_width:4, n_branch:4} but archive shows that
    exact config recently joint-failed. That is useful feedback, but not a hard
    ban: stochastic search can need more samples of the same exact config."""
    e = _evidence_with_failed_recipe([
        _joint_fail_recipe("complexa_beam_default", {"beam_width": 4, "n_branch": 4}),
    ])
    h = _hyp({"complexa_beam": {"beam_width": 4, "n_branch": 4}})
    # Disable warmstart to isolate failed-route feedback.
    cands = build_candidates([h], e, include_warmstart=False)
    assert len(cands) == 1
    assert cands[0].feasibility.all_ok()
    assert any(
        "prior_joint_fail_caution_not_ban" in r
        for r in cands[0].feasibility.reasons
    )


def test_candidate_still_feasible_when_different_config():
    """Same family but different config (beam_width=8 instead of 4) is OK."""
    e = _evidence_with_failed_recipe([
        _joint_fail_recipe("complexa_beam_default", {"beam_width": 4, "n_branch": 4}),
    ])
    h = _hyp({"complexa_beam": {"beam_width": 8, "n_branch": 4}})
    cands = build_candidates([h], e)
    llm_cands = [c for c in cands if c.candidate_id.startswith("cand_")]
    assert len(llm_cands) == 1
    assert llm_cands[0].feasibility.all_ok()


def test_candidate_feasible_when_no_failed_recipes():
    e = _evidence_with_failed_recipe([])
    h = _hyp({"complexa_beam": {"beam_width": 4}})
    # Skip warmstart/fallback to focus this test on dedup behavior alone.
    cands = build_candidates([h], e, include_warmstart=False)
    assert len(cands) == 1
    assert cands[0].feasibility.all_ok()


def test_empty_config_matches_empty_failed_signature():
    """Default config (no overrides) gets scoped caution for an empty-config joint_fail."""
    e = _evidence_with_failed_recipe([
        _joint_fail_recipe("complexa_beam_default", {}),
    ])
    # No default setting is available for a pLDDT-only hypothesis, leaving the
    # configuration empty. Disable warmstart to isolate empty-configuration deduplication.
    h = _hyp(None, axis="pLDDT")  # Disable warmstart to isolate feedback for an empty configuration.
    cands = build_candidates([h], e, include_warmstart=False)
    assert len(cands) == 1
    assert cands[0].feasibility.all_ok()
    assert any(
        "prior_joint_fail_caution_not_ban" in r
        for r in cands[0].feasibility.reasons
    )


def test_stalled_complexa_only_hypotheses_inject_escape_floor():
    """A stalled campaign can retain the proposed retry while adding deterministic
    alternatives.
    """
    e = _evidence_with_failed_recipe([
        _joint_fail_recipe("complexa_beam_default", {"beam_width": 4, "n_branch": 4}),
    ])
    h = _hyp({"complexa_beam": {"beam_width": 4, "n_branch": 4}})
    cands = build_candidates([h], e)  # include_warmstart default True
    # the exact retry is still emitted and feasible, but marked with scoped caution...
    assert any(c.candidate_id.startswith("cand_") and c.feasibility.all_ok()
               for c in cands)
    assert any(
        c.candidate_id.startswith("cand_")
        and any("prior_joint_fail_caution_not_ban" in r for r in c.feasibility.reasons)
        for c in cands
    )
    # ...and because the LLM supplied a feasible route-light generator, the
    # evidence fallback does not inject a second fixed-family escape pool.
    assert not any(c.candidate_id.startswith("evidence_fallback") for c in cands)
    assert any(c.feasibility.all_ok() for c in cands), "stalled tick must not be dead"
