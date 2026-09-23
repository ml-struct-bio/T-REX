"""Test Planner JSON validation, parsing, and coercion of legacy response shapes."""

from __future__ import annotations

from dataclasses import replace
import pytest

from trex.planner import (
    _allowed_evidence_refs,
    _coerce_legacy_shape,
    _evidence_ref_allowed,
    _extract_json as _extract_planner_json,
    _hypothesis_outcomes_tldr,
    _materialize_cards,
    _repair_planner_schema_drift,
    _short_route_id,
    _validate_schema,
    build_evidence_for_prompt,
    evidence_tldr,
)
from trex.supervisor import _extract_json as _extract_supervisor_json


def test_planner_allows_evidence_tldr_alias_citations():
    ev = _empty_evidence()
    allowed = _allowed_evidence_refs(ev, [])
    for ref in (
        "diagnostic_driver_tldr",
        "mode_mixture_ranges",
        "mode_mixture_ranges=low_evidence:explore",
        "diagnostic_driver_tldr=min_ipae:decrease",
        "design_to_target_iptm[src=boltzgen:32]",
        "min_ipae[src=complexa:8]",
        "diagnostic_improvement_score=0.62",
        "diagnostic_improvement_axes=design_to_target_iptm",
        "negative_evidence",
        "negative_evidence=zero_su_route",
        "top_diagnostic_blocker=min_ipae",
        "top_bin_share(live,all)=0.73",
        "strict_su_top_bin_share(live,winners)=0.50",
        "strict_su_top_bin_share(live,strict_records)=0.50",
        "objective_summary.live_su_tm06=4",
        "diversity_summary.recent_strict_records_live_su_bin_share=0.75",
        "strict_duplicate_collapse=YES",
        "dry_since_last_SU=6worker-GPUh",
        "SU/worker-GPUh_recent=0.12",
        "fine_diversity_aux=TM0.8_recent/live_recent",
        "remediation_outcomes",
        "remediation_outcomes=near_miss_rescue",
        "diagnosis_outcomes::binder_scRMSD",
        "hyp_cross_family_escape",
    ):
        assert _evidence_ref_allowed(ref, allowed), ref

    concrete = allowed | {"abcdef1234567890", "route::abc123"}
    assert _evidence_ref_allowed("result_id:abcdef1234567890", concrete)
    assert _evidence_ref_allowed("result::abcdef1234567890", concrete)
    assert _evidence_ref_allowed("route::abc123", concrete)
    assert not _evidence_ref_allowed("result_id:hallucinated", allowed)
    assert not _evidence_ref_allowed("result_fake", allowed)
    assert not _evidence_ref_allowed("route::fake", allowed)


def test_hypothesis_outcomes_tldr_surfaces_resolved_status():
    """Contradicted/retired/supported hypotheses are surfaced prominently;
    active ones are not, and an empty input yields no block."""
    out = _hypothesis_outcomes_tldr([
        {"hypothesis_id": "h1", "status": "contradicted",
         "support_points": 0, "contradiction_points": 3, "descendants_evaluated": 3},
        {"hypothesis_id": "h2", "status": "active"},  # not surfaced
        {"hypothesis_id": "h3", "status": "supported",
         "support_points": 2, "contradiction_points": 0, "descendants_evaluated": 2},
    ])
    assert "HYPOTHESIS OUTCOMES" in out
    assert "h1=contradicted(sup0/con3/n3)" in out
    assert "h3=supported" in out
    assert "h2" not in out
    assert _hypothesis_outcomes_tldr([]) == ""
from trex.schemas import EvidenceSummary, LLMHealthSummary, RouteHealthSummary


def test_coerce_strips_lingering_missing_candidate_requests():
    """Discard deprecated optional fields from older responses."""
    obj = {
        "abstain": False, "confidence": 0.7, "cards": [],
        "rationale": "x", "missing_candidate_requests": [{"requested_family": "x"}],
    }
    out = _coerce_legacy_shape(obj)
    assert "missing_candidate_requests" not in out


def test_coerce_renames_hypotheses_to_cards():
    """Older prompts used `hypotheses`; coerce → `cards`."""
    obj = {"abstain": False, "confidence": 0.6, "hypotheses": [
        {"claim": "x", "mode_affinity": {"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
         "evidence_refs": ["e1"], "predicted_metric_changes": [],
         "preserve_constraints": [], "recommended_action_families": ["complexa_beam"]}
    ], "rationale": "x"}
    out = _coerce_legacy_shape(obj)
    assert "cards" in out and "hypotheses" not in out


def test_coerce_synthesizes_claim_from_title_abstract():
    """Card with title+abstract but no claim → claim = title — abstract."""
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [{
            "title": "Pivot to BindCraft", "abstract": "Beam exhausted",
            "mode_affinity": {"exploit": 0.2, "rescue": 0.3, "explore": 0.5},
            "evidence_refs": ["e1"], "predicted_metric_changes": [],
            "preserve_constraints": [], "recommended_action_families": ["bindcraft"],
        }],
    }
    out = _coerce_legacy_shape(obj)
    assert out["cards"][0]["claim"] == "Pivot to BindCraft — Beam exhausted"


def test_coerce_predicted_metric_changes_dict_to_list():
    """LLM sometimes emits predicted_metric_changes as a dict."""
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [{
            "claim": "iPAE pivot",
            "mode_affinity": {"exploit": 0.2, "rescue": 0.5, "explore": 0.3},
            "evidence_refs": ["e1"],
            "predicted_metric_changes": {
                "iPAE": {"direction": "decrease",
                          "min_relative_deficit_reduction": 0.2}
            },
            "preserve_constraints": [],
            "recommended_action_families": ["complexa_fk_steering"],
        }],
    }
    out = _coerce_legacy_shape(obj)
    pmc = out["cards"][0]["predicted_metric_changes"]
    assert isinstance(pmc, list) and len(pmc) == 1
    assert pmc[0]["axis"] == "iPAE"
    assert pmc[0]["direction"] == "decrease"


# ---------------------------------------------------------------------------
# _validate_schema — load-bearing path
# ---------------------------------------------------------------------------


def _valid_card() -> dict:
    return {
        "claim": "Test claim long enough for minLength",
        "mode_affinity": {"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
        "evidence_refs": ["e1"],
        "predicted_metric_changes": [
            {"axis": "pLDDT", "direction": "increase",
              "min_relative_deficit_reduction": 0.1}
        ],
        "preserve_constraints": [],
        "recommended_action_families": ["complexa_beam"],
        "reasoning_trace": {
            "observed_signal": "Evidence shows a testable signal.",
            "inference": "The route should improve the target axis.",
            "action_implication": "Run the recommended family/config.",
        },
    }


def test_validate_minimal_valid_object():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [_valid_card()],
    }
    ok, why = _validate_schema(obj)
    assert ok, why


def test_validate_rejects_metric_direction_opposite_to_success_contract():
    card = _valid_card()
    card["predicted_metric_changes"][0]["direction"] = "decrease"
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    ok, why = _validate_schema(obj)

    assert not ok
    assert why == "card[0].predicted[0]:direction_mismatch"


@pytest.mark.parametrize("rdr", [-0.1, 1.1, float("nan")])
def test_validate_rejects_invalid_rdr_range(rdr):
    card = _valid_card()
    card["predicted_metric_changes"][0]["min_relative_deficit_reduction"] = rdr
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    ok, why = _validate_schema(obj)

    assert not ok
    assert why == "card[0].predicted[0]:bad_rdr_range"


@pytest.mark.parametrize("abs_delta", [-1.0, 0.0])
def test_validate_rejects_nonpositive_absolute_delta(abs_delta):
    card = _valid_card()
    card["predicted_metric_changes"][0]["min_absolute_delta"] = abs_delta
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    ok, why = _validate_schema(obj)

    assert not ok
    assert why == "card[0].predicted[0]:bad_abs_delta"


@pytest.mark.parametrize(
    ("abs_delta", "expected_abs_delta", "expected_rdr", "repair_tag"),
    [
        (-1.0, 1.0, 0.0, "abs_delta_magnitude"),
        (0.0, None, 0.10, "dropped_zero_abs_delta"),
    ],
)
def test_repair_normalizes_nonpositive_absolute_delta(
    abs_delta, expected_abs_delta, expected_rdr, repair_tag,
):
    card = _valid_card()
    card["predicted_metric_changes"][0]["min_relative_deficit_reduction"] = 0.0
    card["predicted_metric_changes"][0]["min_absolute_delta"] = abs_delta
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    repaired, repairs = _repair_planner_schema_drift(obj, allowed_evidence_refs=None)
    ok, why = _validate_schema(repaired)
    repaired_pc = repaired["cards"][0]["predicted_metric_changes"][0]

    assert ok, why
    assert any(repair_tag in r for r in repairs)
    assert repaired_pc.get("min_absolute_delta") == expected_abs_delta
    assert repaired_pc["min_relative_deficit_reduction"] == expected_rdr


def test_validate_rejects_conflicting_per_axis_baselines():
    card = _valid_card()
    card["predicted_metric_changes"] = [
        {
            "axis": "pLDDT",
            "direction": "increase",
            "baseline_refs": ["baseline_a"],
            "min_relative_deficit_reduction": 0.1,
        },
        {
            "axis": "iPAE",
            "direction": "decrease",
            "baseline_refs": ["baseline_b"],
            "min_relative_deficit_reduction": 0.1,
        },
    ]
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    ok, why = _validate_schema(obj)

    assert not ok
    assert why == "card[0]:inconsistent_baseline_refs"


def test_validate_rejects_mixed_implicit_and_explicit_axis_baselines():
    card = _valid_card()
    card["predicted_metric_changes"] = [
        {
            "axis": "pLDDT", "direction": "increase",
            "baseline_refs": ["baseline_a"],
            "min_relative_deficit_reduction": 0.1,
        },
        {
            "axis": "iPAE", "direction": "decrease",
            "min_relative_deficit_reduction": 0.1,
        },
    ]
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    ok, why = _validate_schema(obj)

    assert not ok
    assert why == "card[0]:inconsistent_baseline_refs"


@pytest.mark.parametrize(
    ("ttl", "reason"),
    [(True, "bad_ttl_type"), ("10", "bad_ttl_type"), (0, "bad_ttl_range"), (21, "bad_ttl_range")],
)
def test_validate_rejects_invalid_ttl(ttl, reason):
    card = _valid_card()
    card["ttl_ticks"] = ttl
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x", "cards": [card]}

    ok, why = _validate_schema(obj)

    assert not ok
    assert why == f"card[0]:{reason}"


def test_validate_requires_reasoning_trace_for_auditable_cards():
    card = _valid_card()
    card.pop("reasoning_trace")
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [card],
    }
    ok, why = _validate_schema(obj)
    assert not ok
    assert "missing:reasoning_trace" in why


def test_preserve_diversity_is_repaired_not_materialized_as_noop():
    card = _valid_card()
    card["preserve_constraints"] = [
        {"axis": "diversity", "max_relative_deficit_increase": 0.05},
        {"axis": "plddt", "max_relative_deficit_increase": 0.05},
    ]
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [card],
    }
    ok, why = _validate_schema(obj)
    assert not ok
    assert "preserve[0]:bad_axis" in why

    repaired, repairs = _repair_planner_schema_drift(obj, allowed_evidence_refs=None)
    ok2, why2 = _validate_schema(repaired)
    assert ok2, why2
    assert any("dropped_bad_axis:diversity" in r for r in repairs)

    cards = _materialize_cards(repaired, target_id="t1", tick_id=1)
    assert [p.axis for p in cards[0].preserve_constraints] == ["pLDDT"]


def test_preserve_invalid_only_card_is_dropped_by_repair():
    card = _valid_card()
    card["preserve_constraints"] = [
        {"axis": "diversity", "max_relative_deficit_increase": 0.05},
    ]
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [card],
    }
    ok, why = _validate_schema(obj)
    assert not ok
    assert "preserve[0]:bad_axis" in why

    repaired, repairs = _repair_planner_schema_drift(obj, allowed_evidence_refs=None)
    assert repaired["cards"] == []
    assert any("dropped_all_invalid_preserve_constraints" in r for r in repairs)


def test_json_extractors_ignore_braces_inside_strings():
    planner_raw = (
        "Planner output follows.```json\n"
        '{"abstain": false, "confidence": 0.7, '
        '"rationale": "brace text {not a boundary}", '
        '"cards": [{"claim": "keep {literal}", '
        '"mode_affinity": {"exploit": 1, "rescue": 0, "explore": 0}, '
        '"evidence_refs": ["e1"], '
        '"predicted_metric_changes": [{"axis": "pLDDT", "direction": "increase", '
        '"min_relative_deficit_reduction": 0.1}], '
        '"preserve_constraints": [], '
        '"recommended_action_families": ["complexa_beam"]}]}\n```'
    )
    obj = _extract_planner_json(planner_raw)
    assert obj is not None
    assert obj["cards"][0]["claim"] == "keep {literal}"

    supervisor_raw = (
        "```json\n"
        '{"confidence": 0.8, "mode_mixture": {"exploit": 0.7, "rescue": 0.2, "explore": 0.1}, '
        '"ranking": [{"candidate_id": "c1", "mode": "exploit", "resource_class": "low", '
        '"score": 1.0, "rationale": "uses {brace} safely"}], '
        '"rationale": "ok"}\n```'
    )
    sup = _extract_supervisor_json(supervisor_raw)
    assert sup is not None
    assert sup["ranking"][0]["rationale"] == "uses {brace} safely"


def test_materialize_cards_keeps_structured_reasoning_trace():
    card = _valid_card()
    card["reasoning_trace"] = {
        "observed_signal": "Recent near-misses pass pLDDT but fail iPAE.",
        "inference": "Interface placement is the limiting axis.",
        "action_implication": "Use a same-family rescue config aimed at iPAE.",
    }
    cards = _materialize_cards(
        {"cards": [card]},
        target_id="t1",
        tick_id=7,
    )
    assert cards[0].reasoning_trace.observed_signal.startswith("Recent near-misses")
    assert "Interface placement" in cards[0].reasoning_trace.inference
    assert "iPAE" in cards[0].reasoning_trace.action_implication


def test_materialize_cards_defaults_missing_reasoning_trace_when_validator_bypassed():
    card = _valid_card()
    card.pop("reasoning_trace")
    cards = _materialize_cards(
        {"cards": [card]},
        target_id="t1",
        tick_id=7,
    )
    assert cards[0].reasoning_trace.observed_signal == ""


def test_validate_recovers_missing_abstain_confidence_rationale():
    """`_coerce_legacy_shape` setdefaults abstain/confidence/rationale so the
    validator recovers from missing fields — load-bearing defensive behavior."""
    for missing in ("abstain", "confidence", "rationale"):
        obj = {
            "abstain": False, "confidence": 0.7, "rationale": "x",
            "cards": [_valid_card()],
        }
        del obj[missing]
        ok, why = _validate_schema(obj)
        assert ok, f"expected recovery for missing {missing}, got {why}"


def test_validate_missing_cards_fails():
    """`cards` is the only field with no default; missing → fail-fast."""
    obj = {"abstain": False, "confidence": 0.7, "rationale": "x"}
    ok, why = _validate_schema(obj)
    assert not ok
    assert "missing_top:cards" in why


def test_validate_rejects_confidence_out_of_range():
    for bad in (-0.1, 1.5, 2.0):
        obj = {"abstain": False, "confidence": bad, "rationale": "x", "cards": []}
        ok, why = _validate_schema(obj)
        assert not ok
        assert "confidence_out_of_range" in why


def test_validate_post_mcr_no_required_field():
    """Deprecated optional fields are not required."""
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [_valid_card()],
    }
    ok, why = _validate_schema(obj)
    assert ok, why
    # And legacy objects still validate after coercion strips MCR
    obj_with_mcr = dict(obj, missing_candidate_requests=[{"x": 1}])
    ok2, why2 = _validate_schema(obj_with_mcr)
    assert ok2, why2


def test_validate_card_evidence_refs_empty_fails():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [{
            **_valid_card(),
            "evidence_refs": [],
        }],
    }
    ok, why = _validate_schema(obj)
    assert not ok
    assert "evidence_refs_empty" in why


def test_validate_card_predicted_changes_empty_fails():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [{
            **_valid_card(),
            "predicted_metric_changes": [],
        }],
    }
    ok, why = _validate_schema(obj)
    assert not ok
    assert "predicted_changes_empty" in why


def test_repair_schema_drift_preserves_safe_planner_card():
    card = _valid_card()
    card["mode_affinity"] = {"exploit": 0.8}
    card["evidence_refs"] = ["made_up_result", "evidence"]
    card["predicted_metric_changes"] = [
        {"axis": "ipae", "direction": "decrease"}
    ]
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [card],
    }
    ok, why = _validate_schema(obj, allowed_evidence_refs={"evidence"})
    assert not ok

    repaired, repairs = _repair_planner_schema_drift(
        obj, allowed_evidence_refs={"evidence"}
    )
    ok2, why2 = _validate_schema(repaired, allowed_evidence_refs={"evidence"})
    assert ok2, why2
    fixed = repaired["cards"][0]
    assert fixed["mode_affinity"]["rescue"] == 0.0
    assert fixed["mode_affinity"]["explore"] == 0.0
    assert fixed["evidence_refs"] == ["evidence"]
    assert fixed["predicted_metric_changes"][0]["axis"] == "iPAE"
    assert fixed["predicted_metric_changes"][0]["min_relative_deficit_reduction"] == 0.10
    assert any("filtered_unknown_evidence_refs" in r for r in repairs)


def test_repair_schema_drift_normalizes_mode_affinity_and_truncates_cards():
    cards = []
    for idx in range(6):
        card = _valid_card()
        card["claim"] = f"Card {idx} with enough text"
        card["mode_affinity"] = {"exploit": 2.0, "rescue": 1.0, "explore": 1.0}
        cards.append(card)
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": cards,
    }
    ok, why = _validate_schema(obj)
    assert not ok
    assert why.startswith("too_many_cards")

    repaired, repairs = _repair_planner_schema_drift(obj, allowed_evidence_refs=None)
    ok2, why2 = _validate_schema(repaired)
    assert ok2, why2
    assert len(repaired["cards"]) == 4
    assert repaired["cards"][0]["mode_affinity"] == {
        "exploit": 0.5, "rescue": 0.25, "explore": 0.25,
    }
    assert any("cards_truncated_to_4" in r for r in repairs)
    assert any("normalized_mode_affinity" in r for r in repairs)


def test_repair_schema_drift_drops_bad_mode_affinity_not_equal_thirds():
    for bad_affinity in (
        {"exploit": 0.0, "rescue": 0.0, "explore": 0.0},
        {"exploit": -0.1, "rescue": 0.5, "explore": 0.6},
        {"exploit": "soon", "rescue": 0.5, "explore": 0.5},
    ):
        card = _valid_card()
        card["mode_affinity"] = bad_affinity
        obj = {
            "abstain": False, "confidence": 0.7, "rationale": "x",
            "cards": [card],
        }
        ok, why = _validate_schema(obj)
        assert not ok
        assert "mode_affinity" in why

        repaired, repairs = _repair_planner_schema_drift(obj, allowed_evidence_refs=None)
        assert repaired["cards"] == []
        assert any("dropped_bad_mode_affinity" in r for r in repairs)


def test_repair_schema_drift_does_not_accept_unsupported_science_axis():
    card = _valid_card()
    card["predicted_metric_changes"] = [
        {"axis": "duplicate_fraction", "direction": "decrease"}
    ]
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "x",
        "cards": [card],
    }
    repaired, _ = _repair_planner_schema_drift(obj, allowed_evidence_refs=None)
    ok, why = _validate_schema(repaired)
    assert not ok
    assert "bad_axis" in why


def _empty_evidence() -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="t1", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=0, remaining_wall_h=10,
        completed_children=0, pending_children=0,
        worker_gpu_h_total=0, worker_gpu_h_last_3_ticks=0,
        strict_count=0, global_new_strict=0,
        run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="low_evidence",
        examples=[], metric_availability={},
    )


def test_keep_always_method_health_even_when_empty():
    """Preserve an empty family map in the prompt evidence."""
    ev = _empty_evidence()
    d = build_evidence_for_prompt(ev)
    assert "method_health" in d
    assert "axis_stats" in d
    assert "joint_patterns" in d
    assert "recipes" in d


def test_drops_only_none_or_empty_string_top_level():
    """None / "" top-level fields dropped; lists/dicts in KEEP_ALWAYS preserved."""
    ev = _empty_evidence()
    d = build_evidence_for_prompt(ev)
    # top_bin_share / duplicate_fraction / su_per_gpu_h_recent are None → dropped
    assert "top_bin_share" not in d
    assert "duplicate_fraction" not in d
    assert "su_per_gpu_h_recent" not in d
    # state_label is a non-empty string → kept
    assert d.get("state_label") == "low_evidence"


def test_prompt_view_collapses_objective_and_diversity_blocks():
    """LLM prompt sees one objective block and one diversity block, not
    scattered denominator / dedup scalars that can be misread as separate
    objectives."""
    ev = replace(
        _empty_evidence(),
        strict_count=12,
        global_new_strict=99,
        run_su_count=4,
        run_su_count_delta=1,
        su_per_gpu_h_recent=0.25,
        duplicate_fraction=0.7,
        top_bin_share=0.8,
        strict_per_su_total=3.0,
        strict_duplicate_collapse_signal=True,
        strict_su_top_bin_share=0.75,
        run_su_per_worker_gpu_h_total=0.4,
        worker_wall_gpu_count=3.0,
        worker_wall_gpu_h_total=12.0,
        run_su_per_worker_wall_gpu_h_total=0.3333,
        run_su_hwm=5,
        run_su_hwm_delta=1,
        run_su_hwm_per_worker_wall_gpu_h_total=0.4167,
        charged_gpu_h_total=20.0,
        run_su_per_charged_gpu_h_total=0.2,
    )
    d = build_evidence_for_prompt(ev)

    assert d["objective_summary"]["headline_objective"] == "live_SU_TM0.6_per_worker_wall_GPUh"
    assert d["objective_summary"]["live_su_tm06"] == 4
    assert d["objective_summary"]["raw_strict_count"] == 12
    charged = d["objective_summary"]["charged_gpu_audit"]
    assert charged["charged_gpu_h_total"] == 20.0
    assert charged["su_per_charged_gpu_h_total"] == 0.2
    assert charged["role"] == "audit_only_not_decision_objective"
    assert d["diversity_summary"]["recent_strict_records_live_su_bin_share"] == 0.75
    assert d["diversity_summary"]["recent_all_scored_live_top_bin_share"] == 0.8
    for noisy_key in (
        "global_new_strict", "run_su_count", "strict_count",
        "run_su_per_worker_wall_gpu_h_total", "charged_gpu_h_total",
        "strict_su_top_bin_share", "top_bin_share",
    ):
        assert noisy_key not in d


def test_parent_artifact_ids_are_compacted_out_of_llm_prompt():
    ev = _empty_evidence()
    ev.parent_artifact_result_ids.extend([f"rid_{i:04d}" for i in range(200)])
    d = build_evidence_for_prompt(ev)
    assert "parent_artifact_result_ids" not in d
    assert d["parent_artifact_result_count"] == 200
    assert d["parent_artifacts_available"] is True


def test_route_values_are_compacted_for_prompt():
    ev = _empty_evidence()
    long_refs = [f"rid_{i:04d}" for i in range(120)]
    ev.route_values.append({
        "scope": "route",
        "status": "promote",
        "family": "complexa_beam",
        "action_family": "complexa_beam",
        "root_family": "complexa_beam",
        "operator_id": "complexa_beam_default",
        "route_role": "generator_with_af2_score_conversion",
        "config_delta": {
            "beam_width": 8, "nsamples": 4, "nsteps": 400,
            "reward_i_ptm_weight": 1.0, "reward_min_ipae_weight": -1.0,
        },
        "attempts": 16,
        "new_su": 3,
        "new_su_recent": 1,
        "near_miss_recent": 2,
        "route_gpu_h": 1.25,
        "recent_new_su_per_route_gpu_h": 0.8,
        "strict_per_su": 2.0,
        "strategy_key": "route::complexa_beam:beam_width=8,nsamples=4#abcdef",
        "evidence_refs": long_refs,
    })
    view = build_evidence_for_prompt(ev)
    row = view["route_values"][0]
    assert row["route_id"].startswith("route::")
    assert row["family"] == "complexa_beam"
    assert row["new_su"] == 3
    assert row["record_recent_new_su_per_route_gpu_h"] == 0.8
    assert row["config_delta"]["beam_width"] == 8
    assert "evidence_refs" not in row
    assert "strategy_key" not in row


def test_parent_bound_route_rows_precede_family_rollups_in_prompt_view():
    ev0 = _empty_evidence()
    route_values = [
        {
            "scope": "family",
            "strategy_key": "family::proteinmpnn_redesign",
            "family": "proteinmpnn_redesign",
            "action_family": "proteinmpnn_redesign",
            "route_role": "sequence_redesign_family_rollup",
            "new_su": 1,
            "new_su_per_route_gpu_h": 1.3,
        },
        {
            "scope": "route",
            "status": "diversify",
            "marginal_status": "productive_but_duplicate",
            "strategy_key": "route::bindcraft:bindcraft_default:max_trajectories=2#dad3aa0305->proteinmpnn_redesign:interface_redesign:sampling_temp=0.1#40f6ab16dc",
            "parent_strategy_key": "route::bindcraft:bindcraft_default:max_trajectories=2#dad3aa0305",
            "family": "proteinmpnn_redesign",
            "root_family": "bindcraft",
            "action_family": "proteinmpnn_redesign",
            "route_role": "sequence_redesign_with_af2_score_conversion",
            "new_su": 1,
            "strict_count": 12,
            "strict_per_su": 12.0,
            "new_su_per_route_gpu_h": 1.36,
        },
    ]
    ev = replace(
        ev0,
        method_health={
            "proteinmpnn_redesign": {
                "family": "proteinmpnn_redesign",
                "attempts": 16,
                "chained_strict_yield_su": 1,
                "chained_su_per_gpu_h": 1.3,
            }
        },
        route_values=route_values,
    )

    view = build_evidence_for_prompt(ev)
    assert view["method_health"]["proteinmpnn_redesign"]["parent_bound_rollup"] is True
    assert view["method_health"]["proteinmpnn_redesign"]["standalone_generator"] is False
    first = view["route_values"][0]
    assert first["scope"] == "route"
    assert first["route_lineage"] == "bindcraft->proteinmpnn_redesign"
    assert first["root_family"] == "bindcraft"
    family = next(r for r in view["route_values"] if r.get("scope") == "family" and r.get("family") == "proteinmpnn_redesign")
    assert family["parent_bound_rollup"] is True
    assert family["standalone_generator"] is False
    assert "exact route rows" in family["interpretation"]

def test_route_values_prompt_prioritizes_gpu_recent_marginal_value_over_stale_lifetime():
    ev = _empty_evidence()
    ev.route_values.extend([
        {
            "scope": "route", "status": "promote",
            "strategy_key": "route::complexa_fk_steering:old:default",
            "family": "complexa_fk_steering", "action_family": "complexa_fk_steering",
            "new_su": 20, "new_su_recent": 0, "new_su_recent_gpu": 0,
            "new_su_per_route_gpu_h": 10.0,
            "gpu_recent_new_su_per_route_gpu_h": 0.0,
            "gpu_recent_route_gpu_h": 3.0,
        },
        {
            "scope": "route", "status": "promote",
            "strategy_key": "route::complexa_beam:fresh:default",
            "family": "complexa_beam", "action_family": "complexa_beam",
            "new_su": 5, "new_su_recent": 1, "new_su_recent_gpu": 1,
            "new_su_per_route_gpu_h": 1.0,
            "gpu_recent_new_su_per_route_gpu_h": 2.0,
            "gpu_recent_route_gpu_h": 0.5,
        },
    ])
    view = build_evidence_for_prompt(ev)
    assert view["route_values"][0]["family"] == "complexa_beam"
    assert view["route_values"][0]["value_rate_for_ranking"] == 2.0
    assert view["route_values"][1]["value_rate_for_ranking"] == 0.0


def test_route_values_prompt_does_not_let_stale_collapse_status_hide_current_su_value():
    ev = _empty_evidence()
    ev.route_values.extend([
        {
            "scope": "route", "status": "collapse_risk",
            "strategy_key": "route::complexa_fk_steering:stale:default",
            "family": "complexa_fk_steering", "action_family": "complexa_fk_steering",
            "marginal_status": "dry_duplicate",
            "new_su": 20, "new_su_recent": 0, "new_su_recent_gpu": 0,
            "new_su_per_route_gpu_h": 10.0,
            "gpu_recent_new_su_per_route_gpu_h": 0.0,
            "gpu_recent_route_gpu_h": 3.0,
        },
        {
            "scope": "route", "status": "healthy",
            "strategy_key": "route::complexa_beam:fresh:default",
            "family": "complexa_beam", "action_family": "complexa_beam",
            "marginal_status": "productive",
            "new_su": 5, "new_su_recent": 1, "new_su_recent_gpu": 1,
            "new_su_per_route_gpu_h": 1.0,
            "gpu_recent_new_su_per_route_gpu_h": 2.0,
            "gpu_recent_route_gpu_h": 0.5,
        },
    ])
    view = build_evidence_for_prompt(ev)
    assert view["route_values"][0]["family"] == "complexa_beam"
    assert view["route_values"][0]["marginal_status"] == "productive"
    assert view["route_values"][1]["marginal_status"] == "dry_duplicate"


def test_route_values_prompt_exposes_medium_recent_without_promoting_over_short_recent():
    from trex.schemas import RouteValueSummary

    ev = _empty_evidence()
    ev.route_values.extend([
        RouteValueSummary(
            strategy_key="route::bindcraft:delayed:default",
            scope="route", family="bindcraft", root_family="bindcraft",
            action_family="bindcraft", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=8.0, new_su=2, new_su_recent=0, new_su_recent_gpu=0,
            medium_recent_new_su=1, medium_recent_route_gpu_h=6.0,
            medium_recent_new_su_per_route_gpu_h=0.167,
            new_su_per_route_gpu_h=0.25,
            status="diversify", marginal_status="delayed_productive_duplicate",
        ),
        RouteValueSummary(
            strategy_key="route::complexa_beam:fresh:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=1.0, new_su=2, new_su_recent=1, new_su_recent_gpu=1,
            new_su_per_route_gpu_h=2.0, gpu_recent_route_gpu_h=0.5,
            gpu_recent_new_su_per_route_gpu_h=2.0,
            status="healthy", marginal_status="productive",
        ),
    ])
    view = build_evidence_for_prompt(ev)
    assert view["route_values"][0]["family"] == "complexa_beam"
    delayed = next(r for r in view["route_values"] if r.get("family") == "bindcraft")
    assert delayed["medium_recent_new_su"] == 1
    assert delayed["medium_recent_new_su_per_route_gpu_h"] == 0.167
    assert delayed["value_rate_for_ranking"] == 0.167
    assert delayed["value_rate_source"] == "medium_recent"
    assert delayed["lifetime_su_per_route_gpu_h"] == 0.25


def test_route_values_prompt_exposes_canonical_quality_evidence():
    from trex.schemas import RouteValueSummary

    ev = _empty_evidence()
    ev.route_values.append(RouteValueSummary(
        strategy_key="route::complexa_beam:quality:default",
        scope="route", family="complexa_beam", root_family="complexa_beam",
        action_family="complexa_beam", scoring_family=None,
        operator_id="op", config_signature="default",
        route_gpu_h=2.0, new_su=3, new_su_recent_gpu=1,
        gpu_recent_route_gpu_h=0.5, gpu_recent_new_su_per_route_gpu_h=2.0,
        strict_quality_n_unique_bins=3,
        strict_quality_median=0.72,
        strict_quality_p25=0.61,
        strict_quality_axis_margins={
            "pLDDT": 1.2, "iPAE": 0.8, "binder_scRMSD": 1.0,
        },
        status="healthy", marginal_status="productive",
    ))

    row = build_evidence_for_prompt(ev)["route_values"][0]

    assert row["strict_quality_n_unique_bins"] == 3
    assert row["strict_quality_median"] == 0.72
    assert row["strict_quality_p25"] == 0.61
    assert row["strict_quality_axis_margins"]["iPAE"] == 0.8


def test_evidence_tldr_includes_current_healthy_route_value_before_stale_collapse():
    from trex.schemas import RouteValueSummary

    ev = _empty_evidence()
    ev.route_values.extend([
        RouteValueSummary(
            strategy_key="route::complexa_fk_steering:stale:default",
            scope="route", family="complexa_fk_steering", root_family="complexa_fk_steering",
            action_family="complexa_fk_steering", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=10.0, new_su=20, new_su_recent=0, new_su_recent_gpu=0,
            new_su_per_route_gpu_h=10.0, gpu_recent_route_gpu_h=3.0,
            gpu_recent_new_su_per_route_gpu_h=0.0,
            status="collapse_risk", marginal_status="dry_duplicate",
        ),
        RouteValueSummary(
            strategy_key="route::complexa_beam:fresh:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=1.0, new_su=2, new_su_recent=1, new_su_recent_gpu=1,
            new_su_per_route_gpu_h=2.0, gpu_recent_route_gpu_h=0.5,
            gpu_recent_new_su_per_route_gpu_h=2.0,
            status="healthy", marginal_status="productive",
        ),
    ])
    tl = evidence_tldr(ev)
    assert "ROUTE VALUE / COST-NORMALIZED FEEDBACK" in tl
    assert tl.index("healthy/productive:route::complexa_beam") < tl.index("collapse_risk/dry_duplicate:route::complexa_fk_steering")
    assert "SU=2(+1)" in tl
    assert "collapse_risk/dry_duplicate:route::complexa_fk_steering" in tl
    assert "rate=10" not in tl


def test_route_values_prompt_uses_objective_summary_dry_time_for_ranking():
    from trex.schemas import RouteValueSummary

    ev = replace(_empty_evidence(), gpu_h_since_last_su=14.0)
    ev.route_values.append(RouteValueSummary(
        strategy_key="route::complexa_beam:stale_record_recent",
        scope="route", family="complexa_beam", root_family="complexa_beam",
        action_family="complexa_beam", scoring_family=None,
        operator_id="op", config_signature="default",
        route_gpu_h=3.0, new_su=4, new_su_recent=1, new_su_recent_gpu=0,
        recent_new_su_per_route_gpu_h=3.0,
        new_su_per_route_gpu_h=1.3, status="promote", marginal_status="observed",
    ))

    row = build_evidence_for_prompt(ev)["route_values"][0]
    assert row["value_rate_source"] == "lifetime_memory"
    assert "value_rate_for_ranking" not in row


def test_route_values_prompt_keeps_lifetime_memory_out_of_ranking_rate():
    from trex.schemas import RouteValueSummary

    ev = _empty_evidence()
    ev.route_values.append(RouteValueSummary(
        strategy_key="route::complexa_beam:old:default",
        scope="route", family="complexa_beam", root_family="complexa_beam",
        action_family="complexa_beam", scoring_family=None,
        operator_id="op", config_signature="default",
        route_gpu_h=2.0, new_su=5, new_su_recent=0, new_su_recent_gpu=0,
        medium_recent_new_su=0, new_su_per_route_gpu_h=2.5,
        status="promote", marginal_status="observed",
    ))

    view = build_evidence_for_prompt(ev)
    row = view["route_values"][0]
    assert row["value_rate_source"] == "lifetime_memory"
    assert "value_rate_for_ranking" not in row
    assert row["lifetime_su_per_route_gpu_h"] == 2.5


def test_evidence_ref_validator_accepts_compact_namespace_refs():
    allowed = {"route_values", "selector_context", "method_health", "state_label"}
    assert _evidence_ref_allowed("route_values::complexa_mcts", allowed)
    assert _evidence_ref_allowed("selector_context.cost_admission", allowed)
    assert _evidence_ref_allowed("method_health::bindcraft", allowed)
    assert _evidence_ref_allowed("state=stalled", allowed)
    assert _evidence_ref_allowed("warmstart", allowed)
    assert _evidence_ref_allowed("EvidenceSummary", allowed)
    assert _evidence_ref_allowed("hypotheses.hyp_0010_00", allowed)
    assert _evidence_ref_allowed("hypothesis_ids.hyp_0010_00", allowed)
    assert _evidence_ref_allowed("cross_family_escape", allowed)
    assert _evidence_ref_allowed("recent_critic_flags_last_tick", allowed)
    assert _evidence_ref_allowed("mode_mixture_ranges", allowed)
    assert _evidence_ref_allowed("exemplar::7f9868d3351b1952", allowed)
    assert _evidence_ref_allowed("example:a955dda2185c4f63", allowed)


def test_evidence_ref_validator_accepts_scalar_value_refs():
    allowed = {"top_bin_share", "strict_su_top_bin_share", "run_SU", "route_values", "route_replay", "route_value_replay"}
    assert _evidence_ref_allowed("top_bin_share=0.73", allowed)
    assert _evidence_ref_allowed("strict_su_top_bin_share=0.50", allowed)
    assert _evidence_ref_allowed("run_SU=0", allowed)
    assert _evidence_ref_allowed("route_replay_v7r017_00_complexa_fk_steering", allowed)
    assert _evidence_ref_allowed("route_value_replay", allowed)


def test_allowed_refs_include_compact_route_ids_and_tick_aliases():
    ev = replace(_empty_evidence(), tick_id="v7r004")
    strategy_key = "route::complexa_beam:complexa_beam_default:beam_width=8#abcdef"
    parent_key = "route::complexa_beam:complexa_beam_default:beam_width=4#123456"
    ev.route_values.append({
        "scope": "route",
        "strategy_key": strategy_key,
        "parent_strategy_key": parent_key,
        "evidence_refs": ["rid_parent", "rid_child"],
    })
    ev.strategy_feedback.append({
        "strategy_key": "complexa_beam::complexa_beam_default::abc",
        "representative_failure_ids": ["rid_fail"],
        "representative_near_miss_ids": ["rid_near"],
    })

    allowed = _allowed_evidence_refs(ev, [])
    assert "tick_v7r004" in allowed
    assert strategy_key in allowed
    assert parent_key in allowed
    assert _short_route_id(strategy_key) in allowed
    assert _short_route_id(parent_key) in allowed
    assert "route::abcdef" in allowed
    assert "route::123456" in allowed
    assert "rid_child" in allowed
    assert "rid_fail" in allowed


def test_materialize_cards_coerces_bad_ttl_ticks_without_crashing():
    """Normalize invalid ttl_ticks without interrupting hypothesis creation."""
    for bad, expected in [
        ("soon", 10),     # non-numeric → default 10
        ("3.5", 3),       # numeric string → int
        (3.9, 3),         # float → int (truncate)
        (999, 20),        # over-range → clamp to 20
        (0, 1),           # under-range → clamp to 1
        (-5, 1),          # negative → clamp to 1
    ]:
        card = _valid_card()
        card["ttl_ticks"] = bad
        cards = _materialize_cards({"cards": [card]}, target_id="t1", tick_id=7)
        assert cards[0].ttl_ticks == expected, (bad, cards[0].ttl_ticks)


def test_materialize_cards_missing_ttl_ticks_defaults_to_10():
    cards = _materialize_cards({"cards": [_valid_card()]}, target_id="t1", tick_id=7)
    assert cards[0].ttl_ticks == 10


def test_filter_cards_to_available_drops_and_trims():
    """A card recommending only unavailable families is dropped (recovers the
    hypothesis slot); a mixed card is trimmed to the available subset; an
    all-available card is unchanged."""
    from trex.planner import _filter_cards_to_available
    good = _valid_card(); good["recommended_action_families"] = ["complexa_beam"]
    mixed = _valid_card(); mixed["recommended_action_families"] = ["bindcraft", "boltzgen"]
    bad = _valid_card(); bad["recommended_action_families"] = ["boltzgen"]
    cards = _materialize_cards({"cards": [good, mixed, bad]}, target_id="t", tick_id=1)
    out = _filter_cards_to_available(cards, {"complexa_beam", "bindcraft"})
    fams = [c.recommended_action_families for c in out]
    assert len(out) == 2                       # boltzgen-only card dropped
    assert ["complexa_beam"] in fams
    assert ["bindcraft"] in fams               # trimmed (boltzgen removed)
    assert all("boltzgen" not in f for f in fams)


def test_filter_cards_to_available_noop_when_empty_or_all_available():
    from trex.planner import _filter_cards_to_available
    c = _valid_card(); c["recommended_action_families"] = ["complexa_beam"]
    cards = _materialize_cards({"cards": [c]}, target_id="t", tick_id=1)
    assert _filter_cards_to_available(cards, set()) == cards          # no avail set → no-op
    assert len(_filter_cards_to_available(cards, {"complexa_beam"})) == 1
