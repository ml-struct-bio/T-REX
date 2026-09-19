"""Q12: Supervisor duplicate-candidate-across-modes prompt + retry.

Tests:
  1. SUPERVISOR_SYSTEM prompt explicitly forbids duplicate-mode.
  2. _validate_schema rejects with the expected fail_reason.
  3. call_supervisor performs scoped one-shot retry when first call
     returns duplicate_candidate_across_modes and second call is clean.
"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

from trex.supervisor import (
    SUPERVISOR_SYSTEM,
    SupervisorCallConfig,
    _allowed_evidence_refs,
    _evidence_ref_allowed,
    _materialize_decisions,
    _validate_schema,
    build_user_prompt,
    call_supervisor,
)
from trex.planner import _short_route_id
from trex.schemas import (
    ActionCandidate, EvidenceSummary, FeasibilityCheck, HypothesisCard,
    PredictedChange, RouteValueSummary,
)
from trex.tests.fixtures.campaign_evidence import case_stalled




def test_supervisor_evidence_ref_accepts_compact_namespace_refs():
    allowed = {"route_values", "selector_context", "method_health", "state_label"}
    assert _evidence_ref_allowed("route_values::complexa_mcts", allowed)
    assert not _evidence_ref_allowed("route::93113c943", allowed)
    assert _evidence_ref_allowed("route::93113c943", allowed | {"route::93113c943"})
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


def test_supervisor_allows_evidence_tldr_alias_citations():
    h, c = _hyp_and_cand("cand_abc")
    _, ev = case_stalled("test-model")
    allowed = _allowed_evidence_refs(ev, [h], [c])
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
        "strict_duplicate_collapse=YES",
        "dry_since_last_SU=6worker-GPUh",
        "SU/worker-GPUh_recent=0.12",
        "fine_diversity_aux=TM0.8_recent/live_recent",
        "remediation_outcomes",
        "remediation_outcomes=near_miss_rescue",
        "diagnostic_i4_mcts",
        "diagnostic_i4_mcts=plddt_i_pae_seed",
        "recent_critic_flags_last_tick=cost_deferral",
        "cross_family_escape=needed",
        "hypothesis_ids.hyp_0020_00",
        "exemplar::49b3a58d7f27058e",
    ):
        assert _evidence_ref_allowed(ref, allowed), ref

    concrete = allowed | {"abcdef1234567890", "route::abc123"}
    assert _evidence_ref_allowed("result_id:abcdef1234567890", concrete)
    assert _evidence_ref_allowed("result::abcdef1234567890", concrete)
    assert _evidence_ref_allowed("route::abc123", concrete)
    assert not _evidence_ref_allowed("result_id:hallucinated", allowed)
    assert not _evidence_ref_allowed("result_fake", allowed)
    assert not _evidence_ref_allowed("route::fake", allowed)


def test_supervisor_evidence_ref_accepts_scalar_values_and_candidate_ids():
    h, c = _hyp_and_cand("cand_abc")
    _, ev = case_stalled("test-model")
    allowed = _allowed_evidence_refs(ev, [h], [c])
    assert "cand_abc" in allowed
    assert _evidence_ref_allowed("cand_abc", allowed)
    assert _evidence_ref_allowed("top_bin_share=0.73", allowed)
    assert _evidence_ref_allowed("run_SU=0", allowed)
    assert _evidence_ref_allowed("route_replay_v7r017_00_complexa_fk_steering", allowed)
    assert _evidence_ref_allowed("route_value_replay", allowed)


def test_supervisor_allowed_refs_include_candidate_hypothesis_ids():
    h, c0 = _hyp_and_cand("cand_abc")
    c = replace(c0, hypothesis_ids=["diagnostic_i4_mcts", h.hypothesis_id])
    _, ev = case_stalled("test-model")
    allowed = _allowed_evidence_refs(ev, [h], [c])
    assert _evidence_ref_allowed("diagnostic_i4_mcts", allowed)
    assert _evidence_ref_allowed(h.hypothesis_id, allowed)


def test_supervisor_allowed_refs_include_compact_route_ids_and_tick_aliases():
    h, c = _hyp_and_cand("c1")
    _, ev0 = case_stalled("test-model")
    ev = replace(ev0, tick_id="v7r004")
    strategy_key = "route::complexa_fk_steering:complexa_fk_steering_default:temperature=0.1#abcdef"
    ev.route_values.append({
        "scope": "route",
        "strategy_key": strategy_key,
        "evidence_refs": ["rid_route"],
    })
    allowed = _allowed_evidence_refs(ev, [h], [c])
    assert "tick_v7r004" in allowed
    assert strategy_key in allowed
    assert _short_route_id(strategy_key) in allowed
    assert "route::abcdef" in allowed
    assert "rid_route" in allowed

    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.2, "explore": 0.3},
        "candidate_decisions": [{
            **_dec("c1", "exploit", 1),
            "evidence_refs": [_short_route_id(strategy_key), "tick_v7r004"],
        }],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1"}, allowed_evidence_refs=allowed)
    assert ok, why


def _dec(cid: str, mode: str, rank: int = 1):
    return {
        "candidate_id": cid, "mode": mode, "rank_in_mode": rank,
        "global_rank": rank,
        "resource_class": "standard",
        "what": "x", "why": "y", "evidence_refs": ["e1"],
        "expected_signal": "z", "stop_or_downgrade_if": "w",
    }


def test_supervisor_system_prompt_forbids_duplicate_mode():
    """The prompt must explicitly state the duplicate-mode constraint
    so the LLM has it in-context."""
    assert "AT MOST ONE mode" in SUPERVISOR_SYSTEM
    assert "duplicate" in SUPERVISOR_SYSTEM.lower()


def test_supervisor_system_prompt_requires_mixture_ranking_coherence():
    assert "Keep mode_mixture coherent with candidate_decisions" in SUPERVISOR_SYSTEM
    assert "do not rely on the Selector" in SUPERVISOR_SYSTEM


def test_supervisor_system_prompt_defines_lexicographic_quality_tier():
    assert "lower rate is at least 85% of the higher rate" in SUPERVISOR_SYSTEM
    assert "MUST\n   rank higher strict_quality_p25" in SUPERVISOR_SYSTEM
    assert "Quality cannot compensate outside" in SUPERVISOR_SYSTEM


def test_supervisor_requires_unique_contiguous_global_launch_order():
    base = {
        "abstain": False,
        "confidence": 0.7,
        "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
        "candidate_decisions": [
            {**_dec("c1", "exploit", 1), "global_rank": 1},
            {**_dec("c2", "rescue", 1), "global_rank": 2},
        ],
    }
    ok, why = _validate_schema(base, known_candidate_ids={"c1", "c2"})
    assert ok, why

    duplicate = {
        **base,
        "candidate_decisions": [
            {**_dec("c1", "exploit", 1), "global_rank": 1},
            {**_dec("c2", "rescue", 1), "global_rank": 1},
        ],
    }
    ok, why = _validate_schema(duplicate, known_candidate_ids={"c1", "c2"})
    assert not ok and why == "dup_global_rank"

    gap = {
        **base,
        "candidate_decisions": [
            {**_dec("c1", "exploit", 1), "global_rank": 1},
            {**_dec("c2", "rescue", 1), "global_rank": 3},
        ],
    }
    ok, why = _validate_schema(gap, known_candidate_ids={"c1", "c2"})
    assert not ok and why == "noncontiguous_global_rank"


def test_supervisor_joins_exact_route_value_to_its_candidate():
    h_a, c_a = _hyp_and_cand("candidate_a", fam="complexa_beam")
    h_b, c_b = _hyp_and_cand("candidate_b", fam="complexa_best_of_n")
    _, ev0 = case_stalled("test-model")
    ev = replace(ev0, route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam_default:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="complexa_beam_default", config_signature="default",
            route_gpu_h=5.0, new_su=5, new_su_recent_gpu=1,
            gpu_recent_route_gpu_h=1.0,
            gpu_recent_new_su_per_route_gpu_h=1.0,
            strict_quality_n_unique_bins=4, strict_quality_p25=0.4,
            strict_quality_median=0.5, status="healthy",
            marginal_status="productive",
        ),
        RouteValueSummary(
            strategy_key="route::complexa_best_of_n:complexa_best_of_n_default:default",
            scope="route", family="complexa_best_of_n",
            root_family="complexa_best_of_n", action_family="complexa_best_of_n",
            scoring_family=None, operator_id="complexa_best_of_n_default",
            config_signature="default", route_gpu_h=5.0, new_su=5,
            new_su_recent_gpu=1, gpu_recent_route_gpu_h=1.08,
            gpu_recent_new_su_per_route_gpu_h=0.92,
            strict_quality_n_unique_bins=4, strict_quality_p25=0.9,
            strict_quality_median=0.9, status="healthy",
            marginal_status="productive",
        ),
    ])

    prompt = build_user_prompt(ev, [h_a, h_b], [c_a, c_b])
    payload = json.loads(prompt[prompt.index("{"):])
    by_id = {row["candidate_id"]: row for row in payload["candidates"]}

    assert by_id["candidate_a"]["matched_exact_route_value"]["value_rate_for_ranking"] == 1.0
    assert by_id["candidate_a"]["matched_exact_route_value"]["strict_quality_p25"] == 0.4
    assert by_id["candidate_b"]["matched_exact_route_value"]["value_rate_for_ranking"] == 0.92
    assert by_id["candidate_b"]["matched_exact_route_value"]["strict_quality_p25"] == 0.9


def test_supervisor_accepts_joined_route_value_citations():
    h, c = _hyp_and_cand("candidate_a", fam="complexa_beam")
    _, ev = case_stalled("test-model")
    allowed = _allowed_evidence_refs(ev, [h], [c])

    assert _evidence_ref_allowed("matched_exact_route_value", allowed)
    assert _evidence_ref_allowed(
        "matched_exact_route_value.strict_quality_p25", allowed
    )
    obj = {
        "abstain": False,
        "confidence": 0.7,
        "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [{
            **_dec("candidate_a", "exploit", 1),
            "evidence_refs": ["matched_exact_route_value"],
        }],
    }
    ok, why = _validate_schema(
        obj,
        known_candidate_ids={"candidate_a"},
        allowed_evidence_refs=allowed,
    )
    assert ok, why


def test_supervisor_system_prompt_forbids_target_family_prior():
    assert "Do not use target identity" in SUPERVISOR_SYSTEM
    assert "No family is the expected winner" in SUPERVISOR_SYSTEM


def test_supervisor_system_prompt_requests_reasoning_trace():
    assert "reasoning_trace" in SUPERVISOR_SYSTEM
    assert "observed_signal" in SUPERVISOR_SYSTEM


def test_validator_repairs_missing_reasoning_trace_for_auditability():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [_dec("c1", "exploit", 1)],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1"})
    assert ok, why

    trace = obj["candidate_decisions"][0]["reasoning_trace"]
    assert trace["observed_signal"] == "cites e1"
    assert trace["inference"] == "y"
    assert trace["action_implication"] == "x; expected signal: z"

    dec = _materialize_decisions(obj)[0]
    assert dec.reasoning_trace.observed_signal == "cites e1"
    assert dec.reasoning_trace.inference == "y"
    assert dec.reasoning_trace.action_implication == "x; expected signal: z"


def test_validator_repairs_blank_reasoning_trace_fields():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [{
            **_dec("c1", "exploit", 1),
            "reasoning_trace": {
                "observed_signal": "",
                "inference": "   ",
                "action_implication": "",
            },
        }],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1"})
    assert ok, why
    trace = obj["candidate_decisions"][0]["reasoning_trace"]
    assert all(trace[k] for k in ("observed_signal", "inference", "action_implication"))


def test_validator_reports_empty_evidence_refs_before_trace_repair():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [{**_dec("c1", "exploit", 1), "evidence_refs": []}],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1"})
    assert not ok
    assert "evidence_refs_empty" in why


def test_validator_rejects_unnormalized_mode_mixture():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 2.0, "rescue": 1.0, "explore": 1.0},
        "candidate_decisions": [_dec("c1", "exploit", 1)],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1"})
    assert not ok
    assert "mode_mixture_not_normalized" in why


def test_validator_downgrades_unsupported_extended_resource_label():
    h, c = _hyp_and_cand("c1", fam="complexa_beam")
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [{**_dec("c1", "exploit", 1), "resource_class": "extended"}],
    }
    ok, why = _validate_schema(
        obj, known_candidate_ids={"c1"},
        candidate_by_id={"c1": c}, hypothesis_by_id={h.hypothesis_id: h},
    )
    assert ok, why
    assert obj["candidate_decisions"][0]["resource_class"] == c.estimated_cost_class



def test_validator_downgrades_extended_self_attested_near_pass_text():
    h, c = _hyp_and_cand("c1", fam="complexa_beam")
    dec = {
        **_dec("c1", "exploit", 1),
        "resource_class": "extended",
        "why": "Repeated near-miss evidence supports an extended run.",
        "expected_signal": "near_miss count should improve",
    }
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [dec],
    }
    ok, why = _validate_schema(
        obj, known_candidate_ids={"c1"},
        candidate_by_id={"c1": c}, hypothesis_by_id={h.hypothesis_id: h},
    )
    assert ok, why
    assert obj["candidate_decisions"][0]["resource_class"] == c.estimated_cost_class


def test_validator_downgrades_extended_from_near_pass_ref_strings_only():
    h, c0 = _hyp_and_cand("c1", fam="complexa_beam")
    c = replace(c0, evidence_refs=["near_miss_count:3", "repeated_near_pass:route_a"])
    dec = {
        **_dec("c1", "exploit", 1),
        "resource_class": "extended",
        "evidence_refs": ["near_miss_count:3"],
    }
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [dec],
    }
    ok, why = _validate_schema(
        obj, known_candidate_ids={"c1"},
        candidate_by_id={"c1": c}, hypothesis_by_id={h.hypothesis_id: h},
    )
    assert ok, why
    assert obj["candidate_decisions"][0]["resource_class"] == c.estimated_cost_class


def test_validator_allows_extended_for_supported_hypothesis():
    h0, c = _hyp_and_cand("c1", fam="complexa_beam")
    h = replace(h0, status="supported", support_points=1.0)
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [{**_dec("c1", "exploit", 1), "resource_class": "extended"}],
    }
    ok, why = _validate_schema(
        obj, known_candidate_ids={"c1"},
        candidate_by_id={"c1": c}, hypothesis_by_id={h.hypothesis_id: h},
    )
    assert ok, why


def test_validator_rejects_resource_class_that_understates_candidate_cost():
    h, c0 = _hyp_and_cand("c1", fam="complexa_mcts")
    c = replace(c0, estimated_cost_class="extended")
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [{**_dec("c1", "exploit", 1), "resource_class": "standard"}],
    }
    ok, why = _validate_schema(
        obj, known_candidate_ids={"c1"},
        candidate_by_id={"c1": c}, hypothesis_by_id={h.hypothesis_id: h},
    )
    assert not ok
    assert "resource_class_understates_candidate_cost" in why


def test_validator_rejects_duplicate_across_modes():
    obj = {
        "abstain": False, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            _dec("c1", "exploit", 1),
            _dec("c2", "rescue", 1),
            _dec("c1", "explore", 1),  # c1 duplicated
        ],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1", "c2"})
    assert not ok
    assert "duplicate_candidate_across_modes" in why


def test_materialize_decisions_keeps_structured_reasoning_trace():
    obj = {
        "candidate_decisions": [
            {
                **_dec("c1", "rescue", 1),
                "reasoning_trace": {
                    "observed_signal": "c1 cites an iPAE-only near-miss.",
                    "inference": "This is a rescue action.",
                    "action_implication": "Rank in rescue with standard cost.",
                },
            }
        ]
    }
    dec = _materialize_decisions(obj)[0]
    assert dec.reasoning_trace.observed_signal.startswith("c1 cites")
    assert dec.reasoning_trace.inference == "This is a rescue action."


def _fake_response(text: str, usage_in: int = 10, usage_out: int = 20):
    r = MagicMock()
    r.text = text
    r.usage = {"prompt_tokens": usage_in, "completion_tokens": usage_out}
    return r


def _hyp_and_cand(cid: str, fam: str = "complexa_beam"):
    h = HypothesisCard(
        hypothesis_id=f"h_{cid}", target_id="t", tick_created=0,
        claim="claim", mode_affinity={"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["e1"], 0.2, None)],
        preserve_constraints=[], recommended_action_families=[fam],
    )
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    a = ActionCandidate(
        candidate_id=cid, hypothesis_ids=[h.hypothesis_id], parent_result_id=None,
        method_family=fam, operator_id=f"{fam}_default", lane_id=fam,
        config_delta={}, downstream_route_plan=[], estimated_cost_class="standard",
        expected_signal="x", evidence_refs=["e1"], feasibility=feas,
    )
    return h, a


def test_call_supervisor_retries_once_on_duplicate_mode_and_recovers():
    """First call returns duplicate-mode JSON; second call is clean.
    call_supervisor must return valid=True after the retry."""
    bad_json = """{
        "abstain": false, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            {"candidate_id": "c1", "mode": "exploit", "rank_in_mode": 1, "global_rank": 1,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"},
            {"candidate_id": "c2", "mode": "rescue", "rank_in_mode": 1, "global_rank": 2,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"},
            {"candidate_id": "c1", "mode": "explore", "rank_in_mode": 1, "global_rank": 3,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"}
        ]
    }"""
    good_json = """{
        "abstain": false, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            {"candidate_id": "c1", "mode": "exploit", "rank_in_mode": 1, "global_rank": 1,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"},
            {"candidate_id": "c2", "mode": "rescue", "rank_in_mode": 1, "global_rank": 2,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"}
        ]
    }"""

    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        _fake_response(bad_json),
        _fake_response(good_json),
    ]

    h1, c1 = _hyp_and_cand("c1")
    h2, c2 = _hyp_and_cand("c2", fam="proteinmpnn_redesign")
    _, ev = case_stalled("test-model")

    with patch("trex.supervisor.create_client", return_value=fake_client):
        out = call_supervisor(ev, [h1, h2], [c1, c2], cfg=SupervisorCallConfig(confidence_threshold=0.0))

    assert out.valid, f"expected retry to recover, got fail_reason={out.fail_reason}"
    assert fake_client.chat.call_count == 2
    # Ensure repair hint was sent in second call (3-message conversation)
    second_call_messages = fake_client.chat.call_args_list[1].args[0]
    assert any("each candidate_id appears in exactly ONE mode" in m["content"]
               for m in second_call_messages if isinstance(m, dict)), \
        "expected repair hint mentioning the one-mode constraint"


def test_call_supervisor_retries_once_on_unknown_candidate_id_and_recovers():
    """Unknown candidate IDs are recoverable because the allowed ID set is known."""
    bad_json_unknown_id = """{
        "abstain": false, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            {"candidate_id": "c_ghost", "mode": "exploit", "rank_in_mode": 1, "global_rank": 1,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"}
        ]
    }"""
    good_json = """{
        "abstain": false, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            {"candidate_id": "c1", "mode": "exploit", "rank_in_mode": 1, "global_rank": 1,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"}
        ]
    }"""
    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        _fake_response(bad_json_unknown_id),
        _fake_response(good_json),
    ]

    h1, c1 = _hyp_and_cand("c1")
    _, ev = case_stalled("test-model")

    with patch("trex.supervisor.create_client", return_value=fake_client):
        out = call_supervisor(ev, [h1], [c1], cfg=SupervisorCallConfig(confidence_threshold=0.0))

    assert out.valid
    assert [d.candidate_id for d in out.candidate_decisions] == ["c1"]
    assert fake_client.chat.call_count == 2
    second_call_messages = fake_client.chat.call_args_list[1].args[0]
    assert any("Use only these candidate_id values: c1" in m["content"]
               for m in second_call_messages if isinstance(m, dict))


def test_call_supervisor_prunes_unknown_candidate_id_if_retry_fails():
    """If retry still fails, keep known decisions and drop only unknown IDs."""
    mixed_json = """{
        "abstain": false, "confidence": 0.7, "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            {"candidate_id": "c_ghost", "mode": "exploit", "rank_in_mode": 1, "global_rank": 1,
             "resource_class": "standard", "what": "bad", "why": "bad",
             "evidence_refs": ["e1"], "expected_signal": "bad",
             "stop_or_downgrade_if": "bad"},
            {"candidate_id": "c1", "mode": "rescue", "rank_in_mode": 1, "global_rank": 2,
             "resource_class": "standard", "what": "x", "why": "y",
             "evidence_refs": ["e1"], "expected_signal": "z",
             "stop_or_downgrade_if": "w"}
        ]
    }"""
    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        _fake_response(mixed_json),
        _fake_response(mixed_json),
    ]

    h1, c1 = _hyp_and_cand("c1")
    _, ev = case_stalled("test-model")

    with patch("trex.supervisor.create_client", return_value=fake_client):
        out = call_supervisor(ev, [h1], [c1], cfg=SupervisorCallConfig(confidence_threshold=0.0))

    assert out.valid, out.fail_reason
    assert out.fail_reason and "schema_repaired" in out.fail_reason
    assert [(d.candidate_id, d.mode, d.rank_in_mode, d.global_rank)
            for d in out.candidate_decisions] == [("c1", "rescue", 1, 1)]
    assert fake_client.chat.call_count == 2
