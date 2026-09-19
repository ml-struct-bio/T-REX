"""Selector unit tests (deterministic, no LLM)."""

from __future__ import annotations

import dataclasses

import pytest

from trex.schemas import (
    ActionCandidate,
    CandidateDecision,
    FeasibilityCheck,
    SupervisorOutput,
)
from trex.refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from trex.selector import SelectorConfig, select_launches


from trex.selector import _fractional_carry_quotas, _realize_quotas

_MIX = {"exploit": 0.5, "rescue": 0.3, "explore": 0.2}
_FEAS = {"exploit", "rescue", "explore"}


def test_quota_realization_all_methods_sum_to_n_slots():
    # The distribution->actions realization is swappable; every method must still
    # allocate exactly n_slots (the engineering invariant the novelty relies on).
    recent = ["exploit"] * 7 + ["rescue"] * 2 + ["explore"]
    for method in ("deterministic_deficit", "largest_remainder", "stochastic"):
        for n in (1, 2, 3, 5):
            q = _realize_quotas(method, _MIX, n, recent, 10, _FEAS)
            assert sum(q.values()) == n, (method, n, q)
            assert all(v >= 0 for v in q.values())


def test_default_realization_is_fractional_carry():
    assert SelectorConfig().quota_realization == "fractional_carry"


def test_fractional_carry_realizes_small_share_without_fixed_window():
    credit = {"exploit": 0.0, "rescue": 0.0, "explore": 0.0}
    counts = {"exploit": 0, "rescue": 0, "explore": 0}
    mix = {"exploit": 0.65, "rescue": 0.25, "explore": 0.10}
    for _ in range(20):
        q, credit = _fractional_carry_quotas(mix, 1, _FEAS, credit)
        for m, n in q.items():
            counts[m] += n
    assert counts["explore"] > 0
    assert counts["rescue"] > 0
    assert counts["exploit"] > counts["rescue"] >= counts["explore"]


def test_fractional_carry_respects_feasibility_and_preserves_credit():
    credit = {"exploit": 0.0, "rescue": 0.8, "explore": 0.0}
    q, after = _fractional_carry_quotas(
        {"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        1,
        {"exploit", "explore"},
        credit,
    )
    assert q["rescue"] == 0
    assert sum(q.values()) == 1
    assert after["rescue"] > credit["rescue"]


def test_deterministic_deficit_realization_remains_available():
    from trex.selector import _windowed_quotas
    recent = ["exploit"] * 8 + ["rescue", "explore"]
    assert _realize_quotas("deterministic_deficit", _MIX, 3, recent, 10, _FEAS) == \
        _windowed_quotas(_MIX, 3, recent, 10, _FEAS)


def test_stochastic_is_seeded_reproducible():
    recent = ["exploit"] * 5
    a = _realize_quotas("stochastic", _MIX, 4, recent, 10, _FEAS)
    b = _realize_quotas("stochastic", _MIX, 4, recent, 10, _FEAS)
    assert a == b  # same state -> same result (reproducible despite sampling)


def test_realization_respects_feasibility():
    # explore infeasible -> no method may allocate an explore slot.
    feas = {"exploit", "rescue"}
    for method in ("deterministic_deficit", "largest_remainder", "stochastic"):
        q = _realize_quotas(method, _MIX, 4, ["exploit"], 10, feas)
        assert q.get("explore", 0) == 0, (method, q)


def test_untrusted_foldseek_su_does_not_drive_route_value_selection():
    from trex.schemas import RouteValueSummary
    from trex.selector import (
        _candidate_route_penalty,
        _candidate_route_rate,
        _has_positive_recent_su,
        family_cost_penalties_for_cfg,
        high_cost_cap_for_evidence,
    )

    ev = dataclasses.replace(
        _evidence("stalled"),
        completed_children=64,
        worker_gpu_h_total=12.0,
        gpu_h_since_last_su=12.0,
        foldseek_su_status="failed",
        foldseek_su_coverage=0.0,
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:op:default",
                scope="route",
                family="bindcraft",
                root_family="bindcraft",
                action_family="bindcraft",
                scoring_family=None,
                operator_id="op",
                config_signature="default",
                route_gpu_h=1.0,
                generator_gpu_h=1.0,
                attempts=1,
                completions=1,
                strict_count=10,
                new_su=10,
                new_su_per_route_gpu_h=10.0,
                new_su_recent_gpu=10,
                gpu_recent_route_gpu_h=1.0,
                gpu_recent_new_su_per_route_gpu_h=10.0,
                status="promote",
                marginal_status="productive",
                evidence_refs=["e1"],
            )
        ],
    )
    cand = _cand("bc", "bindcraft")

    assert _candidate_route_rate(cand, ev) == 0.0
    assert _candidate_route_penalty(cand, ev) == 0
    assert not _has_positive_recent_su(dataclasses.replace(ev, run_su_count_delta=10, su_per_gpu_h_recent=10.0))
    assert family_cost_penalties_for_cfg(ev, SelectorConfig()) == {}
    cap, source = high_cost_cap_for_evidence(
        ev,
        "bindcraft",
        high_cost_pending_cap=1,
        high_cost_pending_strong_cap=3,
    )
    assert (cap, source) != (3, "route_value_best")


def test_realization_feasibility_with_mass_on_excluded_mode():
    # The hard case: the mixture is concentrated on a mode that is INFEASIBLE.
    # Every method must still allocate 0 to it AND keep sum==n (the swappable-
    # realization contract). largest_remainder used to violate both here.
    feas = {"rescue", "explore"}
    mix = {"exploit": 1.0, "rescue": 0.0, "explore": 0.0}
    for method in ("deterministic_deficit", "largest_remainder", "stochastic"):
        q = _realize_quotas(method, mix, 3, ["rescue"], 10, feas)
        assert q.get("exploit", 0) == 0, (method, q)
        assert sum(q.values()) == 3, (method, q)


def _feas(ok: bool = True) -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=ok,
        runtime_bucket_id="rb" if ok else None,
        compiler_ok=ok,
        verifier_ok=ok,
        route_cap_ok=ok,
        cost_ok=ok,
    )


def _cand(cid: str, family: str = "complexa_beam", cost: str = "standard", feas_ok: bool = True, refilter_role: str | None = None) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=cid,
        hypothesis_ids=["h1"],
        parent_result_id=None,
        method_family=family,
        operator_id="op",
        lane_id=family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class=cost,  # type: ignore[arg-type]
        expected_signal="x",
        evidence_refs=["e1"],
        feasibility=_feas(feas_ok),
        refilter_role=refilter_role,  # type: ignore[arg-type]
    )


def _sup_decision(
    cid: str,
    mode: str,
    rank: int,
    rc: str = "standard",
    *,
    global_rank: int | None = None,
) -> CandidateDecision:
    return CandidateDecision(
        candidate_id=cid,
        mode=mode,  # type: ignore[arg-type]
        rank_in_mode=rank,
        resource_class=rc,  # type: ignore[arg-type]
        what="...",
        why="...",
        evidence_refs=["e1"],
        expected_signal="...",
        stop_or_downgrade_if="...",
        global_rank=global_rank,
    )


def _sup(mixture: dict[str, float], decisions: list[CandidateDecision]) -> SupervisorOutput:
    return SupervisorOutput(
        valid=True,
        abstain=False,
        confidence=0.7,
        fail_reason=None,
        mode_mixture=mixture,
        candidate_decisions=decisions,
        rationale="...",
        raw_text="",
        usage={},
    )


def _evidence(state: str = "productive"):
    """Minimal evidence stub with attributes the selector reads."""
    from trex.schemas import (
        EvidenceSummary,
        LLMHealthSummary,
        RouteHealthSummary,
    )

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
        state_label=state, examples=[], metric_availability={},  # type: ignore[arg-type]
    )


# ---- tests ---------------------------------------------------------------


def test_fresh_global_priority_controls_immediate_cross_mode_launch():
    candidates = [
        _cand("fast_exploit", "complexa_beam"),
        _cand("near_miss_rescue", "proteinmpnn_redesign"),
        _cand("counterfactual", "complexa_mcts", cost="extended"),
    ]
    decisions = [
        _sup_decision("fast_exploit", "exploit", 1, global_rank=2),
        _sup_decision(
            "near_miss_rescue", "rescue", 1, "low", global_rank=1,
        ),
        _sup_decision(
            "counterfactual", "explore", 1, "extended", global_rank=3,
        ),
    ]
    launches, debug = select_launches(
        _evidence("rescue_rich"),
        candidates,
        _sup({"exploit": 0.70, "rescue": 0.20, "explore": 0.10}, decisions),
        cfg=SelectorConfig(available_slots=1),
        mode_credit={"exploit": 3.0, "rescue": -2.0, "explore": 0.0},
    )

    started = [x for x in launches if x.status == "launched"]
    assert [x.candidate_id for x in started] == ["near_miss_rescue"]
    assert debug["global_priority_used"]
    assert debug["quota_realization"] == "global_priority"
    assert debug["launch_modes"] == {"near_miss_rescue": "rescue"}


def test_global_priority_fills_multiple_slots_in_global_order():
    candidates = [
        _cand("bc", "bindcraft", cost="diagnostic"),
        _cand("beam", "complexa_beam"),
        _cand("redesign", "proteinmpnn_redesign", cost="low"),
    ]
    decisions = [
        _sup_decision("bc", "exploit", 1, "diagnostic", global_rank=3),
        _sup_decision("beam", "exploit", 2, global_rank=1),
        _sup_decision("redesign", "rescue", 1, "low", global_rank=2),
    ]
    launches, debug = select_launches(
        _evidence("productive"),
        candidates,
        _sup({"exploit": 0.60, "rescue": 0.30, "explore": 0.10}, decisions),
        cfg=SelectorConfig(available_slots=2),
    )

    assert [
        x.candidate_id for x in launches if x.status == "launched"
    ] == ["beam", "redesign"]
    assert debug["global_priority_order"] == ["beam", "redesign", "bc"]


def test_legacy_decisions_without_global_rank_keep_fractional_fallback():
    decisions = [
        _sup_decision("exploit", "exploit", 1),
        _sup_decision("rescue", "rescue", 1),
    ]
    launches, debug = select_launches(
        _evidence("productive"),
        [_cand("exploit"), _cand("rescue", "proteinmpnn_redesign", cost="low")],
        _sup({"exploit": 0.8, "rescue": 0.2, "explore": 0.0}, decisions),
        cfg=SelectorConfig(available_slots=1),
    )
    assert not debug["global_priority_used"]
    assert debug["quota_realization"] == "fractional_carry"
    assert len([x for x in launches if x.status == "launched"]) == 1


def test_production_defaults_keep_only_bounded_quota_safeguards():
    cfg = SelectorConfig()

    assert cfg.productive_wall_momentum_guard
    assert cfg.realize_cross_family_escape_floor
    assert not cfg.realize_repeated_support_probe
    assert not cfg.low_cost_near_miss_rescue_floor
    assert not cfg.marginal_diversity_override


def test_cross_family_escape_floor_launches_even_if_supervisor_omits_it():
    # Regression for BetV1/SC2RBD-style dry runs: Builder creates evidence_fallback
    # cross-family candidates, but the Supervisor may only rank stale same-root
    # replay candidates. Selector must not let that erase the escape floor.
    e = dataclasses.replace(
        _evidence("deep_stall"),
        gpu_h_since_last_su=14.0,
        worker_gpu_h_total=50.0,
        run_su_count=22,
        run_su_count_delta=0,
    )
    cands = [
        _cand("complexa_1", "complexa_fk_steering"),
        _cand("complexa_2", "complexa_fk_steering"),
        _cand("complexa_3", "complexa_fk_steering"),
        _cand("evidence_fallback_cross_family_t1_00_bindcraft", "bindcraft", cost="diagnostic"),
        _cand("evidence_fallback_cross_family_t1_01_boltzgen", "boltzgen", cost="diagnostic"),
    ]
    sup = _sup(
        mixture={"exploit": 0.9, "rescue": 0.05, "explore": 0.05},
        decisions=[
            _sup_decision("complexa_1", "exploit", 1),
            _sup_decision("complexa_2", "exploit", 2),
            _sup_decision("complexa_3", "exploit", 3),
        ],
    )
    launches, dbg = select_launches(
        e, cands, sup, cfg=SelectorConfig(available_slots=3)
    )
    launched = {l.candidate_id for l in launches if l.status == "launched"}
    assert "evidence_fallback_cross_family_t1_00_bindcraft" in launched
    assert sum(cid.startswith("evidence_fallback_cross_family") for cid in launched) == 1
    assert len(launched) == 3
    assert dbg["forced_cross_family_escape_floor"]["replaced"]


def test_cross_family_escape_floor_does_not_replace_rank1_single_slot():
    # Live CD45 regression: a one-slot tick had an explicit rank-1 explore
    # candidate, but the deterministic floor replaced it with a repeated
    # BoltzGen fallback. A safeguard may replace omitted/lower-ranked work, not
    # become the policy by erasing the LLM's strongest current decision.
    e = dataclasses.replace(
        _evidence("deep_stall"),
        gpu_h_since_last_su=14.0,
        worker_gpu_h_total=20.0,
    )
    ranked = _cand("ranked_rescue", "complexa_best_of_n")
    fallback = _cand(
        "evidence_fallback_cross_family_t1_00_boltzgen",
        "boltzgen",
        cost="diagnostic",
    )
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        decisions=[_sup_decision("ranked_rescue", "rescue", 1)],
    )

    launches, dbg = select_launches(
        e,
        [ranked, fallback],
        sup,
        cfg=SelectorConfig(available_slots=1),
    )
    launched = [l.candidate_id for l in launches if l.status == "launched"]

    assert launched == ["ranked_rescue"]
    assert dbg["forced_cross_family_escape_floor"] == {}
    assert (
        "cross_family_escape_floor_skipped:rank1_protected"
        in dbg["redistribute_log"]
    )

def test_cross_family_escape_floor_skips_backlog_saturated_deferred_generator():
    # TNF-alpha regression: a dry route-deferred diagnostic generator with a
    # large unscored backlog should not be revived by the cross-family escape
    # floor as another fresh generation. The Supervisor's concrete rescue card
    # should keep the slot.
    from trex.schemas import RouteValueSummary

    e = dataclasses.replace(
        _evidence("deep_stall"),
        gpu_h_since_last_su=25.0,
        worker_gpu_h_total=25.0,
        diagnostic_chain_backlog={
            "by_family": {
                "boltzgen": {
                    "accepted_artifacts": 176,
                    "completed_refilters": 13,
                    "unscored_artifacts": 163,
                    "pending_chain_candidates": 163,
                    "proxy_promising_pending_refilter": 0,
                    "native_or_proxy_pending_refilter": 0,
                }
            }
        },
        method_health={
            "boltzgen": {
                "strict_yield_su": 0,
                "chained_strict_yield_su": 0,
                "near_miss_yield": 0,
                "near_miss_yield_recent": 0,
            }
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::boltzgen:op:default",
                scope="route",
                family="boltzgen",
                root_family="boltzgen",
                action_family="boltzgen",
                scoring_family=None,
                operator_id="op",
                config_signature="default",
                route_gpu_h=2.0,
                attempts=10,
                completions=10,
                new_su=0,
                near_miss_count=0,
                status="defer",
                marginal_status="dry_low_quality",
            )
        ],
    )
    rescue = _cand("rescue_fk", "complexa_fk_steering")
    bestn = _cand("bestn", "complexa_best_of_n")
    boltz = _cand("evidence_fallback_cross_family_t1_00_boltzgen", "boltzgen", cost="diagnostic")
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        decisions=[
            _sup_decision("rescue_fk", "rescue", 1),
            _sup_decision("bestn", "explore", 1),
            _sup_decision("evidence_fallback_cross_family_t1_00_boltzgen", "explore", 2, rc="diagnostic"),
        ],
    )

    launches, dbg = select_launches(e, [rescue, bestn, boltz], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]

    assert launched == ["rescue_fk"]
    assert dbg["route_deferred_candidate_ids"] == ["evidence_fallback_cross_family_t1_00_boltzgen"]
    assert dbg["backlog_saturated_route_deferred_candidate_ids"] == [
        "evidence_fallback_cross_family_t1_00_boltzgen"
    ]
    assert dbg["forced_cross_family_escape_floor"] == {}


def test_promising_backlog_does_not_make_fallback_saturated_or_override_rank1():
    # Positive control: if the diagnostic backlog still has a promising pending
    # artifact, the escape floor remains available. This preserves the existing
    # cross-family policy and only removes stale/no-signal backlog floods.
    from trex.schemas import RouteValueSummary

    e = dataclasses.replace(
        _evidence("deep_stall"),
        gpu_h_since_last_su=25.0,
        worker_gpu_h_total=25.0,
        diagnostic_chain_backlog={
            "by_family": {
                "boltzgen": {
                    "accepted_artifacts": 176,
                    "completed_refilters": 13,
                    "unscored_artifacts": 163,
                    "pending_chain_candidates": 163,
                    "proxy_promising_pending_refilter": 1,
                    "native_or_proxy_pending_refilter": 0,
                }
            }
        },
        method_health={"boltzgen": {"strict_yield_su": 0, "chained_strict_yield_su": 0}},
        route_values=[
            RouteValueSummary(
                strategy_key="route::boltzgen:op:default",
                scope="route",
                family="boltzgen",
                root_family="boltzgen",
                action_family="boltzgen",
                scoring_family=None,
                operator_id="op",
                config_signature="default",
                route_gpu_h=2.0,
                attempts=10,
                completions=10,
                new_su=0,
                near_miss_count=0,
                status="defer",
                marginal_status="dry_low_quality",
            )
        ],
    )
    rescue = _cand("rescue_fk", "complexa_fk_steering")
    bestn = _cand("bestn", "complexa_best_of_n")
    boltz = _cand("evidence_fallback_cross_family_t1_00_boltzgen", "boltzgen", cost="diagnostic")
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        decisions=[
            _sup_decision("rescue_fk", "rescue", 1),
            _sup_decision("bestn", "explore", 1),
            _sup_decision("evidence_fallback_cross_family_t1_00_boltzgen", "explore", 2, rc="diagnostic"),
        ],
    )

    launches, dbg = select_launches(e, [rescue, bestn, boltz], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]

    assert launched == ["rescue_fk"]
    assert dbg["route_deferred_candidate_ids"] == ["evidence_fallback_cross_family_t1_00_boltzgen"]
    assert dbg["backlog_saturated_route_deferred_candidate_ids"] == []
    assert dbg["forced_cross_family_escape_floor"] == {}
    assert (
        "cross_family_escape_floor_skipped:rank1_protected"
        in dbg["redistribute_log"]
    )


def test_hybrid_budget_follows_ranking_not_contradictory_scalar():
    # The scalar mode_mixture is exploit-heavy, but the RANKED decisions favor
    # explore. With the hybrid flag ON (opt-in) the budget derives from the
    # ranking → an explore candidate launches; with the flag OFF (the v7_3
    # DEFAULT, scalar-driven) explore is starved. This pins both flag behaviors.
    e = _evidence("productive")
    cands = [
        _cand("c1", "complexa_beam"), _cand("c2", "complexa_beam"),
        _cand("c3", "bindcraft", cost="diagnostic"),
        _cand("c4", "bindcraft", cost="diagnostic"),
    ]
    sup = _sup(
        mixture={"exploit": 0.9, "rescue": 0.05, "explore": 0.05},   # scalar: exploit-heavy
        decisions=[
            _sup_decision("c3", "explore", 1, rc="diagnostic"),       # ranking: explore-heavy
            _sup_decision("c4", "explore", 2, rc="diagnostic"),
            _sup_decision("c1", "exploit", 1),
            _sup_decision("c2", "exploit", 2),
        ],
    )
    on, dbg_on = select_launches(
        e, cands, sup,
        cfg=SelectorConfig(available_slots=2, derive_mixture_from_ranking=True),
    )
    sel_on = {l.candidate_id for l in on if l.status == "launched"}
    assert sel_on & {"c3", "c4"}, sel_on                # an explore candidate launched
    assert dbg_on["source"] == "supervisor_ranking_derived"

    off, dbg_off = select_launches(
        e, cands, sup,
        cfg=SelectorConfig(available_slots=2, derive_mixture_from_ranking=False),
    )
    sel_off = {l.candidate_id for l in off if l.status == "launched"}
    assert not (sel_off & {"c3", "c4"}), sel_off        # explore starved under the scalar
    assert dbg_off["source"] == "supervisor_llm_scalar"


def test_high_cost_capacity_pressure_prefers_launchable_alternative():
    e = dataclasses.replace(
        _evidence("stalled"),
        pending_family_load={
            "by_family": {
                "bindcraft": {
                    "running": 1,
                    "queued": 0,
                    "pending_total": 1,
                    "inflight_gpu_h": 0.8,
                }
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
    )
    cands = [
        _cand("bc", "bindcraft", cost="extended"),
        _cand("cx", "complexa_beam", cost="standard"),
    ]
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        decisions=[
            _sup_decision("bc", "explore", 1, rc="extended"),
            _sup_decision("cx", "explore", 2, rc="standard"),
        ],
    )
    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1))
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert selected == ["cx"]
    assert dbg["capacity_pressure_families"] == ["bindcraft"]
    assert dbg["capacity_deferred_candidate_ids"] == ["bc"]
    assert dbg["rank1_not_launched"][0]["family"] == "bindcraft"
    assert dbg["rank1_not_launched"][0]["reason"].startswith(
        "capacity_pressure_high_cost_running_cap:family=bindcraft"
    )



def test_high_cost_queued_only_does_not_capacity_block_fresh_candidate():
    e = dataclasses.replace(
        _evidence("stalled"),
        pending_family_load={
            "by_family": {
                "bindcraft": {
                    "running": 0,
                    "queued": 1,
                    "pending_total": 1,
                    "inflight_gpu_h": 0.0,
                }
            },
            "high_cost_families": ["bindcraft"],
        },
    )
    cands = [
        _cand("bc_fresh", "bindcraft", cost="extended"),
        _cand("cx", "complexa_beam", cost="standard"),
    ]
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        decisions=[
            _sup_decision("bc_fresh", "explore", 1, rc="extended"),
            _sup_decision("cx", "explore", 2, rc="standard"),
        ],
    )

    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1))

    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert selected == ["bc_fresh"]
    assert dbg["capacity_pressure_families"] == []
    assert dbg["capacity_deferred_candidate_ids"] == []
    assert dbg["rank1_not_launched"] == []

def test_deep_stall_productive_high_cost_family_gets_promoted_cap():
    e = dataclasses.replace(
        _evidence("deep_stall"),
        method_health={
            "bindcraft": {
                "chained_su_per_gpu_h_recent": 0.20,
                "chained_strict_yield_su": 2,
                "near_miss_yield_recent": 0,
            }
        },
        pending_family_load={
            "by_family": {
                "bindcraft": {
                    "running": 1,
                    "queued": 0,
                    "pending_total": 1,
                    "inflight_gpu_h": 1.5,
                }
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
    )
    cands = [_cand("bc", "bindcraft", cost="extended")]
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        decisions=[_sup_decision("bc", "explore", 1, rc="extended")],
    )

    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1))

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert "bindcraft" not in dbg["capacity_blocked_families"]


def test_stale_high_cost_telemetry_at_cap_is_ignored_by_default():
    e = dataclasses.replace(
        _evidence("stalled"),
        pending_family_load={
            "by_family": {
                "bindcraft": {
                    "running": 1,
                    "queued": 0,
                    "pending_total": 1,
                    "inflight_gpu_h": 0.8,
                }
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
    )
    cands = [_cand("bc", "bindcraft", cost="extended")]
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        decisions=[_sup_decision("bc", "explore", 1, rc="extended")],
    )

    launches, dbg = select_launches(
        e,
        cands,
        sup,
        cfg=SelectorConfig(available_slots=1, high_cost_pending_families=()),
    )

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert not [l for l in launches if l.status == "rejected" and l.candidate_id == "bc"]
    assert dbg["capacity_pressure_families"] == []
    assert dbg["n_capacity_blocked_candidates"] == 0
    assert dbg["n_capacity_pressure_candidates"] == 0
    assert dbg["rank1_not_launched"] == []


def test_repeated_supervisor_support_opens_one_extra_high_cost_probe():
    e = dataclasses.replace(
        _evidence("stalled"),
        pending_family_load={
            "by_family": {
                "bindcraft": {"running": 1, "queued": 0, "pending_total": 1, "inflight_gpu_h": 0.8}
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
        execution_realization={
            "by_family": {
                "bindcraft": {"proposed": 12, "selected": 4, "started": 1, "dispatch_deferred": 2, "selected_not_started": 1}
            },
            "under_started_families": ["bindcraft"],
        },
    )
    cands = [_cand("bc", "bindcraft", cost="extended")]
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        decisions=[_sup_decision("bc", "explore", 1, rc="extended")],
    )

    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1))

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert "bindcraft" not in dbg["capacity_blocked_families"]


def test_single_su_high_cost_route_value_current_signal_opens_promoted_cap_only():
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    ev = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=1,
        foldseek_su_status="ok",
        foldseek_su_coverage=1.0,
        method_health={
            "bindcraft": {
                "chained_su_per_gpu_h_recent": 1.0,
                "chained_su_per_gpu_h": 1.0,
                "chained_strict_yield_su": 1,
            }
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:op:default",
                scope="route",
                family="bindcraft",
                root_family="bindcraft",
                action_family="bindcraft",
                scoring_family="structure_refilter",
                operator_id="op",
                config_signature="default",
                route_gpu_h=1.0,
                generator_gpu_h=1.0,
                attempts=1,
                completions=1,
                strict_count=1,
                new_su=1,
                new_su_per_route_gpu_h=1.0,
                new_su_recent_gpu=1,
                gpu_recent_route_gpu_h=1.0,
                gpu_recent_new_su_per_route_gpu_h=1.0,
                status="promote",
                marginal_status="productive",
            )
        ],
    )

    cap, source = high_cost_cap_for_evidence(
        ev,
        "bindcraft",
        high_cost_pending_cap=1,
        high_cost_pending_promoted_cap=2,
        high_cost_pending_strong_cap=3,
        high_cost_strong_min_su=2,
        high_cost_strong_min_recent_su_per_gpu_h=0.5,
    )
    assert cap == 2
    assert source in {"promoted", "route_value_promoted", "single_su_high_value"}


def test_repeated_support_does_not_tax_high_su_easy_productive_duplicate():
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=40,
        pending_family_load={
            "by_family": {
                "bindcraft": {"running": 1, "queued": 0, "pending_total": 1, "inflight_gpu_h": 0.8}
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
        execution_realization={
            "by_family": {
                "bindcraft": {"proposed": 12, "selected": 4, "started": 1, "dispatch_deferred": 2, "selected_not_started": 1}
            },
            "under_started_families": ["bindcraft"],
        },
    )
    cands = [_cand("bc", "bindcraft", cost="extended")]
    sup = _sup(
        mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        decisions=[_sup_decision("bc", "explore", 1, rc="extended")],
    )

    launches, dbg = select_launches(
        e,
        cands,
        sup,
        cfg=SelectorConfig(available_slots=1, high_cost_pending_families=()),
    )

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert dbg["capacity_pressure_families"] == []


def test_repeated_support_rank_one_gets_realized_even_if_scalar_mixture_underallocates_mode():
    e = dataclasses.replace(
        _evidence("stalled"),
        execution_realization={
            "by_family": {
                "bindcraft": {"proposed": 18, "selected": 3, "started": 1, "dispatch_deferred": 0, "selected_not_started": 2}
            },
            "under_started_families": ["bindcraft"],
        },
    )
    bc = _cand("bc", "bindcraft", cost="extended")
    cx = _cand("cx", "complexa_mcts", cost="standard")
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[
            _sup_decision("bc", "explore", 1, rc="extended"),
            _sup_decision("cx", "exploit", 1, rc="standard"),
        ],
    )

    launches, dbg = select_launches(
        e,
        [bc, cx],
        sup,
        cfg=SelectorConfig(available_slots=1, realize_repeated_support_probe=True),
    )

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert dbg["forced_repeated_support_probe"] == {
        "candidate_id": "bc", "family": "bindcraft", "mode": "explore", "donor_mode": "exploit",
    }


def test_repeated_support_quota_override_does_not_fire_on_high_su_easy_target():
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=40,
        execution_realization={
            "by_family": {
                "bindcraft": {"proposed": 18, "selected": 3, "started": 1, "dispatch_deferred": 0, "selected_not_started": 0}
            },
            "under_started_families": ["bindcraft"],
        },
    )
    bc = _cand("bc", "bindcraft", cost="extended")
    cx = _cand("cx", "complexa_beam", cost="standard")
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[
            _sup_decision("bc", "explore", 1, rc="extended"),
            _sup_decision("cx", "exploit", 1, rc="standard"),
        ],
    )

    launches, dbg = select_launches(
        e,
        [bc, cx],
        sup,
        cfg=SelectorConfig(available_slots=1, realize_repeated_support_probe=True),
    )

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["cx"]
    assert dbg["forced_repeated_support_probe"] == {}


def test_repeated_support_quota_override_fires_for_dry_productive_duplicate_collapse():
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=40,
        strict_duplicate_collapse_signal=True,
        gpu_h_since_last_su=1.25,
        execution_realization={
            "by_family": {
                "bindcraft": {"proposed": 18, "selected": 3, "started": 1, "dispatch_deferred": 0, "selected_not_started": 2}
            },
            "under_started_families": ["bindcraft"],
        },
    )
    bc = _cand("bc", "bindcraft", cost="extended")
    cx = _cand("cx", "complexa_beam", cost="standard")
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[
            _sup_decision("bc", "explore", 1, rc="extended"),
            _sup_decision("cx", "exploit", 1, rc="standard"),
        ],
    )

    launches, dbg = select_launches(
        e,
        [bc, cx],
        sup,
        cfg=SelectorConfig(available_slots=1, realize_repeated_support_probe=True),
    )

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert dbg["forced_repeated_support_probe"]["family"] == "bindcraft"


def test_select_respects_supervisor_ranking():
    e = _evidence("productive")
    cands = [
        _cand("c1", "complexa_beam"),
        _cand("c2", "proteinmpnn_redesign", cost="low"),
        _cand("c3", "bindcraft", cost="diagnostic"),
    ]
    sup = _sup(
        mixture={"exploit": 0.7, "rescue": 0.2, "explore": 0.1},
        decisions=[
            _sup_decision("c1", "exploit", 1),
            _sup_decision("c2", "rescue", 1, rc="low"),
            _sup_decision("c3", "explore", 1, rc="diagnostic"),
        ],
    )
    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=3))
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert "c1" in selected
    # v7_3 hybrid: the budget is now derived from the ranked decisions (the source
    # label reflects that); still a supervisor source, never fallback.
    assert dbg["source"].startswith("supervisor")


def test_select_drops_infeasible():
    e = _evidence("productive")
    cands = [
        _cand("c1", feas_ok=False),
        _cand("c2", "proteinmpnn_redesign", feas_ok=True),
    ]
    sup = _sup(
        mixture={"exploit": 0.5, "rescue": 0.5, "explore": 0.0},
        decisions=[
            _sup_decision("c1", "exploit", 1),
            _sup_decision("c2", "rescue", 1),
        ],
    )
    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=2))
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert "c1" not in selected
    assert "c2" in selected


def test_fallback_uses_default_mixture():
    e = _evidence("stalled")
    cands = [_cand("c1"), _cand("c2", "bindcraft", cost="diagnostic")]
    sup = SupervisorOutput(
        valid=False, abstain=False, confidence=0.0, fail_reason="parse_fail",
        mode_mixture={}, candidate_decisions=[], rationale="", raw_text="", usage={},
    )
    launches, dbg = select_launches(
        e, cands, sup, cfg=SelectorConfig(available_slots=2), fallback_reason="parse_fail"
    )
    assert dbg["source"].startswith("fallback")
    # stalled defaults: exploit 0.20, rescue 0.35, explore 0.45 → clamped to floor 0.25/0.25
    assert dbg["clamped_mixture"]["explore"] >= 0.25


def test_batch_diversity_caps_family_repetition():
    e = _evidence("productive")
    cands = [_cand(f"c{i}", "complexa_beam") for i in range(5)]
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[_sup_decision(f"c{i}", "exploit", i + 1) for i in range(5)],
    )
    launches, _ = select_launches(
        e, cands, sup, cfg=SelectorConfig(available_slots=5, diversity_max_per_family=2)
    )
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    # diversity cap of 2 per family with all-same family means only 2 chosen initially,
    # then overshoot fills remainder
    assert len(selected) >= 2


def test_no_feasible_yields_no_launches():
    e = _evidence("productive")
    cands = [_cand("c1", feas_ok=False), _cand("c2", feas_ok=False)]
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[_sup_decision("c1", "exploit", 1)],
    )
    launches, _ = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=3))
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert selected == []


def test_selected_candidate_is_not_also_logged_rejected():
    e = _evidence("stalled")
    cands = [_cand("c1"), _cand("c2"), _cand("c3")]
    sup = SupervisorOutput(
        valid=False,
        abstain=False,
        confidence=0.0,
        fail_reason="parse_fail",
        mode_mixture={},
        candidate_decisions=[],
        rationale="",
        raw_text="",
        usage={},
    )
    launches, _ = select_launches(
        e,
        cands,
        sup,
        cfg=SelectorConfig(available_slots=2),
        fallback_reason="parse_fail",
    )
    launched = {l.candidate_id for l in launches if l.status == "launched"}
    rejected = {l.candidate_id for l in launches if l.status == "rejected"}
    assert not (launched & rejected)


def test_hinted_candidate_unranked_by_supervisor_is_grouped_and_launched():
    """Regression (2026-05-29): an auto-chain candidate merged into the pool
    AFTER the supervisor ranked the fresh candidates is NOT in sup_decs, so it
    must resolve its mode from candidate_mode_hint EVEN on the supervisor-ranked
    path — otherwise it gets no mode, never enters a quota, and only backfill
    (which never fires under busy=3/3) could launch it → it accumulates
    unlaunched and the proteinMPNN→AF2 rescue chain stays inert."""
    e = _evidence("rescue_rich")
    fresh = _cand("c_fresh", family="complexa_beam")
    chain = _cand("chain_t1_mpnn_to_refilter_001", family="structure_refilter", refilter_role=CANONICAL_SCORE_CONVERSION)
    # Supervisor ranked ONLY the fresh candidate (exploit); the chain candidate
    # was merged afterward and is absent from candidate_decisions.
    sup = _sup({"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
               [_sup_decision("c_fresh", "exploit", 1)])
    launches, dbg = select_launches(
        e, [fresh, chain], sup,
        cfg=SelectorConfig(available_slots=2),
        candidate_mode_hint={"chain_t1_mpnn_to_refilter_001": "rescue"},
    )
    launched = {l.candidate_id: l for l in launches if l.status == "launched"}
    assert "chain_t1_mpnn_to_refilter_001" in launched, (
        "hinted chain candidate must launch, not be stranded"
    )
    # R3 (2026-06-01): the chain candidate still COMPETES in the rescue quota via
    # its mode_hint, but its RECORDED realised mode is "chain_refilter" (a
    # deterministic system lane) so it does not consume the LLM rescue K-window.
    assert launched["chain_t1_mpnn_to_refilter_001"].resource_class_concrete.get("mode") == "chain_refilter"


def test_manual_structure_refilter_records_requested_ere_mode():
    e = _evidence("rescue_rich")
    ref = _cand(
        "cand_manual_refilter", family="structure_refilter", cost="low",
        refilter_role=PARENT_MODEL_REFOLD,
    )
    sup = _sup({"exploit": 0.0, "rescue": 1.0, "explore": 0.0}, [
        _sup_decision("cand_manual_refilter", "rescue", 1, rc="low"),
    ])
    launches, _dbg = select_launches(e, [ref], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l for l in launches if l.status == "launched"]
    assert len(launched) == 1
    assert launched[0].resource_class_concrete.get("mode") == "rescue"
    assert launched[0].resource_class_concrete.get("refilter_role") == PARENT_MODEL_REFOLD


def test_canonical_refilter_records_chain_mode_without_chain_prefix():
    e = _evidence("rescue_rich")
    ref = _cand(
        "cand_system_refilter", family="structure_refilter", cost="low",
        refilter_role=CANONICAL_SCORE_CONVERSION,
    )
    sup = _sup({"exploit": 0.0, "rescue": 1.0, "explore": 0.0}, [
        _sup_decision("cand_system_refilter", "rescue", 1, rc="low"),
    ])
    launches, _dbg = select_launches(e, [ref], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l for l in launches if l.status == "launched"]
    assert len(launched) == 1
    assert launched[0].resource_class_concrete.get("mode") == "chain_refilter"
    assert launched[0].resource_class_concrete.get("refilter_role") == CANONICAL_SCORE_CONVERSION


# ---- GAP 1: cost-aware family admission (soft within-mode demotion) ----------

def _evidence_with_mh(mh: dict):
    from dataclasses import replace
    return replace(_evidence("productive"), method_health=mh)


_WASTEFUL_MH = {
    # bindcraft: 4 GPU-h, 0 SU, 0 near-miss -> genuinely dead lane
    "bindcraft": {"cumulative_gpu_h": 4.0, "strict_yield_su": 0, "near_miss_yield": 0},
    # complexa_beam: productive
    "complexa_beam": {"cumulative_gpu_h": 2.0, "strict_yield_su": 2, "near_miss_yield": 1},
}


def test_cost_aware_tiebreak_demotes_wasteful_family_below_rank():
    """A wasteful family (>=3 GPU-h, 0 SU, 0 near-miss) loses its slot to a
    healthy family even when the LLM ranked it #1 — within the same mode."""
    e = _evidence_with_mh(_WASTEFUL_MH)
    cands = [_cand("c1", "bindcraft", cost="diagnostic"), _cand("c2", "complexa_beam")]
    sup = _sup({"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
               [_sup_decision("c1", "exploit", 1, rc="diagnostic"),   # wasteful, rank 1
                _sup_decision("c2", "exploit", 2)])                    # healthy, rank 2
    launches, _ = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1),
                                  tick_id="t")
    assert {l.candidate_id for l in launches if l.status == "launched"} == {"c2"}


def test_cost_aware_tiebreak_off_respects_rank():
    e = _evidence_with_mh(_WASTEFUL_MH)
    cands = [_cand("c1", "bindcraft", cost="diagnostic"), _cand("c2", "complexa_beam")]
    sup = _sup({"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
               [_sup_decision("c1", "exploit", 1, rc="diagnostic"),
                _sup_decision("c2", "exploit", 2)])
    cfg = SelectorConfig(available_slots=1, cost_aware_family_tiebreak=False)
    launches, _ = select_launches(e, cands, sup, cfg=cfg, tick_id="t")
    assert {l.candidate_id for l in launches if l.status == "launched"} == {"c1"}  # rank-1 wins


def test_cost_aware_tiebreak_protects_near_miss_lane():
    """A 0-SU family that is still producing NEAR-MISSES is NOT demoted (it's
    making progress — the rescue path needs it)."""
    from trex.selector import _wasteful_families
    mh = {"complexa_fk_steering": {"cumulative_gpu_h": 5.0, "strict_yield_su": 0,
                                   "near_miss_yield": 4}}
    assert _wasteful_families(_evidence_with_mh(mh), waste_gpu_h=3.0) == set()


def test_cost_aware_tiebreak_gives_undersampled_family_a_shot():
    """Below the waste GPU-h threshold a 0-SU family is NOT demoted (fair shot)."""
    from trex.selector import _wasteful_families
    mh = {"boltzgen": {"cumulative_gpu_h": 1.0, "strict_yield_su": 0, "near_miss_yield": 0}}
    assert _wasteful_families(_evidence_with_mh(mh), waste_gpu_h=3.0) == set()


def test_cost_aware_tiebreak_demotes_dominated_low_yield_family_with_some_su():
    """A family with nonzero SU can still be the wrong next launch when it
    has burned enough GPU-h and is strongly dominated by another family on
    chained/direct SU per worker-GPU-h. This catches the BindCraft-heavy failure
    mode from the live T-ReX traces without naming a target."""
    e = _evidence_with_mh({
        "bindcraft": {
            "cumulative_gpu_h": 20.0,
            "strict_yield_su": 0,
            "chained_strict_yield_su": 4,
            "near_miss_yield": 0,
            "chained_su_per_gpu_h": 0.20,
            "chained_su_per_gpu_h_recent": 0.0,
            "near_miss_yield_recent": 0,
        },
        "complexa_mcts": {
            "cumulative_gpu_h": 4.0,
            "strict_yield_su": 0,
            "chained_strict_yield_su": 16,
            "near_miss_yield": 2,
            "chained_su_per_gpu_h": 4.00,
            "chained_su_per_gpu_h_recent": 4.00,
            "near_miss_yield_recent": 1,
        },
    })
    cands = [_cand("c_bad", "bindcraft", cost="extended"), _cand("c_good", "complexa_mcts")]
    sup = _sup({"exploit": 1.0, "rescue": 0.0, "explore": 0.0}, [
        _sup_decision("c_bad", "exploit", 1, rc="extended"),
        _sup_decision("c_good", "exploit", 2),
    ])
    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1), tick_id="t")
    selected = {l.candidate_id for l in launches if l.status == "launched"}
    assert selected == {"c_good"}
    assert dbg["family_cost_penalties"]["bindcraft"] >= 2


def test_cost_aware_deferred_family_does_not_force_bad_mode_quota():
    """If the LLM asks for explore but the only explore candidate is an
    evidence-dominated family, selector should redistribute to a healthier
    feasible family instead of launching the dominated lane or leaving a slot
    idle. The E/R/X mixture remains the strategy; feasibility is evidence-aware."""
    e = _evidence_with_mh({
        "bindcraft": {
            "cumulative_gpu_h": 18.0,
            "strict_yield_su": 0,
            "chained_strict_yield_su": 2,
            "near_miss_yield": 0,
            "chained_su_per_gpu_h": 0.11,
            "chained_su_per_gpu_h_recent": 0.0,
            "near_miss_yield_recent": 0,
        },
        "complexa_beam": {
            "cumulative_gpu_h": 5.0,
            "strict_yield_su": 0,
            "chained_strict_yield_su": 15,
            "near_miss_yield": 1,
            "chained_su_per_gpu_h": 3.00,
            "chained_su_per_gpu_h_recent": 3.00,
        },
    })
    cands = [_cand("c_explore_bad", "bindcraft", cost="extended"), _cand("c_exploit_good", "complexa_beam")]
    sup = _sup({"exploit": 0.0, "rescue": 0.0, "explore": 1.0}, [
        _sup_decision("c_explore_bad", "explore", 1, rc="extended"),
        _sup_decision("c_exploit_good", "exploit", 1),
    ])
    launches, dbg = select_launches(e, cands, sup, cfg=SelectorConfig(available_slots=1), tick_id="t")
    selected = {l.candidate_id for l in launches if l.status == "launched"}
    assert selected == {"c_exploit_good"}
    assert "bindcraft" in dbg["cost_deferred_families"]
    assert dbg["quotas_final"]["exploit"] == 1


def test_cost_aware_keeps_under_sampled_and_recent_productive_families_primary():
    from trex.selector import _family_cost_penalties

    e = _evidence_with_mh({
        "bindcraft": {"cumulative_gpu_h": 2.0, "strict_yield_su": 0, "near_miss_yield": 0},
        "complexa_beam": {"cumulative_gpu_h": 3.0, "chained_su_per_gpu_h": 2.0},
    })
    assert "bindcraft" not in _family_cost_penalties(e, waste_gpu_h=3.0)

    e = _evidence_with_mh({
        "bindcraft": {
            "cumulative_gpu_h": 20.0,
            "chained_strict_yield_su": 3,
            "chained_su_per_gpu_h": 0.15,
            "chained_su_per_gpu_h_recent": 0.80,
        },
        "complexa_beam": {"cumulative_gpu_h": 4.0, "chained_su_per_gpu_h": 3.0},
    })
    assert "bindcraft" not in _family_cost_penalties(e, waste_gpu_h=3.0)


def test_cost_aware_comparator_ignores_tiny_lucky_best_rate():
    """A one-SU, sub-GPU-hour hit is not mature enough to define the global
    "best" rate used to suppress expensive alternatives."""
    from trex.selector import _family_cost_penalties

    e = _evidence_with_mh({
        "complexa_beam": {
            "cumulative_gpu_h": 0.2,
            "strict_yield_su": 1,
            "su_per_gpu_h": 5.0,
            "su_per_gpu_h_recent": 5.0,
        },
        "bindcraft": {
            "cumulative_gpu_h": 12.0,
            "strict_yield_su": 0,
            "chained_strict_yield_su": 2,
            "near_miss_yield": 1,
            "chained_su_per_gpu_h": 0.25,
            "chained_su_per_gpu_h_recent": 0.0,
            "near_miss_yield_recent": 0,
        },
    })
    penalties = _family_cost_penalties(e, waste_gpu_h=3.0)
    assert penalties.get("bindcraft", 0) < 2


# F2 (2026-06-18): cost-deferral must not defeat the collapse explore floor.

def test_f2_deferred_explore_family_preserved_under_collapse_clamp():
    """Under a Category-B clamp (stalled), the explore FLOOR must still be able to
    launch a deferred (dead) explore family — deferral must not zero explore."""
    from dataclasses import replace
    mh = {
        "bindcraft": {"cumulative_gpu_h": 10.0, "strict_yield_su": 0,
                      "chained_strict_yield_su": 0, "near_miss_yield": 0},
        "complexa_beam": {"cumulative_gpu_h": 3.0, "strict_yield_su": 3,
                          "near_miss_yield": 1},
    }
    e = replace(_evidence("stalled"), method_health=mh)  # stalled -> cat_b_on
    cands = [_cand("x1", "complexa_beam"),
             _cand("e1", "bindcraft", cost="diagnostic")]  # only explore family, deferred
    sup = _sup({"exploit": 0.5, "rescue": 0.0, "explore": 0.5},
               [_sup_decision("x1", "exploit", 1),
                _sup_decision("e1", "explore", 1, rc="diagnostic")])
    launches, _ = select_launches(
        e, cands, sup, cfg=SelectorConfig(available_slots=3), tick_id="t",
        recent_modes=["exploit"] * 9)
    launched = {l.candidate_id for l in launches if l.status == "launched"}
    assert "e1" in launched  # deferred explore family still launched under collapse


def test_f6_family_cost_penalties_for_cfg_matches_inline_and_toggle():
    from trex.selector import (
        _family_cost_penalties, family_cost_penalties_for_cfg,
    )
    e = _evidence_with_mh(_WASTEFUL_MH)
    cfg = SelectorConfig()
    assert family_cost_penalties_for_cfg(e, cfg) == _family_cost_penalties(
        e,
        waste_gpu_h=cfg.cost_aware_waste_gpu_h,
        low_yield_gpu_h=cfg.cost_aware_low_yield_gpu_h,
        min_best_su_per_gpu_h=cfg.cost_aware_min_best_su_per_gpu_h,
        best_min_gpu_h=cfg.cost_aware_best_min_gpu_h,
        best_min_su=cfg.cost_aware_best_min_su,
        best_rate_prior_gpu_h=cfg.cost_aware_best_rate_prior_gpu_h,
        dominated_fraction=cfg.cost_aware_dominated_fraction,
        min_su_per_gpu_h=cfg.cost_aware_min_su_per_gpu_h,
    )
    assert family_cost_penalties_for_cfg(
        e, SelectorConfig(cost_aware_family_tiebreak=False)) == {}


# ---- allocation-001 (2026-06-18): fallback within-mode ranking uses SU/GPU-h ----


def test_alloc001_fallback_tiebreak_prefers_higher_su_per_gpuh():
    """On the supervisor-FALLBACK path (no CandidateDecision → d is None), the
    within-mode order must prefer the higher recent SU/GPU-h family, NOT fall
    through to cost-class then alphabetical candidate_id. The higher-rate family is
    named alphabetically LAST so an id-tiebreak would (wrongly) pick the weaker."""
    from trex.selector import _tiebreak_key
    e = _evidence("productive")
    e.method_health.update({
        "aaa_weak": {"su_per_gpu_h_recent": 0.2, "chained_su_per_gpu_h_recent": 0.0},
        "zzz_strong": {"su_per_gpu_h_recent": 1.0, "chained_su_per_gpu_h_recent": 0.0},
    })
    weak = _cand("c_weak", "aaa_weak")
    strong = _cand("c_strong", "zzz_strong")
    assert _tiebreak_key(strong, {}, e) < _tiebreak_key(weak, {}, e)


def test_alloc001_fallback_tiebreak_uses_chained_rate_for_diagnostic_only():
    """Diagnostic-only generators have su_per_gpu_h_recent=0 and earn SU only via
    chained refilter, so the fallback tiebreak must read chained_su_per_gpu_h_recent."""
    from trex.selector import _tiebreak_key
    e = _evidence("productive")
    e.method_health.update({
        "aaa_weak": {"su_per_gpu_h_recent": 0.0, "chained_su_per_gpu_h_recent": 0.1},
        "zzz_strong": {"su_per_gpu_h_recent": 0.0, "chained_su_per_gpu_h_recent": 0.9},
    })
    assert _tiebreak_key(_cand("c_strong", "zzz_strong"), {}, e) < \
           _tiebreak_key(_cand("c_weak", "aaa_weak"), {}, e)


def test_alloc001_ranked_path_unchanged_by_rate():
    """No-op guard: on the supervisor-RANKED path (d present) rate_key=0.0 for all,
    so the LLM rank still decides even when it ranks the lower-rate family #1."""
    from trex.selector import _tiebreak_key
    e = _evidence("productive")
    e.method_health.update({
        "aaa_weak": {"su_per_gpu_h_recent": 0.2},
        "zzz_strong": {"su_per_gpu_h_recent": 1.0},
    })
    weak = _cand("c_weak", "aaa_weak")
    strong = _cand("c_strong", "zzz_strong")
    decs = {
        "c_weak": _sup_decision("c_weak", "exploit", 1),
        "c_strong": _sup_decision("c_strong", "exploit", 2),
    }
    assert _tiebreak_key(weak, decs, e) < _tiebreak_key(strong, decs, e)


def test_ranked_path_route_value_is_tiebreak_not_rank_override():
    import dataclasses
    from trex.schemas import RouteValueSummary

    e = dataclasses.replace(_evidence("stalled"), route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_mcts:op:family_rollup",
            scope="route",
            family="complexa_mcts",
            root_family="complexa_mcts",
            action_family="complexa_mcts",
            scoring_family=None,
            operator_id="op",
            config_signature="family_rollup",
            route_gpu_h=2.0,
            new_su=4,
            new_su_recent=2,
            new_su_per_route_gpu_h=2.0,
            recent_new_su_per_route_gpu_h=2.0,
            status="promote",
        )
    ])
    bc = _cand("bc", "bindcraft", cost="extended")
    mcts = _cand("mcts", "complexa_mcts", cost="standard")
    sup = _sup({"exploit": 0.0, "rescue": 0.0, "explore": 1.0}, [
        _sup_decision("bc", "explore", 1, rc="extended"),
        _sup_decision("mcts", "explore", 2),
    ])

    launches, dbg = select_launches(e, [bc, mcts], sup, cfg=SelectorConfig(available_slots=1))

    assert [l.candidate_id for l in launches if l.status == "launched"] == ["bc"]
    assert dbg["rank1_not_launched"] == []


def test_selector_defers_bad_exact_route_config_without_banning_family():
    import dataclasses
    from trex.evidence_reducer import canonical_config_signature
    from trex.schemas import RouteValueSummary

    e = dataclasses.replace(_evidence("productive"), route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam:beam_width=8",
            scope="route",
            family="complexa_beam",
            root_family="complexa_beam",
            action_family="complexa_beam",
            scoring_family=None,
            operator_id="op",
            config_signature=canonical_config_signature({"beam_width": 8}),
            config_delta={"beam_width": 8},
            route_gpu_h=9.0,
            attempts=8,
            new_su=0,
            new_su_recent=0,
            near_miss_recent=0,
            status="defer",
        ),
    ])
    good = dataclasses.replace(_cand("good", "complexa_beam"), config_delta={"beam_width": 4})
    bad = dataclasses.replace(_cand("bad", "complexa_beam"), config_delta={"beam_width": 8})
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[
            _sup_decision("bad", "exploit", 1),
            _sup_decision("good", "exploit", 2),
        ],
    )

    launches, dbg = select_launches(
        e, [bad, good], sup, cfg=SelectorConfig(available_slots=1)
    )
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert selected == ["good"]
    assert dbg["route_deferred_candidate_ids"] == ["bad"]


def test_selector_keeps_mode_last_candidate_despite_route_deferral():
    import dataclasses
    from trex.evidence_reducer import canonical_config_signature
    from trex.schemas import RouteValueSummary

    e = dataclasses.replace(_evidence("productive"), route_values=[
        RouteValueSummary(
            strategy_key="route::complexa_beam:complexa_beam:beam_width=8",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None, operator_id="op",
            config_signature=canonical_config_signature({"beam_width": 8}),
            config_delta={"beam_width": 8}, route_gpu_h=9.0, status="defer",
        ),
    ])
    only = dataclasses.replace(_cand("only", "complexa_beam"), config_delta={"beam_width": 8})
    sup = _sup(
        mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        decisions=[_sup_decision("only", "exploit", 1)],
    )

    launches, dbg = select_launches(
        e, [only], sup, cfg=SelectorConfig(available_slots=1)
    )
    selected = [l.candidate_id for l in launches if l.status == "launched"]
    assert selected == ["only"]
    assert dbg["route_deferred_candidate_ids"] == []


def test_selector_exact_route_penalty_respects_parent_root_context():
    import dataclasses
    from trex.evidence_reducer import canonical_config_signature

    cx_key = "route::complexa_beam:op:beam_width=4"
    bg_key = "route::boltzgen:op:num_designs=16"
    mp_sig = canonical_config_signature({})
    e = dataclasses.replace(_evidence("productive"), route_values=[
        {"strategy_key": cx_key, "scope": "route", "family": "complexa_beam", "action_family": "complexa_beam", "operator_id": "op", "config_signature": "beam_width=4", "status": "healthy", "evidence_refs": ["cx_parent"]},
        {"strategy_key": bg_key, "scope": "route", "family": "boltzgen", "action_family": "boltzgen", "operator_id": "op", "config_signature": "num_designs=16", "status": "healthy", "evidence_refs": ["bg_parent"]},
        {"strategy_key": f"{cx_key}->proteinmpnn_redesign:op:{mp_sig}", "scope": "route", "family": "proteinmpnn_redesign", "root_family": "complexa_beam", "action_family": "proteinmpnn_redesign", "operator_id": "op", "config_signature": mp_sig, "parent_strategy_key": cx_key, "status": "collapse_risk", "evidence_refs": ["mp_cx"]},
    ])
    mp_cx = dataclasses.replace(_cand("mp_cx", "proteinmpnn_redesign", cost="low"), parent_result_id="cx_parent")
    mp_bg = dataclasses.replace(_cand("mp_bg", "proteinmpnn_redesign", cost="low"), parent_result_id="bg_parent")
    sup = _sup({"exploit": 1.0, "rescue": 0.0, "explore": 0.0}, [
        _sup_decision("mp_cx", "exploit", 1, rc="low"),
        _sup_decision("mp_bg", "exploit", 2, rc="low"),
    ])

    launches, dbg = select_launches(e, [mp_cx, mp_bg], sup, cfg=SelectorConfig(available_slots=1))
    assert [l.candidate_id for l in launches if l.status == "launched"] == ["mp_bg"]
    assert dbg["route_deferred_candidate_ids"] == ["mp_cx"]

def test_effective_mode_window_k_shortens_only_under_urgent_evidence():
    from trex.selector import effective_mode_window_k

    cfg = SelectorConfig(mode_window_k=10)
    assert effective_mode_window_k(_evidence("productive"), cfg) == 10
    assert effective_mode_window_k(_evidence("productive_duplicate"), cfg) == 8
    assert effective_mode_window_k(_evidence("stalled"), cfg) == 6
    assert effective_mode_window_k(_evidence("rescue_rich"), cfg) == 6
    assert effective_mode_window_k(_evidence("strict_duplicate_collapse"), cfg) == 4
    assert effective_mode_window_k(_evidence("deep_stall"), cfg) == 4

    fixed = SelectorConfig(mode_window_k=10, adaptive_mode_window_k=False)
    assert effective_mode_window_k(_evidence("strict_duplicate_collapse"), fixed) == 10


def test_selector_debug_reports_effective_mode_window_k():
    c = _cand("c1", "complexa_beam")
    sup = _sup({"exploit": 1.0, "rescue": 0.0, "explore": 0.0}, [
        _sup_decision("c1", "exploit", 1),
    ])
    launches, dbg = select_launches(
        _evidence("strict_duplicate_collapse"),
        [c],
        sup,
        cfg=SelectorConfig(available_slots=1, mode_window_k=10),
        recent_modes=["exploit"] * 10,
    )
    assert [l.candidate_id for l in launches if l.status == "launched"] == ["c1"]
    assert dbg["mode_window_k_configured"] == 10
    assert dbg["mode_window_k_effective"] == 4
    assert dbg["mode_window"]["window_k"] == 4


def test_selector_demotes_duplicate_collapsed_exact_route_without_banning_family():
    import dataclasses
    from trex.evidence_reducer import canonical_config_signature
    from trex.schemas import RouteValueSummary

    e = dataclasses.replace(_evidence("productive"), route_values=[
        RouteValueSummary(
            strategy_key="route::proteinmpnn_redesign:op:default",
            scope="route",
            family="proteinmpnn_redesign",
            root_family="proteinmpnn_redesign",
            action_family="proteinmpnn_redesign",
            scoring_family=None,
            operator_id="op",
            config_signature=canonical_config_signature({}),
            config_delta={},
            route_gpu_h=5.0,
            attempts=20,
            strict_count=103,
            new_su=1,
            new_su_recent=0,
            strict_per_su=103.0,
            duplicate_bin_fraction=0.99,
            status="collapse_risk",
        ),
    ])
    duplicate_route = _cand("dup", "proteinmpnn_redesign", cost="low")
    new_route = dataclasses.replace(
        _cand("new", "proteinmpnn_redesign", cost="low"),
        config_delta={"sampling_temp": 0.25},
    )
    sup = _sup({"exploit": 1.0, "rescue": 0.0, "explore": 0.0}, [
        _sup_decision("dup", "exploit", 1, rc="low"),
        _sup_decision("new", "exploit", 2, rc="low"),
    ])

    launches, dbg = select_launches(
        e, [duplicate_route, new_route], sup, cfg=SelectorConfig(available_slots=1)
    )
    assert [l.candidate_id for l in launches if l.status == "launched"] == ["new"]
    assert dbg["route_deferred_candidate_ids"] == ["dup"]


def test_high_cost_strong_evidence_opens_two_slot_cap_by_default():
    import dataclasses
    from trex.selector import (
        high_cost_cap_for_evidence,
        high_cost_capacity_block_reason_for_cfg,
    )

    e = dataclasses.replace(
        _evidence("productive"),
        completed_children=96,
        worker_gpu_h_total=6.0,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 2.5,
                "chained_strict_yield_su": 3,
                "chained_su_per_gpu_h_recent": 1.2,
                "chained_su_per_gpu_h": 1.0,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 4.0,
                "strict_yield_su": 1,
                "su_per_gpu_h_recent": 0.2,
            },
        },
        pending_family_load={
            "by_family": {
                "bindcraft": {"running": 1, "queued": 0, "pending_total": 1, "inflight_gpu_h": 0.8}
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (2, "strong_evidence")
    assert high_cost_capacity_block_reason_for_cfg(e, SelectorConfig(), "bindcraft") is None
    cap_cfg = SelectorConfig(high_cost_pending_families=("bindcraft", "complexa_mcts"))

    e_at_cap = dataclasses.replace(
        e,
        pending_family_load={
            "by_family": {
                "bindcraft": {"running": 2, "queued": 0, "pending_total": 2, "inflight_gpu_h": 1.6}
            },
            "high_cost_families": ["bindcraft", "complexa_mcts"],
        },
    )
    assert high_cost_capacity_block_reason_for_cfg(e_at_cap, SelectorConfig(), "bindcraft") is not None
    reason = high_cost_capacity_block_reason_for_cfg(e_at_cap, cap_cfg, "bindcraft")
    assert reason is not None and "cap=2" in reason and "cap_source=strong_evidence" in reason


def test_high_cost_batch_cap_blocks_third_global_priority_bindcraft():
    import dataclasses

    e = dataclasses.replace(
        _evidence("productive"),
        completed_children=96,
        worker_gpu_h_total=6.0,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 2.5,
                "chained_strict_yield_su": 3,
                "chained_su_per_gpu_h_recent": 1.2,
                "chained_su_per_gpu_h": 1.0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 4.0,
                "strict_yield_su": 1,
                "su_per_gpu_h_recent": 0.2,
            },
        },
    )
    cands = [
        _cand("bc1", "bindcraft", cost="extended"),
        _cand("bc2", "bindcraft", cost="extended"),
        _cand("bc3", "bindcraft", cost="extended"),
        _cand("cx1", "complexa_beam", cost="low"),
    ]
    sup = _sup(
        {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        [
            _sup_decision("bc1", "exploit", 1, global_rank=1),
            _sup_decision("bc2", "exploit", 2, global_rank=2),
            _sup_decision("bc3", "exploit", 3, global_rank=3),
            _sup_decision("cx1", "exploit", 4, global_rank=4),
        ],
    )

    launches, dbg = select_launches(
        e,
        cands,
        sup,
        cfg=SelectorConfig(available_slots=3, diversity_max_per_family=3),
    )
    launched = [x.candidate_id for x in launches if x.status == "launched"]

    assert launched == ["bc1", "bc2", "cx1"]
    assert dbg["high_cost_batch_hard_caps"] == {"bindcraft": 2}
    held = [x for x in dbg["global_priority_not_launched"] if x["candidate_id"] == "bc3"]
    assert held and held[0]["reason"].startswith("high_cost_batch_hard_cap")


def test_high_cost_single_su_high_value_opens_promoted_two_slot_cap_only():
    import dataclasses
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("stalled"),
        completed_children=80,
        worker_gpu_h_total=8.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 1.4,
                "chained_strict_yield_su": 1,
                "chained_strict_yield_su_recent": 1,
                "chained_su_per_gpu_h_recent": 0.9,
                "chained_su_per_gpu_h": 0.7,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 4.0,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert cap == 2
    assert source in {"promoted", "single_su_high_value"}


def test_selector_route_rate_ignores_inflated_record_recent_for_score_conversion_route():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import _candidate_route_rate

    replay = _cand("route_replay_t_00_bindcraft", "bindcraft")
    e = dataclasses.replace(
        _evidence("productive"),
        route_values=[RouteValueSummary(
            strategy_key="route::bindcraft:bindcraft_default:default",
            scope="route", family="bindcraft", root_family="bindcraft",
            action_family="bindcraft", scoring_family="structure_refilter",
            operator_id="op", config_signature="default",
            route_role="generator_with_af2_score_conversion",
            route_gpu_h=0.662, generator_gpu_h=0.638,
            canonical_refilter_gpu_h=0.024, new_su=2,
            record_recent_new_su=2, record_recent_route_gpu_h=0.024,
            record_recent_new_su_per_route_gpu_h=83.333,
            new_su_per_route_gpu_h=3.021, status="promote",
            marginal_status="productive",
        )],
    )

    assert abs(_candidate_route_rate(replay, e) - 3.021) < 1e-9


def test_high_cost_route_value_best_ignores_inflated_record_recent_for_score_conversion_route():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("strict_duplicate_collapse"),
        run_su_count=2,
        completed_children=120,
        worker_gpu_h_total=16.0,
        strict_duplicate_collapse_signal=True,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 0.662,
                "chained_strict_yield_su": 2,
                "chained_su_per_gpu_h": 3.021,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 2.0,
                "strict_yield_su": 2,
                "su_per_gpu_h_recent": 10.0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family="bindcraft",
                action_family="bindcraft", scoring_family="structure_refilter",
                operator_id="bindcraft_default", config_signature="default",
                route_role="generator_with_af2_score_conversion",
                route_gpu_h=0.662, generator_gpu_h=0.638,
                canonical_refilter_gpu_h=0.024, new_su=2,
                record_recent_new_su=2, record_recent_route_gpu_h=0.024,
                record_recent_new_su_per_route_gpu_h=83.333,
                new_su_per_route_gpu_h=3.021, status="promote",
                marginal_status="productive",
            ),
            RouteValueSummary(
                strategy_key="route::complexa_beam:op:default",
                scope="route", family="complexa_beam", root_family="complexa_beam",
                action_family="complexa_beam", scoring_family=None,
                operator_id="op", config_signature="default",
                route_gpu_h=2.0, new_su=2, new_su_recent_gpu=1,
                gpu_recent_new_su_per_route_gpu_h=10.0,
                new_su_per_route_gpu_h=1.0, status="healthy",
                marginal_status="productive",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) != (3, "route_value_best")


def test_high_cost_route_value_single_su_does_not_open_full_cap_under_collapse():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("strict_duplicate_collapse"),
        run_su_count=2,
        completed_children=120,
        worker_gpu_h_total=16.0,
        strict_duplicate_collapse_signal=True,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 4.6,
                "chained_strict_yield_su": 1,
                "chained_su_per_gpu_h": 0.217,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 3.0,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family=None,
                action_family="bindcraft", scoring_family=None,
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=4.6, new_su=1, new_su_recent=0,
                new_su_per_route_gpu_h=0.217, recent_route_gpu_h=1.0,
                recent_new_su_per_route_gpu_h=None, strict_count=3,
                strict_per_su=3.0, duplicate_bin_fraction=0.0,
                status="healthy",
            ),
            RouteValueSummary(
                strategy_key="route::complexa_beam:op:default",
                scope="route", family="complexa_beam", root_family=None,
                action_family="complexa_beam", scoring_family=None,
                operator_id="op", config_signature="default",
                route_gpu_h=3.0, new_su=0, status="observed",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (1, "default")


def test_high_cost_route_value_single_su_current_winner_opens_promoted_cap_only():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("strict_duplicate_collapse"),
        run_su_count=2,
        completed_children=120,
        worker_gpu_h_total=16.0,
        strict_duplicate_collapse_signal=True,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 4.6,
                "chained_strict_yield_su": 1,
                "chained_su_per_gpu_h": 0.217,
                "chained_su_per_gpu_h_recent": 0.217,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 3.0,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family=None,
                action_family="bindcraft", scoring_family="structure_refilter",
                route_role="generator_with_af2_score_conversion",
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=4.6, new_su=1,
                new_su_recent_gpu=1, gpu_recent_route_gpu_h=4.6,
                gpu_recent_new_su_per_route_gpu_h=0.217,
                new_su_per_route_gpu_h=0.217, strict_count=3,
                strict_per_su=3.0, duplicate_bin_fraction=0.0,
                status="healthy", marginal_status="productive",
            ),
            RouteValueSummary(
                strategy_key="route::complexa_beam:op:default",
                scope="route", family="complexa_beam", root_family=None,
                action_family="complexa_beam", scoring_family=None,
                operator_id="op", config_signature="default",
                route_gpu_h=3.0, new_su=0, status="observed",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert cap == 2
    assert source in {"promoted", "route_value_promoted", "single_su_high_value"}


def test_high_cost_route_value_repeated_current_su_opens_two_slot_cap():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("productive"),
        run_su_count=4,
        completed_children=160,
        worker_gpu_h_total=12.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 4.0,
                "chained_strict_yield_su": 2,
                "chained_su_per_gpu_h": 0.8,
                "chained_su_per_gpu_h_recent": 0.8,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 3.0,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family="bindcraft",
                action_family="bindcraft", scoring_family="structure_refilter",
                route_role="generator_with_af2_score_conversion",
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=2.5, new_su=2,
                new_su_recent_gpu=2, gpu_recent_route_gpu_h=2.5,
                gpu_recent_new_su_per_route_gpu_h=0.8,
                new_su_per_route_gpu_h=0.8, strict_count=2,
                strict_per_su=1.0, duplicate_bin_fraction=0.0,
                status="healthy", marginal_status="productive",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (2, "strong_evidence")


def test_high_cost_route_value_repeated_current_route_opens_two_slot_cap_without_method_recent():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("productive"),
        run_su_count=4,
        completed_children=160,
        worker_gpu_h_total=12.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 4.0,
                "chained_strict_yield_su": 2,
                "chained_su_per_gpu_h": 0.8,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield_recent": 0,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 3.0,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family="bindcraft",
                action_family="bindcraft", scoring_family="structure_refilter",
                route_role="generator_with_af2_score_conversion",
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=2.5, new_su=2,
                new_su_recent_gpu=2, gpu_recent_route_gpu_h=2.5,
                gpu_recent_new_su_per_route_gpu_h=0.8,
                new_su_per_route_gpu_h=0.8, strict_count=2,
                strict_per_su=1.0, duplicate_bin_fraction=0.0,
                status="healthy", marginal_status="productive",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (2, "route_value_best")


def test_high_cost_recent_near_only_does_not_open_full_cap():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("rescue_rich"),
        run_su_count=3,
        completed_children=160,
        worker_gpu_h_total=18.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 8.0,
                "chained_strict_yield_su": 2,
                "chained_su_per_gpu_h": 0.7,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield": 5,
                "near_miss_yield_recent": 2,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 4.0,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:near_only",
                scope="route", family="bindcraft", root_family="bindcraft",
                action_family="bindcraft", scoring_family="structure_refilter",
                route_role="generator_with_af2_score_conversion",
                operator_id="bindcraft_default", config_signature="near_only",
                route_gpu_h=8.0, generator_gpu_h=7.0,
                canonical_refilter_gpu_h=1.0, new_su=2,
                new_su_recent_gpu=0, gpu_recent_route_gpu_h=2.5,
                gpu_recent_new_su_per_route_gpu_h=0.0,
                medium_recent_new_su=0, medium_recent_route_gpu_h=5.0,
                medium_recent_new_su_per_route_gpu_h=0.0,
                near_miss_recent=2,
                new_su_per_route_gpu_h=0.7,
                status="healthy", marginal_status="delayed_productive",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert cap == 2
    assert source in {"promoted", "route_value_promoted"}


def test_high_cost_tnf_like_no_su_deep_stall_never_opens_full_bindcraft_cap():
    import dataclasses
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("deep_stall"),
        run_su_count=0,
        completed_children=96,
        worker_gpu_h_total=24.0,
        gpu_h_since_last_su=24.0,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 6.5,
                "strict_yield_su": 0,
                "chained_strict_yield_su": 0,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield": 2,
                "near_miss_yield_recent": 0,
                "timeout_count": 0,
            },
            "complexa_mcts": {
                "cumulative_gpu_h": 1.8,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
        execution_realization={
            "by_family": {
                "bindcraft": {
                    "proposed": 8,
                    "selected": 3,
                    "started": 1,
                    "selected_not_started": 2,
                    "dispatch_deferred": 1,
                }
            }
        },
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert cap <= 2
    assert source != "strong_evidence"
    assert source != "route_value_best"


def test_high_cost_route_value_single_su_duplicate_route_does_not_open_full_cap():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("strict_duplicate_collapse"),
        run_su_count=3,
        completed_children=140,
        worker_gpu_h_total=20.0,
        strict_duplicate_collapse_signal=True,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 5.0,
                "chained_strict_yield_su": 1,
                "chained_su_per_gpu_h": 0.2,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield_recent": 0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family=None,
                action_family="bindcraft", scoring_family=None,
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=5.0, new_su=1, new_su_recent=0,
                new_su_per_route_gpu_h=0.2, strict_count=20,
                strict_per_su=20.0, duplicate_bin_fraction=0.90,
                status="collapse_risk",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (1, "default")


def test_high_cost_route_value_requires_mature_su_even_with_stale_competitor():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("strict_duplicate_collapse"),
        run_su_count=2,
        completed_children=180,
        worker_gpu_h_total=21.0,
        strict_duplicate_collapse_signal=True,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 7.2,
                "chained_strict_yield_su": 1,
                "chained_su_per_gpu_h": 0.139,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield_recent": 0,
            },
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::proteinmpnn_redesign:dead_duplicate",
                scope="route", family="proteinmpnn_redesign", root_family="boltzgen",
                action_family="proteinmpnn_redesign", scoring_family=None,
                operator_id="proteinmpnn_redesign_default", config_signature="default",
                route_gpu_h=0.72, new_su=1, new_su_recent=0,
                new_su_per_route_gpu_h=1.38, recent_new_su_per_route_gpu_h=None,
                strict_count=31, strict_per_su=31.0, duplicate_bin_fraction=0.95,
                status="collapse_risk",
            ),
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family=None,
                action_family="bindcraft", scoring_family=None,
                operator_id="bindcraft_default", config_signature="default",
                route_gpu_h=7.2, new_su=1, new_su_recent=0,
                new_su_per_route_gpu_h=0.139, recent_new_su_per_route_gpu_h=None,
                strict_count=3, strict_per_su=3.0, duplicate_bin_fraction=0.67,
                status="diversify",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (1, "default")


def test_high_cost_stale_lifetime_su_does_not_open_extra_bindcraft_slots():
    import dataclasses
    from trex.schemas import RouteValueSummary
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("stalled"),
        run_su_count=2,
        completed_children=180,
        worker_gpu_h_total=18.0,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 8.0,
                "chained_strict_yield_su": 2,
                "chained_su_per_gpu_h": 0.7,
                "chained_su_per_gpu_h_recent": 0.0,
                "near_miss_yield_recent": 0,
            },
        },
        execution_realization={
            "by_family": {
                "bindcraft": {
                    "proposed": 12,
                    "selected": 8,
                    "started": 2,
                    "selected_not_started": 6,
                    "dispatch_deferred": 6,
                }
            }
        },
        route_values=[
            RouteValueSummary(
                strategy_key="route::bindcraft:bindcraft_default:default",
                scope="route", family="bindcraft", root_family="bindcraft",
                action_family="bindcraft", scoring_family="structure_refilter",
                operator_id="bindcraft_default", config_signature="default",
                route_role="generator_with_af2_score_conversion",
                route_gpu_h=8.0, generator_gpu_h=7.2,
                canonical_refilter_gpu_h=0.8, new_su=2,
                new_su_recent_gpu=0, gpu_recent_route_gpu_h=6.0,
                gpu_recent_new_su_per_route_gpu_h=0.0,
                medium_recent_new_su=0, medium_recent_route_gpu_h=6.0,
                medium_recent_new_su_per_route_gpu_h=0.0,
                new_su_per_route_gpu_h=0.7,
                status="healthy", marginal_status="productive",
            ),
        ],
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (1, "default")


def test_high_cost_single_su_requires_enough_effective_compute():
    import dataclasses
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("low_evidence"),
        run_su_count=1,
        completed_children=12,
        worker_gpu_h_total=1.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "complexa_mcts": {
                "cumulative_gpu_h": 0.16,
                "strict_yield_su": 1,
                "su_per_gpu_h_recent": 6.25,
                "su_per_gpu_h": 6.25,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 0.2,
                "strict_yield_su": 0,
                "su_per_gpu_h_recent": 0.0,
            },
        },
    )

    cap, source = high_cost_cap_for_evidence(e, "complexa_mcts")
    assert (cap, source) == (2, "promoted")


def test_high_cost_single_su_does_not_beat_better_cheap_route():
    import dataclasses
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=18,
        completed_children=160,
        worker_gpu_h_total=8.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 1.4,
                "chained_strict_yield_su": 1,
                "chained_strict_yield_su_recent": 1,
                "chained_su_per_gpu_h_recent": 0.9,
                "chained_su_per_gpu_h": 0.7,
            },
            "complexa_beam": {
                "cumulative_gpu_h": 3.0,
                "strict_yield_su": 12,
                "su_per_gpu_h_recent": 3.0,
            },
        },
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (2, "promoted")


def test_high_cost_cap_stays_conservative_on_easy_productive_without_collapse():
    import dataclasses
    from trex.selector import high_cost_cap_for_evidence

    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=40,
        completed_children=200,
        worker_gpu_h_total=12.0,
        gpu_h_since_last_su=3.0,
        strict_duplicate_collapse_signal=False,
        method_health={
            "bindcraft": {"cumulative_gpu_h": 0.0, "strict_yield_su": 0, "near_miss_yield": 0},
            "complexa_beam": {"cumulative_gpu_h": 4.0, "strict_yield_su": 24, "su_per_gpu_h_recent": 4.0},
        },
    )

    cap, source = high_cost_cap_for_evidence(e, "bindcraft")
    assert (cap, source) == (1, "default")


def test_low_cost_near_miss_rescue_floor_preempts_high_cost_explore_single_slot():
    explore = _cand("cand_explore_bindcraft", "bindcraft", cost="diagnostic")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign", cost="low"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("stalled"),
        near_miss_count=1,
        production_near_miss_ids=["near_miss_1"],
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
    )
    sup = _sup(
        {"exploit": 0.1, "rescue": 0.3, "explore": 0.6},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="diagnostic"),
            _sup_decision(rescue.candidate_id, "rescue", 1, rc="low"),
        ],
    )

    launches, dbg = select_launches(
        e,
        [explore, rescue],
        sup,
        cfg=SelectorConfig(available_slots=1, low_cost_near_miss_rescue_floor=True),
    )
    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [rescue.candidate_id]
    assert dbg["forced_near_miss_rescue_floor"]["candidate_id"] == rescue.candidate_id
    assert any("low_cost_near_miss_rescue_floor" in x for x in dbg["redistribute_log"])


@pytest.mark.parametrize(
    ("near_miss_dedup_status", "near_miss_dedup_coverage"),
    [("no_binary", 0.0), ("disabled", None)],
)
def test_degraded_near_miss_dedup_does_not_force_rescue_floor(
    near_miss_dedup_status,
    near_miss_dedup_coverage,
):
    explore = _cand("cand_explore_bindcraft", "bindcraft", cost="diagnostic")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign", cost="low"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("stalled"),
        near_miss_count=1,
        production_near_miss_ids=["near_miss_1"],
        near_miss_dedup_status=near_miss_dedup_status,
        near_miss_dedup_coverage=near_miss_dedup_coverage,
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
    )
    sup = _sup(
        {"exploit": 0.1, "rescue": 0.3, "explore": 0.6},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="diagnostic"),
            _sup_decision(rescue.candidate_id, "rescue", 1, rc="low"),
        ],
    )

    launches, dbg = select_launches(e, [explore, rescue], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [explore.candidate_id]
    assert dbg["forced_near_miss_rescue_floor"] == {}


def test_degraded_near_miss_exemplars_do_not_reenable_rescue_floor_after_count_zeroing():
    explore = _cand("cand_explore_bindcraft_after_zeroed_nm", "bindcraft", cost="diagnostic")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn_after_zeroed_nm", "proteinmpnn_redesign", cost="low"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("stalled"),
        near_miss_count=0,
        production_near_miss_ids=["near_miss_1"],
        near_miss_dedup_status="disabled",
        near_miss_dedup_coverage=None,
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
    )
    sup = _sup(
        {"exploit": 0.1, "rescue": 0.3, "explore": 0.6},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="diagnostic"),
            _sup_decision(rescue.candidate_id, "rescue", 1, rc="low"),
        ],
    )

    launches, dbg = select_launches(e, [explore, rescue], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [explore.candidate_id]
    assert dbg["forced_near_miss_rescue_floor"] == {}


def test_low_cost_near_miss_rescue_floor_rejects_canonical_score_conversion():
    explore = _cand("cand_explore_bindcraft", "bindcraft", cost="diagnostic")
    canonical = dataclasses.replace(
        _cand("cand_score_conversion", "structure_refilter", cost="low", refilter_role=CANONICAL_SCORE_CONVERSION),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("stalled"),
        near_miss_count=1,
        production_near_miss_ids=["near_miss_1"],
    )
    sup = _sup(
        {"exploit": 0.1, "rescue": 0.3, "explore": 0.6},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="diagnostic"),
            _sup_decision(canonical.candidate_id, "rescue", 1, rc="low"),
        ],
    )

    launches, dbg = select_launches(e, [explore, canonical], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [explore.candidate_id]
    assert dbg["forced_near_miss_rescue_floor"] == {}


def test_low_cost_near_miss_rescue_floor_does_not_tax_productive_easy_target():
    explore = _cand("cand_explore_bindcraft", "bindcraft", cost="diagnostic")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign", cost="low"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        near_miss_count=1,
        production_near_miss_ids=["near_miss_1"],
        run_su_count=12,
        run_su_count_delta=1,
        su_per_gpu_h_recent=2.0,
    )
    sup = _sup(
        {"exploit": 0.1, "rescue": 0.3, "explore": 0.6},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="diagnostic"),
            _sup_decision(rescue.candidate_id, "rescue", 1, rc="low"),
        ],
    )

    launches, dbg = select_launches(e, [explore, rescue], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [explore.candidate_id]
    assert dbg["forced_near_miss_rescue_floor"] == {}


def test_low_cost_near_miss_rescue_floor_is_one_shot_per_observed_parent_route():
    from trex.schemas import RouteValueSummary

    explore = _cand("cand_explore_bindcraft", "bindcraft", cost="diagnostic")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign", cost="low"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("stalled"),
        near_miss_count=1,
        production_near_miss_ids=["near_miss_1"],
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_mcts:op:parent->proteinmpnn_redesign:op:default",
            scope="route", family="proteinmpnn_redesign", root_family="complexa_mcts",
            action_family="proteinmpnn_redesign", scoring_family=None,
            operator_id="op", config_signature="family_rollup",
            route_gpu_h=0.2, attempts=1, completions=1, evidence_refs=["near_miss_1"],
        )],
        run_su_count_delta=0,
        su_per_gpu_h_recent=0.0,
    )
    sup = _sup(
        {"exploit": 0.1, "rescue": 0.3, "explore": 0.6},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="diagnostic"),
            _sup_decision(rescue.candidate_id, "rescue", 1, rc="low"),
        ],
    )

    launches, dbg = select_launches(e, [explore, rescue], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [explore.candidate_id]
    assert dbg["forced_near_miss_rescue_floor"] == {}


def test_gpu_recent_duplicate_route_keeps_exploit_when_still_buying_su():
    from trex.schemas import RouteValueSummary

    exploit = _cand('route_replay_t_00_complexa_fk_steering', 'complexa_fk_steering')
    rescue = dataclasses.replace(
        _cand('cand_rescue_mpnn', 'proteinmpnn_redesign'),
        parent_result_id='near_miss_1',
    )
    e = dataclasses.replace(
        _evidence('productive_duplicate'),
        duplicate_fraction=0.85,
        route_values=[RouteValueSummary(
            strategy_key='route::complexa_fk_steering:op:default',
            scope='route', family='complexa_fk_steering', root_family='complexa_fk_steering',
            action_family='complexa_fk_steering', scoring_family=None,
            operator_id='op', config_signature='default',
            route_gpu_h=5.0, new_su=10, new_su_recent=0,
            new_su_recent_gpu=1, new_su_per_route_gpu_h=2.0,
            recent_route_gpu_h=0.5, recent_new_su_per_route_gpu_h=None,
            gpu_recent_route_gpu_h=3.0, gpu_recent_new_su_per_route_gpu_h=0.33,
            strict_count=100, strict_per_su=10.0, duplicate_bin_fraction=0.85,
            status='diversify', marginal_status='productive_but_duplicate',
        )],
    )
    sup = _sup(
        {'exploit': 0.55, 'rescue': 0.45, 'explore': 0.0},
        [_sup_decision(exploit.candidate_id, 'exploit', 1), _sup_decision(rescue.candidate_id, 'rescue', 1)],
    )

    launches, dbg = select_launches(
        e,
        [exploit, rescue],
        sup,
        cfg=SelectorConfig(available_slots=1, marginal_diversity_override=True),
    )
    launched = [l.candidate_id for l in launches if l.status == 'launched']
    assert launched == [exploit.candidate_id]
    assert not any('marginal_diversity_override' in x for x in dbg['redistribute_log'])


def test_dry_duplicate_exact_replay_yields_single_slot_to_material_rescue():
    from trex.schemas import RouteValueSummary

    exploit = _cand('route_replay_t_00_complexa_fk_steering', 'complexa_fk_steering')
    rescue = dataclasses.replace(
        _cand('cand_rescue_mpnn', 'proteinmpnn_redesign'),
        parent_result_id='near_miss_1',
    )
    e = dataclasses.replace(
        _evidence('productive_duplicate'),
        duplicate_fraction=0.85,
        route_values=[RouteValueSummary(
            strategy_key='route::complexa_fk_steering:op:default',
            scope='route', family='complexa_fk_steering', root_family='complexa_fk_steering',
            action_family='complexa_fk_steering', scoring_family=None,
            operator_id='op', config_signature='default',
            route_gpu_h=5.0, new_su=10, new_su_recent=0,
            new_su_recent_gpu=0, new_su_per_route_gpu_h=2.0,
            recent_route_gpu_h=1.0, recent_new_su_per_route_gpu_h=None,
            gpu_recent_route_gpu_h=3.0, gpu_recent_new_su_per_route_gpu_h=None,
            strict_count=100, strict_per_su=10.0, duplicate_bin_fraction=0.85,
            status='collapse_risk', marginal_status='dry_duplicate',
        )],
    )
    sup = _sup(
        {'exploit': 0.55, 'rescue': 0.45, 'explore': 0.0},
        [_sup_decision(exploit.candidate_id, 'exploit', 1), _sup_decision(rescue.candidate_id, 'rescue', 1)],
    )

    launches, dbg = select_launches(
        e,
        [exploit, rescue],
        sup,
        cfg=SelectorConfig(available_slots=1, marginal_diversity_override=True),
    )
    launched = [l.candidate_id for l in launches if l.status == 'launched']
    assert launched == [rescue.candidate_id]
    assert any('marginal_diversity_override' in x for x in dbg['redistribute_log'])


def test_lifetime_only_stale_route_replay_yields_single_slot_to_material_rescue():
    from trex.schemas import RouteValueSummary

    exploit = _cand("route_replay_t_00_complexa_beam", "complexa_beam")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        gpu_h_since_last_su=13.0,
        duplicate_fraction=0.85,
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=2.0, new_su=5, new_su_recent=0, new_su_recent_gpu=0,
            new_su_per_route_gpu_h=2.5, recent_route_gpu_h=0.0,
            record_recent_route_gpu_h=0.0, strict_count=40, strict_per_su=8.0,
            duplicate_bin_fraction=0.80, status="diversify", marginal_status="observed",
        )],
    )
    sup = _sup(
        {"exploit": 0.55, "rescue": 0.45, "explore": 0.0},
        [_sup_decision(exploit.candidate_id, "exploit", 1), _sup_decision(rescue.candidate_id, "rescue", 1)],
    )

    launches, dbg = select_launches(
        e,
        [exploit, rescue],
        sup,
        cfg=SelectorConfig(available_slots=1, marginal_diversity_override=True),
    )

    launched = [l.candidate_id for l in launches if l.status == "launched"]
    assert launched == [rescue.candidate_id]
    assert any("marginal_diversity_override" in x for x in dbg["redistribute_log"])


def test_record_recent_only_stale_route_replay_yields_single_slot_to_material_rescue():
    from trex.schemas import RouteValueSummary

    exploit = _cand("route_replay_t_00_complexa_beam", "complexa_beam")
    rescue = dataclasses.replace(
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign"),
        parent_result_id="near_miss_1",
    )
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        gpu_h_since_last_su=13.0,
        duplicate_fraction=0.85,
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=0.7, new_su=1, new_su_recent=1, record_recent_new_su=1,
            new_su_recent_gpu=0, new_su_per_route_gpu_h=1.4,
            record_recent_new_su_per_route_gpu_h=1.4, recent_route_gpu_h=0.7,
            record_recent_route_gpu_h=0.7, strict_count=11, strict_per_su=11.0,
            duplicate_bin_fraction=0.82, status="diversify",
            marginal_status="productive_but_duplicate",
        )],
    )
    sup = _sup(
        {"exploit": 0.55, "rescue": 0.45, "explore": 0.0},
        [_sup_decision(exploit.candidate_id, "exploit", 1), _sup_decision(rescue.candidate_id, "rescue", 1)],
    )

    launches, dbg = select_launches(
        e,
        [exploit, rescue],
        sup,
        cfg=SelectorConfig(available_slots=1, marginal_diversity_override=True),
    )

    assert [l.candidate_id for l in launches if l.status == "launched"] == [rescue.candidate_id]
    assert any("marginal_diversity_override" in x for x in dbg["redistribute_log"])

def test_medium_recent_route_not_treated_as_dry_duplicate_replay():
    from trex.schemas import RouteValueSummary

    exploit = _cand('route_replay_t_00_bindcraft', 'bindcraft')
    rescue = dataclasses.replace(
        _cand('cand_rescue_mpnn', 'proteinmpnn_redesign'),
        parent_result_id='near_miss_1',
    )
    e = dataclasses.replace(
        _evidence('productive_duplicate'),
        route_values=[RouteValueSummary(
            strategy_key='route::bindcraft:op:default',
            scope='route', family='bindcraft', root_family='bindcraft',
            action_family='bindcraft', scoring_family=None,
            operator_id='op', config_signature='default',
            route_gpu_h=8.0, new_su=2, new_su_recent=0,
            new_su_recent_gpu=0, medium_recent_new_su=1,
            new_su_per_route_gpu_h=0.25,
            gpu_recent_route_gpu_h=3.0, gpu_recent_new_su_per_route_gpu_h=0.0,
            medium_recent_route_gpu_h=6.0,
            medium_recent_new_su_per_route_gpu_h=0.167,
            strict_count=20, strict_per_su=10.0, duplicate_bin_fraction=0.85,
            status='diversify', marginal_status='delayed_productive_duplicate',
        )],
    )
    sup = _sup(
        {'exploit': 0.55, 'rescue': 0.45, 'explore': 0.0},
        [_sup_decision(exploit.candidate_id, 'exploit', 1), _sup_decision(rescue.candidate_id, 'rescue', 1)],
    )

    launches, dbg = select_launches(e, [exploit, rescue], sup, cfg=SelectorConfig(available_slots=1))
    launched = [l.candidate_id for l in launches if l.status == 'launched']
    assert launched == [exploit.candidate_id]
    assert not any('marginal_diversity_override' in x for x in dbg['redistribute_log'])


def test_productive_but_duplicate_route_penalty_does_not_override_supervisor_rank():
    from trex.schemas import RouteValueSummary

    replay = _cand("route_replay_t_00_complexa_fk_steering", "complexa_fk_steering")
    alt = _cand("cand_alt_complexa_beam", "complexa_beam")
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_fk_steering:op:default",
            scope="route", family="complexa_fk_steering", root_family="complexa_fk_steering",
            action_family="complexa_fk_steering", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=5.0, new_su=10, new_su_recent=0,
            new_su_recent_gpu=1, new_su_per_route_gpu_h=2.0,
            gpu_recent_route_gpu_h=3.0, gpu_recent_new_su_per_route_gpu_h=0.33,
            strict_count=100, strict_per_su=10.0, duplicate_bin_fraction=0.85,
            status="collapse_risk", marginal_status="productive_but_duplicate",
        )],
    )
    sup = _sup(
        {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        [_sup_decision(replay.candidate_id, "exploit", 1), _sup_decision(alt.candidate_id, "exploit", 2)],
    )

    launches, _dbg = select_launches(e, [replay, alt], sup, cfg=SelectorConfig(available_slots=1))
    assert [l.candidate_id for l in launches if l.status == "launched"] == [replay.candidate_id]


def test_selector_best_route_row_prefers_current_gpu_recent_value_over_stale_status():
    from trex.schemas import RouteValueSummary

    replay = _cand("route_replay_t_00_complexa_beam", "complexa_beam")
    alt = _cand("cand_alt_bindcraft", "bindcraft")
    e = dataclasses.replace(
        _evidence("productive_duplicate"),
        route_values=[
            RouteValueSummary(
                strategy_key="route::complexa_beam:op:default",
                scope="route", family="complexa_beam", root_family="complexa_beam", action_family="complexa_beam", scoring_family=None,
                operator_id="op", config_signature="default",
                route_gpu_h=10.0, new_su=20, new_su_recent=0, new_su_recent_gpu=0,
                new_su_per_route_gpu_h=10.0, gpu_recent_route_gpu_h=3.0,
                gpu_recent_new_su_per_route_gpu_h=0.0,
                strict_count=100, strict_per_su=10.0, duplicate_bin_fraction=0.85,
                status="collapse_risk", marginal_status="dry_duplicate",
            ),
            RouteValueSummary(
                strategy_key="route::complexa_beam:op:default",
                scope="route", family="complexa_beam", root_family="complexa_beam", action_family="complexa_beam", scoring_family=None,
                operator_id="op", config_signature="default",
                route_gpu_h=1.0, new_su=2, new_su_recent=1, new_su_recent_gpu=1,
                new_su_per_route_gpu_h=2.0, gpu_recent_route_gpu_h=0.5,
                gpu_recent_new_su_per_route_gpu_h=2.0,
                strict_count=2, strict_per_su=1.0, duplicate_bin_fraction=0.0,
                status="healthy", marginal_status="productive",
            ),
        ],
    )
    sup = _sup(
        {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        [_sup_decision(replay.candidate_id, "exploit", 1), _sup_decision(alt.candidate_id, "exploit", 2)],
    )

    launches, dbg = select_launches(e, [replay, alt], sup, cfg=SelectorConfig(available_slots=1))
    assert [l.candidate_id for l in launches if l.status == "launched"] == [replay.candidate_id]
    assert replay.candidate_id not in dbg["route_deferred_candidate_ids"]
    assert not any("marginal_diversity_override" in x for x in dbg["redistribute_log"])


def test_productive_wall_momentum_guard_preserves_exploit_slots_when_su_is_still_arriving():
    from trex.schemas import RouteValueSummary

    e1 = _cand("cand_exploit_beam_a", "complexa_beam")
    e2 = _cand("cand_exploit_beam_b", "complexa_beam")
    rescue = _cand("cand_rescue_mpnn", "proteinmpnn_redesign")
    explore = _cand("cand_explore_boltz", "boltzgen")
    ev = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=8,
        run_su_count_delta=2,
        run_su_hwm=8,
        run_su_hwm_delta=2,
        su_per_gpu_h_recent=1.2,
        gpu_h_since_last_su=0.4,
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=2.0, new_su=8, new_su_recent_gpu=2,
            gpu_recent_route_gpu_h=1.0, gpu_recent_new_su_per_route_gpu_h=2.0,
            new_su_per_route_gpu_h=4.0, status="promote", marginal_status="productive",
        )],
    )
    sup = _sup(
        {"exploit": 0.05, "rescue": 0.60, "explore": 0.35},
        [
            _sup_decision(e1.candidate_id, "exploit", 1),
            _sup_decision(e2.candidate_id, "exploit", 2),
            _sup_decision(rescue.candidate_id, "rescue", 1),
            _sup_decision(explore.candidate_id, "explore", 1),
        ],
    )

    launches, dbg = select_launches(
        ev, [e1, e2, rescue, explore], sup, cfg=SelectorConfig(available_slots=3)
    )
    launched_modes = [l.resource_class_concrete.get("mode") for l in launches if l.status == "launched"]

    assert launched_modes.count("exploit") == 2
    assert any("productive_wall_momentum_guard" in x for x in dbg["redistribute_log"])


def test_productive_wall_momentum_guard_respects_overspent_exploit_credit():
    from trex.schemas import RouteValueSummary

    exploit = _cand("cand_exploit_beam", "complexa_beam")
    rescue = _cand("cand_rescue_mpnn", "proteinmpnn_redesign")
    explore = _cand("cand_explore_boltz", "boltzgen")
    ev = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=8,
        run_su_count_delta=2,
        run_su_hwm=8,
        run_su_hwm_delta=2,
        su_per_gpu_h_recent=1.2,
        gpu_h_since_last_su=0.4,
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=2.0, new_su=8, new_su_recent_gpu=2,
            gpu_recent_route_gpu_h=1.0, gpu_recent_new_su_per_route_gpu_h=2.0,
            new_su_per_route_gpu_h=4.0, status="promote", marginal_status="productive",
        )],
    )
    sup = _sup(
        {"exploit": 0.05, "rescue": 0.60, "explore": 0.35},
        [
            _sup_decision(exploit.candidate_id, "exploit", 1),
            _sup_decision(rescue.candidate_id, "rescue", 1),
            _sup_decision(explore.candidate_id, "explore", 1),
        ],
    )

    launches, dbg = select_launches(
        ev,
        [exploit, rescue, explore],
        sup,
        cfg=SelectorConfig(available_slots=1),
        mode_credit={"exploit": -3.0, "rescue": 3.0, "explore": 3.0},
    )
    launched = [l for l in launches if l.status == "launched"]

    assert [l.candidate_id for l in launched] == [rescue.candidate_id]
    assert dbg["quotas_raw"] == {"exploit": 0, "rescue": 1, "explore": 0}
    assert dbg["quotas_final"] == dbg["quotas_raw"]
    assert any(
        "productive_wall_momentum_guard_skipped:exploit_credit=" in reason
        for reason in dbg["redistribute_log"]
    )


def test_specific_probe_is_not_undone_by_productive_momentum():
    from trex.schemas import RouteValueSummary

    exploit = _cand("cand_exploit_beam", "complexa_beam")
    explore = _cand("cand_explore_bindcraft", "bindcraft", cost="extended")
    ev = dataclasses.replace(
        _evidence("productive"),
        run_su_count=1,
        run_su_count_delta=1,
        run_su_hwm=1,
        run_su_hwm_delta=1,
        su_per_gpu_h_recent=1.0,
        gpu_h_since_last_su=0.2,
        execution_realization={
            "by_family": {
                "bindcraft": {
                    "proposed": 18,
                    "selected": 3,
                    "started": 1,
                    "dispatch_deferred": 0,
                    "selected_not_started": 2,
                },
            },
            "under_started_families": ["bindcraft"],
        },
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=1.0, new_su=1, new_su_recent_gpu=1,
            gpu_recent_route_gpu_h=1.0, gpu_recent_new_su_per_route_gpu_h=1.0,
            new_su_per_route_gpu_h=1.0, status="promote", marginal_status="productive",
        )],
    )
    sup = _sup(
        {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        [
            _sup_decision(explore.candidate_id, "explore", 1, rc="extended"),
            _sup_decision(exploit.candidate_id, "exploit", 1),
        ],
    )

    launches, dbg = select_launches(
        ev,
        [exploit, explore],
        sup,
        cfg=SelectorConfig(available_slots=1, realize_repeated_support_probe=True),
        mode_credit={"exploit": 0.0, "rescue": 0.0, "explore": 0.0},
    )

    launched = [l for l in launches if l.status == "launched"]
    assert [(l.candidate_id, l.resource_class_concrete["mode"]) for l in launched] == [
        (explore.candidate_id, "explore"),
    ]
    assert dbg["forced_repeated_support_probe"]["candidate_id"] == explore.candidate_id
    assert dbg["quotas_final"] == {"exploit": 0, "rescue": 0, "explore": 1}


def test_productive_single_slot_ticks_track_llm_mixture_longitudinally():
    from trex.schemas import RouteValueSummary

    candidates = [
        _cand("cand_exploit_beam", "complexa_beam"),
        _cand("cand_rescue_mpnn", "proteinmpnn_redesign"),
        _cand("cand_explore_boltz", "boltzgen"),
    ]
    mixture = {"exploit": 0.50, "rescue": 0.35, "explore": 0.15}
    sup = _sup(mixture, [
        _sup_decision(candidates[0].candidate_id, "exploit", 1),
        _sup_decision(candidates[1].candidate_id, "rescue", 1),
        _sup_decision(candidates[2].candidate_id, "explore", 1),
    ])
    ev = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=8,
        run_su_count_delta=2,
        run_su_hwm=8,
        run_su_hwm_delta=2,
        su_per_gpu_h_recent=1.2,
        gpu_h_since_last_su=0.4,
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=2.0, new_su=8, new_su_recent_gpu=2,
            gpu_recent_route_gpu_h=1.0, gpu_recent_new_su_per_route_gpu_h=2.0,
            new_su_per_route_gpu_h=4.0, status="promote", marginal_status="productive",
        )],
    )
    credit = {mode: 0.0 for mode in mixture}
    counts = {mode: 0 for mode in mixture}

    for tick in range(30):
        launches, _dbg = select_launches(
            ev, candidates, sup, cfg=SelectorConfig(available_slots=1),
            tick_id=f"t{tick}", mode_credit=credit,
        )
        launched = next(l for l in launches if l.status == "launched")
        mode = launched.resource_class_concrete["mode"]
        counts[mode] += 1
        for key, share in mixture.items():
            credit[key] = max(-3.0, min(3.0, credit[key] + share))
        credit[mode] = max(-3.0, min(3.0, credit[mode] - 1.0))

    for mode, share in mixture.items():
        assert abs(counts[mode] - 30 * share) <= 1.5, (counts, credit)


def test_productive_wall_momentum_guard_does_not_expand_high_cost_route_from_target_rate():
    from trex.schemas import RouteValueSummary

    bc1 = _cand("cand_exploit_bc_a", "bindcraft", cost="extended")
    bc2 = _cand("cand_exploit_bc_b", "bindcraft", cost="extended")
    rescue = _cand("cand_rescue_mpnn", "proteinmpnn_redesign")
    explore = _cand("cand_explore_boltz", "boltzgen")
    ev = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=8,
        run_su_count_delta=2,
        run_su_hwm=8,
        run_su_hwm_delta=2,
        su_per_gpu_h_recent=1.2,
        gpu_h_since_last_su=0.4,
        route_values=[RouteValueSummary(
            strategy_key="route::bindcraft:op:default",
            scope="route", family="bindcraft", root_family="bindcraft",
            action_family="bindcraft", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=4.0, new_su=2, new_su_recent_gpu=2,
            gpu_recent_route_gpu_h=2.0, gpu_recent_new_su_per_route_gpu_h=1.0,
            new_su_per_route_gpu_h=0.5, status="promote", marginal_status="productive",
        )],
    )
    sup = _sup(
        {"exploit": 0.05, "rescue": 0.60, "explore": 0.35},
        [
            _sup_decision(bc1.candidate_id, "exploit", 1, rc="extended"),
            _sup_decision(bc2.candidate_id, "exploit", 2, rc="extended"),
            _sup_decision(rescue.candidate_id, "rescue", 1),
            _sup_decision(explore.candidate_id, "explore", 1),
        ],
    )

    launches, dbg = select_launches(
        ev, [bc1, bc2, rescue, explore], sup, cfg=SelectorConfig(available_slots=3)
    )
    launched_modes = [l.resource_class_concrete.get("mode") for l in launches if l.status == "launched"]

    assert launched_modes.count("exploit") == 1
    assert not any("target=2" in x for x in dbg["redistribute_log"])


def test_productive_wall_momentum_guard_does_not_protect_long_dry_route():
    from trex.schemas import RouteValueSummary

    exploit = _cand("cand_exploit_beam", "complexa_beam")
    rescue = _cand("cand_rescue_mpnn", "proteinmpnn_redesign")
    explore = _cand("cand_explore_boltz", "boltzgen")
    ev = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=8,
        run_su_count_delta=0,
        run_su_hwm=8,
        run_su_hwm_delta=0,
        su_per_gpu_h_recent=0.0,
        gpu_h_since_last_su=8.0,
        route_values=[RouteValueSummary(
            strategy_key="route::complexa_beam:op:default",
            scope="route", family="complexa_beam", root_family="complexa_beam",
            action_family="complexa_beam", scoring_family=None,
            operator_id="op", config_signature="default",
            route_gpu_h=2.0, new_su=8, new_su_recent_gpu=0,
            gpu_recent_route_gpu_h=3.0, gpu_recent_new_su_per_route_gpu_h=0.0,
            new_su_per_route_gpu_h=4.0, status="diversify", marginal_status="dry_duplicate",
        )],
    )
    sup = _sup(
        {"exploit": 0.0, "rescue": 0.55, "explore": 0.45},
        [
            _sup_decision(exploit.candidate_id, "exploit", 1),
            _sup_decision(rescue.candidate_id, "rescue", 1),
            _sup_decision(explore.candidate_id, "explore", 1),
        ],
    )

    launches, dbg = select_launches(
        ev, [exploit, rescue, explore], sup, cfg=SelectorConfig(available_slots=2)
    )
    launched_ids = [l.candidate_id for l in launches if l.status == "launched"]

    assert exploit.candidate_id not in launched_ids
    assert not any("productive_wall_momentum_guard" in x for x in dbg["redistribute_log"])
