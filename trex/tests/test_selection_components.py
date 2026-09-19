"""Direct contracts for deterministic Selector policy and admission."""

from __future__ import annotations

from types import SimpleNamespace

from trex.schemas import (
    ActionCandidate,
    CandidateDecision,
    FeasibilityCheck,
    SupervisorOutput,
)
from trex.selection import (
    BatchDiversityPolicy,
    CandidateAdmissionRequest,
    ModeQuotaRequest,
    PlannedEvidenceSnapshot,
    QuotaCandidateContext,
    RankingCandidateContext,
    SelectionEmissionRequest,
    SelectionRankingRequest,
    SelectionPolicyRequest,
    admit_candidates,
    emit_selection_decisions,
    rank_candidates,
    realize_mode_quotas,
    resolve_batch_diversity_policy,
    resolve_selection_policy,
)


def _candidate(
    candidate_id: str,
    family: str,
    *,
    feasible: bool = True,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        hypothesis_ids=[],
        parent_result_id=None,
        method_family=family,
        operator_id=f"{family}_default",
        lane_id=family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="test",
        evidence_refs=[],
        feasibility=FeasibilityCheck(
            backend_healthy=feasible,
            runtime_bucket_id="runtime-v1",
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
        ),
    )


def _supervisor(
    decisions: list[CandidateDecision],
) -> SupervisorOutput:
    return SupervisorOutput(
        valid=True,
        abstain=False,
        confidence=0.9,
        fail_reason=None,
        mode_mixture={"exploit": 0.2, "rescue": 0.6, "explore": 0.2},
        candidate_decisions=decisions,
        rationale="test",
        raw_text="{}",
        usage={},
    )


def _evidence() -> SimpleNamespace:
    return SimpleNamespace(
        state_label="productive",
        top_bin_share=None,
        strict_su_top_bin_share=None,
        panel_ready_bins_covered=0,
        recent_fallback_high=False,
    )


def test_policy_resolves_supervisor_priority_then_unranked_hint() -> None:
    ranked = _candidate("ranked", "complexa_beam")
    unranked = _candidate("unranked", "boltzgen")
    decision = CandidateDecision(
        candidate_id=ranked.candidate_id,
        mode="rescue",
        rank_in_mode=1,
        global_rank=1,
        resource_class="standard",
        what="run ranked",
        why="test",
        evidence_refs=[],
        expected_signal="test",
        stop_or_downgrade_if="test",
    )

    result = resolve_selection_policy(SelectionPolicyRequest(
        evidence=_evidence(),  # type: ignore[arg-type]
        candidates=(ranked, unranked),
        supervisor_output=_supervisor([decision]),
        fallback_reason=None,
        candidate_mode_hints={unranked.candidate_id: "explore"},
        derive_mixture_from_ranking=False,
        realize_global_priority=True,
        all_explore_backends_unhealthy=False,
        route_backlog_saturated=False,
        panel_live=False,
    ))

    assert result.source == "supervisor_llm_scalar"
    assert result.use_supervisor_ranks
    assert result.global_priority_used
    assert result.mode_for(ranked) == "rescue"
    assert result.mode_for(unranked) == "explore"
    assert result.has_mode_information
    assert result.clamp_log[0].startswith("clamp:cat_b_off")


def test_policy_fallback_uses_mode_hints_without_supervisor_priority() -> None:
    candidate = _candidate("fallback", "boltzgen")
    result = resolve_selection_policy(SelectionPolicyRequest(
        evidence=_evidence(),  # type: ignore[arg-type]
        candidates=(candidate,),
        supervisor_output=_supervisor([]),
        fallback_reason="parse_fail",
        candidate_mode_hints={candidate.candidate_id: "explore"},
        derive_mixture_from_ranking=False,
        realize_global_priority=True,
        all_explore_backends_unhealthy=False,
        route_backlog_saturated=False,
        panel_live=False,
    ))

    assert result.source == "fallback(parse_fail)"
    assert not result.supervisor_used
    assert not result.use_supervisor_ranks
    assert not result.global_priority_used
    assert result.mode_for(candidate) == "explore"


def test_admission_orders_capacity_family_and_route_deferrals() -> None:
    capacity = _candidate("capacity", "bindcraft")
    healthy = _candidate("healthy", "complexa_beam")
    cost = _candidate("cost", "boltzgen")
    route = _candidate("route", "proteinmpnn_redesign")
    route_peer = _candidate("route-peer", "complexa_beam")
    infeasible = _candidate("infeasible", "unknown_family", feasible=False)
    candidates = (capacity, healthy, cost, route, route_peer, infeasible)
    modes = {
        capacity.candidate_id: "exploit",
        healthy.candidate_id: "exploit",
        cost.candidate_id: "explore",
        route.candidate_id: "rescue",
        route_peer.candidate_id: "rescue",
        infeasible.candidate_id: "explore",
    }

    result = admit_candidates(CandidateAdmissionRequest(
        candidates=candidates,
        candidate_modes=modes,
        has_mode_information=True,
        global_priority_used=False,
        category_b_enabled=False,
        family_cost_penalties={"boltzgen": 2},
        capacity_block_reasons={"bindcraft": "running_cap"},
        candidate_route_penalties={"route": 2},
        defer_penalty=2,
    ))

    assert [c.candidate_id for c in result.eligible_candidates] == [
        "healthy",
        "route-peer",
    ]
    assert [c.candidate_id for c in result.capacity_deferred_candidates] == [
        "capacity"
    ]
    assert [c.candidate_id for c in result.cost_deferred_candidates] == ["cost"]
    assert [c.candidate_id for c in result.route_deferred_candidates] == [
        "route"
    ]
    assert result.feasibility_by_mode == {
        "exploit": True,
        "rescue": True,
        "explore": False,
    }

    collapse_result = admit_candidates(CandidateAdmissionRequest(
        candidates=candidates,
        candidate_modes=modes,
        has_mode_information=True,
        global_priority_used=False,
        category_b_enabled=True,
        family_cost_penalties={"boltzgen": 2},
        capacity_block_reasons={"bindcraft": "running_cap"},
        candidate_route_penalties={"route": 2},
        defer_penalty=2,
    ))
    assert "cost" in {
        candidate.candidate_id
        for candidate in collapse_result.eligible_candidates
    }
    assert collapse_result.cost_deferred_candidates == ()
    assert collapse_result.feasibility_by_mode["explore"]


def _quota_candidate(
    candidate_id: str,
    family: str,
    mode: str,
    priority: int,
    *,
    cost_rank: int = 2,
    repeated_probe: bool = False,
    near_miss_rescue: bool = False,
    dry_duplicate_replay: bool = False,
    material_diversity: bool = False,
) -> QuotaCandidateContext:
    return QuotaCandidateContext(
        candidate_id=candidate_id,
        method_family=family,
        mode=mode,
        parent_result_id="parent-1" if near_miss_rescue else None,
        priority=priority,
        effective_cost_rank=cost_rank,
        repeated_support_probe_eligible=repeated_probe,
        low_cost_near_miss_rescue_eligible=near_miss_rescue,
        dry_duplicate_exact_replay=dry_duplicate_replay,
        material_diversity_candidate=material_diversity,
    )


def _quota_request(
    candidates: tuple[QuotaCandidateContext, ...],
    **overrides: object,
) -> ModeQuotaRequest:
    values: dict[str, object] = {
        "mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "available_slots": 1,
        "feasibility_by_mode": {
            "exploit": True,
            "rescue": True,
            "explore": True,
        },
        "candidates": candidates,
        "quota_realization": "largest_remainder",
        "recent_modes": [],
        "effective_window_k": 10,
        "mode_credit": None,
        "mode_credit_cap": 3.0,
        "has_mode_information": True,
        "use_supervisor_ranks": True,
        "global_priority_used": False,
        "productive_wall_momentum_guard": False,
        "productive_wall_target_slots": 0,
        "productive_wall_min_exploit_credit": -1.0,
        "repeated_support_probe_enabled": False,
        "low_cost_near_miss_rescue_floor": False,
        "evidence_allows_near_miss_rescue": False,
        "marginal_diversity_override": False,
    }
    values.update(overrides)
    return ModeQuotaRequest(**values)  # type: ignore[arg-type]


def test_quota_repairs_record_repeated_support_before_near_miss_floor() -> None:
    exploit = _quota_candidate("exploit", "complexa_beam", "exploit", 2)
    explore = _quota_candidate(
        "explore",
        "bindcraft",
        "explore",
        1,
        repeated_probe=True,
    )
    rescue = _quota_candidate(
        "rescue",
        "proteinmpnn_redesign",
        "rescue",
        0,
        cost_rank=0,
        near_miss_rescue=True,
    )

    result = realize_mode_quotas(_quota_request(
        (exploit, explore, rescue),
        repeated_support_probe_enabled=True,
        low_cost_near_miss_rescue_floor=True,
        evidence_allows_near_miss_rescue=True,
    ))

    assert result.raw_quotas == {"exploit": 1, "rescue": 0, "explore": 0}
    assert result.final_quotas == {"exploit": 0, "rescue": 1, "explore": 0}
    assert result.forced_repeated_support_probe == {
        "candidate_id": "explore",
        "family": "bindcraft",
        "mode": "explore",
        "donor_mode": "exploit",
    }
    assert result.forced_near_miss_rescue_floor == {
        "candidate_id": "rescue",
        "family": "proteinmpnn_redesign",
        "parent_result_id": "parent-1",
        "donor_mode": "explore",
        "donor_candidate_id": "explore",
    }
    assert result.redistribution_log[-2:] == (
        "forced_repeated_support_probe:bindcraft:exploit->explore",
        "low_cost_near_miss_rescue_floor:explore->rescue:"
        "proteinmpnn_redesign:parent=parent-1",
    )


def test_quota_marginal_diversity_override_uses_priority_order() -> None:
    exploit = _quota_candidate(
        "stale-replay",
        "complexa_beam",
        "exploit",
        0,
        dry_duplicate_replay=True,
    )
    rescue = _quota_candidate(
        "diverse-rescue",
        "proteinmpnn_redesign",
        "rescue",
        1,
        material_diversity=True,
    )

    result = realize_mode_quotas(_quota_request(
        (rescue, exploit),
        marginal_diversity_override=True,
    ))

    assert result.final_quotas == {"exploit": 0, "rescue": 1, "explore": 0}
    assert result.redistribution_log[-1] == (
        "marginal_diversity_override:complexa_beam->"
        "proteinmpnn_redesign:rescue"
    )


def _ranking_candidate(
    candidate: ActionCandidate,
    mode: str,
    priority: int,
    *,
    rank_in_mode: int = 1,
    global_rank: int | None = None,
) -> RankingCandidateContext:
    return RankingCandidateContext(
        candidate=candidate,
        mode=mode,
        priority=priority,
        rank_in_mode=rank_in_mode,
        global_rank=global_rank,
    )


def test_diversity_policy_reports_recent_overrepresented_family() -> None:
    result = resolve_batch_diversity_policy(
        state_label="productive_duplicate",
        category_b_enabled=True,
        available_slots=3,
        default_max_per_family=2,
        duplicate_fraction=0.8,
        recent_started_families=("bindcraft", "bindcraft", "complexa_beam"),
        duplicate_pressure_fraction=0.75,
        family_share_cap=0.6,
        family_window_k=3,
    )

    assert result.max_per_family == 2
    assert result.overrepresented_families == frozenset({"bindcraft"})


def test_ranking_backfills_cost_then_route_deferred_candidates() -> None:
    primary = _ranking_candidate(
        _candidate("primary", "complexa_beam"),
        "exploit",
        0,
    )
    cost_deferred = _ranking_candidate(
        _candidate("cost", "bindcraft"),
        "rescue",
        1,
    )
    route_deferred = _ranking_candidate(
        _candidate("route", "proteinmpnn_redesign"),
        "explore",
        2,
    )

    result = rank_candidates(SelectionRankingRequest(
        eligible_candidates=(primary,),
        cost_deferred_candidates=(cost_deferred,),
        route_deferred_candidates=(route_deferred,),
        cross_family_escape_candidates=(),
        quotas={"exploit": 3, "rescue": 0, "explore": 0},
        available_slots=3,
        diversity_policy=BatchDiversityPolicy(3, frozenset()),
        high_cost_batch_hard_caps={},
        has_mode_information=True,
        global_priority_used=False,
        top_n_rejected_per_mode=3,
        cross_family_escape_floor_enabled=False,
    ))

    assert [
        candidate.candidate_id
        for candidate in result.selected_candidates
    ] == ["primary", "cost", "route"]
    assert result.selection_modes == {
        "primary": "exploit",
        "cost": "rescue",
        "route": "explore",
    }


def test_ranking_cross_family_floor_replaces_non_rank1_candidate() -> None:
    primary = _ranking_candidate(
        _candidate("primary", "complexa_beam"),
        "exploit",
        0,
        rank_in_mode=2,
    )
    cross_family = _ranking_candidate(
        _candidate(
            "evidence_fallback_cross_family_bindcraft",
            "bindcraft",
        ),
        "explore",
        1,
        rank_in_mode=2,
    )

    result = rank_candidates(SelectionRankingRequest(
        eligible_candidates=(primary,),
        cost_deferred_candidates=(),
        route_deferred_candidates=(),
        cross_family_escape_candidates=(cross_family,),
        quotas={"exploit": 1, "rescue": 0, "explore": 0},
        available_slots=1,
        diversity_policy=BatchDiversityPolicy(1, frozenset()),
        high_cost_batch_hard_caps={},
        has_mode_information=True,
        global_priority_used=False,
        top_n_rejected_per_mode=3,
        cross_family_escape_floor_enabled=True,
    ))

    assert result.selected_candidates == (cross_family.candidate,)
    assert result.forced_cross_family_escape_floor == {
        "target_cross": 1,
        "added": [],
        "replaced": [{
            "out": "primary",
            "in": "evidence_fallback_cross_family_bindcraft",
        }],
    }


def test_emission_records_rejection_and_rank1_non_launch_reason() -> None:
    selected = _candidate("selected", "complexa_beam")
    rejected = _candidate("rejected", "bindcraft")
    rejected_decision = CandidateDecision(
        candidate_id=rejected.candidate_id,
        mode="rescue",
        rank_in_mode=1,
        global_rank=None,
        resource_class="standard",
        what="run rejected",
        why="test",
        evidence_refs=[],
        expected_signal="test",
        stop_or_downgrade_if="test",
    )

    result = emit_selection_decisions(SelectionEmissionRequest(
        tick_id="tick_test",
        candidates=(selected, rejected),
        selected_candidates=(selected,),
        selection_modes={selected.candidate_id: "exploit"},
        rejected_per_mode={
            "exploit": (),
            "rescue": (rejected,),
            "explore": (),
        },
        supervisor_decisions={rejected.candidate_id: rejected_decision},
        quotas={"exploit": 1, "rescue": 0, "explore": 0},
        source="supervisor_llm_scalar",
        fallback_used=False,
        global_priority_used=False,
        evidence_snapshot=PlannedEvidenceSnapshot(
            completed_children=5,
            run_su_count=2,
            state_label="productive",
        ),
        capacity_blocked_families=frozenset(),
        capacity_block_reasons={},
        high_cost_batch_hard_caps={},
        route_deferred_candidate_ids=frozenset(),
        cost_deferred_families=frozenset(),
    ))

    assert [
        (decision.candidate_id, decision.status)
        for decision in result.launch_decisions
    ] == [("selected", "launched"), ("rejected", "rejected")]
    assert result.rank1_not_launched == ({
        "candidate_id": "rejected",
        "family": "bindcraft",
        "mode": "rescue",
        "reason": "not_selected_in_rescue_quota(0)",
    },)
    assert result.launch_modes == {"selected": "exploit"}
