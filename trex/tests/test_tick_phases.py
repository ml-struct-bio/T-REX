"""Direct contracts for the typed phases composing a scientific live tick."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from trex.candidate_builder import BuilderConfig
from trex.critic import CriticCallConfig
from trex.lifecycle import LifecycleConfig
from trex.planner import PlannerCallConfig
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    HypothesisCard,
    LLMHealthSummary,
    LaunchDecision,
    PlannerOutput,
    PredictedChange,
    PreserveConstraint,
    ResultRecord,
    RouteHealthSummary,
    SupervisorOutput,
    TargetConstraint,
)
from trex.selector import SelectorConfig
from trex.supervisor import SupervisorCallConfig
from trex.tick import (
    CandidatePhaseRequest,
    EvidenceAssemblyRequest,
    FoldseekConfig,
    LifecyclePhaseRequest,
    LiveTickSummaryRequest,
    PrePlannerLifecycleRequest,
    ProposalPhaseDependencies,
    ProposalPhaseRequest,
    ResultClusteringRequest,
    SelectionPhaseRequest,
    SequenceDedupConfig,
    SupervisionPhaseDependencies,
    SupervisionPhaseRequest,
    assemble_tick_evidence,
    build_candidate_phase,
    build_evidence_only_summary,
    build_live_tick_summary,
    cluster_evidence_results,
    retire_expired_hypotheses,
    run_proposal_phase,
    select_live_tick_launches,
    run_supervision_phase,
    update_lifecycle_phase,
)


def _feasibility(ok: bool = True) -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=ok,
        runtime_bucket_id="runtime-v1",
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )


def _candidate(
    candidate_id: str,
    method_family: str = "complexa_beam",
    *,
    hypothesis_ids: list[str] | None = None,
    feasible: bool = True,
    parent_result_id: str | None = None,
    baseline_result_id: str | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        hypothesis_ids=hypothesis_ids or ["hypothesis-1"],
        parent_result_id=parent_result_id,
        method_family=method_family,
        operator_id=f"{method_family}_default",
        lane_id=method_family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="Generate evidence.",
        evidence_refs=["evidence-1"],
        feasibility=_feasibility(feasible),
        baseline_result_id=baseline_result_id,
    )


def _result(
    result_id: str,
    *,
    parent_ids: list[str] | None = None,
    backend_family: str = "complexa_beam",
    i_pae: float = 0.7,
    binder_sc_rmsd: float = 1.5,
    artifacts: dict[str, str] | None = None,
    bins: dict[str, str] | None = None,
) -> ResultRecord:
    metrics = {
        "pLDDT": 92.0,
        "iPAE": i_pae,
        "binder_scRMSD": binder_sc_rmsd,
    }
    return ResultRecord(
        result_id=result_id,
        parent_ids=parent_ids or [],
        target_id="target-1",
        backend_family=backend_family,
        runtime_bucket_id="runtime-v1",
        metrics=metrics,
        metrics_calibrated=dict(metrics),
        route_lineage=[],
        gpu_h=0.5,
        exit_status="ok",
        artifacts=artifacts or {},
        bins=bins or {},
    )


def _hypothesis(hypothesis_id: str = "hypothesis-1") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        target_id="target-1",
        tick_created=0,
        claim="Canonical refiltering improves iPAE.",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["baseline"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["baseline"], 0.2, None)
        ],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["complexa_beam"],
    )


def _supervisor_output() -> SupervisorOutput:
    return SupervisorOutput(
        valid=True,
        abstain=False,
        confidence=0.8,
        fail_reason=None,
        mode_mixture={"exploit": 0.4, "rescue": 0.4, "explore": 0.2},
        candidate_decisions=[],
        rationale="Use current evidence.",
        raw_text="{}",
        usage={},
    )


def test_candidate_phase_exposes_feasibility_and_refilter_lineage(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "parent.cif"
    structure_path.write_text("data_parent\n")
    source = _result("source", artifacts={"cif_path": str(structure_path)})
    refilter = _result(
        "refiltered",
        parent_ids=["chain-1", "source"],
        backend_family="structure_refilter",
        bins={"refilter_source": "source"},
    )
    feasible = _candidate("new-feasible")
    infeasible = _candidate("new-infeasible", feasible=False)
    pending_chain = _candidate("chain_pending", "structure_refilter")
    archived_chain = _candidate("chain_archived", "structure_refilter")
    prior_launch = LaunchDecision(
        launch_id="launch-1",
        tick_id="tick-1",
        candidate_id="chain_archived",
        status="launched",
        resource_class_concrete={},
        why="test",
    )

    with patch(
        "trex.tick.candidates.build_candidates",
        return_value=[feasible, infeasible],
    ) as build_candidates:
        result = build_candidate_phase(CandidatePhaseRequest(
            hypotheses=(_hypothesis(),),
            evidence=MagicMock(),
            builder_config=BuilderConfig(),
            include_warmstart=False,
            warmstart_completed_families=frozenset({"bindcraft"}),
            results=(source, refilter),
            existing_actions=(pending_chain, archived_chain),
            prior_launches=(prior_launch,),
            spawning_actions={},
        ))

    assert result.parent_structure_available
    assert result.structure_refilter_result_ids == frozenset({"refiltered"})
    assert result.structure_refilter_scored_source_ids == frozenset({"source"})
    assert result.feasible_candidates == (feasible,)
    assert result.infeasible_candidate_count == 1
    assert result.pending_chain_candidates == (pending_chain,)
    assert build_candidates.call_args.kwargs["parent_pdb_available"] is True


def test_selection_phase_derives_mode_hints_and_backlog_saturation() -> None:
    hypothesis = _hypothesis()
    candidate = _candidate("candidate-1")
    evidence_fallback = _candidate(
        "evidence_fallback_1", hypothesis_ids=[]
    )
    launch = LaunchDecision(
        launch_id="launch-1",
        tick_id="tick-2",
        candidate_id="candidate-1",
        status="launched",
        resource_class_concrete={"mode": "rescue"},
        why="test",
    )
    selector_debug = {
        "source": "supervisor",
        "raw_mixture": {},
        "clamped_mixture": {},
        "clamp_log": [],
        "quotas_final": {},
    }

    with patch(
        "trex.tick.selection.select_launches",
        return_value=([launch], selector_debug),
    ) as select_launches:
        result = select_live_tick_launches(SelectionPhaseRequest(
            evidence=MagicMock(),
            candidates=(candidate, evidence_fallback),
            supervisor_output=_supervisor_output(),
            selector_config=SelectorConfig(),
            tick_id="tick-2",
            fallback_reason=None,
            hypotheses=(hypothesis,),
            recent_realized_modes=("exploit",),
            mode_credit={"rescue": 0.5},
            recent_started_families=("bindcraft",),
            route_health=RouteHealthSummary(
                raw_routed=20,
                score_files_completed=1,
                backlog_used=19,
                backlog_cap=20,
                near_miss_conversion=None,
                strict_conversion=None,
                panel_ready_conversion=None,
            ),
        ))

    assert result.launches == (launch,)
    assert result.route_backlog_saturated
    assert result.candidate_mode_hints == {
        "candidate-1": "rescue",
        "evidence_fallback_1": "explore",
    }
    assert select_launches.call_args.kwargs["route_backlog_saturated"] is True
    assert select_launches.call_args.kwargs["recent_modes"] == ["exploit"]


def test_proposal_phase_preserves_low_confidence_cards_and_audit_inputs() -> None:
    hypothesis = _hypothesis()
    planner_output = PlannerOutput(
        valid=True,
        abstain=False,
        confidence=0.4,
        fail_reason="low_confidence(0.400)",
        cards=[hypothesis],
        rationale="Rescue evidence is still actionable.",
        raw_text="{}",
        usage={},
    )
    call_planner = MagicMock(return_value=planner_output)
    build_llm_record = MagicMock(return_value="planner-audit-record")
    build_prompt_audit = MagicMock(return_value={"audit": "ok"})
    archive = SimpleNamespace(iter_records=lambda record_type: iter(()))
    evidence = SimpleNamespace(state_label="low_evidence", recipes=[])
    registry = SimpleNamespace(
        feasible_families=lambda: ["bindcraft", "boltzgen"]
    )
    clock_values = iter((10.0, 12.5))

    with patch("trex.tick.proposal.default_registry", return_value=registry), patch(
        "trex.tick.proposal.build_user_prompt", return_value="planner payload"
    ):
        result = run_proposal_phase(
            ProposalPhaseRequest(
                archive=archive,  # type: ignore[arg-type]
                evidence=evidence,  # type: ignore[arg-type]
                active_hypotheses=(),
                warmstart_missing_families=frozenset({"bindcraft"}),
                warmstart_seen=False,
                unavailable_families=frozenset({"boltzgen"}),
                planner_config=PlannerCallConfig(),
                critic_config=CriticCallConfig(enabled=False),
                tick_id="tick-2",
                tick_number=2,
                reuse_prior_plan=False,
                bootstrap_seed_action_families=("bindcraft", "boltzgen"),
                schema_version="test-schema",
            ),
            ProposalPhaseDependencies(
                call_planner=call_planner,
                build_llm_call_record=build_llm_record,
                build_prompt_audit_snapshot=build_prompt_audit,
            ),
            clock=lambda: next(clock_values),
        )

    assert result.available_families == ("bindcraft",)
    assert result.seed_action_families == ("bindcraft",)
    assert result.planner_latency_seconds == 2.5
    assert result.hypotheses == (hypothesis,)
    assert result.fallback_reason == "low_confidence(0.400)"
    assert result.records_to_append == ("planner-audit-record", hypothesis)
    assert call_planner.call_args.kwargs["available_families"] == ["bindcraft"]
    assert build_prompt_audit.call_args.kwargs["seed_action_families"] == [
        "bindcraft"
    ]


def test_supervision_phase_reuses_original_confidence_without_llm_call() -> None:
    hypothesis = _hypothesis()
    candidate = _candidate("candidate-1")
    planner_output = PlannerOutput(
        valid=True,
        abstain=False,
        confidence=0.4,
        fail_reason="low_confidence(0.400)",
        cards=[hypothesis],
        rationale="test",
        raw_text="{}",
        usage={},
    )
    call_supervisor = MagicMock()
    build_llm_record = MagicMock(return_value="supervisor-audit-record")
    selector_context = {"quota_realization": "fractional_carry"}

    with patch(
        "trex.tick.supervision.build_supervisor_selector_context",
        return_value=selector_context,
    ), patch(
        "trex.tick.supervision.build_user_prompt",
        return_value="supervisor payload",
    ):
        result = run_supervision_phase(
            SupervisionPhaseRequest(
                evidence=MagicMock(),
                hypotheses=(hypothesis,),
                candidates=(candidate,),
                planner_output=planner_output,
                planner_fallback_reason="low_confidence(0.400)",
                selector_config=SelectorConfig(),
                supervisor_config=SupervisorCallConfig(model="test-model"),
                recent_realized_modes=("exploit",),
                mode_credit={"rescue": 0.5},
                diagnostic_chain_backlog={},
                pending_family_load={},
                execution_realization={},
                tick_id="tick-2",
                reuse_prior_plan=True,
                reuse_mode_mixture={
                    "exploit": 0.2,
                    "rescue": 0.6,
                    "explore": 0.2,
                },
                reuse_confidence=0.35,
                use_unified_reasoner=False,
            ),
            SupervisionPhaseDependencies(
                call_supervisor=call_supervisor,
                build_llm_call_record=build_llm_record,
                build_prompt_audit_snapshot=MagicMock(return_value={}),
            ),
        )

    assert result.supervisor_output.confidence == 0.35
    assert result.supervisor_output.mode_mixture["rescue"] == 0.6
    assert result.fallback_reason is None
    assert result.selector_context == selector_context
    assert result.llm_call_record == "supervisor-audit-record"
    call_supervisor.assert_not_called()


def test_lifecycle_phase_returns_records_without_archive_side_effects() -> None:
    hypothesis = _hypothesis()
    baseline = _result("baseline", i_pae=0.7)
    candidate = _candidate(
        "candidate-1", baseline_result_id="baseline"
    )
    descendants = (
        _result("descendant-1", parent_ids=["candidate-1"], i_pae=0.3),
        _result("descendant-2", parent_ids=["candidate-1"], i_pae=0.25),
    )

    result = update_lifecycle_phase(LifecyclePhaseRequest(
        hypotheses=(hypothesis,),
        actions=(candidate,),
        results=(baseline, *descendants),
        spawning_actions={},
        current_tick=2,
        lifecycle_config=LifecycleConfig(),
    ))

    assert len(result.updated_hypotheses) == 1
    assert result.updated_hypotheses[0].status == "supported"
    assert result.update_summaries[0]["hypothesis_id"] == "hypothesis-1"
    assert result.update_summaries[0]["n_descendants"] == 2


def test_evidence_phases_keep_unclustered_strict_result_out_of_official_su() -> None:
    strict_result = _result(
        "strict-result",
        i_pae=0.2,
        binder_sc_rmsd=1.4,
    )
    target = TargetConstraint(target_id="target-1", target_class="test")
    clustering = cluster_evidence_results(ResultClusteringRequest(
        results=[strict_result],
        target=target,
        spawning_actions={},
        previous_evidence=None,
        tick_id_int=1,
        window_size=60,
        foldseek_config=FoldseekConfig(enabled=False),
        sequence_config=SequenceDedupConfig(enabled=False),
    ))
    evidence = assemble_tick_evidence(EvidenceAssemblyRequest(
        target=target,
        tick_id="tick-1",
        tick_id_int=1,
        elapsed_wall_h=1.0,
        remaining_wall_h=47.0,
        pending_children=0,
        inflight_gpu_h=0.0,
        results=[strict_result],
        hypotheses=[],
        spawning_actions={},
        prior_evidence=[],
        clustering=clustering,
        window_size=60,
        worker_wall_gpu_count=3,
        planner_model="test-model",
        enable_exemplars=False,
        route_health_summary=RouteHealthSummary(
            raw_routed=0,
            score_files_completed=0,
            backlog_used=0,
            backlog_cap=4,
            near_miss_conversion=None,
            strict_conversion=None,
            panel_ready_conversion=None,
        ),
        recent_fallback_rate=0.0,
        charged_gpu_count=None,
        charged_gpu_h_total=None,
        charged_gpu_h_recent=None,
        charged_gpu_h_scope="unavailable",
        recent_ticks_history=[],
        diagnostic_chain_backlog={},
        llm_health=LLMHealthSummary(
            model="test-model",
            last_calls_window=[],
            parse_fail_rate=0.0,
            schema_fail_rate=0.0,
            timeout_count=0,
            median_latency_s=0.0,
        ),
        pending_family_load={},
        dispatch_realization={},
        execution_realization={},
    ))

    assert clustering.strict_result_ids == frozenset({"strict-result"})
    assert clustering.foldseek_su_status == "disabled"
    assert "foldseek_su" not in strict_result.bins
    assert evidence.strict_total == 1
    assert evidence.evidence.strict_count == 1
    assert evidence.evidence.run_su_count == 0
    assert (
        evidence.evidence.production_panel_status
        == "degraded_untrusted_structure_dedup"
    )


def test_preplanner_ttl_phase_is_pure_and_evidence_only_is_non_mutating() -> None:
    expired = _hypothesis("expired")
    lifecycle_config = LifecycleConfig()
    normal = retire_expired_hypotheses(PrePlannerLifecycleRequest(
        active_hypotheses=(expired,),
        all_hypotheses=(expired,),
        actions=(),
        results=(),
        current_tick=expired.ttl_ticks,
        lifecycle_config=lifecycle_config,
    ))
    monitoring = retire_expired_hypotheses(PrePlannerLifecycleRequest(
        active_hypotheses=(expired,),
        all_hypotheses=(expired,),
        actions=(),
        results=(),
        current_tick=expired.ttl_ticks,
        lifecycle_config=lifecycle_config,
        evidence_only=True,
    ))

    assert normal.active_hypotheses == ()
    assert len(normal.retired_hypotheses) == 1
    assert normal.retired_hypotheses[0].status == "retired"
    assert expired.status == "active"
    assert monitoring.active_hypotheses == (expired,)
    assert monitoring.all_hypotheses == (expired,)
    assert monitoring.retired_hypotheses == ()


def test_summary_phase_preserves_monitoring_and_full_tick_contracts() -> None:
    evidence = SimpleNamespace(
        state_label="stalled",
        elapsed_wall_h=2.0,
        remaining_wall_h=46.0,
        worker_wall_gpu_h_total=6.0,
        run_su_per_worker_wall_gpu_h_total=0.5,
        run_su_count=3,
        run_su_count_delta=1,
        gpu_h_since_last_su=0.0,
        top_bin_share=0.4,
        duplicate_fraction=0.2,
        near_miss_count=2,
        production_panel_status="ok",
        production_panel_selected_ids=["result-1"],
        production_panel_value=1.5,
        strict_su_tm08_recent_count=None,
        strict_su_live_recent_count=3,
        strict_su_tm05_recent_count=None,
        strict_su_tm08_live_split_ratio=None,
        strict_su_tm08_split_ratio=None,
        recipes=[],
        recent_fallback_high=False,
        axis_stats={},
    )
    evidence_only = build_evidence_only_summary(
        tick_id="tick-2",
        target_id="target-1",
        evidence=evidence,  # type: ignore[arg-type]
        all_result_count=8,
        window_result_count=4,
        strict_total=3,
        strict_window=1,
    )
    candidate = _candidate("candidate-1")
    planner_output = PlannerOutput(
        valid=True,
        abstain=False,
        confidence=0.8,
        fail_reason=None,
        cards=[_hypothesis()],
        rationale="test",
        raw_text="{}",
        usage={},
    )
    launch = LaunchDecision(
        launch_id="launch-1",
        tick_id="tick-2",
        candidate_id="candidate-1",
        status="launched",
        resource_class_concrete={},
        why="test",
    )
    full_summary = build_live_tick_summary(LiveTickSummaryRequest(
        tick_id="tick-2",
        target_id="target-1",
        evidence=evidence,  # type: ignore[arg-type]
        all_result_count=8,
        window_result_count=4,
        strict_total=3,
        strict_window=1,
        critic_enabled=True,
        critic_flags=("flag-1",),
        planner_output=planner_output,
        planner_latency_seconds=1.234,
        candidates=(candidate,),
        supervisor_output=_supervisor_output(),
        supervisor_latency_seconds=2.345,
        selector_debug={
            "source": "supervisor",
            "raw_mixture": {},
            "clamped_mixture": {},
            "clamp_log": [],
            "quotas_final": {},
        },
        launches=(launch,),
        lifecycle_updates=(),
        fallback_reason=None,
    ))

    assert evidence_only["evidence_only"] is True
    assert evidence_only["evidence"]["run_su_count"] == 3
    assert full_summary["evidence"]["n_all_results"] == 8
    assert full_summary["planner"]["latency_s"] == 1.23
    assert full_summary["supervisor"]["latency_s"] == 2.35
    assert full_summary["selector"]["n_launched"] == 1
    assert full_summary["fallback_used"] is False
