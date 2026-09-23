"""Regression tests for selection-to-dispatch accounting."""

from __future__ import annotations

from trex.archive import Archive
from trex.live_tick import (
    _dispatch_realization_summary,
    _execution_realization_summary,
    _mode_credit_from_history,
    _recent_realized_modes,
    _recent_started_families,
    _route_health_from_archive,
)
from trex.controller import (
    _dispatch_record_metadata,
    _drop_stale_scientific_pending,
    _should_pause_initial_no_evidence_prefetch,
)
from trex.schemas import (
    ActionCandidate,
    CandidateDecision,
    DispatchRecord,
    EvidenceSummary,
    FeasibilityCheck,
    LaunchDecision,
    LLMHealthSummary,
    RouteHealthSummary,
    SupervisorDecision,
    SupervisorOutput,
)
from trex.selector import SelectorConfig, select_launches


def _launch(
    cid: str,
    mode: str,
    launch_id: str | None = None,
    *,
    planned_completed_children: int = 0,
    planned_state_label: str = "low_evidence",
) -> LaunchDecision:
    return LaunchDecision(
        launch_id=launch_id or f"L_{cid}",
        tick_id="v7r001",
        candidate_id=cid,
        status="launched",
        resource_class_concrete={
            "class": "standard",
            "mode": mode,
            "planned_completed_children": planned_completed_children,
            "planned_run_su_count": 0,
            "planned_state_label": planned_state_label,
        },
        why="test",
    )


def _dispatch(cid: str, status: str, launch_id: str | None = None, why: str = "") -> DispatchRecord:
    return DispatchRecord(
        dispatch_id=f"D_{cid}_{status}",
        tick_id="v7r002",
        candidate_id=cid,
        status=status,  # type: ignore[arg-type]
        launch_id=launch_id or f"L_{cid}",
        why=why,
    )


def _feas() -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=True,
        runtime_bucket_id="rb",
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )


def _cand(cid: str, family: str) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=cid,
        hypothesis_ids=["h"],
        parent_result_id=None,
        method_family=family,
        operator_id=f"{family}_op",
        lane_id=family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="test",
        evidence_refs=["ev"],
        feasibility=_feas(),
    )


def _decision(cid: str, rank: int) -> CandidateDecision:
    return CandidateDecision(
        candidate_id=cid,
        mode="explore",
        rank_in_mode=rank,
        global_rank=rank,
        resource_class="standard",
        what="test",
        why="test",
        evidence_refs=["ev"],
        expected_signal="test",
        stop_or_downgrade_if="test",
    )


def test_unselected_proposals_are_not_reported_as_under_started() -> None:
    actions = [_cand(f"bc_{i}", "bindcraft") for i in range(6)]
    supervisors = [SupervisorDecision(
        tick_id="v7r001",
        mode_mixture={"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        candidate_decisions=[
            _decision(c.candidate_id, i) for i, c in enumerate(actions, 1)
        ],
        clamps_applied=[],
        fallback_used=False,
        rationale="test",
    )]
    launches = [_launch(actions[0].candidate_id, "explore")]
    dispatches = [_dispatch(actions[0].candidate_id, "started")]

    summary = _execution_realization_summary(actions, launches, dispatches, supervisors)
    row = summary["by_family"]["bindcraft"]

    assert row["proposal_to_start_gap"] == 5
    assert row["selection_to_start_gap"] == 0
    assert summary["under_started_families"] == []


def test_dispatch_realization_uses_dispatch_metadata_when_action_missing():
    launch = _launch("cand_missing_action", "explore")
    dispatch = DispatchRecord(
        dispatch_id="d_started",
        tick_id="v7r002",
        candidate_id="cand_missing_action",
        status="started",
        launch_id=launch.launch_id,
        method_family="bindcraft",
        supervisor_mode="explore",
        score_credit_basis="requires_canonical_score_conversion",
    )

    summary = _dispatch_realization_summary([], [launch], [dispatch], recent_started_window_k=10)
    assert summary["started_by_family"] == {"bindcraft": 1}
    assert summary["recent_started_families"] == ["bindcraft"]
    assert _recent_started_families([], [dispatch]) == ["bindcraft"]

    exec_summary = _execution_realization_summary([], [launch], [dispatch], [])
    assert exec_summary["by_family"]["bindcraft"]["started"] == 1


def test_route_health_treats_high_cost_defer_as_pending_not_terminal():
    action = _cand("cand_route", "structure_refilter")
    launch = _launch(action.candidate_id, "rescue")
    defer = DispatchRecord(
        dispatch_id="d_defer",
        tick_id="v7r002",
        candidate_id=action.candidate_id,
        status="dispatch_failed",
        launch_id=launch.launch_id,
        method_family="structure_refilter",
        why="high_cost_inflight_cap: cap=1",
    )
    health = _route_health_from_archive([action], [], [launch], [defer], backlog_cap=4)
    assert health.backlog_used == 1
    assert health.raw_routed == 1


def test_route_health_treats_permanent_dispatch_failure_as_terminal():
    action = _cand("cand_route", "structure_refilter")
    launch = _launch(action.candidate_id, "rescue")
    failure = DispatchRecord(
        dispatch_id="d_fail",
        tick_id="v7r002",
        candidate_id=action.candidate_id,
        status="dispatch_failed",
        launch_id=launch.launch_id,
        method_family="structure_refilter",
        why="permanent dispatch skip",
    )
    health = _route_health_from_archive([action], [], [launch], [failure], backlog_cap=4)
    assert health.backlog_used == 0
    assert health.raw_routed == 1


def test_route_health_uses_dispatch_metadata_when_action_missing():
    dispatch = DispatchRecord(
        dispatch_id="d_started",
        tick_id="v7r002",
        candidate_id="cand_score_route",
        status="started",
        method_family="structure_refilter",
    )
    health = _route_health_from_archive(
        [], [], [], [dispatch], backlog_cap=4
    )
    assert health.backlog_used == 1
    assert health.raw_routed == 1


def test_dispatch_record_metadata_preserves_route_role_vocabulary():
    canonical = _cand("chain_1", "structure_refilter")
    canonical = canonical.__class__(
        **{**canonical.__dict__, "refilter_role": "canonical_score_conversion", "supervisor_mode": "rescue"}
    )
    meta = _dispatch_record_metadata(cand=canonical, launch_decision=_launch(canonical.candidate_id, "rescue"))
    assert meta["method_family"] == "structure_refilter"
    assert meta["operator_id"] == "structure_refilter_op"
    assert meta["supervisor_mode"] == "rescue"
    assert meta["refilter_role"] == "canonical_score_conversion"
    assert meta["score_credit_basis"] == "official_score_conversion"

    direct = _cand("cx_1", "complexa_beam")
    meta = _dispatch_record_metadata(cand=direct, launch_decision=_launch(direct.candidate_id, "exploit"))
    assert meta["method_family"] == "complexa_beam"
    assert meta["score_credit_basis"] == "direct_official_strict_metrics"

    diagnostic = _cand("bc_1", "bindcraft")
    meta = _dispatch_record_metadata(cand=diagnostic, launch_decision=_launch(diagnostic.candidate_id, "explore"))
    assert meta["method_family"] == "bindcraft"
    assert meta["score_credit_basis"] == "requires_canonical_score_conversion"


def test_legacy_dispatch_record_without_metadata_still_reconstructs():
    rec = DispatchRecord(
        dispatch_id="d_legacy", tick_id="v7r001", candidate_id="cand", status="started"
    )
    assert rec.method_family is None
    assert rec.score_credit_basis is None


def _evidence() -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="v7r010", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=1.0, remaining_wall_h=1.0, completed_children=10, pending_children=0,
        worker_gpu_h_total=10.0, worker_gpu_h_last_3_ticks=3.0,
        strict_count=8, global_new_strict=2, run_su_count=2, run_su_count_delta=0,
        su_per_gpu_h_recent=0.0, duplicate_fraction=0.9, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 4, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="productive_duplicate",  # type: ignore[arg-type]
        examples=[], metric_availability={},
    )


def _decision(cid: str, rank: int) -> CandidateDecision:
    return CandidateDecision(
        candidate_id=cid,
        mode="exploit",
        rank_in_mode=rank,
        resource_class="standard",  # type: ignore[arg-type]
        what="test",
        why="test",
        evidence_refs=["ev"],
        expected_signal="test",
        stop_or_downgrade_if="test",
    )


def test_recent_modes_use_started_dispatch_not_selected_launches() -> None:
    launches = [_launch("beam", "exploit"), _launch("mcts", "explore")]
    dispatches = [
        _dispatch("beam", "started"),
        _dispatch("mcts", "dispatch_failed", why="stale scientific prefetch"),
    ]

    assert _recent_realized_modes(launches, dispatches) == ["exploit"]
    assert _recent_realized_modes(launches, []) == ["exploit", "explore"]


def test_mode_credit_spends_only_worker_started_modes() -> None:
    launches = [
        _launch("beam", "exploit", launch_id="L_beam"),
        _launch("mpnn", "rescue", launch_id="L_mpnn"),
    ]
    dispatches = [_dispatch("beam", "started", launch_id="L_beam")]
    supervisors = [SupervisorDecision(
        tick_id="v7r001",
        mode_mixture={"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
        candidate_decisions=[],
        clamps_applied=[],
        fallback_used=False,
        rationale="test",
    )]

    credit = _mode_credit_from_history(supervisors, launches, dispatches, cap=3.0)
    assert credit == {"exploit": 0.0, "rescue": 0.6, "explore": 0.4}


def test_dispatch_realization_summary_separates_selected_started_and_stale() -> None:
    actions = [_cand("beam", "complexa_beam"), _cand("mcts", "complexa_mcts")]
    launches = [_launch("beam", "exploit"), _launch("mcts", "explore")]
    dispatches = [
        _dispatch("beam", "started"),
        _dispatch("mcts", "dispatch_failed", why="stale scientific prefetch"),
    ]

    summary = _dispatch_realization_summary(actions, launches, dispatches, recent_started_window_k=10)
    assert summary["selected_by_mode"] == {"exploit": 1, "explore": 1, "rescue": 0}
    assert summary["started_by_mode"] == {"exploit": 1, "explore": 0, "rescue": 0}
    assert summary["stale_dropped_by_mode"] == {"exploit": 0, "explore": 1, "rescue": 0}
    assert summary["selected_not_started_by_mode"] == {"exploit": 0, "explore": 1, "rescue": 0}
    assert summary["started_by_family"] == {"complexa_beam": 1}
    assert summary["stale_dropped_by_family"] == {"complexa_mcts": 1}
    assert summary["selected_not_started_by_family"] == {"complexa_mcts": 1}
    assert summary["recent_started_window_k"] == 10
    assert "window_k" not in summary


def test_dispatch_realization_summary_counts_high_cost_defer() -> None:
    actions = [_cand("bc", "bindcraft")]
    launches = [_launch("bc", "explore")]
    dispatches = [
        _dispatch(
            "bc",
            "dispatch_failed",
            why="high_cost_inflight_cap:bindcraft:active=1 cap=1 state=stalled",
        ),
    ]

    summary = _dispatch_realization_summary(actions, launches, dispatches, recent_started_window_k=10)

    assert summary["dispatch_deferred_by_mode"] == {"exploit": 0, "explore": 1, "rescue": 0}
    assert summary["dispatch_deferred_by_family"] == {"bindcraft": 1}
    assert summary["selected_not_started_by_family"] == {"bindcraft": 1}
    assert summary["dispatch_defer_reason_counts"] == {"high_cost_inflight_cap": 1}


def test_soft_stale_policy_keeps_normal_evidence_updates(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TREX_PREFETCH_STALE_POLICY", raising=False)
    archive = Archive(tmp_path)
    archive.append(_launch(
        "mcts", "explore",
        planned_completed_children=10,
        planned_state_label="stalled",
    ))
    pending = ["mcts"]

    dropped = _drop_stale_scientific_pending(
        archive,
        pending,
        current_completed_children=99,
        current_run_su_count=7,
        current_state_label="productive_duplicate",
        tick_id="v7probe999",
    )

    assert dropped == []
    assert pending == ["mcts"]
    assert list(archive.iter_records(DispatchRecord)) == []


def test_initial_no_evidence_prefetch_is_dropped_even_under_soft_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TREX_PREFETCH_STALE_POLICY", raising=False)
    archive = Archive(tmp_path)
    archive.append(_launch("best_of_n", "explore"))
    pending = ["best_of_n"]

    dropped = _drop_stale_scientific_pending(
        archive,
        pending,
        current_completed_children=1,
        current_run_su_count=0,
        current_state_label="low_evidence",
        tick_id="v7probe002",
    )

    assert dropped == ["best_of_n"]
    assert pending == []
    rec = list(archive.iter_records(DispatchRecord))[0]
    assert rec.status == "dispatch_failed"
    assert "initial_no_evidence_prefetch" in rec.why


def test_initial_no_evidence_prefetch_pause_helper() -> None:
    assert _should_pause_initial_no_evidence_prefetch(
        completed_children=0, busy_count=3, pool_size=3
    )
    assert not _should_pause_initial_no_evidence_prefetch(
        completed_children=1, busy_count=3, pool_size=3
    )
    assert not _should_pause_initial_no_evidence_prefetch(
        completed_children=0, busy_count=2, pool_size=3
    )


def test_hard_stale_policy_preserves_old_count_based_drop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TREX_PREFETCH_STALE_POLICY", "hard")
    archive = Archive(tmp_path)
    archive.append(_launch("mcts", "explore"))
    pending = ["mcts"]

    dropped = _drop_stale_scientific_pending(
        archive,
        pending,
        current_completed_children=99,
        current_run_su_count=7,
        current_state_label="productive_duplicate",
        tick_id="v7probe999",
    )

    assert dropped == ["mcts"]
    assert pending == []
    assert list(archive.iter_records(DispatchRecord))[0].status == "dispatch_failed"


def test_soft_stale_policy_keeps_scientific_pending_on_deep_stall(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TREX_PREFETCH_STALE_POLICY", raising=False)
    archive = Archive(tmp_path)
    archive.append(_launch(
        "beam", "exploit",
        planned_completed_children=10,
        planned_state_label="stalled",
    ))
    archive.append(_launch(
        "mpnn", "rescue",
        planned_completed_children=10,
        planned_state_label="stalled",
    ))
    archive.append(_launch(
        "mcts", "explore",
        planned_completed_children=10,
        planned_state_label="stalled",
    ))
    pending = ["beam", "mpnn", "mcts"]

    dropped = _drop_stale_scientific_pending(
        archive,
        pending,
        current_completed_children=99,
        current_run_su_count=0,
        current_state_label="deep_stall",
        tick_id="v7probe999",
    )

    assert dropped == []
    assert pending == ["beam", "mpnn", "mcts"]
    assert list(archive.iter_records(DispatchRecord)) == []


def test_duplicate_pressure_demotes_overrepresented_family_when_alternative_exists() -> None:
    candidates = [_cand("beam", "complexa_beam"), _cand("mcts", "complexa_mcts")]
    sup = SupervisorOutput(
        valid=True,
        abstain=False,
        confidence=0.9,
        fail_reason=None,
        mode_mixture={"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        candidate_decisions=[_decision("beam", 1), _decision("mcts", 2)],
        rationale="test",
        raw_text="{}",
        usage={},
    )

    launches, debug = select_launches(
        _evidence(),
        candidates,
        sup,
        cfg=SelectorConfig(available_slots=1),
        tick_id="v7r010",
        recent_modes=["exploit"] * 10,
        recent_started_families=["complexa_beam"] * 10,
    )

    selected = [L for L in launches if L.status == "launched"]
    assert selected[0].candidate_id == "mcts"
    assert debug["overrepresented_families"] == ["complexa_beam"]
