"""Regression tests for worker timeouts, incremental output recovery, and evaluation
backfill.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from trex.archive import Archive
from trex.controller import (
    BINDCRAFT_FLOOR_S,
    BINDCRAFT_YIELD_WINDOW_S,
    HARD_CEILING_S,
    _WorkerSlot,
    _parse_and_archive_slot,
    _worker_timeout_reason,
    _auto_chain_cap_for_diagnostic_launch,
    _acquire_controller_lock,
    _archive_resume_state,
    _cheap_checkpoint_evidence,
    _checkpoint_charged_gpu_count,
    _read_controller_checkpoint,
    _write_controller_checkpoint,
    _bindcraft_accepted_count,
    _bindcraft_scoreable_final_count,
    _bindcraft_progress_count,
    _chain_backfill_ids,
    _chain_route_tranche_cap,
    _chain_refilter_reserve_limit,
    _chain_refilter_recent_share_cap,
    _dispatch_pending_to_free_slots,
    _fence_started_dispatch_for_recovery,
    _high_value_score_conversion_pending,
    _score_conversion_feedback_inflight,
    _queue_chain_refilter_reserve,
    _pid_start_ticks,
    _recover_undispatched_launch_ids,
    _score_conversion_reserve_room,
    _select_score_conversion_parents,
    _noncanonical_residue_names,
    PermanentDispatchSkip,
)
from trex.schemas import (
    ActionCandidate,
    DispatchRecord,
    EvidenceSummary,
    FeasibilityCheck,
    LaunchDecision,
    LLMHealthSummary,
    ResultRecord,
    RouteHealthSummary,
    RouteValueSummary,
    TargetConstraint,
)


def test_bindcraft_accepted_count(tmp_path: Path):
    assert _bindcraft_accepted_count(tmp_path) is None        # nothing yet
    assert _bindcraft_accepted_count(None) is None
    designs = tmp_path / "designs"; designs.mkdir()
    csv = designs / "final_design_stats.csv"
    csv.write_text("design,score\n")                          # header only -> 0
    assert _bindcraft_accepted_count(tmp_path) == 0
    with open(csv, "a") as fh:
        fh.write("d1,0.9\nd2,0.8\n")                          # 2 accepted
    assert _bindcraft_accepted_count(tmp_path) == 2


def test_bindcraft_scoreable_final_count_includes_rejected_but_not_trajectory(tmp_path: Path):
    designs = tmp_path / "designs"
    accepted = designs / "Accepted"
    rejected = designs / "Rejected"
    trajectory = designs / "Trajectory"
    for path in (accepted, rejected, trajectory):
        path.mkdir(parents=True, exist_ok=True)
    (accepted / "a.pdb").write_text("MODEL\nEND\n")
    (rejected / "r1.pdb").write_text("MODEL\nEND\n")
    (rejected / "r2.pdb").write_text("MODEL\nEND\n")
    (trajectory / "t.pdb").write_text("MODEL\nEND\n")

    assert _bindcraft_accepted_count(tmp_path) == 1
    assert _bindcraft_scoreable_final_count(tmp_path) == 3


def test_bindcraft_progress_count_includes_intermediate_outputs(tmp_path: Path):
    assert _bindcraft_progress_count(tmp_path) is None
    designs = tmp_path / "designs"
    designs.mkdir()
    (designs / "final_design_stats.csv").write_text("design,score\n")
    (designs / "trajectory_stats.csv").write_text("design,plddt\nt1,0.80\nt2,0.82\n")
    (designs / "mpnn_design_stats.csv").write_text("design,score\nm1,-1.0\n")
    traj = designs / "Trajectory"
    traj.mkdir()
    (traj / "t1.pdb").write_text("MODEL\nEND\n")
    # final accepted count is still zero, but the worker is clearly progressing.
    assert _bindcraft_accepted_count(tmp_path) == 0
    assert _bindcraft_progress_count(tmp_path) >= 4


def _started(cid: str, i: int = 0) -> DispatchRecord:
    return DispatchRecord(
        dispatch_id=f"d_{cid}_{i}", tick_id="t", candidate_id=cid,
        status="started", worker_slot="0", gpu_id="0", why="history",
    )

def _would_kill(
    elapsed,
    ceiling,
    *,
    fam,
    now,
    last_yield_at,
    primed,
    accepted_count=None,
    progress_count=None,
    prev_yield_n=0,
):
    """Use the controller's timeout decision helper. BindCraft ignores the
    reference ceiling and is killed only by the no-yield productivity watchdog.
    `primed` mirrors the loop's first post-floor probe."""
    reason, _, _ = _worker_timeout_reason(
        fam,
        elapsed_slot_s=elapsed,
        ceiling_s=ceiling,
        now_t=now,
        accepted_count=accepted_count,
        progress_count=progress_count,
        prev_yield_n=prev_yield_n if primed else -1,
        last_yield_at=last_yield_at,
    )
    return reason is not None


def test_adaptive_kill_decision():
    now = time.time()
    bc_ceiling = HARD_CEILING_S["bindcraft"]        # reference only for BindCraft
    mcts_ceiling = HARD_CEILING_S["complexa_mcts"]  # 2.5 h
    # A newly primed worker beyond the minimum runtime must receive its full yield window.
    assert not _would_kill(BINDCRAFT_FLOOR_S + 1, bc_ceiling, fam="bindcraft",
                           now=now, last_yield_at=now, primed=True)
    # non-bindcraft under its ceiling is NEVER killed (no liveness probe) ...
    assert not _would_kill(mcts_ceiling - 1, mcts_ceiling, fam="complexa_mcts",
                           now=now, last_yield_at=0.0, primed=False)
    # ... but a non-bindcraft worker over its ceiling IS killed (hang backstop)
    assert _would_kill(mcts_ceiling + 1, mcts_ceiling, fam="complexa_mcts",
                       now=now, last_yield_at=0.0, primed=False)
    # bindcraft accepted/final scoreable output increased -> KEEP. Trajectory
    # liveness alone is not a productivity signal because it can self-lock the
    # high-cost cap while no AF2-scoreable artifacts appear.
    assert not _would_kill(9000, bc_ceiling, fam="bindcraft",
                           now=now,
                           last_yield_at=now - (BINDCRAFT_YIELD_WINDOW_S + 60),
                           primed=True, accepted_count=4, progress_count=4,
                           prev_yield_n=3)
    # bindcraft busy but NO accepted/final growth for > yield window -> KILL
    assert _would_kill(9000, bc_ceiling, fam="bindcraft", now=now,
                       last_yield_at=now - (BINDCRAFT_YIELD_WINDOW_S + 60),
                       primed=True, accepted_count=4, progress_count=4,
                       prev_yield_n=4)
    # bindcraft below the floor -> never killed
    assert not _would_kill(1000, bc_ceiling, fam="bindcraft",
                           now=now, last_yield_at=now - 9999, primed=False)
    # Continue while new accepted outputs arrive.
    assert not _would_kill(bc_ceiling + 1000, bc_ceiling, fam="bindcraft",
                           now=now, last_yield_at=now - 9999, primed=True,
                           accepted_count=4, prev_yield_n=3)
    # Stop after the configured interval without new accepted outputs.
    assert _would_kill(bc_ceiling + 1000, bc_ceiling, fam="bindcraft",
                       now=now,
                       last_yield_at=now - (BINDCRAFT_YIELD_WINDOW_S + 60),
                       primed=True, accepted_count=4, prev_yield_n=4)


def test_timeout_dicts_cover_enabled_families():
    enabled = {"bindcraft", "complexa_beam", "boltzgen",
               "complexa_fk_steering", "complexa_mcts", "proteinmpnn_redesign",
               "structure_refilter"}
    assert enabled <= set(HARD_CEILING_S), f"missing: {enabled - set(HARD_CEILING_S)}"
    # every ceiling must be positive
    for f in enabled:
        assert HARD_CEILING_S[f] > 0, f
    # BindCraft's productivity watchdog is independent of the reference ceiling.
    # Keep it long enough for hard-target accepted designs to emerge after the
    # 1.5h floor, without introducing an absolute cap on productive runs.
    assert BINDCRAFT_FLOOR_S == 5400
    assert BINDCRAFT_YIELD_WINDOW_S == 5400


def test_bindcraft_incremental_parse_dedupes_and_delta_charges(tmp_path: Path, monkeypatch):
    import trex.controller as controller

    arc = Archive(tmp_path / "inc")
    cand = ActionCandidate(
        candidate_id="cand_bc", hypothesis_ids=[], parent_result_id=None,
        method_family="bindcraft", operator_id="op", lane_id="lane",
        config_delta={}, downstream_route_plan=[], estimated_cost_class="extended",
        expected_signal="smoke", evidence_refs=[],
        feasibility=FeasibilityCheck(True, "rb", True, True, True, True, []),
    )
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r000"
    target = TargetConstraint(target_id="smoke", target_class="test")

    def rec(rid: str) -> ResultRecord:
        return ResultRecord(
            result_id=rid, parent_ids=[cand.candidate_id], target_id="smoke",
            backend_family="bindcraft", runtime_bucket_id="rb",
            metrics={}, metrics_calibrated={}, route_lineage=["bindcraft"],
            gpu_h=999.0, exit_status="ok", bins={}, artifacts={},
        )

    seq = [
        [rec("bc_A")],
        [rec("bc_A")],
        [rec("bc_A"), rec("bc_B")],
        [rec("bc_A"), rec("bc_B")],
    ]
    monkeypatch.setattr(controller, "parse_bindcraft_output", lambda out_dir, ctx: seq.pop(0))

    assert _parse_and_archive_slot(
        slot, rc=0, archive=arc, target=target, chain_seq_ref=[0],
        elapsed_gpu_h=1.0, incremental=True,
    ) == 1
    assert _parse_and_archive_slot(
        slot, rc=0, archive=arc, target=target, chain_seq_ref=[0],
        elapsed_gpu_h=2.0, incremental=True,
    ) == 0
    assert _parse_and_archive_slot(
        slot, rc=0, archive=arc, target=target, chain_seq_ref=[0],
        elapsed_gpu_h=3.0, incremental=True,
    ) == 1
    assert _parse_and_archive_slot(
        slot, rc=0, archive=arc, target=target, chain_seq_ref=[0],
        elapsed_gpu_h=3.5, incremental=False,
    ) == 0

    records = list(arc.iter_records(ResultRecord))
    ids = [r.result_id for r in records]
    assert ids.count("bc_A") == 1
    assert ids.count("bc_B") == 1
    assert abs(sum(float(r.gpu_h) for r in records) - 3.5) < 1e-9


def _chain(cid: str, ok: bool = True) -> ActionCandidate:
    feas = FeasibilityCheck(ok, "rb1", True, True, True, ok)
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=["seed"], parent_result_id="r0",
        method_family="structure_refilter", operator_id="af2_multimer",
        lane_id="structure_refilter", config_delta={},
        downstream_route_plan=[], estimated_cost_class="low",
        expected_signal="auto_chain", evidence_refs=["r0"], feasibility=feas,
    )


def test_backfill_returns_feasible_unlaunched_chains(tmp_path: Path):
    arc = Archive(tmp_path / "a")
    arc.append(_chain("chain_t0_mpnn_to_structure_refilter_001"))
    arc.append(_chain("chain_t0_mpnn_to_structure_refilter_002"))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=5)
    assert set(ids) == {"chain_t0_mpnn_to_structure_refilter_001",
                        "chain_t0_mpnn_to_structure_refilter_002"}


def test_backfill_excludes_already_dispatched(tmp_path: Path):
    arc = Archive(tmp_path / "b")
    arc.append(_chain("chain_a"))
    arc.append(_chain("chain_b"))
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="t0", candidate_id="chain_a", status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="prev"))
    arc.append(DispatchRecord(
        dispatch_id="d0", launch_id="l0", tick_id="t0",
        candidate_id="chain_a", status="started",
    ))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=5)
    assert ids == ["chain_b"]


def test_backfill_recovers_launch_intent_without_dispatch(tmp_path: Path):
    arc = Archive(tmp_path / "b2")
    arc.append(_chain("chain_a"))
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="t0", candidate_id="chain_a", status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="queued-but-never-dispatched"))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=5)
    assert ids == ["chain_a"]


def test_startup_recovers_selected_but_not_dispatched_candidate(tmp_path: Path):
    arc = Archive(tmp_path / "b3")
    cand = _chain("chain_a")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id="chain_a",
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected-before-crash",
    ))
    seen: set[str] = set()
    ids = _recover_undispatched_launch_ids(arc, chain_backfill_seen=seen)
    assert ids == ["chain_a"]
    assert seen == {"chain_a"}


def test_startup_recovers_legacy_started_candidate_only_with_explicit_opt_in(
    tmp_path: Path, monkeypatch,
):
    arc = Archive(tmp_path / "b3_started")
    cand = _chain("chain_a")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id="chain_a",
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected-before-crash",
    ))
    monkeypatch.setenv("TREX_RECOVER_LEGACY_STARTED", "1")
    arc.append(DispatchRecord(
        dispatch_id="d0", launch_id="l0", tick_id="v7r001",
        candidate_id="chain_a", status="started",
    ))
    seen: set[str] = set()
    ids = _recover_undispatched_launch_ids(arc, chain_backfill_seen=seen)
    assert ids == ["chain_a"]
    assert seen == {"chain_a"}


def test_startup_does_not_duplicate_legacy_started_candidate_by_default(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.delenv("TREX_RECOVER_LEGACY_STARTED", raising=False)
    arc = Archive(tmp_path / "legacy_started_default")
    arc.append(_chain("chain_a"))
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id="chain_a",
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected-before-crash",
    ))
    arc.append(DispatchRecord(
        dispatch_id="d0", launch_id="l0", tick_id="v7r001",
        candidate_id="chain_a", status="started",
    ))

    assert _recover_undispatched_launch_ids(arc) == []


def test_startup_recovers_identity_tracked_dead_worker(tmp_path: Path):
    arc = Archive(tmp_path / "dead_started")
    arc.append(_chain("chain_a"))
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id="chain_a",
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected-before-crash",
    ))
    arc.append(DispatchRecord(
        dispatch_id="d0", launch_id="l0", tick_id="v7r001",
        candidate_id="chain_a", status="started",
        worker_pid=999_999_999, worker_pgid=999_999_999,
        worker_host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    ))

    assert _recover_undispatched_launch_ids(arc) == ["chain_a"]


def test_restart_fencing_detects_pid_reuse_without_signalling_current_process():
    current_start = _pid_start_ticks(os.getpid())
    assert current_start is not None
    dispatch = DispatchRecord(
        dispatch_id="d_reused", tick_id="v7r001", candidate_id="chain_a",
        status="started", worker_pid=os.getpid(), worker_pgid=os.getpgrp(),
        worker_pid_start_ticks=current_start + 1,
        worker_host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )

    assert _fence_started_dispatch_for_recovery(dispatch)


def test_startup_recovery_excludes_dispatch_terminal_status(tmp_path: Path):
    arc = Archive(tmp_path / "b4")
    cand = _chain("chain_a")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id="chain_a",
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected",
    ))
    arc.append(DispatchRecord(
        dispatch_id="d0", launch_id="l0", tick_id="v7r001",
        candidate_id="chain_a", status="dispatch_failed",
    ))
    assert _recover_undispatched_launch_ids(arc) == []


def test_startup_recovery_keeps_high_cost_defer_recoverable(tmp_path: Path):
    arc = Archive(tmp_path / "b4_high_cost")
    cand = _chain("chain_a")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id="chain_a",
        status="launched",
        resource_class_concrete={"class": "standard", "mode": "explore", "source": "x"},
        why="selected",
    ))
    arc.append(DispatchRecord(
        dispatch_id="d0", launch_id="l0", tick_id="v7r001",
        candidate_id="chain_a", status="dispatch_failed",
        why="high_cost_inflight_cap:bindcraft:active=1 cap=1 state=stalled",
    ))
    assert _recover_undispatched_launch_ids(arc) == ["chain_a"]


def test_deep_stall_throttled_chain_is_backfill_recoverable(tmp_path: Path):
    arc = Archive(tmp_path / "deep_stall_chain_recover")
    arc.append(_chain("chain_a"))
    arc.append(DispatchRecord(
        dispatch_id="deep_stall_throttle_v7r010_x", tick_id="v7r010",
        candidate_id="chain_a", status="dispatch_failed",
        why="current tick state=deep_stall; cancelled deterministic diagnostic score-conversion reserve before dispatch",
    ))
    assert _chain_backfill_ids(arc, seen=set(), max_n=2) == ["chain_a"]


def test_backfill_respects_seen_and_cap(tmp_path: Path):
    arc = Archive(tmp_path / "c")
    for i in range(5):
        arc.append(_chain(f"chain_{i}"))
    # cap
    assert len(_chain_backfill_ids(arc, seen=set(), max_n=2)) == 2
    # seen set excludes
    ids = _chain_backfill_ids(arc, seen={"chain_0", "chain_1", "chain_2"}, max_n=5)
    assert set(ids) == {"chain_3", "chain_4"}


def test_backfill_ignores_non_chain_and_infeasible(tmp_path: Path):
    arc = Archive(tmp_path / "d")
    arc.append(_chain("chain_ok"))
    arc.append(_chain("chain_bad", ok=False))   # infeasible
    # a non-chain refilter (warmstart) must not be picked up
    nc = _chain("warmstart_refilter")
    arc.append(nc)
    ids = _chain_backfill_ids(arc, seen=set(), max_n=5)
    assert ids == ["chain_ok"]


def _generator_candidate(cid: str, family: str, cfg: dict | None = None) -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=["seed"], parent_result_id=None,
        method_family=family, operator_id=f"{family}_default",
        lane_id=family, config_delta=dict(cfg or {}),
        downstream_route_plan=["structure_refilter"], estimated_cost_class="medium",
        expected_signal="generator", evidence_refs=[], feasibility=feas,
    )


def _diagnostic_result(rid: str, family: str, action_id: str) -> ResultRecord:
    return ResultRecord(
        result_id=rid, parent_ids=[action_id], target_id="t", backend_family=family,
        runtime_bucket_id="rb1", metrics={}, metrics_calibrated={},
        route_lineage=[family], gpu_h=0.1, exit_status="ok",
        artifacts={"pdb_path": f"/tmp/{rid}.pdb"}, bins={},
    )


def _chain_for_parent(cid: str, parent_rid: str) -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=["seed"], parent_result_id=parent_rid,
        method_family="structure_refilter", operator_id="af2_multimer",
        lane_id="structure_refilter", config_delta={}, downstream_route_plan=[],
        estimated_cost_class="low", expected_signal="auto_chain",
        evidence_refs=[parent_rid], feasibility=feas,
    )


def _route_value_evidence(row: RouteValueSummary) -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="ev", target_id="t", target_class="test", schema_version="v",
        elapsed_wall_h=1.0, remaining_wall_h=1.0,
        completed_children=0, pending_children=0,
        worker_gpu_h_total=1.0, worker_gpu_h_last_3_ticks=1.0,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="stalled", examples=[], metric_availability={},
        route_values=[row],
    )


def test_archive_resume_state_preserves_round_identity_and_elapsed_wall(tmp_path: Path):
    arc = Archive(tmp_path / "resume_state")
    row = RouteValueSummary(
        strategy_key="family::complexa_beam", scope="family",
        family="complexa_beam", root_family="complexa",
        action_family="complexa_beam", scoring_family=None,
        operator_id=None, config_signature="",
    )
    evidence = replace(
        _route_value_evidence(row), tick_id="v7r009", elapsed_wall_h=7.5,
    )
    arc.append(evidence)
    arc.append(LaunchDecision(
        launch_id="l12", tick_id="v7r012", candidate_id="candidate_12",
        status="launched", resource_class_concrete={}, why="selected",
    ))

    round_base, elapsed_offset_h = _archive_resume_state(arc)

    assert round_base == 12
    assert elapsed_offset_h == pytest.approx(7.5)


def test_archive_resume_state_uses_newer_barrier_checkpoint(tmp_path: Path):
    arc = Archive(tmp_path / "resume_checkpoint")
    row = RouteValueSummary(
        strategy_key="family::complexa_beam", scope="family",
        family="complexa_beam", root_family="complexa",
        action_family="complexa_beam", scoring_family=None,
        operator_id=None, config_signature="",
    )
    arc.append(replace(
        _route_value_evidence(row),
        target_id="t",
        tick_id="v7r001",
        elapsed_wall_h=0.1,
    ))
    checkpoint_path = _write_controller_checkpoint(
        arc,
        target_id="t",
        round_id=1,
        elapsed_wall_h=0.75,
        remaining_wall_h=46.25,
        evidence={"state_label": "productive", "run_su_count": 2},
    )

    assert checkpoint_path.name == "controller_checkpoint.json"
    assert _read_controller_checkpoint(arc)["evidence"]["run_su_count"] == 2
    round_base, elapsed_offset_h = _archive_resume_state(arc, target_id="t")
    assert round_base == 1
    assert elapsed_offset_h == pytest.approx(0.75)


def test_archive_resume_state_ignores_foreign_or_malformed_checkpoint(tmp_path: Path):
    arc = Archive(tmp_path / "resume_bad_checkpoint")
    (arc.root / "controller_checkpoint.json").write_text("not-json")
    assert _read_controller_checkpoint(arc) is None
    assert _archive_resume_state(arc, target_id="t") == (0, 0.0)

    _write_controller_checkpoint(
        arc,
        target_id="other",
        round_id=9,
        elapsed_wall_h=9.0,
        remaining_wall_h=38.0,
        evidence={},
    )
    assert _archive_resume_state(arc, target_id="t") == (0, 0.0)


def test_cheap_checkpoint_evidence_refreshes_wall_accounting(monkeypatch):
    latest = SimpleNamespace(state_label="productive_duplicate", run_su_count=3)
    payload = _cheap_checkpoint_evidence(
        latest,
        latest_state="low_evidence",
        elapsed_wall_h=2.5,
        remaining_wall_h=44.5,
        charged_gpu_count=4,
        worker_wall_gpu_count=3,
    )

    assert payload["state_label"] == "productive_duplicate"
    assert payload["run_su_count"] == 3
    assert payload["elapsed_wall_h"] == 2.5
    assert payload["charged_gpu_h_total"] == 10.0
    assert payload["worker_wall_gpu_h_total"] == 7.5
    assert payload["run_su_per_charged_gpu_h_total"] == 0.3
    assert payload["run_su_per_worker_wall_gpu_h_total"] == 0.4

    monkeypatch.setenv("TREX_CHARGED_GPUS", "bad")
    assert _checkpoint_charged_gpu_count(default=4) == 4


def test_controller_archive_lock_is_exclusive(tmp_path: Path):
    arc = Archive(tmp_path / "locked_archive")
    first_fd = _acquire_controller_lock(arc.root)
    try:
        with pytest.raises(SystemExit, match="active controller"):
            _acquire_controller_lock(arc.root)
    finally:
        os.close(first_fd)
    second_fd = _acquire_controller_lock(arc.root)
    os.close(second_fd)


def test_backfill_fair_probe_preserves_underscore_families(tmp_path: Path, monkeypatch):
    """Underscore-containing diagnostic families must not collapse to the
    wrong source family when the fair-probe cap is applied."""
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "2")
    arc = Archive(tmp_path / "fair_underscore")
    for i in range(5):
        arc.append(_chain(f"chain_t0_boltzgen_best_of_n_to_structure_refilter_{i:03d}"))
        arc.append(_chain(f"chain_t0_bindcraft_fk_steering_to_structure_refilter_{i:03d}"))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=10)
    assert len(ids) == 4
    assert sum("boltzgen_best_of_n" in cid for cid in ids) == 2
    assert sum("bindcraft_fk_steering" in cid for cid in ids) == 2


def test_backfill_fair_probe_is_exact_config_scoped(tmp_path: Path, monkeypatch):
    """Different generator configs are different routes, so each gets its own probe."""
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "1")
    arc = Archive(tmp_path / "fair_config")
    arc.append(_generator_candidate("gen_bw4", "boltzgen", {"budget": 4}))
    arc.append(_generator_candidate("gen_bw8", "boltzgen", {"budget": 8}))
    for i in range(3):
        rid = f"r4_{i}"
        arc.append(_diagnostic_result(rid, "boltzgen", "gen_bw4"))
        arc.append(_chain_for_parent(f"chain_t0_boltzgen_to_structure_refilter_4_{i}", rid))
    for i in range(3):
        rid = f"r8_{i}"
        arc.append(_diagnostic_result(rid, "boltzgen", "gen_bw8"))
        arc.append(_chain_for_parent(f"chain_t0_boltzgen_to_structure_refilter_8_{i}", rid))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=6)
    assert len(ids) == 2
    assert sum("_4_" in cid for cid in ids) == 1
    assert sum("_8_" in cid for cid in ids) == 1


def test_backfill_route_value_promotion_can_exceed_probe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "1")
    from trex.evidence_reducer import route_component_key

    arc = Archive(tmp_path / "fair_promote")
    cfg = {"budget": 4}
    arc.append(_generator_candidate("gen_bw4", "boltzgen", cfg))
    route_key = "route::" + route_component_key(
        "boltzgen", "boltzgen_default", cfg
    )
    prior = RouteValueSummary(
        strategy_key=route_key, scope="route", family="boltzgen",
        root_family=None, action_family="boltzgen", scoring_family=None,
        operator_id="boltzgen_default", config_signature="budget=4",
        config_delta=cfg, canonical_score_conversion_count=0,
    )
    arc.append(replace(_route_value_evidence(prior), tick_id="ev0"))
    arc.append(replace(_route_value_evidence(RouteValueSummary(
        strategy_key=route_key, scope="route", family="boltzgen",
        root_family=None, action_family="boltzgen", scoring_family=None,
        operator_id="boltzgen_default", config_signature="budget=4",
        config_delta=cfg, route_gpu_h=2.0, new_su=2, new_su_recent=1,
        new_su_per_route_gpu_h=1.0, recent_route_gpu_h=1.0,
        recent_new_su_per_route_gpu_h=1.0, status="promote",
        canonical_score_conversion_count=1,
    )), tick_id="ev1"))
    for i in range(4):
        rid = f"r4_{i}"
        arc.append(_diagnostic_result(rid, "boltzgen", "gen_bw4"))
        arc.append(_chain_for_parent(f"chain_t0_boltzgen_to_structure_refilter_4_{i}", rid))
    arc.append(DispatchRecord(
        dispatch_id="d0", tick_id="t0", candidate_id="chain_t0_boltzgen_to_structure_refilter_4_0",
        status="started",
    ))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=4)
    assert len(ids) == 3


def test_backfill_unpromoted_route_stops_after_probe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "2")
    arc = Archive(tmp_path / "fair_stop")
    for i in range(5):
        cid = f"chain_t0_boltzgen_best_of_n_to_structure_refilter_{i:03d}"
        arc.append(_chain(cid))
        if i < 2:
            arc.append(DispatchRecord(
                dispatch_id=f"d{i}", tick_id="t0", candidate_id=cid, status="started",
            ))
    assert _chain_backfill_ids(arc, seen=set(), max_n=5) == []


def test_proxy_promising_chain_does_not_bypass_spent_fair_probe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "1")
    arc = Archive(tmp_path / "fair_proxy_bypass")
    cfg = {"budget": 4}
    arc.append(_generator_candidate("gen_bg", "boltzgen", cfg))
    for i in range(2):
        rid = f"bg_proxy_{i}"
        rec = _diagnostic_result(rid, "boltzgen", "gen_bg")
        rec.bins.update({
            "boltzgen_design_iptm": "0.86",
            "boltzgen_design_to_target_iptm": "0.70",
        })
        arc.append(rec)
        arc.append(_chain_for_parent(f"chain_t0_boltzgen_to_structure_refilter_proxy_{i}", rid))
    arc.append(DispatchRecord(
        dispatch_id="d_proxy_0", tick_id="t0",
        candidate_id="chain_t0_boltzgen_to_structure_refilter_proxy_0",
        status="started",
    ))

    ids = _chain_backfill_ids(arc, seen=set(), max_n=5)
    assert ids == []
    assert _chain_backfill_ids(
        arc, seen=set(), max_n=1, allow_exhausted_background=True,
    ) == ["chain_t0_boltzgen_to_structure_refilter_proxy_1"]


def test_route_tranche_advances_once_per_new_evidence_signal():
    def row(count: int, *, su: int = 0, strict: int = 0, near: int = 0,
            quality_bins: int = 0, diag: float = 0.0) -> RouteValueSummary:
        return RouteValueSummary(
            strategy_key="route::x", scope="route", family="bindcraft",
            root_family=None, action_family="bindcraft", scoring_family=None,
            operator_id="bindcraft", config_signature="default",
            canonical_score_conversion_count=count,
            new_su=su, strict_count=strict, near_miss_count=near,
            strict_quality_n_unique_bins=quality_bins,
            diagnostic_improvement_score=diag,
        )

    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=None, prior_route_row=None,
    ) == 4
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(4), prior_route_row=row(0),
    ) == 4
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(4, strict=1),
        prior_route_row=row(0),
    ) == 4
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(4, strict=1, quality_bins=1),
        prior_route_row=row(0),
    ) == 8
    # A partial snapshot may finish the already-authorized tranche, never jump
    # directly to the following one.
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(5, strict=1, quality_bins=1),
        prior_route_row=row(4, strict=1, quality_bins=1),
    ) == 8
    # Lifetime success is insufficient: the 8->16 transition needs marginal
    # signal from the newly completed tranche.
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(8, strict=8, quality_bins=1),
        prior_route_row=row(4, strict=1, quality_bins=1),
    ) == 8
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(8, strict=8, near=1, quality_bins=1),
        prior_route_row=row(4, strict=1, quality_bins=1),
    ) == 16
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(16, strict=1, near=1, diag=0.2),
        prior_route_row=row(8, strict=1, near=1),
    ) == 32
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(32, su=2),
        prior_route_row=row(16),
    ) == 64
    # Productive routes do not hard-stop at the configured ladder tail.
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(64, su=3),
        prior_route_row=row(32, su=2),
    ) == 128
    # Partial completion of an elastic tail tranche keeps the active tail cap.
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(96, su=3),
        prior_route_row=row(64, su=3),
    ) == 128
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(128, su=3),
        prior_route_row=row(64, su=3),
    ) == 128
    assert _chain_route_tranche_cap(
        fair_cap=4, route_row=row(128, su=4),
        prior_route_row=row(64, su=3),
    ) == 256


def test_route_evidence_snapshots_canonical_conversion_count():
    from trex.evidence_reducer import build_route_values

    generator = _generator_candidate("gen_bg", "boltzgen", {"budget": 4})
    parent = _diagnostic_result("bg_parent", "boltzgen", generator.candidate_id)
    chain = _chain_for_parent("chain_bg_001", parent.result_id)
    converted = ResultRecord(
        result_id="af2_bg_001", parent_ids=[chain.candidate_id, parent.result_id],
        target_id="t", backend_family="structure_refilter",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 84.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=["structure_refilter"],
        gpu_h=0.01, exit_status="ok", artifacts={},
        bins={
            "refilter_role": "canonical_score_conversion",
            "refilter_source": parent.result_id,
            "refilter_source_family": "boltzgen",
        },
    )
    rows = build_route_values(
        [parent, converted],
        [parent, converted],
        {parent.result_id: generator, converted.result_id: chain},
    )
    route = next(
        r for r in rows if r.scope == "route" and r.action_family == "boltzgen"
    )
    family = next(
        r for r in rows if r.scope == "family" and r.action_family == "boltzgen"
    )
    assert route.canonical_score_conversion_count == 1
    assert family.canonical_score_conversion_count == 1


class _FakeProc:
    def poll(self):
        return None


def test_score_conversion_feedback_refresh_detects_high_value_scoring(tmp_path: Path):
    arc = Archive(tmp_path / "score_before_plan")
    pdb = tmp_path / "bc_parent.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 93.0,
            "bindcraft_native_iPAE": 0.22,
            "bindcraft_native_binder_RMSD": 1.12,
        },
        metrics_calibrated={}, route_lineage=[], gpu_h=0.4, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
        bins={"bindcraft_filter_status": "rejected"},
    )
    chain = _chain("chain_t0_bindcraft_to_structure_refilter_001")
    arc.append(parent)
    arc.append(chain)
    pool = [_WorkerSlot(slot_id=0, gpu_id="0", proc=_FakeProc(), cand=chain)]

    assert _high_value_score_conversion_pending(arc)
    assert _score_conversion_feedback_inflight(arc, pool)


def test_running_score_conversion_does_not_hold_generator_dispatch(tmp_path: Path):
    arc = Archive(tmp_path / "score_before_plan_dispatch")
    chain = _chain("chain_t0_bindcraft_to_structure_refilter_001")
    gen = ActionCandidate(
        candidate_id="cand_new_bindcraft", hypothesis_ids=["h"], parent_result_id=None,
        method_family="bindcraft", operator_id="bindcraft", lane_id="bindcraft",
        config_delta={}, downstream_route_plan=[], estimated_cost_class="extended",
        expected_signal="fresh generator", evidence_refs=[], feasibility=FeasibilityCheck(True, "rb1", True, True, True, True),
    )
    arc.append(chain)
    arc.append(gen)
    pending = [gen.candidate_id, chain.candidate_id]
    cand_by_id = {gen.candidate_id: gen, chain.candidate_id: chain}
    pool = [_WorkerSlot(slot_id=0, gpu_id="0"), _WorkerSlot(slot_id=1, gpu_id="1")]

    def fake_dispatch(cand, *, gpu_id, archive, target, target_pdb, round_id, archive_root):
        out_dir = tmp_path / f"out_{cand.candidate_id}"
        out_dir.mkdir(exist_ok=True)
        return _FakeProc(), out_dir, "", ""

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id,
        archive=arc, target=None, target_pdb=str(tmp_path / "target.pdb"),
        archive_root=tmp_path, round_id=7, dispatch_fn=fake_dispatch,
        dispatch_retries={}, dispatch_defer_seen=set(),
    )

    assert n == 2
    assert {slot.cand.candidate_id for slot in pool} == {
        gen.candidate_id, chain.candidate_id,
    }
    assert pending == []


def test_chain_refilter_reserve_limit_is_bounded(monkeypatch):
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    assert _chain_refilter_reserve_limit(0) == 0
    assert _chain_refilter_reserve_limit(3) == 1
    monkeypatch.setenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", "2")
    assert _chain_refilter_reserve_limit(3) == 2
    assert _chain_refilter_reserve_limit(1) == 1
    monkeypatch.setenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", "bad")
    assert _chain_refilter_reserve_limit(3) == 1


def test_chain_refilter_reserve_adaptive_scales_to_backlog(monkeypatch):
    """Adaptive reserve adds one lane only for a large score-conversion backlog."""
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", raising=False)
    monkeypatch.delenv("TREX_CHAIN_REFILTER_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("TREX_CHAIN_REFILTER_HIGH_BACKLOG_MIN", raising=False)
    # no backlog -> nothing reserved
    assert _chain_refilter_reserve_limit(3, backlog_n=0) == 0
    # A small backlog keeps one background scoring lane.
    assert _chain_refilter_reserve_limit(3, backlog_n=10) == 1
    # backlog smaller than the cap -> reserve only the backlog
    assert _chain_refilter_reserve_limit(3, backlog_n=1) == 1
    # A large backlog may use a second lane, while preserving one generation slot.
    assert _chain_refilter_reserve_limit(3, backlog_n=99) == 2
    assert _chain_refilter_reserve_limit(8, backlog_n=99) == 2
    monkeypatch.setenv("TREX_CHAIN_REFILTER_HIGH_BACKLOG_MIN", "1000")
    assert _chain_refilter_reserve_limit(8, backlog_n=99) == 1
    monkeypatch.delenv("TREX_CHAIN_REFILTER_HIGH_BACKLOG_MIN", raising=False)
    # 2 free -> reserve 1, generate 1
    assert _chain_refilter_reserve_limit(2, backlog_n=99) == 1
    # With one free slot, score conversion reserves it; no second slot is free.
    assert _chain_refilter_reserve_limit(1, backlog_n=99) == 1
    # 0 free -> 0
    assert _chain_refilter_reserve_limit(0, backlog_n=99) == 0


def test_chain_refilter_reserve_adaptive_respects_fraction_env(monkeypatch):
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    monkeypatch.setenv("TREX_CHAIN_REFILTER_MAX_CONCURRENT", "8")
    # fraction 1.0 -> may take all free slots except the protected generation one
    monkeypatch.setenv("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", "1.0")
    assert _chain_refilter_reserve_limit(3, backlog_n=10) == 2   # room-1 still caps
    assert _chain_refilter_reserve_limit(8, backlog_n=99) == 7
    # fraction 0.0 -> floor still applies (default floor 1), bounded by room-1
    monkeypatch.setenv("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", "0.0")
    assert _chain_refilter_reserve_limit(3, backlog_n=10) == 1
    # malformed -> default 0.5
    monkeypatch.setenv("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", "oops")
    assert _chain_refilter_reserve_limit(3, backlog_n=10) == 2


def test_chain_backfill_caps_weak_score_conversion_by_family(tmp_path: Path, monkeypatch):
    """Low-value diagnostic backlog should not turn into unbounded AF2 scoring.

    Native/proxy/promoted parents can bypass this cap elsewhere; this fixture uses
    legacy weak BoltzGen chains with no parent metrics, so the family cap applies.
    """
    monkeypatch.setenv("TREX_CHAIN_REFILTER_WEAK_FAMILY_PROBE_CAP", "5")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "99")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAMILY_PER_TICK_CAP", "99")
    arc = Archive(tmp_path / "weakfam")
    for i in range(12):
        arc.append(_chain(f"chain_t0_boltzgen_to_structure_refilter_{i:03d}"))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=12)
    assert len(ids) == 5
    assert all("boltzgen" in cid for cid in ids)


def test_chain_backfill_family_tick_cap_prevents_route_group_flood(tmp_path: Path, monkeypatch):
    """Many weak route groups from one family should not monopolize one tick."""
    monkeypatch.setenv("TREX_CHAIN_REFILTER_WEAK_FAMILY_PROBE_CAP", "99")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "99")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAMILY_PER_TICK_CAP", "3")
    arc = Archive(tmp_path / "tickfam")
    for route in range(6):
        gen = f"gen_route_{route}"
        arc.append(_generator_candidate(gen, "boltzgen", {"route": route}))
        for i in range(2):
            rid = f"tickfam_{route}_{i}"
            arc.append(_diagnostic_result(rid, "boltzgen", gen))
            arc.append(_chain_for_parent(
                f"chain_t0_boltzgen_to_structure_refilter_route{route}_{i:03d}", rid
            ))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=12)
    assert len(ids) == 3
    assert all("boltzgen" in cid for cid in ids)


def test_chain_backfill_family_tick_cap_persists_across_reserve_calls(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_WEAK_FAMILY_PROBE_CAP", "99")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "99")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAMILY_PER_TICK_CAP", "3")
    arc = Archive(tmp_path / "persistent_tick_cap")
    candidate_ids = []
    for i in range(6):
        cid = f"chain_t0_boltzgen_to_structure_refilter_{i:03d}"
        candidate_ids.append(cid)
        arc.append(_chain(cid))
    for i, cid in enumerate(candidate_ids[:3]):
        arc.append(LaunchDecision(
            launch_id=f"reserve_{i}", tick_id="v7r007", candidate_id=cid,
            status="launched", resource_class_concrete={}, why="reserve",
        ))

    assert _chain_backfill_ids(
        arc, seen=set(), max_n=6, admission_tick_id="v7r007",
    ) == []


def test_chain_backfill_successive_halving_limits_promoted_route(tmp_path: Path, monkeypatch):
    """A productive route expands by one tranche, not to all pending children."""
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "4")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_ROUTE_TRANCHES", "4,8,16")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAMILY_PER_TICK_HIGH_VALUE_CAP", "99")
    from trex.evidence_reducer import route_component_key

    arc = Archive(tmp_path / "route_tranche")
    cfg = {"budget": 4}
    arc.append(_generator_candidate("gen_bg", "boltzgen", cfg))
    route_key = "route::" + route_component_key("boltzgen", "boltzgen_default", cfg)
    prior = RouteValueSummary(
        strategy_key=route_key, scope="route", family="boltzgen",
        root_family=None, action_family="boltzgen", scoring_family=None,
        operator_id="boltzgen_default", config_signature="budget=4",
        config_delta=cfg, canonical_score_conversion_count=0,
    )
    arc.append(replace(_route_value_evidence(prior), tick_id="ev0"))
    arc.append(replace(_route_value_evidence(RouteValueSummary(
        strategy_key=route_key, scope="route", family="boltzgen",
        root_family=None, action_family="boltzgen", scoring_family=None,
        operator_id="boltzgen_default", config_signature="budget=4",
        config_delta=cfg, route_gpu_h=2.0, new_su=1, new_su_recent=1,
        new_su_per_route_gpu_h=0.5, recent_route_gpu_h=1.0,
        recent_new_su_per_route_gpu_h=1.0, status="promote",
        canonical_score_conversion_count=4,
    )), tick_id="ev1"))
    for i in range(20):
        rid = f"bg_tranche_{i}"
        arc.append(_diagnostic_result(rid, "boltzgen", "gen_bg"))
        cid = f"chain_t0_boltzgen_to_structure_refilter_tranche_{i:03d}"
        arc.append(_chain_for_parent(cid, rid))
        if i < 4:
            arc.append(DispatchRecord(
                dispatch_id=f"d_tranche_{i}", tick_id="t0", candidate_id=cid, status="started",
            ))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=20)
    assert len(ids) == 4  # served 4 -> next tranche 8, so only 4 more now


def test_chain_backfill_su_gpuh_route_expands_despite_duplicate_label(tmp_path: Path, monkeypatch):
    """Duplicate-heavy labels are warnings; positive SU/GPU-h remains the objective."""
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "4")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_ROUTE_TRANCHES", "4,8,16")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_FAMILY_PER_TICK_HIGH_VALUE_CAP", "99")
    from trex.evidence_reducer import route_component_key

    arc = Archive(tmp_path / "route_duplicate_but_su")
    cfg = {"budget": 4}
    arc.append(_generator_candidate("gen_bg_dup", "boltzgen", cfg))
    route_key = "route::" + route_component_key("boltzgen", "boltzgen_default", cfg)
    prior = RouteValueSummary(
        strategy_key=route_key, scope="route", family="boltzgen",
        root_family=None, action_family="boltzgen", scoring_family=None,
        operator_id="boltzgen_default", config_signature="budget=4",
        config_delta=cfg, canonical_score_conversion_count=0,
    )
    arc.append(replace(_route_value_evidence(prior), tick_id="ev0"))
    arc.append(replace(_route_value_evidence(RouteValueSummary(
        strategy_key=route_key, scope="route", family="boltzgen",
        root_family=None, action_family="boltzgen", scoring_family=None,
        operator_id="boltzgen_default", config_signature="budget=4",
        config_delta=cfg, route_gpu_h=2.0, new_su=1, new_su_recent=0,
        new_su_per_route_gpu_h=0.5, recent_route_gpu_h=1.0,
        recent_new_su_per_route_gpu_h=None, strict_count=20,
        strict_per_su=20.0, duplicate_bin_fraction=0.90, status="collapse_risk",
        canonical_score_conversion_count=4,
    )), tick_id="ev1"))
    for i in range(20):
        rid = f"bg_dup_{i}"
        arc.append(_diagnostic_result(rid, "boltzgen", "gen_bg_dup"))
        cid = f"chain_t0_boltzgen_to_structure_refilter_dup_{i:03d}"
        arc.append(_chain_for_parent(cid, rid))
        if i < 4:
            arc.append(DispatchRecord(
                dispatch_id=f"d_dup_{i}", tick_id="t0", candidate_id=cid, status="started",
            ))
    ids = _chain_backfill_ids(arc, seen=set(), max_n=20)
    assert len(ids) == 4


def test_chain_refilter_reserve_uses_one_background_lane(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", raising=False)
    arc = Archive(tmp_path / "drain")
    # two source families so the round-robin yields >1 distinct chain
    for i in range(3):
        arc.append(_chain(f"chain_t0_bindcraft_to_structure_refilter_{i:03d}"))
        arc.append(_chain(f"chain_t0_boltzgen_to_structure_refilter_{i:03d}"))
    pending: list[str] = []
    seen: set[str] = set()
    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r003", queue_room=3,
        source="chain_refilter_reserve", why="drain",
    )
    assert len(ids) == 1
    assert pending == ids           # prepended ahead of (empty) generator queue
    assert seen == set(ids)
    assert _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r003", queue_room=2,
        source="chain_refilter_reserve", why="already occupied",
    ) == []


def test_chain_refilter_reserve_per_round_cap(tmp_path: Path, monkeypatch):
    """A1: max_reserve_cap bounds a SINGLE call so predispatch+refill reserves
    can't compound to fill the pool. With cap=1, only 1 chain is reserved even
    though backlog + queue_room would allow 2."""
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", raising=False)
    arc = Archive(tmp_path / "roundcap")
    for i in range(3):
        arc.append(_chain(f"chain_t0_bindcraft_to_structure_refilter_{i:03d}"))
        arc.append(_chain(f"chain_t0_boltzgen_to_structure_refilter_{i:03d}"))
    pending: list[str] = []
    seen: set[str] = set()
    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r004", queue_room=3,
        source="chain_refilter_reserve", why="roundcap", max_reserve_cap=1,
    )
    assert len(ids) == 1            # capped to the round's remaining budget
    # cap=0 reserves nothing even with backlog + room
    ids0 = _queue_chain_refilter_reserve(
        arc, [], set(), tick_id="v7r004b", queue_room=3,
        source="chain_refilter_reserve", why="roundcap0", max_reserve_cap=0,
    )
    assert ids0 == []
    # cap=None still obeys the single asynchronous scoring-lane default.
    ids_none = _queue_chain_refilter_reserve(
        arc, [], set(), tick_id="v7r004c", queue_room=3,
        source="chain_refilter_reserve", why="roundcap_none",
    )
    assert len(ids_none) == 1


def test_score_conversion_reserve_can_use_single_slot_when_feedback_pending(tmp_path: Path, monkeypatch):
    """Allow bounded evaluation work when a single worker is free and artifacts await scores."""
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    arc = Archive(tmp_path / "single_slot_score")
    pdb = tmp_path / "bc_parent.pdb"
    pdb.write_text("ATOM\n")
    arc.append(_generator_candidate("cand_bc", "bindcraft"))
    parent = ResultRecord(**{
        **_diagnostic_result("bc_parent", "bindcraft", "cand_bc").__dict__,
        "artifacts": {"pdb_path": str(pdb)},
    })
    arc.append(parent)
    arc.append(_chain_for_parent("chain_bc_to_structure_refilter_001", "bc_parent"))

    room, used_single_slot = _score_conversion_reserve_room(arc, 1)
    assert room == 1
    assert used_single_slot is True

    pending = ["stale_generator_candidate"]
    seen: set[str] = set()
    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r_score", queue_room=room,
        source="chain_refilter_reserve", why="single-slot regression",
    )
    assert ids == ["chain_bc_to_structure_refilter_001"]
    assert pending[0] == "chain_bc_to_structure_refilter_001"


def test_score_conversion_reserve_keeps_single_slot_for_llm_without_backlog(tmp_path: Path):
    arc = Archive(tmp_path / "single_slot_empty")
    room, used_single_slot = _score_conversion_reserve_room(arc, 1)
    assert room == 0
    assert used_single_slot is False


def test_chain_refilter_reserve_preempts_stale_pending_work(tmp_path: Path, monkeypatch):
    """The reserved diagnostic scoring lane must not sit behind an already
    primed generator queue; otherwise accepted diagnostic artifacts can wait
    multiple worker completions before getting canonical strict/SU scoring."""
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    arc = Archive(tmp_path / "e")
    arc.append(_chain("chain_diag_to_structure_refilter_001"))
    pending = ["stale_generator_candidate"]
    seen: set[str] = set()

    ids = _queue_chain_refilter_reserve(
        arc,
        pending,
        seen,
        tick_id="v7r002",
        queue_room=1,
        source="chain_refilter_reserve",
        why="test",
    )

    assert ids == ["chain_diag_to_structure_refilter_001"]
    assert pending == [
        "chain_diag_to_structure_refilter_001",
        "stale_generator_candidate",
    ]
    assert seen == {"chain_diag_to_structure_refilter_001"}
    launched = [
        L for L in arc.iter_records(LaunchDecision)
        if L.candidate_id == "chain_diag_to_structure_refilter_001"
    ]
    assert launched and launched[0].status == "launched"
    assert launched[0].resource_class_concrete.get("mode") == "chain_refilter"


def test_dispatch_records_worker_start_for_queued_candidate(tmp_path: Path):
    arc = Archive(tmp_path / "f")
    cand = _chain("chain_dispatch_me")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id=cand.candidate_id,
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected",
    ))
    pool = [_WorkerSlot(slot_id=0, gpu_id="1")]
    pending = [cand.candidate_id]

    class _FakeProc:
        pid = 424_242

    def _fake_dispatch(candidate, **kwargs):
        assert candidate.candidate_id == cand.candidate_id
        return _FakeProc(), tmp_path / "out", str(tmp_path / "parent.pdb"), "parent_r0"

    n = _dispatch_pending_to_free_slots(
        pool,
        pending,
        {cand.candidate_id: cand},
        archive=arc,
        target=None,
        target_pdb=None,
        archive_root=tmp_path,
        round_id=1,
        dispatch_fn=_fake_dispatch,
        dispatch_retries={},
    )

    assert n == 1
    assert pending == []
    dispatches = list(arc.iter_records(DispatchRecord))
    assert len(dispatches) == 1
    assert dispatches[0].status == "started"
    assert dispatches[0].launch_id == "l0"
    assert dispatches[0].candidate_id == cand.candidate_id
    assert dispatches[0].worker_pid == 424_242
    assert dispatches[0].worker_pgid == 424_242
    assert dispatches[0].worker_host == socket.gethostname()


def test_dispatch_skips_duplicate_pending_candidate_in_same_controller(tmp_path: Path):
    """A chain selected by the selector and re-added by backfill must not launch
    twice before the first worker finishes."""
    arc = Archive(tmp_path / "g")
    cand = _chain("chain_duplicate")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0",
        tick_id="v7r001",
        candidate_id=cand.candidate_id,
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected",
    ))
    pool = [_WorkerSlot(slot_id=0, gpu_id="1"), _WorkerSlot(slot_id=1, gpu_id="2")]
    pending = [cand.candidate_id, cand.candidate_id]
    calls = []

    def _fake_dispatch(candidate, **kwargs):
        calls.append(candidate.candidate_id)
        return object(), tmp_path / f"out_{len(calls)}", str(tmp_path / "parent.pdb"), "parent_r0"

    n = _dispatch_pending_to_free_slots(
        pool,
        pending,
        {cand.candidate_id: cand},
        archive=arc,
        target=None,
        target_pdb=None,
        archive_root=tmp_path,
        round_id=1,
        dispatch_fn=_fake_dispatch,
        dispatch_retries={},
    )

    assert n == 1
    assert calls == [cand.candidate_id]
    assert pending == []
    dispatches = list(arc.iter_records(DispatchRecord))
    assert len(dispatches) == 1


def test_permanent_dispatch_skip_is_not_retried(tmp_path: Path):
    """Deterministic invalid parents should leave the queue immediately.

    Noncanonical BoltzGen PDBs are a permanent precondition failure, not a
    transient parent-file race. Retrying them pollutes the score-conversion
    backlog and can repeatedly consume reserve attempts.
    """
    arc = Archive(tmp_path / "permanent")
    cand = _chain("chain_bad_boltzgen_to_structure_refilter_001")
    arc.append(cand)
    arc.append(LaunchDecision(
        launch_id="l0",
        tick_id="v7r001",
        candidate_id=cand.candidate_id,
        status="launched",
        resource_class_concrete={"class": "low", "mode": "chain_refilter", "source": "test"},
        why="selected",
    ))
    pool = [_WorkerSlot(slot_id=0, gpu_id="1")]
    pending = [cand.candidate_id]
    retries = {}

    def _fake_dispatch(candidate, **kwargs):
        raise PermanentDispatchSkip("noncanonical binder residues ['UNK']")

    n = _dispatch_pending_to_free_slots(
        pool,
        pending,
        {cand.candidate_id: cand},
        archive=arc,
        target=None,
        target_pdb=None,
        archive_root=tmp_path,
        round_id=1,
        dispatch_fn=_fake_dispatch,
        dispatch_retries=retries,
    )

    assert n == 0
    assert pending == []
    assert retries == {}
    dispatches = list(arc.iter_records(DispatchRecord))
    assert len(dispatches) == 1
    assert dispatches[0].status == "dispatch_failed"
    assert "noncanonical binder residues" in dispatches[0].why


def _bc_result(rid, *, exit_status):
    # Real schema: a bindcraft record is DIAGNOSTIC-only — it never carries the
    # strict keys (the parser emits bindcraft_native_*). So metrics={} here.
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family="bindcraft",
        runtime_bucket_id="rb1", metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=2.5, exit_status=exit_status, bins={},  # type: ignore[arg-type]
    )


def _bc_chain_strict(rid):
    # Canonical strict SU is minted by the auto-chained structure_refilter child
    # and credited back to bindcraft via bins["refilter_source_family"] (the real
    # production attribution — same key method_health uses).
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source_family": "bindcraft", "foldseek_su": "c1"},
    )


def _bc_chain_near(rid):
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.25, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source_family": "bindcraft", "foldseek_near_miss": "n1"},
    )


def test_family_circuit_broken(tmp_path: Path):
    """Trip the BindCraft circuit breaker after the configured number of unproductive timeouts."""
    from trex.controller import _family_circuit_broken
    arc = Archive(tmp_path / "cb")
    assert not _family_circuit_broken(arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0)   # empty
    arc.append(_bc_result("b1", exit_status="timeout"))
    assert not _family_circuit_broken(arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0)   # 1 < 2
    arc.append(_bc_result("b2", exit_status="timeout"))
    assert _family_circuit_broken(arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0)       # 2 >= 2, 0 strict
    assert not _family_circuit_broken(arc, "complexa_beam", max_timeouts=2, min_gpu_h=3.0)  # other fam


def test_family_circuit_not_broken_if_lineage_strict(tmp_path: Path):
    """2 bindcraft timeouts BUT its auto-chained refilter minted a strict SU
    (credited via refilter_source_family) -> NOT broken (productive lane). This
    is the SC2RBD/BetV1 case the lineage-aware strict credit must protect."""
    from trex.controller import _family_circuit_broken
    arc = Archive(tmp_path / "cb2")
    arc.append(_bc_result("b1", exit_status="timeout"))
    arc.append(_bc_result("b2", exit_status="timeout"))
    arc.append(_bc_chain_strict("r1"))  # bindcraft minted SU via refilter chain
    assert not _family_circuit_broken(arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0)


def test_family_circuit_not_broken_if_lineage_near_miss(tmp_path: Path):
    """Near-miss signal means the lane is diagnosable, not dead."""
    from trex.controller import _family_circuit_broken
    arc = Archive(tmp_path / "cb_near")
    arc.append(_bc_result("b1", exit_status="timeout"))
    arc.append(_bc_result("b2", exit_status="timeout"))
    arc.append(_bc_chain_near("n1"))
    assert not _family_circuit_broken(arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0)


def test_deep_stall_keeps_one_native_strict_like_chain_reserve(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    arc = Archive(tmp_path / "native_strict_like")
    pdb = tmp_path / "bc_parent.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 92.0,
            "bindcraft_native_iPAE": 0.10,
            "bindcraft_native_binder_RMSD": 1.0,
        },
        metrics_calibrated={}, route_lineage=[], gpu_h=1.0, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
    )
    arc.append(parent)
    arc.append(_chain("chain_t0_bindcraft_to_structure_refilter_001"))
    pending: list[str] = []
    seen: set[str] = set()

    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r010", queue_room=3,
        source="chain_refilter_reserve", why="test", throttled=True,
    )

    assert ids == ["chain_t0_bindcraft_to_structure_refilter_001"]
    assert pending == ids
    assert seen == set(ids)


def test_deep_stall_keeps_one_bindcraft_native_strict_like_chain_reserve(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", raising=False)
    arc = Archive(tmp_path / "bindcraft_native_strict_like")
    pdb = tmp_path / "bc_parent.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 93.0,
            "bindcraft_native_iPAE": 0.15,
            "bindcraft_native_binder_RMSD": 1.0,
        },
        metrics_calibrated={}, route_lineage=[], gpu_h=0.4, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
    )
    arc.append(parent)
    arc.append(_chain("chain_t0_bindcraft_to_structure_refilter_001"))
    pending: list[str] = []
    seen: set[str] = set()

    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r010", queue_room=3,
        source="chain_refilter_reserve", why="test", throttled=True,
    )

    assert ids == ["chain_t0_bindcraft_to_structure_refilter_001"]
    assert pending == ids
    assert seen == set(ids)


def test_deep_stall_suppresses_chain_reserve_without_native_strict_like(tmp_path: Path):
    arc = Archive(tmp_path / "ordinary")
    arc.append(_chain("chain_t0_boltzgen_to_structure_refilter_001"))
    pending: list[str] = []
    seen: set[str] = set()

    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r010", queue_room=3,
        source="chain_refilter_reserve", why="test", throttled=True,
    )

    assert ids == []
    assert pending == []
    assert seen == set()


def test_deep_stall_keeps_one_chain_reserve_for_any_unscored_diagnostic_artifact(tmp_path: Path):
    arc = Archive(tmp_path / "ordinary_with_parent")
    pdb = tmp_path / "boltzgen_parent.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    arc.append(ResultRecord(
        result_id="r0", parent_ids=["bg_cand"], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.4,
        exit_status="ok", artifacts={"pdb_path": str(pdb)},
    ))
    arc.append(_chain("chain_t0_boltzgen_to_structure_refilter_001"))
    pending: list[str] = []
    seen: set[str] = set()

    ids = _queue_chain_refilter_reserve(
        arc, pending, seen, tick_id="v7r010", queue_room=3,
        source="chain_refilter_reserve", why="test", throttled=True,
    )

    assert ids == ["chain_t0_boltzgen_to_structure_refilter_001"]
    assert pending == ids
    assert seen == set(ids)


def test_direct_complexa_with_canonical_axes_never_auto_chains_for_score_conversion(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_AUTO_CHAIN_CAP_DIAGNOSTIC", "4")
    arc = Archive(tmp_path / "cap_direct")
    records = []
    for i in range(7):
        records.append(ResultRecord(
            result_id=f"cx{i}", parent_ids=[], target_id="t",
            backend_family="complexa_beam", runtime_bucket_id="rb1",
            metrics={
                "pLDDT": 94.0,
                "iPAE": 0.12,
                "binder_scRMSD": 0.8,
                "complexa_native_pLDDT": 94.0,
                "complexa_native_iPAE": 0.12,
                "complexa_native_binder_scRMSD": 0.8,
            },
            metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
            exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"cx{i}.pdb")},
            bins={"strict_score_source": "complexa_af2folding_canonical"},
        ))

    assert _auto_chain_cap_for_diagnostic_launch(arc, "complexa_beam", records) == 0


def test_complexa_missing_canonical_axes_falls_back_to_score_conversion(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_AUTO_CHAIN_CAP_DIAGNOSTIC", "4")
    arc = Archive(tmp_path / "cap_fallback")
    records = []
    for i in range(7):
        metrics = {}
        if i == 3:
            # Native-only metrics are diagnostic provenance, not canonical strict axes.
            metrics = {
                "complexa_native_pLDDT": 94.0,
                "complexa_native_iPAE": 0.12,
                "complexa_native_binder_scRMSD": 0.8,
            }
        records.append(ResultRecord(
            result_id=f"cx_missing{i}", parent_ids=[], target_id="t",
            backend_family="complexa_beam", runtime_bucket_id="rb1",
            metrics=metrics, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
            exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"cx_missing{i}.pdb")},
            bins={"needs_canonical_score_conversion": "1"},
        ))

    assert _auto_chain_cap_for_diagnostic_launch(arc, "complexa_beam", records) == 7


def test_score_conversion_scores_all_diagnostic_artifacts_in_priority_order(tmp_path: Path, monkeypatch):
    arc = Archive(tmp_path / "cap_high_signal")
    records = []
    for i in range(12):
        metrics = {}
        if i == 0:
            metrics = {
                "complexa_native_pLDDT": 96.0,
                "complexa_native_iPAE": 0.10,
                "complexa_native_binder_scRMSD": 0.7,
            }
        records.append(ResultRecord(
            result_id=f"cx_hi{i}", parent_ids=[], target_id="t",
            backend_family="boltzgen", runtime_bucket_id="rb1",
            metrics=metrics, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
            exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"cx_hi{i}.pdb")},
        ))

    cap = _auto_chain_cap_for_diagnostic_launch(arc, "boltzgen", records)
    assert cap == 12
    scored = [(float(100 - i), r) for i, r in enumerate(records)]
    selected = _select_score_conversion_parents(scored, cap)
    selected_ids = [r.result_id for _, r in selected]
    assert selected_ids[:4] == [f"cx_hi{i}" for i in range(4)]
    assert len({int(x.removeprefix("cx_hi")) for x in selected_ids[4:]}) >= 3
    assert selected_ids[-1] == "cx_hi11"


def test_auto_chain_cap_scores_all_without_signal(tmp_path: Path, monkeypatch):
    arc = Archive(tmp_path / "cap_default")
    records = [ResultRecord(
        result_id=f"bg{i}", parent_ids=[], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"bg{i}.pdb")},
    ) for i in range(7)]

    assert _auto_chain_cap_for_diagnostic_launch(arc, "boltzgen", records) == 7


def test_auto_chain_env_cap_does_not_leave_diagnostic_artifacts_unscored(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_AUTO_CHAIN_CAP_DIAGNOSTIC", "4")
    arc = Archive(tmp_path / "cap_env")
    records = [ResultRecord(
        result_id=f"bg{i}", parent_ids=[], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"bg{i}.pdb")},
    ) for i in range(9)]

    assert _auto_chain_cap_for_diagnostic_launch(
        arc, "boltzgen", records, source_candidate=None) == 9


def test_bindcraft_env_cap_does_not_leave_accepted_artifacts_unscored(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_AUTO_CHAIN_CAP_BINDCRAFT", "1")
    arc = Archive(tmp_path / "cap_bc_env")
    records = [ResultRecord(
        result_id=f"bc{i}", parent_ids=[], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"bc{i}.pdb")},
    ) for i in range(4)]

    assert _auto_chain_cap_for_diagnostic_launch(
        arc, "bindcraft", records, source_candidate=None) == 4


def test_select_score_conversion_parents_keeps_same_sequence_distinct_artifacts(tmp_path: Path):
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    a.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    b.write_text("ATOM      1  CA  GLY A   1       1.0   0.0   0.0  1.00 90.00           C\nEND\n")
    recs = []
    for rid, path, score in (("mp_a", a, 2.0), ("mp_b", b, 1.0)):
        recs.append((score, ResultRecord(
            result_id=rid, parent_ids=[], target_id="t",
            backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
            metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
            exit_status="ok", bins={"mpnn_sequence": "ACDEFG"},
            artifacts={"pdb_path": str(path)},
        )))
    selected = _select_score_conversion_parents(recs, cap=4)
    assert {r.result_id for _, r in selected} == {"mp_a", "mp_b"}


def test_chain_backfill_drains_complexa_fallback_missing_canonical_axes(tmp_path: Path):
    arc = Archive(tmp_path / "complexa_fallback")
    pdb = tmp_path / "cx.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    parent = ResultRecord(
        result_id="cx_missing", parent_ids=[], target_id="t",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={"complexa_native_pLDDT": 93.0}, metrics_calibrated={},
        route_lineage=[], gpu_h=0.2, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
    )
    chain = ActionCandidate(
        candidate_id="chain_t0_complexa_beam_to_structure_refilter_001",
        hypothesis_ids=["h"], parent_result_id=parent.result_id,
        method_family="structure_refilter", operator_id="af2_multimer",
        lane_id="structure_refilter", config_delta={}, downstream_route_plan=[],
        estimated_cost_class="low", expected_signal="auto_chain:complexa_beam->structure_refilter",
        evidence_refs=[parent.result_id], feasibility=FeasibilityCheck(True, "rb1", True, True, True, True),
    )
    arc.append(parent)
    arc.append(chain)
    assert _chain_backfill_ids(arc, seen=set(), max_n=5) == [chain.candidate_id]

def test_select_score_conversion_parents_dedupes_identical_artifact_paths(tmp_path: Path):
    pdb = tmp_path / "same.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    dup_low = ResultRecord(
        result_id="dup_low", parent_ids=[], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(pdb)},
    )
    dup_high = ResultRecord(
        result_id="dup_high", parent_ids=[], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(pdb)},
    )
    unique = ResultRecord(
        result_id="unique", parent_ids=[], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / "unique.pdb")},
    )

    selected = _select_score_conversion_parents(
        [(1.0, dup_low), (5.0, dup_high), (3.0, unique)], cap=10
    )
    selected_ids = [r.result_id for _, r in selected]
    assert selected_ids == ["dup_high", "unique"]


def test_auto_chain_cap_matches_parent_bound_route_key(tmp_path: Path, monkeypatch):
    from trex.evidence_reducer import route_component_key

    arc = Archive(tmp_path / "cap_parent_bound")
    root_action = _generator_candidate("cand_cx", "complexa_beam", {"beam_width": 4})
    root = _diagnostic_result("cx_parent", "complexa_beam", root_action.candidate_id)
    mp_base = _generator_candidate("cand_mp", "proteinmpnn_redesign", {"temperature": 0.1})
    mp = ActionCandidate(**{**mp_base.__dict__, "parent_result_id": root.result_id})
    arc.append(root_action)
    arc.append(root)
    arc.append(mp)

    records = [ResultRecord(
        result_id=f"mp{i}", parent_ids=[mp.candidate_id], target_id="t",
        backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"mp{i}.pdb")},
    ) for i in range(7)]

    root_comp = route_component_key("complexa_beam", root_action.operator_id, root_action.config_delta)
    action_comp = route_component_key("proteinmpnn_redesign", mp.operator_id, mp.config_delta)
    arc.append(_route_value_evidence(RouteValueSummary(
        strategy_key=f"route::{root_comp}->{action_comp}", scope="route",
        family="proteinmpnn_redesign", root_family="complexa_beam",
        action_family="proteinmpnn_redesign", scoring_family=None,
        operator_id=mp.operator_id, config_signature=action_comp.split(":", 2)[-1],
        parent_strategy_key=f"route::{root_comp}", route_gpu_h=4.0,
        new_su=3, new_su_recent=1, new_su_per_route_gpu_h=0.75,
        recent_new_su_per_route_gpu_h=0.75, status="promote",
    )))

    assert _auto_chain_cap_for_diagnostic_launch(
        arc, "proteinmpnn_redesign", records, source_candidate=mp) == 7


def test_chain_refilter_recent_share_cap_blocks_plumbing_monopoly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_MAX_SHARE", "0.60")
    arc = Archive(tmp_path / "share_cap")
    for i in range(35):
        arc.append(LaunchDecision(
            launch_id=f"lc{i}", tick_id="t", candidate_id=f"chain_old_{i}",
            status="launched", resource_class_concrete={"mode": "chain_refilter"},
            why="history"))
        arc.append(_started(f"chain_old_{i}", i))
    for i in range(5):
        arc.append(LaunchDecision(
            launch_id=f"lg{i}", tick_id="t", candidate_id=f"cand_old_{i}",
            status="launched", resource_class_concrete={"mode": "explore"},
            why="history"))
        arc.append(_started(f"cand_old_{i}", i))
    assert _chain_refilter_recent_share_cap(arc, 3, ["chain_new"]) == 0


def test_chain_refilter_recent_share_cap_bootstrap_blocks_early_burst(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_BOOTSTRAP_MAX", "4")
    arc = Archive(tmp_path / "share_bootstrap")
    for i in range(4):
        arc.append(LaunchDecision(
            launch_id=f"lc{i}", tick_id="t", candidate_id=f"chain_old_{i}",
            status="launched", resource_class_concrete={"mode": "chain_refilter"},
            why="history"))
        arc.append(_started(f"chain_old_{i}", i))
    for i in range(3):
        arc.append(LaunchDecision(
            launch_id=f"lg{i}", tick_id="t", candidate_id=f"cand_old_{i}",
            status="launched", resource_class_concrete={"mode": "explore"},
            why="history"))
        arc.append(_started(f"cand_old_{i}", i))
    assert _chain_refilter_recent_share_cap(arc, 3, ["chain_new"]) == 0


def test_chain_refilter_recent_share_cap_default_bootstrap_uses_share_budget(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_MAX_SHARE", "0.60")
    monkeypatch.delenv("TREX_CHAIN_REFILTER_BOOTSTRAP_MAX", raising=False)
    arc = Archive(tmp_path / "share_bootstrap_default")
    for i in range(4):
        arc.append(LaunchDecision(
            launch_id=f"lc{i}", tick_id="t", candidate_id=f"chain_old_{i}",
            status="launched", resource_class_concrete={"mode": "chain_refilter"},
            why="history"))
        arc.append(_started(f"chain_old_{i}", i))
    for i in range(3):
        arc.append(LaunchDecision(
            launch_id=f"lg{i}", tick_id="t", candidate_id=f"cand_old_{i}",
            status="launched", resource_class_concrete={"mode": "explore"},
            why="history"))
        arc.append(_started(f"cand_old_{i}", i))

    # The default bootstrap allowance follows max_share times min_n; an explicit
    # environment override can lower it.
    assert _chain_refilter_recent_share_cap(arc, 3, ["chain_new"]) == 3


def test_chain_refilter_recent_share_cap_keeps_bindcraft_native_escape_slot(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_MAX_SHARE", "0.60")
    arc = Archive(tmp_path / "share_escape")
    pdb = tmp_path / "native_parent.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    parent = ResultRecord(
        result_id="r_native", parent_ids=["gen"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={"bindcraft_native_pLDDT": 94.0, "bindcraft_native_iPAE": 0.10,
                 "bindcraft_native_binder_RMSD": 0.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
    )
    arc.append(parent)
    arc.append(_chain_for_parent("chain_native", "r_native"))
    for i in range(35):
        arc.append(LaunchDecision(
            launch_id=f"lc{i}", tick_id="t", candidate_id=f"chain_old_{i}",
            status="launched", resource_class_concrete={"mode": "chain_refilter"},
            why="history"))
        arc.append(_started(f"chain_old_{i}", i))
    for i in range(5):
        arc.append(LaunchDecision(
            launch_id=f"lg{i}", tick_id="t", candidate_id=f"cand_old_{i}",
            status="launched", resource_class_concrete={"mode": "explore"},
            why="history"))
        arc.append(_started(f"cand_old_{i}", i))
    assert _chain_refilter_recent_share_cap(arc, 3, ["chain_native"]) == 1


def test_boltzgen_noncanonical_binder_residue_detector(tmp_path: Path):
    pdb = tmp_path / "bad_binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\n"
        "ATOM      2  CA  UNK B   1       1.0   0.0   0.0  1.00 90.00           C\n"
        "ATOM      3  CA  MSE C   1       2.0   0.0   0.0  1.00 90.00           C\n"
        "END\n"
    )
    assert _noncanonical_residue_names(pdb, chain_id="B") == {"UNK"}
    assert _noncanonical_residue_names(pdb, chain_id="A") == set()
    assert _noncanonical_residue_names(pdb) == {"UNK", "MSE"}


def test_f5_breaker_prebuilt_index_matches_self_built(tmp_path: Path):
    """A supplied archive index must preserve the breaker decision."""
    from trex.controller import _family_circuit_broken
    from trex.live_tick import _index_actions_by_spawned_result

    def _decide(arc, prebuilt):
        if not prebuilt:
            return _family_circuit_broken(arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0)
        res = list(arc.iter_records(ResultRecord))
        return _family_circuit_broken(
            arc, "bindcraft", max_timeouts=2, min_gpu_h=3.0,
            results=res, by_result_id={r.result_id: r for r in res},
            spawning_actions=_index_actions_by_spawned_result(
                list(arc.iter_records(ActionCandidate)), res),
        )

    arc = Archive(tmp_path / "f5dead")
    arc.append(_bc_result("b1", exit_status="timeout"))
    arc.append(_bc_result("b2", exit_status="timeout"))
    assert _decide(arc, False) == _decide(arc, True) is True   # dead lane

    arc.append(_bc_chain_strict("r1"))   # bindcraft minted SU via refilter chain
    assert _decide(arc, False) == _decide(arc, True) is False  # productive via lineage


def test_chain_refilter_share_cap_ignores_failed_launch_intents(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_BOOTSTRAP_MAX", "4")
    arc = Archive(tmp_path / "share_failed_intents")
    for i in range(4):
        cid = f"chain_failed_{i}"
        arc.append(LaunchDecision(
            launch_id=f"lc{i}", tick_id="t", candidate_id=cid,
            status="launched", resource_class_concrete={"mode": "chain_refilter"},
            why="history"))
        arc.append(DispatchRecord(
            dispatch_id=f"df{i}", tick_id="t", candidate_id=cid,
            status="dispatch_failed", why="history"))

    assert _chain_refilter_recent_share_cap(arc, 3, ["chain_new"]) == 3


def test_chain_refilter_share_cap_scans_for_later_bindcraft_escape_candidate(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_MAX_SHARE", "0.60")
    arc = Archive(tmp_path / "share_later_escape")
    pdb = tmp_path / "native_parent_later.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n")
    arc.append(ResultRecord(
        result_id="r_native_later", parent_ids=["gen"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={"bindcraft_native_pLDDT": 94.0, "bindcraft_native_iPAE": 0.10,
                 "bindcraft_native_binder_RMSD": 0.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
    ))
    arc.append(_chain_for_parent("chain_native_later", "r_native_later"))
    for i in range(35):
        cid = f"chain_old_{i}"
        arc.append(LaunchDecision(
            launch_id=f"lc{i}", tick_id="t", candidate_id=cid,
            status="launched", resource_class_concrete={"mode": "chain_refilter"},
            why="history"))
        arc.append(_started(cid, i))
    for i in range(5):
        cid = f"cand_old_{i}"
        arc.append(LaunchDecision(
            launch_id=f"lg{i}", tick_id="t", candidate_id=cid,
            status="launched", resource_class_concrete={"mode": "explore"},
            why="history"))
        arc.append(_started(cid, i))

    assert _chain_refilter_recent_share_cap(
        arc, 3, ["chain_plain_first", "chain_native_later"]
    ) == 1


def test_auto_chain_cap_bad_env_uses_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_AUTO_CHAIN_CAP_DIAGNOSTIC", "not_an_int")
    records = [ResultRecord(
        result_id=f"bg{i}", parent_ids=[], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / f"bg{i}.pdb")},
    ) for i in range(5)]
    assert _auto_chain_cap_for_diagnostic_launch(
        Archive(tmp_path / "cap_bad_env"), "boltzgen", records
    ) == 5
