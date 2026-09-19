"""Auto-chain dispatch (bug fix 2026-05-28).

The controller enqueues downstream refilter ActionCandidates (candidate_id
"chain_…") into the archive after a diagnostic-only generator finishes.
T-ReX drains those candidates through the event-controller reserve/backfill lane,
not through the scientific E/R/X Selector pool.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

from trex.archive import Archive
from trex.refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from trex.controller import (
    _chain_refilter_recent_share_cap,
    _diagnostic_refilter_score,
)
from trex.live_tick import (
    LiveTickConfig,
    _diagnostic_chain_backlog,
    _native_strict_like_pending_score_conversion,
    _proxy_promising_pending_score_conversion,
    run_live_tick,
)
from trex.schemas import (
    ActionCandidate,
    DispatchRecord,
    FeasibilityCheck,
    LaunchDecision,
    PlannerOutput,
    ResultRecord,
    SupervisorOutput,
    TargetConstraint,
)


def _result(rid: str, pdb: str) -> ResultRecord:
    m = {"pLDDT": 85.0, "iPAE": 0.3, "binder_scRMSD": 1.5}
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics=m, metrics_calibrated=dict(m), route_lineage=[],
        gpu_h=0.4, exit_status="ok", artifacts={"pdb_path": pdb},
    )


def _chain_candidate(parent_rid: str) -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id="chain_t000_boltzgen_to_structure_refilter_001",
        hypothesis_ids=["seed"], parent_result_id=parent_rid,
        method_family="structure_refilter", operator_id="af2_multimer",
        lane_id="structure_refilter", config_delta={},
        downstream_route_plan=[], estimated_cost_class="low",
        expected_signal="auto_chain:boltzgen->structure_refilter",
        evidence_refs=[parent_rid], feasibility=feas,
    )


def _planner_empty() -> PlannerOutput:
    return PlannerOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        cards=[], rationale="x", raw_text='{"cards": []}', usage={},
    )


def _sup_ok() -> SupervisorOutput:
    return SupervisorOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        mode_mixture={"exploit": 0.4, "rescue": 0.4, "explore": 0.2},
        candidate_decisions=[], rationale="y",
        raw_text='{"mode_mixture": {}}', usage={},
    )


def test_pending_chain_candidate_is_not_dispatched_by_live_tick_selector(tmp_path: Path):
    arc = Archive(tmp_path / "arc")
    pdb = tmp_path / "seed.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    # Seed BoltzGen results (so the reducer runs) + the enqueued chain candidate
    # the controller would have appended after parsing the BoltzGen batch.
    for i in range(4):
        arc.append(_result(f"r{i}", str(pdb)))
    chain = _chain_candidate("r0")
    arc.append(chain)

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_empty()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    launched = [
        L for L in arc.iter_records(LaunchDecision)
        if L.status == "launched" and L.candidate_id == chain.candidate_id
    ]
    assert not launched, "chain score-conversion leaked into E/R/X selector launches"


def test_chain_candidate_not_reselected_once_selected(tmp_path: Path):
    """A chain candidate that already has a launch intent must not be
    re-selected on the next tick (no duplicate queueing)."""
    arc = Archive(tmp_path / "arc2")
    pdb = tmp_path / "seed.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    for i in range(4):
        arc.append(_result(f"r{i}", str(pdb)))
    chain = _chain_candidate("r0")
    arc.append(chain)
    # Pre-existing launch intent for the chain candidate.
    arc.append(LaunchDecision(
        launch_id="launch_prev_000", tick_id="t_000",
        candidate_id=chain.candidate_id, status="launched",
        resource_class_concrete={"class": "low", "source": "x", "mode": "rescue"},
        why="prev",
    ))
    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_empty()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )
    launched_this_tick = [
        L for L in arc.iter_records(LaunchDecision)
        if L.status == "launched" and L.candidate_id == chain.candidate_id
        and L.tick_id == "t_001"
    ]
    assert not launched_this_tick, "already-selected chain candidate was re-selected"


def test_diagnostic_chain_backlog_ignores_direct_scored_complexa(tmp_path: Path):
    pdb = tmp_path / "cx.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    parent = ResultRecord(
        result_id="cx1", parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={
            "pLDDT": 94.0,
            "iPAE": 0.18,
            "binder_scRMSD": 1.1,
            "complexa_native_pLDDT": 94.0,
            "complexa_native_iPAE": 0.18,
            "complexa_native_binder_scRMSD": 1.1,
        },
        metrics_calibrated={}, route_lineage=[],
        gpu_h=0.4, exit_status="ok", artifacts={"pdb_path": str(pdb)},
    )
    chain = _chain_candidate(parent.result_id)
    got = _diagnostic_chain_backlog([chain], [parent], [])
    assert got["total_unscored_diagnostic_artifacts"] == 0
    assert "complexa_beam" not in got["by_family"]


def test_complexa_auto_chain_score_prefers_native_strict_like_parent():
    good = ResultRecord(
        result_id="cx_good", parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={
            "complexa_native_pLDDT": 95.0,
            "complexa_native_iPAE": 0.12,
            "complexa_native_binder_scRMSD": 0.9,
            "ipTM": 0.78,
            "avg_ipsae": 0.60,
        },
        metrics_calibrated={}, route_lineage=[],
        gpu_h=0.4, exit_status="ok", artifacts={},
    )
    weak = ResultRecord(
        result_id="cx_weak", parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={
            "complexa_native_pLDDT": 72.0,
            "complexa_native_iPAE": 0.44,
            "complexa_native_binder_scRMSD": 3.4,
            "ipTM": 0.35,
        },
        metrics_calibrated={}, route_lineage=[],
        gpu_h=0.4, exit_status="ok", artifacts={},
    )
    assert _diagnostic_refilter_score(good) > _diagnostic_refilter_score(weak)
    assert _diagnostic_refilter_score(good) > 10.0


def test_diagnostic_chain_backlog_exposes_unscored_bindcraft(tmp_path: Path):
    pdb = tmp_path / "bc.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    parent = ResultRecord(
        result_id="bc1", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 93.0,
            "bindcraft_native_iPAE": 0.19,
            "bindcraft_native_binder_RMSD": 0.85,
        },
        metrics_calibrated={}, route_lineage=[],
        gpu_h=1.0, exit_status="ok", artifacts={"pdb_path": str(pdb)},
    )
    chain = _chain_candidate(parent.result_id)
    got = _diagnostic_chain_backlog([chain], [parent], [])
    bc = got["by_family"]["bindcraft"]
    assert got["total_unscored_diagnostic_artifacts"] == 1
    assert got["pending_chain_candidates"] == 1
    assert bc["unscored_artifacts"] == 1
    assert bc["native_strict_like_pending_refilter"] == 1


def test_bindcraft_accepted_and_native_near_pass_are_proxy_promising(tmp_path: Path):
    accepted_pdb = tmp_path / "Accepted" / "bc_acc.pdb"
    rejected_pdb = tmp_path / "Rejected" / "bc_rej.pdb"
    far_pdb = tmp_path / "Rejected" / "bc_far.pdb"
    accepted_pdb.parent.mkdir()
    rejected_pdb.parent.mkdir(exist_ok=True)
    for p in (accepted_pdb, rejected_pdb, far_pdb):
        p.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
        )
    accepted = ResultRecord(
        result_id="bc_acc", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 70.0,
            "bindcraft_native_iPAE": 0.60,
            "bindcraft_native_binder_RMSD": 4.0,
        },
        metrics_calibrated={}, route_lineage=[], gpu_h=1.0, exit_status="ok",
        artifacts={"pdb_path": str(accepted_pdb)},
        bins={"bindcraft_filter_status": "accepted"},
    )
    near = ResultRecord(
        result_id="bc_near", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 93.0,
            "bindcraft_native_iPAE": 0.220,
            "bindcraft_native_binder_RMSD": 1.12,
        },
        metrics_calibrated={}, route_lineage=[], gpu_h=1.0, exit_status="ok",
        artifacts={"pdb_path": str(rejected_pdb)},
        bins={"bindcraft_filter_status": "rejected"},
    )
    far = ResultRecord(
        result_id="bc_far", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={
            "bindcraft_native_pLDDT": 60.0,
            "bindcraft_native_iPAE": 0.80,
            "bindcraft_native_binder_RMSD": 6.0,
        },
        metrics_calibrated={}, route_lineage=[], gpu_h=1.0, exit_status="ok",
        artifacts={"pdb_path": str(far_pdb)},
        bins={"bindcraft_filter_status": "rejected"},
    )

    assert _proxy_promising_pending_score_conversion(accepted)
    assert _proxy_promising_pending_score_conversion(near)
    assert not _proxy_promising_pending_score_conversion(far)

    got = _diagnostic_chain_backlog(
        [_chain_candidate(accepted.result_id), _chain_candidate(near.result_id), _chain_candidate(far.result_id)],
        [accepted, near, far],
        [],
    )
    bc = got["by_family"]["bindcraft"]
    assert got["total_unscored_diagnostic_artifacts"] == 3
    assert got["proxy_promising_pending_refilter"] == 2
    assert got["native_or_proxy_pending_refilter"] == 2
    assert bc["proxy_promising_pending_refilter"] == 2


def test_boltzgen_proxy_promising_is_visible_and_ranked(tmp_path: Path):
    pdb_good = tmp_path / "bg_good.cif"
    pdb_bad = tmp_path / "bg_bad.cif"
    pdb_good.write_text("data_good\n")
    pdb_bad.write_text("data_bad\n")
    good = ResultRecord(
        result_id="bg_good", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.5, exit_status="ok",
        bins={
            "boltzgen_design_iptm": "0.86",
            "boltzgen_design_to_target_iptm": "0.70",
            "boltzgen_min_design_to_target_pae": "5.0",
        },
        artifacts={"pdb_path": str(pdb_good)},
    )
    weak = ResultRecord(
        result_id="bg_weak", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.5, exit_status="ok",
        bins={"boltzgen_design_iptm": "0.35"},
        artifacts={"pdb_path": str(pdb_bad)},
    )

    assert not _native_strict_like_pending_score_conversion(good)
    assert _proxy_promising_pending_score_conversion(good)
    assert not _proxy_promising_pending_score_conversion(weak)
    assert _diagnostic_refilter_score(good) > _diagnostic_refilter_score(weak)

    got = _diagnostic_chain_backlog([_chain_candidate(good.result_id)], [good, weak], [])
    bg = got["by_family"]["boltzgen"]
    assert got["total_unscored_diagnostic_artifacts"] == 2
    assert got["proxy_promising_pending_refilter"] == 1
    assert got["native_or_proxy_pending_refilter"] == 1
    assert bg["proxy_promising_pending_refilter"] == 1


def test_proteinmpnn_proxy_promising_is_visible_and_ranked(tmp_path: Path):
    pdb_good = tmp_path / "mpnn_good.pdb"
    pdb_bad = tmp_path / "mpnn_bad.pdb"
    pdb_good.write_text("ATOM      1  CA  ALA B   1      0.0   0.0   0.0  1.00 90.00           C\n")
    pdb_bad.write_text("ATOM      1  CA  ALA B   1      0.0   0.0   0.0  1.00 90.00           C\n")
    good = ResultRecord(
        result_id="mpnn_good", parent_ids=[], target_id="t1",
        backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.1, exit_status="ok",
        bins={"mpnn_global_score": "0.95", "mpnn_seq_recovery": "0.42"},
        artifacts={"pdb_path": str(pdb_good)},
    )
    weak = ResultRecord(
        result_id="mpnn_weak", parent_ids=[], target_id="t1",
        backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.1, exit_status="ok",
        bins={"mpnn_global_score": "2.10", "mpnn_seq_recovery": "0.99"},
        artifacts={"pdb_path": str(pdb_bad)},
    )

    assert not _native_strict_like_pending_score_conversion(good)
    assert _proxy_promising_pending_score_conversion(good)
    assert not _proxy_promising_pending_score_conversion(weak)
    assert _diagnostic_refilter_score(good) > _diagnostic_refilter_score(weak)

    got = _diagnostic_chain_backlog([_chain_candidate(good.result_id)], [good, weak], [])
    mpnn = got["by_family"]["proteinmpnn_redesign"]
    assert got["total_unscored_diagnostic_artifacts"] == 2
    assert got["proxy_promising_pending_refilter"] == 1
    assert got["native_or_proxy_pending_refilter"] == 1
    assert mpnn["proxy_promising_pending_refilter"] == 1


def test_boltzgen_proxy_promising_keeps_one_share_cap_escape(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_WINDOW", "20")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_SHARE_MIN_N", "1")
    monkeypatch.setenv("TREX_CHAIN_REFILTER_MAX_SHARE", "0.0")
    arc = Archive(tmp_path / "proxy_escape")
    pdb = tmp_path / "bg.cif"
    pdb.write_text("data_bg\n")
    parent = ResultRecord(
        result_id="bg_parent", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.5, exit_status="ok",
        bins={
            "boltzgen_design_iptm": "0.86",
            "boltzgen_design_to_target_iptm": "0.70",
        },
        artifacts={"pdb_path": str(pdb)},
    )
    chain = _chain_candidate(parent.result_id)
    arc.append(parent)
    arc.append(chain)
    for i in range(20):
        arc.append(DispatchRecord(
            dispatch_id=f"old_{i}", tick_id="t0",
            candidate_id=f"chain_old_{i}", status="started",
            worker_slot="0", gpu_id="0", why="history",
        ))

    assert _chain_refilter_recent_share_cap(arc, 3, [chain.candidate_id]) == 1


def test_diagnostic_chain_backlog_ignores_synthetic_no_artifact_records(tmp_path: Path):
    synth = ResultRecord(
        result_id="synth_bindcraft_1",
        parent_ids=[],
        target_id="t1",
        backend_family="bindcraft",
        runtime_bucket_id="rb1",
        metrics={},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=1.0,
        exit_status="no_artifacts",  # type: ignore[arg-type]
        artifacts={},
    )
    got = _diagnostic_chain_backlog([], [synth], [])
    assert got["total_unscored_diagnostic_artifacts"] == 0
    assert got["by_family"] == {}


def test_diagnostic_chain_backlog_distinguishes_queued_from_dispatched(tmp_path: Path):
    pdb = tmp_path / "bc.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    parent = ResultRecord(
        result_id="bc1", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=1.0, exit_status="ok", artifacts={"pdb_path": str(pdb)},
    )
    chain = _chain_candidate(parent.result_id)
    launch = LaunchDecision(
        launch_id="l0", tick_id="v7r001", candidate_id=chain.candidate_id,
        status="launched",
        resource_class_concrete={"class": "low", "mode": "rescue", "source": "x"},
        why="selected",
    )
    queued = _diagnostic_chain_backlog([chain], [parent], [launch], [])
    assert queued["queued_chain_candidates"] == 1
    assert queued["dispatched_chain_candidates"] == 0
    assert queued["launched_chain_candidates"] == 0

    dispatched = _diagnostic_chain_backlog(
        [chain],
        [parent],
        [launch],
        [DispatchRecord(
            dispatch_id="d0", launch_id="l0", tick_id="v7r001",
            candidate_id=chain.candidate_id, status="started",
        )],
    )
    assert dispatched["queued_chain_candidates"] == 0
    assert dispatched["dispatched_chain_candidates"] == 1
    assert dispatched["launched_chain_candidates"] == 1


# ---- registry-001 (2026-06-18): lazy re-chain of stranded diagnostic artifacts ----


def _scored_refilter(rid: str, source_rid: str) -> ResultRecord:
    m = {"pLDDT": 92.0, "iPAE": 0.2, "binder_scRMSD": 1.0}
    return ResultRecord(
        result_id=rid, parent_ids=["chain_x", source_rid], target_id="t1",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics=m, metrics_calibrated=dict(m), route_lineage=[], gpu_h=0.05,
        exit_status="ok", bins={"refilter_source": source_rid, "refilter_role": CANONICAL_SCORE_CONVERSION, "foldseek_su": "cA"},
    )


def test_lazy_rechain_does_not_refilter_canonical_score_conversion_record(tmp_path: Path):
    """Canonical AF2 score-conversion records are already scored.

    Regression for a live BetV1 no-op where lazy_rechain minted
    structure_refilter->structure_refilter candidates and four were dispatched.
    """
    from trex.controller import (
        _lazy_rechain_stranded_diagnostic_artifacts,
        _record_needs_score_conversion,
    )
    arc = Archive(tmp_path / "arc_noop")
    pdb = tmp_path / "af2_refilter_prediction.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    scored = ResultRecord(
        result_id="rf_scored", parent_ids=["chain_x", "bg_parent"], target_id="t1",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 91.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={"pLDDT": 91.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        route_lineage=[], gpu_h=0.05, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
        bins={
            "refilter_source": "bg_parent",
            "refilter_role": CANONICAL_SCORE_CONVERSION,
            "foldseek_su": "cA",
        },
    )
    arc.append(scored)

    assert not _record_needs_score_conversion("structure_refilter", scored)
    minted = _lazy_rechain_stranded_diagnostic_artifacts(
        arc, tick_id="v7r001", chain_seq_ref=[0], target_buffer=10
    )
    assert minted == 0
    assert not [
        c for c in arc.iter_records(ActionCandidate)
        if "structure_refilter_to_structure_refilter" in c.candidate_id
    ]


def test_lazy_rechain_mints_only_stranded_artifacts(tmp_path: Path):
    """A diagnostic artifact that (a) already has a chain_* candidate or (b) is
    already scored (refilter_source) must NOT be re-chained; only the stranded
    backbones get a fresh chain_* -> structure_refilter candidate, drainable by
    _chain_backfill_ids."""
    from trex.controller import (
        _lazy_rechain_stranded_diagnostic_artifacts, _chain_backfill_ids)
    arc = Archive(tmp_path / "arc")
    for i in range(6):
        pdb = tmp_path / f"seed_{i}.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        arc.append(_result(f"r{i}", str(pdb)))   # boltzgen diagnostic artifacts r0..r5
    arc.append(_chain_candidate("r0"))            # r0 already has a chain candidate
    arc.append(_scored_refilter("rf_r1", "r1"))   # r1 already scored
    minted = _lazy_rechain_stranded_diagnostic_artifacts(
        arc, tick_id="v7r001", chain_seq_ref=[0], target_buffer=10)
    assert minted == 4                            # r2,r3,r4,r5 stranded
    chain_parents = {
        c.parent_result_id for c in arc.iter_records(ActionCandidate)
        if c.candidate_id.startswith("chain_v7r001_")
    }
    assert chain_parents == {"r2", "r3", "r4", "r5"}
    # minted candidates are feasible, but the backfill drains only the route's
    # bounded fair-probe tranche in one call; the pre-existing r0 chain consumes
    # one of those probe slots.
    drainable = _chain_backfill_ids(arc, set(), 10)
    assert len(drainable) == 4
    assert len([c for c in drainable if c.startswith("chain_v7r001_")]) == 3


def test_lazy_rechain_is_bounded_by_target_buffer_and_idempotent(tmp_path: Path):
    """Minting only tops up to target_buffer pending candidates (no archive bloat),
    and a second call after the buffer is full mints nothing more."""
    from trex.controller import _lazy_rechain_stranded_diagnostic_artifacts
    arc = Archive(tmp_path / "arc")
    for i in range(8):
        pdb = tmp_path / f"seed_{i}.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        arc.append(_result(f"r{i}", str(pdb)))
    seq = [0]
    first = _lazy_rechain_stranded_diagnostic_artifacts(
        arc, tick_id="v7r001", chain_seq_ref=seq, target_buffer=3)
    assert first == 3                              # bounded by buffer, not all 8
    # buffer already full of pending candidates → second call mints nothing
    second = _lazy_rechain_stranded_diagnostic_artifacts(
        arc, tick_id="v7r002", chain_seq_ref=seq, target_buffer=3)
    assert second == 0



def _evidence_with_route_status(family: str, status: str):
    from trex.schemas import (
        EvidenceSummary,
        LLMHealthSummary,
        RouteHealthSummary,
        RouteValueSummary,
    )
    return EvidenceSummary(
        tick_id="t_ev", target_id="t1", target_class="test", schema_version="v",
        elapsed_wall_h=1.0, remaining_wall_h=10.0,
        completed_children=0, pending_children=0,
        worker_gpu_h_total=1.0, worker_gpu_h_last_3_ticks=1.0,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats={}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label="productive", examples=[], metric_availability={},
        route_values=[RouteValueSummary(
            strategy_key=f"family::{family}", scope="family", family=family,
            root_family=None, action_family=family, scoring_family=None,
            operator_id=f"{family}_default", config_signature="family_rollup",
            route_gpu_h=3.0, new_su=2 if status == "promote" else 0,
            new_su_per_route_gpu_h=0.8 if status == "promote" else 0.0,
            recent_new_su_per_route_gpu_h=0.8 if status == "promote" else 0.0,
            status=status,
        )],
    )


def test_diagnostic_refilter_score_uses_route_value_as_tiebreak_only():
    r = ResultRecord(
        result_id="bg1", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.4,
        exit_status="ok", artifacts={}, bins={"boltzgen_design_iptm": 0.50},
    )
    base = _diagnostic_refilter_score(r)
    promoted = _diagnostic_refilter_score(r, _evidence_with_route_status("boltzgen", "promote"))
    deferred = _diagnostic_refilter_score(r, _evidence_with_route_status("boltzgen", "defer"))
    assert promoted > base > deferred


def test_diagnostic_chain_backlog_counts_complexa_missing_canonical_axes(tmp_path: Path):
    pdb = tmp_path / "cx_missing_axes.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    parent = ResultRecord(
        result_id="cx_missing", parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={"complexa_native_pLDDT": 94.0, "complexa_native_iPAE": 0.18},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.4,
        exit_status="ok", artifacts={"pdb_path": str(pdb)},
    )
    chain = _chain_candidate(parent.result_id)
    got = _diagnostic_chain_backlog([chain], [parent], [])
    cx = got["by_family"]["complexa_beam"]
    assert got["total_unscored_diagnostic_artifacts"] == 1
    assert cx["accepted_artifacts"] == 1
    assert cx["pending_chain_candidates"] == 1


def test_advisory_parent_model_refold_does_not_clear_score_conversion_backlog(tmp_path: Path):
    pdb = tmp_path / "bc.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=[], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={"bindcraft_native_pLDDT": 95.0}, metrics_calibrated={},
        route_lineage=[], gpu_h=1.0, exit_status="ok",
        artifacts={"pdb_path": str(pdb)},
    )
    advisory = ResultRecord(
        result_id="adv", parent_ids=["cand_adv", parent.result_id], target_id="t1",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.2, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", bins={"refilter_source": parent.result_id, "refilter_role": PARENT_MODEL_REFOLD},
        artifacts={"pdb_path": str(tmp_path / "adv.pdb")},
    )
    canonical = ResultRecord(
        result_id="canon", parent_ids=["chain_x", parent.result_id], target_id="t1",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.2, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", bins={"refilter_source": parent.result_id, "refilter_role": CANONICAL_SCORE_CONVERSION},
        artifacts={"pdb_path": str(tmp_path / "canon.pdb")},
    )
    advisory_backlog = _diagnostic_chain_backlog([], [parent, advisory], [])
    assert advisory_backlog["by_family"]["bindcraft"]["unscored_artifacts"] == 1
    canonical_backlog = _diagnostic_chain_backlog([], [parent, canonical], [])
    assert canonical_backlog["by_family"]["bindcraft"]["completed_refilters"] == 1
    assert canonical_backlog["by_family"]["bindcraft"]["unscored_artifacts"] == 0


def test_boltzgen_aggregate_artifact_is_not_score_conversion_parent(tmp_path: Path):
    """Regression for SC2RBD: root boltzgen/design.cif caused parse-error loops."""
    from trex.live_tick import _record_needs_canonical_score_conversion
    from trex.controller import _record_needs_score_conversion

    aggregate = tmp_path / "design.cif"
    aggregate.write_text("data_aggregate\n")
    rec = ResultRecord(
        result_id="bg_aggregate", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.01,
        exit_status="ok",
        bins={"boltzgen_design_id": "design", "boltzgen_orphan_cif": "1"},
        artifacts={"pdb_path": str(aggregate)},
    )

    assert not _record_needs_score_conversion("boltzgen", rec)
    assert not _record_needs_canonical_score_conversion(rec)


def test_lazy_rechain_does_not_retry_parse_failed_score_conversion_parent(tmp_path: Path):
    """A terminal parse_failed chain blocks re-minting the same parent."""
    from trex.controller import _lazy_rechain_stranded_diagnostic_artifacts

    arc = Archive(tmp_path / "arc_parse_failed")
    pdb = tmp_path / "seed_valid.cif"
    pdb.write_text("data_valid\n")
    parent = _result("r_failed_parent", str(pdb))
    chain = _chain_candidate(parent.result_id)
    arc.append(parent)
    arc.append(chain)
    arc.append(DispatchRecord(
        dispatch_id="d_parse_failed", tick_id="v7r001",
        candidate_id=chain.candidate_id, status="parse_failed",
        worker_slot="0", gpu_id="0", why="missing af2 result json",
    ))

    minted = _lazy_rechain_stranded_diagnostic_artifacts(
        arc, tick_id="v7r002", chain_seq_ref=[0], target_buffer=10
    )

    assert minted == 0
    chain_candidates = [
        c for c in arc.iter_records(ActionCandidate)
        if c.method_family == "structure_refilter" and c.parent_result_id == parent.result_id
    ]
    assert len(chain_candidates) == 1


def test_chain_backfill_skips_existing_aggregate_boltzgen_chain(tmp_path: Path):
    """Existing legacy chain candidates for aggregate design.cif must not drain."""
    from trex.controller import _chain_backfill_ids

    arc = Archive(tmp_path / "arc_aggregate_chain")
    aggregate = tmp_path / "design.cif"
    aggregate.write_text("data_aggregate\n")
    parent = ResultRecord(
        result_id="bg_aggregate", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.01,
        exit_status="ok",
        bins={"boltzgen_design_id": "design", "boltzgen_orphan_cif": "1"},
        artifacts={"pdb_path": str(aggregate)},
    )
    chain = _chain_candidate(parent.result_id)
    arc.append(parent)
    arc.append(chain)

    assert _chain_backfill_ids(arc, set(), 10) == []


def test_auto_chain_cap_zero_for_unscoreable_boltzgen_aggregate(tmp_path: Path):
    """Even diagnostic-only families must not bypass score-conversion eligibility."""
    from trex.controller import (
        _auto_chain_cap_for_diagnostic_launch,
        _record_needs_score_conversion,
    )

    aggregate = tmp_path / "design.cif"
    aggregate.write_text("data_aggregate\n")
    rec = ResultRecord(
        result_id="bg_aggregate", parent_ids=[], target_id="t1",
        backend_family="boltzgen", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.01,
        exit_status="ok",
        bins={"boltzgen_design_id": "design", "boltzgen_orphan_cif": "1"},
        artifacts={"pdb_path": str(aggregate)},
    )

    eligible = [r for r in [rec] if _record_needs_score_conversion("boltzgen", r)]
    assert eligible == []
    assert _auto_chain_cap_for_diagnostic_launch(Archive(tmp_path / "arc"), "boltzgen", eligible) == 0
