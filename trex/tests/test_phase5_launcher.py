"""Tests for the staged-output launcher simulation; no real workers or Slurm submissions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trex.archive import Archive
from trex.phase5_launcher import (
    DaemonConfig,
    PARSER_BY_FAMILY,
    WorkerSlot,
    _expected_output_dir,
    _free_slot,
    _get_candidate_family,
    _iter_unscheduled_launches,
    _lookup_candidate,
    daemon_main_loop,
    is_complete,
    submit_test_mode,
)
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    LaunchDecision,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _feas() -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=True, runtime_bucket_id="rb_v7", compiler_ok=True,
        verifier_ok=True, route_cap_ok=True, cost_ok=True,
    )


def _candidate(cid: str = "c1", family: str = "bindcraft") -> ActionCandidate:
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=["h1"], parent_result_id=None,
        method_family=family, operator_id="op", lane_id=family,
        config_delta={"max_trajectories": 4}, downstream_route_plan=[],
        estimated_cost_class="diagnostic",  # type: ignore[arg-type]
        expected_signal="x", evidence_refs=["e1"], feasibility=_feas(),
    )


def _launch(cid: str = "c1", status: str = "launched") -> LaunchDecision:
    return LaunchDecision(
        launch_id=f"L_{cid}", tick_id="t1", candidate_id=cid,
        status=status,  # type: ignore[arg-type]
        resource_class_concrete={"class": "standard", "source": "test"},
        why="unit test", fallback=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parser_registry_covers_v7_core_families():
    """Check the parser registry supported by the launcher simulation."""
    required = {"bindcraft", "complexa_beam", "complexa_best_of_n",
                 "complexa_fk_steering", "complexa_mcts"}
    assert required.issubset(set(PARSER_BY_FAMILY)), (
        f"missing: {required - set(PARSER_BY_FAMILY)}"
    )


def test_free_slot_clears_all_fields():
    s = WorkerSlot(
        slot_id="s0", job_id="JOB123", candidate_id="c1",
        launched_at=12345.0, output_dir=Path("/tmp/x"),
    )
    _free_slot(s)
    assert s.job_id is None
    assert s.candidate_id is None
    assert s.launched_at is None
    assert s.output_dir is None


def test_submit_test_mode_writes_jsonl_and_returns_job_id(tmp_path: Path):
    log = tmp_path / "daemon_submissions.jsonl"
    cand = _candidate(cid="c1", family="bindcraft")
    jid = submit_test_mode(cand, tmp_path / "out", log)
    assert jid.startswith("TEST_")
    assert log.exists()
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["candidate_id"] == "c1"
    assert entries[0]["method_family"] == "bindcraft"
    assert entries[0]["stub_job_id"] == jid


def test_is_complete_false_when_no_output_dir():
    s = WorkerSlot(slot_id="s0")
    assert is_complete(s, mode="test") is False


def test_is_complete_test_mode_requires_sentinel(tmp_path: Path):
    """test mode → completion = sentinel file exists at
    `output_dir/designs/final_design_stats.csv`."""
    out = tmp_path / "out"
    out.mkdir()
    s = WorkerSlot(slot_id="s0", job_id="TEST_x", output_dir=out)
    assert not is_complete(s, mode="test")  # no sentinel yet
    designs = out / "designs"
    designs.mkdir()
    (designs / "final_design_stats.csv").write_text("design,pLDDT\n")
    assert is_complete(s, mode="test")


def test_is_complete_live_mode_not_implemented(tmp_path: Path):
    s = WorkerSlot(slot_id="s0", job_id="J1", output_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        is_complete(s, mode="live")


def test_lookup_candidate_returns_match(tmp_path: Path):
    arc = Archive(tmp_path)
    arc.append(_candidate(cid="c1"))
    arc.append(_candidate(cid="c2"))
    c = _lookup_candidate(arc, "c2")
    assert c is not None and c.candidate_id == "c2"


def test_lookup_candidate_none_for_missing(tmp_path: Path):
    arc = Archive(tmp_path)
    assert _lookup_candidate(arc, "missing") is None
    assert _lookup_candidate(arc, None) is None


def test_get_candidate_family(tmp_path: Path):
    arc = Archive(tmp_path)
    arc.append(_candidate(cid="cX", family="complexa_fk_steering"))
    assert _get_candidate_family(arc, "cX") == "complexa_fk_steering"
    assert _get_candidate_family(arc, "nope") is None


def test_iter_unscheduled_launches_filters_by_status(tmp_path: Path):
    arc = Archive(tmp_path)
    arc.append(_launch(cid="c1", status="launched"))
    arc.append(_launch(cid="c2", status="rejected"))  # should not yield
    arc.append(_launch(cid="c3", status="launched"))
    out = list(_iter_unscheduled_launches(arc))
    cids = sorted(L.candidate_id for L in out)
    assert cids == ["c1", "c3"]


def test_iter_unscheduled_launches_excludes_already_submitted(tmp_path: Path):
    arc = Archive(tmp_path)
    arc.append(_launch(cid="c1"))
    arc.append(_launch(cid="c2"))
    # Pre-populate submission log to mark c1 as already-submitted
    (tmp_path / "daemon_submissions.jsonl").write_text(
        json.dumps({"candidate_id": "c1"}) + "\n"
    )
    out = list(_iter_unscheduled_launches(arc))
    assert [L.candidate_id for L in out] == ["c2"]


def test_iter_unscheduled_launches_tolerates_corrupt_log(tmp_path: Path):
    """Garbage in submission log should not crash the daemon."""
    arc = Archive(tmp_path)
    arc.append(_launch(cid="c1"))
    (tmp_path / "daemon_submissions.jsonl").write_text(
        "not_json\n{}\n{\"candidate_id\": \"c2\"}\n"
    )
    out = list(_iter_unscheduled_launches(arc))
    # c1 not in log → yielded; c2 in log → excluded
    assert [L.candidate_id for L in out] == ["c1"]


def test_expected_output_dir_test_mode_uses_override(tmp_path: Path):
    cfg = DaemonConfig(
        archive_root=tmp_path, target_id="t", n_slots=3,
        mode="test", test_output_root=tmp_path / "preset",
    )
    slot = WorkerSlot(slot_id="s0")
    cand = _candidate()
    out = _expected_output_dir(cfg, slot, cand)
    assert out == tmp_path / "preset"


def test_expected_output_dir_live_mode_derives_path(tmp_path: Path):
    cfg = DaemonConfig(
        archive_root=tmp_path, target_id="t", n_slots=3,
        mode="live", test_output_root=None,
    )
    slot = WorkerSlot(slot_id="s2")
    cand = _candidate(family="complexa_beam")
    out = _expected_output_dir(cfg, slot, cand)
    assert out == tmp_path / "worker_outputs" / "complexa_beam" / "s2"


# ---------------------------------------------------------------------------
# Daemon main loop — test mode end-to-end
# ---------------------------------------------------------------------------


def _make_bindcraft_fixture(out_root: Path) -> Path:
    """Stage a minimal BindCraft-style output dir that the parser accepts.

    The bindcraft parser reads `designs/final_design_stats.csv` and pulls
    per-design metrics. A single-row fixture is sufficient.
    """
    designs = out_root / "designs"
    designs.mkdir(parents=True, exist_ok=True)
    csv_path = designs / "final_design_stats.csv"
    csv_path.write_text(
        "Design,pLDDT,InterfacePAE,Binder_RMSD,DesignTime\n"
        "design_001,92.5,8.3,0.85,123.5\n"
    )
    return out_root


def test_daemon_main_loop_test_mode_submits_and_parses(tmp_path: Path):
    """End-to-end: archive has 1 LaunchDecision + 1 ActionCandidate, pre-
    staged fixture has the bindcraft sentinel → daemon submits, detects
    completion, parses, appends ResultRecord, exits soft (no more work).
    """
    arc_root = tmp_path / "arc"
    arc = Archive(arc_root)
    arc.append(_candidate(cid="c1", family="bindcraft"))
    arc.append(_launch(cid="c1"))

    out_root = tmp_path / "fixture_out"
    _make_bindcraft_fixture(out_root)

    cfg = DaemonConfig(
        archive_root=arc_root, target_id="t", n_slots=1,
        poll_interval_s=0.01,  # tight loop for unit test
        max_wall_h=0.01,        # 36s safety cap
        mode="test", test_output_root=out_root,
    )
    summary = daemon_main_loop(cfg)
    assert summary["n_submitted"] >= 1
    # Soft exit reached: all slots idle + no new launches
    assert summary["n_completed"] >= 1
    # Submissions log written
    log = arc_root / "daemon_submissions.jsonl"
    assert log.exists()


def test_daemon_main_loop_unknown_family_logs_parse_error(tmp_path: Path):
    """If a launched candidate has a family with no parser, the daemon
    must log the error and free the slot — not crash."""
    arc_root = tmp_path / "arc"
    arc = Archive(arc_root)
    arc.append(_candidate(cid="c1", family="rfdiffusion"))  # not in PARSER_BY_FAMILY
    arc.append(_launch(cid="c1"))

    out_root = tmp_path / "fixture_out"
    _make_bindcraft_fixture(out_root)  # sentinel triggers completion

    cfg = DaemonConfig(
        archive_root=arc_root, target_id="t", n_slots=1,
        poll_interval_s=0.01, max_wall_h=0.01,
        mode="test", test_output_root=out_root,
    )
    summary = daemon_main_loop(cfg)
    assert summary["n_parse_errors"] >= 1
    err_log = arc_root / "parse_errors.jsonl"
    assert err_log.exists()
    entries = [json.loads(l) for l in err_log.read_text().splitlines()]
    assert any("no parser for family" in e["msg"] for e in entries)
