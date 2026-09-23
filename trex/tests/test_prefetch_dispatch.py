"""Test queue dispatch and replenishment with mocked worker slots."""

from __future__ import annotations

import dataclasses

from trex.controller import (
    _dispatch_pending_to_free_slots,
    _high_cost_dispatch_cap,
)
from trex.schemas import EvidenceSummary, LaunchDecision, LLMHealthSummary, RouteHealthSummary


class _Slot:
    def __init__(self, gid: str):
        self.slot_id = int(gid) if str(gid).isdigit() else 0
        self.gpu_id = gid
        self.proc = None
        self.cand = None
        self.out_dir = None
        self.parent_pdb_str = None
        self.parent_result_id = None
        self.tick_id = None
        self.launched_at = None

    @property
    def busy(self) -> bool:
        return self.proc is not None


class _Cand:
    def __init__(self, cid: str, family: str = "", expected_signal: str = "", supervisor_mode: str = ""):
        self.candidate_id = cid
        self.method_family = family
        self.expected_signal = expected_signal
        self.supervisor_mode = supervisor_mode


def _ok_dispatch(cand, *, gpu_id, **kw):
    # (proc, out_dir, parent_pdb_str, parent_result_id)
    return (object(), f"/out/{gpu_id}", "parent.pdb", "r_parent")


def _evidence(state: str = "stalled") -> EvidenceSummary:
    return EvidenceSummary(
        target_id="t", target_class="test", schema_version="test",
        tick_id="v7r001", elapsed_wall_h=0.0, remaining_wall_h=10.0,
        completed_children=0, pending_children=1, worker_gpu_h_total=0.0,
        worker_gpu_h_last_3_ticks=0.0, strict_count=0, global_new_strict=0,
        run_su_count=0, run_su_count_delta=0, su_per_gpu_h_recent=None,
        duplicate_fraction=None, top_bin_share=None, axis_stats={}, joint_patterns=[],
        near_miss_count=0, panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={},
        route_health=RouteHealthSummary(0, 0, 0, 0, None, None, None),
        llm_health=LLMHealthSummary("test", [], 0.0, 0.0, 0, 0.0),
        state_label=state, examples=[], metric_availability={},
    )


class _Archive:
    def __init__(self, evidence):
        self.evidence = evidence
        self.records = []

    def iter_records(self, kind):
        if kind is EvidenceSummary:
            return iter([self.evidence])
        return iter([r for r in self.records if isinstance(r, kind)])

    def append(self, rec):
        self.records.append(rec)
        return None


def test_high_cost_dispatch_cap_is_on_by_default_and_uses_alternative():
    busy = _Slot("1")
    busy.proc = object()
    busy.cand = _Cand("bc_running", "bindcraft")
    free = _Slot("2")
    pool = [busy, free]
    pending = ["bc_selected", "cx_good"]
    cand_by_id = {
        "bc_selected": _Cand("bc_selected", "bindcraft"),
        "cx_good": _Cand("cx_good", "complexa_beam"),
    }
    archive = _Archive(_evidence("stalled"))

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id, archive=archive, target=None,
        target_pdb="t", archive_root="/a", round_id=1, dispatch_fn=_ok_dispatch,
        dispatch_retries={},
    )

    assert n == 1
    assert free.cand.candidate_id == "cx_good"
    assert pending == ["bc_selected"]


def test_high_cost_dispatch_cap_keeps_feedback_lane_under_strong_evidence():
    busy_a = _Slot("1")
    busy_a.proc = object()
    busy_a.cand = _Cand("bc_running_a", "bindcraft")
    busy_b = _Slot("2")
    busy_b.proc = object()
    busy_b.cand = _Cand("bc_running_b", "bindcraft")
    free = _Slot("3")
    pool = [busy_a, busy_b, free]
    pending = ["bc_third", "cx_good"]
    cand_by_id = {
        "bc_third": _Cand("bc_third", "bindcraft"),
        "cx_good": _Cand("cx_good", "complexa_beam"),
    }
    evidence = dataclasses.replace(
        _evidence("productive"),
        completed_children=96,
        worker_gpu_h_total=6.0,
        method_health={
            "bindcraft": {
                "cumulative_gpu_h": 2.5,
                "chained_strict_yield_su": 3,
                "chained_su_per_gpu_h_recent": 1.2,
                "chained_su_per_gpu_h": 1.0,
            }
        },
    )
    archive = _Archive(evidence)

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id, archive=archive, target=None,
        target_pdb="t", archive_root="/a", round_id=1, dispatch_fn=_ok_dispatch,
        dispatch_retries={},
    )

    assert n == 1
    assert free.cand.candidate_id == "cx_good"
    assert pending == ["bc_third"]


def test_high_cost_cap_can_be_disabled_for_explicit_ablation(monkeypatch):
    monkeypatch.setenv("TREX_HIGH_COST_PENDING_FAMILIES", "")
    busy = _Slot("1")
    busy.proc = object()
    busy.cand = _Cand("bc_running", "bindcraft")
    free = _Slot("2")
    pool = [busy, free]
    pending = ["bc_blocked", "cx_good"]
    cand_by_id = {
        "bc_blocked": _Cand("bc_blocked", "bindcraft"),
        "cx_good": _Cand("cx_good", "complexa_beam"),
    }
    archive = _Archive(_evidence("stalled"))

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id, archive=archive, target=None,
        target_pdb="t", archive_root="/a", round_id=1, dispatch_fn=_ok_dispatch,
        dispatch_retries={}, dispatch_defer_seen=set(),
    )

    assert n == 1
    assert free.cand.candidate_id == "bc_blocked"
    assert pending == ["cx_good"]

def test_high_cost_cap_defer_audit_is_not_repeated_for_same_reason(monkeypatch, capsys):
    monkeypatch.setenv("TREX_HIGH_COST_PENDING_FAMILIES", "bindcraft")
    busy = _Slot("1")
    busy.proc = object()
    busy.cand = _Cand("bc_running", "bindcraft")
    archive = _Archive(_evidence("stalled"))
    seen = set()

    for _ in range(2):
        free = _Slot("2")
        pending = ["bc_blocked"]
        n = _dispatch_pending_to_free_slots(
            [busy, free], pending, {"bc_blocked": _Cand("bc_blocked", "bindcraft")},
            archive=archive, target=None, target_pdb="t", archive_root="/a",
            round_id=1, dispatch_fn=_ok_dispatch, dispatch_retries={},
            dispatch_defer_seen=seen,
        )
        assert n == 0
        assert pending == ["bc_blocked"]
        assert free.cand is None

    assert len(archive.records) == 1
    out = capsys.readouterr().out
    assert out.count("[dispatch_defer]") == 1

def test_repeated_support_dispatch_cap_matches_selector_guard(monkeypatch):
    monkeypatch.setenv("TREX_HIGH_COST_PENDING_FAMILIES", "bindcraft")
    e = dataclasses.replace(
        _evidence("stalled"),
        execution_realization={
            "by_family": {
                "bindcraft": {"proposed": 12, "selected": 4, "started": 1, "dispatch_deferred": 2, "selected_not_started": 1}
            }
        },
    )
    assert _high_cost_dispatch_cap(e, "bindcraft") == 2

    easy = dataclasses.replace(
        _evidence("productive_duplicate"),
        run_su_count=40,
        execution_realization=e.execution_realization,
    )
    assert _high_cost_dispatch_cap(easy, "bindcraft") == 1

    dry_duplicate = dataclasses.replace(
        easy,
        strict_duplicate_collapse_signal=True,
        gpu_h_since_last_su=1.25,
    )
    assert _high_cost_dispatch_cap(dry_duplicate, "bindcraft") == 2


def test_dispatch_prioritizes_current_route_value_over_stale_lifetime_memory():
    pool = [_Slot("1")]
    pending = ["old_mcts", "fresh_bc"]
    cand_by_id = {
        "old_mcts": _Cand(
            "old_mcts",
            "complexa_mcts",
            expected_signal="route_value_replay complexa_mcts source=lifetime_memory no_recent_signal=1",
            supervisor_mode="exploit",
        ),
        "fresh_bc": _Cand(
            "fresh_bc",
            "bindcraft",
            expected_signal="route_value_replay bindcraft new_su_per_gpu_h=1.2 source=gpu_recent",
            supervisor_mode="exploit",
        ),
    }
    archive = _Archive(_evidence("productive"))
    archive.records.extend([
        LaunchDecision(
            launch_id="old",
            tick_id="v7r040",
            candidate_id="old_mcts",
            status="launched",
            resource_class_concrete={"mode": "exploit"},
            why="old lifetime replay",
        ),
        LaunchDecision(
            launch_id="fresh",
            tick_id="v7r045",
            candidate_id="fresh_bc",
            status="launched",
            resource_class_concrete={"mode": "exploit"},
            why="fresh gpu_recent route",
        ),
    ])

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id, archive=archive, target=None,
        target_pdb="t", archive_root="/a", round_id=45, dispatch_fn=_ok_dispatch,
        dispatch_retries={},
    )

    assert n == 1
    assert pool[0].cand.candidate_id == "fresh_bc"
    assert pending == ["old_mcts"]


def test_dispatch_prioritizes_newer_supervisor_candidate_over_old_cross_family_escape():
    pool = [_Slot("1")]
    pending = ["old_cross_bg", "fresh_bc"]
    cand_by_id = {
        "old_cross_bg": _Cand(
            "old_cross_bg",
            "boltzgen",
            expected_signal="cross_family_escape dominant_root=complexa family=boltzgen",
            supervisor_mode="explore",
        ),
        "fresh_bc": _Cand(
            "fresh_bc",
            "bindcraft",
            expected_signal="Supervisor pivot: bindcraft is the best current hypothesis",
            supervisor_mode="exploit",
        ),
    }
    archive = _Archive(_evidence("productive"))
    archive.records.extend([
        LaunchDecision(
            launch_id="old_cross",
            tick_id="v7r061",
            candidate_id="old_cross_bg",
            status="launched",
            resource_class_concrete={"mode": "explore"},
            why="old cross-family floor",
        ),
        LaunchDecision(
            launch_id="fresh_bc",
            tick_id="v7r064",
            candidate_id="fresh_bc",
            status="launched",
            resource_class_concrete={"mode": "exploit"},
            why="fresh supervisor rank1 bindcraft pivot",
        ),
    ])

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id, archive=archive, target=None,
        target_pdb="t", archive_root="/a", round_id=64, dispatch_fn=_ok_dispatch,
        dispatch_retries={},
    )

    assert n == 1
    assert pool[0].cand.candidate_id == "fresh_bc"
    assert pending == ["old_cross_bg"]


def test_fills_all_free_slots_in_order():
    pool = [_Slot("1"), _Slot("2"), _Slot("3")]
    pending = ["c1", "c2", "c3", "c4"]
    cand_by_id = {c: _Cand(c) for c in pending}
    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id,
        archive=None, target=None, target_pdb="t.pdb",
        archive_root="/a", round_id=1, dispatch_fn=_ok_dispatch,
    )
    assert n == 3
    assert all(s.busy for s in pool)
    assert pending == ["c4"], "one candidate should remain queued"
    assert [s.cand.candidate_id for s in pool] == ["c1", "c2", "c3"]


def test_skips_infeasible_then_dispatches_next():
    pool = [_Slot("1")]
    pending = ["bad", "good"]
    cand_by_id = {"bad": _Cand("bad"), "good": _Cand("good")}

    def dispatch(cand, *, gpu_id, **kw):
        return None if cand.candidate_id == "bad" else _ok_dispatch(cand, gpu_id=gpu_id)

    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id,
        archive=None, target=None, target_pdb="t.pdb",
        archive_root="/a", round_id=1, dispatch_fn=dispatch,
    )
    assert n == 1
    assert pool[0].busy and pool[0].cand.candidate_id == "good"
    assert pending == [], "infeasible 'bad' dropped, 'good' dispatched"


def test_busy_slots_untouched_and_empty_queue_is_noop():
    busy = _Slot("1"); busy.proc = object()  # already running
    free = _Slot("2")
    pool = [busy, free]
    # empty queue → nothing dispatched, busy slot untouched
    n = _dispatch_pending_to_free_slots(
        pool, [], {}, archive=None, target=None, target_pdb="t",
        archive_root="/a", round_id=1, dispatch_fn=_ok_dispatch,
    )
    assert n == 0
    assert free.proc is None and busy.busy

    # one queued → only the free slot is filled, busy slot keeps its proc
    orig_proc = busy.proc
    n = _dispatch_pending_to_free_slots(
        pool, ["c1"], {"c1": _Cand("c1")}, archive=None, target=None,
        target_pdb="t", archive_root="/a", round_id=2, dispatch_fn=_ok_dispatch,
    )
    assert n == 1
    assert free.busy and busy.proc is orig_proc


def test_transient_dispatch_failure_requeued_for_retry():
    """Requeue transient dispatch failures within the retry limit."""
    pool = [_Slot("1")]
    pending = ["c1"]
    retries: dict[str, int] = {}
    n = _dispatch_pending_to_free_slots(
        pool, pending, {"c1": _Cand("c1")}, archive=None, target=None,
        target_pdb="t", archive_root="/a", round_id=1,
        dispatch_fn=lambda *a, **k: None,  # always (transiently) fails
        dispatch_retries=retries,
    )
    assert n == 0
    assert pending == ["c1"], "transient failure re-queued for a later round"
    assert retries["c1"] == 1


def test_dispatch_retry_is_bounded_no_infinite_requeue():
    """After _MAX_DISPATCH_RETRIES, a persistently-failing candidate is dropped."""
    from trex.controller import _MAX_DISPATCH_RETRIES
    pool = [_Slot("1")]
    pending = ["c1"]
    retries: dict[str, int] = {}
    for _ in range(_MAX_DISPATCH_RETRIES):  # slot never fills (dispatch fails) → stays free
        _dispatch_pending_to_free_slots(
            pool, pending, {"c1": _Cand("c1")}, archive=None, target=None,
            target_pdb="t", archive_root="/a", round_id=1,
            dispatch_fn=lambda *a, **k: None, dispatch_retries=retries,
        )
        assert pending == ["c1"]
    _dispatch_pending_to_free_slots(
        pool, pending, {"c1": _Cand("c1")}, archive=None, target=None,
        target_pdb="t", archive_root="/a", round_id=1,
        dispatch_fn=lambda *a, **k: None, dispatch_retries=retries,
    )
    assert pending == [], "dropped after the retry cap (no infinite re-queue)"


def test_no_retries_dict_preserves_original_drop_behavior():
    """Backward-compat: without dispatch_retries, a failed dispatch is dropped."""
    pool = [_Slot("1")]
    pending = ["bad"]
    n = _dispatch_pending_to_free_slots(
        pool, pending, {"bad": _Cand("bad")}, archive=None, target=None,
        target_pdb="t", archive_root="/a", round_id=1,
        dispatch_fn=lambda *a, **k: None,
    )
    assert n == 0 and pending == [], "no retries dict → original drop behavior"


def test_missing_candidate_id_is_skipped():
    pool = [_Slot("1")]
    pending = ["ghost", "real"]
    cand_by_id = {"real": _Cand("real")}  # 'ghost' not resolvable
    n = _dispatch_pending_to_free_slots(
        pool, pending, cand_by_id,
        archive=None, target=None, target_pdb="t",
        archive_root="/a", round_id=1, dispatch_fn=_ok_dispatch,
    )
    assert n == 1
    assert pool[0].cand.candidate_id == "real"
    assert pending == []
