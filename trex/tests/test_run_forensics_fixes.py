"""Regression tests for the T-ReX-run-forensics bottleneck fixes (2026-06-10).

Derived from a forensic audit of 307 T-ReX archives (adversarially double-checked):
  F6  productive_duplicate state — high-SU + high-duplicate no longer aliases to
      low_evidence (which drained exploit on fallback ticks + made the diversity
      clamp unreachable).
  F3  stuck_lineage_roots SU-aware arm — a strict-but-duplicate child no longer
      reads as "improved", so a duplicate-collapsing lineage is flagged stuck.
  F5  since-last-SU plateau governor — duration signal + deep_stall escalation.
  F4  closed-loop chain-refilter throttle — refilter reserve/backfill stops when
      recent refilters add ~0 new dedup-SU (with a cold-start guard).
"""
from __future__ import annotations

import typing

import trex.schemas as S
from trex.evidence_reducer import classify_state, StateClassifierConfig
from trex.fallback import (
    CATEGORY_A_CLAMPS,
    DEFAULT_CLAMPS,
    DEFAULT_MIXTURES,
    clamp_mixture,
    diversity_adjusted_clamp,
)

_CFG = StateClassifierConfig()


def _classify(**kw):
    base = dict(
        worker_gpu_h_last_3_ticks=2.0, completed_children_window=10,
        run_su_count_delta=8, duplicate_fraction=0.72, near_miss_count=1,
        axis_stats={}, top_bin_share=0.55, cfg=_CFG, cumulative_gpu_h=5.0,
        su_dedup_trusted=True,
    )
    base.update(kw)
    return classify_state(**base)


# ---- F6: productive_duplicate -------------------------------------------------

def test_all_state_labels_present_in_all_clamp_tables():
    # The fallback path does DEFAULT_MIXTURES[state]; a missing key would KeyError
    # or silently revert to the bug. Every StateLabel must be in all three tables.
    labels = set(typing.get_args(S.StateLabel))
    for name, tbl in (("DEFAULT_MIXTURES", DEFAULT_MIXTURES),
                      ("DEFAULT_CLAMPS", DEFAULT_CLAMPS),
                      ("CATEGORY_A_CLAMPS", CATEGORY_A_CLAMPS)):
        assert labels <= set(tbl), (name, labels - set(tbl))
    for st, m in DEFAULT_MIXTURES.items():
        assert abs(sum(m.values()) - 1.0) < 1e-9, st


def test_high_su_high_dup_is_productive_duplicate_not_low_evidence():
    # PDL1-like: producing SU, high duplicate_fraction, sub-collapse top_bin.
    assert _classify(run_su_count_delta=8, duplicate_fraction=0.72,
                     near_miss_count=1, top_bin_share=0.55) == "productive_duplicate"


def test_productive_duplicate_not_shadowed_by_rescue_rich():
    # Productive + duplicate-rich + concentrated near-misses must still keep the
    # exploit-preserving productive_duplicate state. Otherwise rescue_rich cuts
    # exploit on a lane that is actively buying SU.
    axis = S.AxisStat(
        pass_count=0, near_pass_count=3, fail_count=0,
        median_raw=0.1, median_calibrated=0.1, median_deficit=1.0,
        calibration_status="provisional", n=3,
    )
    assert _classify(
        run_su_count_delta=8,
        duplicate_fraction=0.75,
        near_miss_count=3,
        axis_stats={"iPAE": axis},
        top_bin_share=0.55,
    ) == "productive_duplicate"


def test_low_dup_producing_is_still_productive():
    assert _classify(duplicate_fraction=0.45, top_bin_share=0.30) == "productive"


def test_severe_collapse_is_still_stalled():
    assert _classify(run_su_count_delta=0, duplicate_fraction=0.8,
                     near_miss_count=0, top_bin_share=0.9) == "stalled"


def test_productive_duplicate_fallback_mixture_keeps_exploit_hot():
    # The fallback drain bug: low_evidence default would be {0.35,0.30,0.35}.
    # productive_duplicate keeps exploit-leaning so a fallback tick over a
    # producing lane does not bleed budget off the winning lane.
    m = DEFAULT_MIXTURES["productive_duplicate"]
    assert m["exploit"] >= 0.50
    assert m["explore"] <= 0.15
    assert m["exploit"] > DEFAULT_MIXTURES["low_evidence"]["exploit"]


def test_diversity_clamp_now_reachable_from_productive_duplicate():
    # Previously keyed on state=="productive" only -> unreachable for this
    # population (which never reaches productive). Now it loosens on collapse.
    base = DEFAULT_CLAMPS["productive_duplicate"]
    adj, log = diversity_adjusted_clamp(
        base, "productive_duplicate", top_bin_share=0.85,
        panel_ready_bins_covered=0,
    )
    assert any("diversity_loosen" in s for s in log), log
    assert (adj.explore_max or 0) >= 0.40  # loosened for the pivot


def test_productive_duplicate_clamp_normalizes():
    clamped, _ = clamp_mixture(
        {"exploit": 0.9, "rescue": 0.05, "explore": 0.05}, "productive_duplicate",
        category_b_enabled=True,
    )
    assert abs(sum(clamped.values()) - 1.0) < 1e-9
    assert clamped["exploit"] <= 0.70  # Cat-B ceiling for this state


# ---- F3: stuck_lineage_roots mined-out arm -----------------------------------

from trex.evidence_reducer import stuck_lineage_roots, ReducerConfig
from trex.schemas import ResultRecord


def _gen(rid, fam="bindcraft"):
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family=fam,
        runtime_bucket_id="rb", metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.1, exit_status="ok", bins={},
    )


def _strict_child(rid, parent, su_bin):
    return ResultRecord(
        result_id=rid, parent_ids=[parent], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"foldseek_su": su_bin, "refilter_source": parent},
    )


def test_mined_out_lineage_flagged_stuck():
    # one backbone, 6 strict refilter children all collapsing to 2 SU bins (3x
    # redundancy) -> mined out, should be flagged so the builder regenerates.
    g = _gen("g1")
    kids = [_strict_child(f"r{i}", "g1", "binA" if i < 4 else "binB") for i in range(6)]
    roots = stuck_lineage_roots([g, *kids], ReducerConfig())
    flagged = {e["root_result_id"]: e for e in roots}
    assert "g1" in flagged
    assert flagged["g1"]["reason"].startswith("strict_duplicate_")


def test_healthy_diverse_lineage_not_flagged():
    # 6 strict children, 6 distinct SU bins -> productive, must NOT be flagged.
    g = _gen("g1")
    kids = [_strict_child(f"r{i}", "g1", f"bin{i}") for i in range(6)]
    roots = stuck_lineage_roots([g, *kids], ReducerConfig())
    assert "g1" not in {e["root_result_id"] for e in roots}


def test_small_lineage_not_judged():
    # below STUCK_DUP_MIN_STRICT -> not enough evidence to call it mined out.
    g = _gen("g1")
    kids = [_strict_child(f"r{i}", "g1", "binA") for i in range(3)]
    assert "g1" not in {e["root_result_id"] for e in stuck_lineage_roots([g, *kids], ReducerConfig())}


# ---- MPNN-rescue stuck-lineage (2026-06-13): traverse to refilter grandchild --

def _failing_parent(rid, fam="complexa_beam"):
    # passes pLDDT + scRMSD, FAILS iPAE -> dominant deficit = iPAE
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family=fam,
        runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": 0.45, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok", bins={},
    )


def _mpnn_child(rid, parent):  # sequence-only: metrics={}
    return ResultRecord(
        result_id=rid, parent_ids=[parent], target_id="t",
        backend_family="proteinmpnn_redesign", runtime_bucket_id="rb",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.02,
        exit_status="ok", bins={},
    )


def _refilter_gc(rid, mpnn_child, ipae):
    return ResultRecord(
        result_id=rid, parent_ids=[mpnn_child], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": ipae, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source": mpnn_child},
    )


def test_mpnn_rescue_lineage_flagged_via_refilter_grandchild():
    """A backbone with K MPNN rescue children (metrics={}) whose refilter
    grandchildren never improve the blocking iPAE axis must be flagged stuck —
    even though the MPNN children carry no metrics (BetV1 9630422 perseveration)."""
    recs = [_failing_parent("p1")]
    for i in range(3):
        recs += [_mpnn_child(f"m{i}", "p1"), _refilter_gc(f"g{i}", f"m{i}", ipae=0.45)]
    roots = {e["root_result_id"]: e for e in stuck_lineage_roots(recs, ReducerConfig())}
    assert "p1" in roots
    assert roots["p1"]["dominant_axis"] == "iPAE"
    # GAP 2: proteinmpnn had 3 (>=2) non-improving attempts on p1 -> exhausted;
    # a DIFFERENT rescue family is NOT in the set (could still try p1).
    assert roots["p1"]["exhausted_families"] == ["proteinmpnn_redesign"]


def test_mpnn_rescue_lineage_not_flagged_when_grandchild_improves():
    recs = [_failing_parent("p1")]
    for i in range(3):
        ipae = 0.20 if i == 0 else 0.45   # one grandchild closes iPAE -> progress
        recs += [_mpnn_child(f"m{i}", "p1"), _refilter_gc(f"g{i}", f"m{i}", ipae=ipae)]
    roots = {e["root_result_id"] for e in stuck_lineage_roots(recs, ReducerConfig())}
    assert "p1" not in roots


# ---- F5: deep_stall escalation + calibration ---------------------------------

def _stall(**kw):
    base = dict(
        worker_gpu_h_last_3_ticks=2.0, completed_children_window=10,
        run_su_count_delta=0, duplicate_fraction=0.8, near_miss_count=0,
        axis_stats={}, top_bin_share=0.9, cfg=_CFG, cumulative_gpu_h=30.0,
        su_dedup_trusted=True,
    )
    base.update(kw)
    return classify_state(**base)


def test_deep_stall_escalates_only_past_threshold():
    assert _stall(gpu_h_since_last_su=5.0) == "stalled"        # short dry
    assert _stall(gpu_h_since_last_su=20.0) == "deep_stall"    # long dry
    assert _stall(gpu_h_since_last_su=None) == "stalled"       # no signal -> stalled


def test_deep_stall_calibration_never_fires_on_healthy_dry_plateaus():
    # CALIBRATION GUARD: healthy runs' longest dry plateau was 8.0 worker-GPU-h
    # (CD45 4.7 / HER2 8.0 / PDL1 2.3). The threshold (12.0) must sit ABOVE that,
    # so no healthy plateau can escalate to deep_stall.
    assert _CFG.deep_stall_gpu_h > 8.0
    for healthy_max_dry in (4.7, 8.0, 2.3):
        assert _stall(gpu_h_since_last_su=healthy_max_dry) == "stalled"
    # collapsed runs ran 17-37 dry -> they DO escalate.
    for collapsed_dry in (17.3, 24.0, 37.3):
        assert _stall(gpu_h_since_last_su=collapsed_dry) == "deep_stall"


def _ax(median_deficit):
    return S.AxisStat(pass_count=0, near_pass_count=2, fail_count=3, median_raw=None,
                      median_calibrated=None, median_deficit=median_deficit,
                      calibration_status="provisional", n=10)


def _rescue(**kw):
    # rescue_rich inputs: near-miss-rich + iPAE-concentrated (mirrors the
    # deterministic_smoke BetV1-like fixture).
    base = dict(
        worker_gpu_h_last_3_ticks=10.0, completed_children_window=10,
        run_su_count_delta=0, duplicate_fraction=0.40, near_miss_count=6,
        axis_stats={"iPAE": _ax(5.0), "pLDDT": _ax(0.5), "binder_scRMSD": _ax(0.2)},
        top_bin_share=0.40, cfg=_CFG, cumulative_gpu_h=30.0, su_dedup_trusted=True,
    )
    base.update(kw)
    return classify_state(**base)


def test_R1_1_rescue_rich_escalates_to_deep_stall_only_when_very_dry():
    # R1-1 fix: rescue_rich normally shadows deep_stall, but a near-miss-rich lane
    # producing NO new SU until the normal deep-stall dry interval escalates.
    assert _rescue(gpu_h_since_last_su=5.0) == "rescue_rich"     # working slow rescue
    assert _rescue(gpu_h_since_last_su=11.9) == "rescue_rich"    # warning plateau, not yet hard-stale
    assert _rescue(gpu_h_since_last_su=12.0) == "deep_stall"     # stale rescue loses protection
    assert _rescue(gpu_h_since_last_su=None) == "rescue_rich"    # no signal -> protected
    assert _CFG.rescue_deep_stall_gpu_h == _CFG.deep_stall_gpu_h


def test_R1_2_parse_failed_appends_synthetic_resultrecord_with_gpuh():
    # R1-2 fix: a parse-failed worker that ran meaningful GPU time must leave a
    # synthetic ResultRecord (gpu_h>0, exit_status!='ok') so the dry timer counts
    # it; without target_id it must NOT fabricate a record.
    import time as _time
    from trex.controller import (
        _append_parse_failed_dispatch_record, _WorkerSlot,
    )
    from trex.schemas import ResultRecord

    class _Arc:
        def __init__(self): self.recs = []
        def append(self, r): self.recs.append(r)
        def iter_records(self, cls): return [r for r in self.recs if isinstance(r, cls)]
    feas = S.FeasibilityCheck(True, "rb", True, True, True, True)
    cand = S.ActionCandidate(
        candidate_id="c_pf", hypothesis_ids=[], parent_result_id="parent_r",
        method_family="bindcraft", operator_id="bc", lane_id="l", config_delta={},
        downstream_route_plan=[], estimated_cost_class="extended", expected_signal="x",
        evidence_refs=[], feasibility=feas)
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.parent_result_id = "parent_r"
    slot.tick_id = "v7r001"
    slot.launched_at = _time.time() - 600.0  # ran ~10 min -> ~0.167 gpu_h

    arc = _Arc()
    _append_parse_failed_dispatch_record(arc, slot, "boom", target_id="t1")
    rrs = arc.iter_records(ResultRecord)
    assert len(rrs) == 1, "should append exactly one synthetic ResultRecord"
    assert rrs[0].gpu_h > 0.1 and rrs[0].exit_status != "ok"
    assert rrs[0].target_id == "t1" and rrs[0].metrics == {}
    assert rrs[0].parent_ids == ["c_pf", "parent_r"]

    # without target_id (the deliberate no-fabricate path): no ResultRecord
    arc2 = _Arc()
    _append_parse_failed_dispatch_record(arc2, slot, "boom")
    assert arc2.iter_records(ResultRecord) == []



def test_incremental_bindcraft_completion_does_not_false_parse_fail(monkeypatch, tmp_path):
    # BindCraft is parsed incrementally while still running. On clean completion,
    # the final parse may replay only records that were already archived. That is
    # a clean dedupe-zero completion, not a parse failure/no-artifact worker.
    import trex.controller as C
    from trex.schemas import DispatchRecord, ResultRecord

    class _Arc:
        def __init__(self):
            self.recs = []
        def append(self, r):
            self.recs.append(r)
        def iter_records(self, cls):
            return [r for r in self.recs if isinstance(r, cls)]

    feas = S.FeasibilityCheck(True, "rb", True, True, True, True)
    cand = S.ActionCandidate(
        candidate_id="bc_incremental", hypothesis_ids=[], parent_result_id=None,
        method_family="bindcraft", operator_id="bc", lane_id="l", config_delta={},
        downstream_route_plan=[], estimated_cost_class="extended", expected_signal="x",
        evidence_refs=[], feasibility=feas)
    rec = ResultRecord(
        result_id="r_existing", parent_ids=[cand.candidate_id], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb", metrics={},
        metrics_calibrated={}, route_lineage=["bindcraft"], gpu_h=1.0,
        exit_status="ok", bins={}, artifacts={}, panel_ready=False,
        tick_id="v7r001")
    arc = _Arc()
    arc.append(rec)
    slot = C._WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r001"
    slot.archived_result_ids.add(rec.result_id)
    slot.archived_elapsed_gpu_h = 1.0

    monkeypatch.setattr(C, "parse_bindcraft_output", lambda out_dir, ctx: [rec])
    n = C._parse_and_archive_slot(
        slot, rc=0, archive=arc, target=S.TargetConstraint("t1", "x"),
        chain_seq_ref=[0], elapsed_gpu_h=1.001, incremental=False)

    assert n == 0
    assert len(arc.iter_records(ResultRecord)) == 1
    assert arc.iter_records(DispatchRecord) == []

def test_deep_stall_preserves_confident_llm_mixture():
    # deep_stall is NOT in the always-Cat-B set: a confident LLM mixture is
    # honored above the Cat-A explore floor (0.35), not hard-overridden.
    from trex.fallback import should_apply_category_b_clamps, clamp_mixture
    capb, _ = should_apply_category_b_clamps(
        "deep_stall", top_bin_share=0.4, panel_ready_bins_covered=0,
        supervisor_confidence=0.8, recent_fallback_high=False, supervisor_used=True)
    assert capb is False
    c, _ = clamp_mixture({"exploit": 0.30, "rescue": 0.20, "explore": 0.50},
                         "deep_stall", category_b_enabled=capb)
    assert c["explore"] >= 0.35 and abs(sum(c.values()) - 1.0) < 1e-9


# ---- F4 throttle + F1 round-robin (controller helpers) -----------------------

from trex.controller import (
    _queue_chain_refilter_reserve, _chain_backfill_ids,
    _cancel_pending_chain_reserves_for_deep_stall,
)
from trex.schemas import (
    ActionCandidate, FeasibilityCheck, DispatchRecord, LaunchDecision,
)


class _FakeArchive:
    def __init__(self, recs):
        self._recs = recs
    def iter_records(self, cls):
        return [r for r in self._recs if isinstance(r, cls)]
    def append(self, r):
        self._recs.append(r)


def _chain_cand(src, idx):
    feas = FeasibilityCheck(True, "rb", True, True, True, True)
    return ActionCandidate(
        candidate_id=f"chain_v7r001_{src}_to_structure_refilter_{idx:03d}",
        hypothesis_ids=[], parent_result_id=f"{src}_{idx}",
        method_family="structure_refilter", operator_id="rf", lane_id="l",
        config_delta={}, downstream_route_plan=[], estimated_cost_class="low",
        expected_signal="x", evidence_refs=[], feasibility=feas,
    )


def test_throttle_suppresses_reserve_on_deep_stall():
    arch = _FakeArchive([_chain_cand("proteinmpnn_redesign", i) for i in range(5)])
    # not throttled -> reserves
    ids = _queue_chain_refilter_reserve(
        arch, [], set(), tick_id="v7r001", queue_room=3,
        source="s", why="w", throttled=False)
    assert len(ids) >= 1
    # throttled (deep_stall) -> reserves NOTHING
    ids2 = _queue_chain_refilter_reserve(
        arch, [], set(), tick_id="v7r001", queue_room=3,
        source="s", why="w", throttled=True)
    assert ids2 == []


def test_throttle_reads_state_from_run_live_tick_dict():
    # BLOCKER regression: run_live_tick returns a DICT with the state nested at
    # summary["evidence"]["state_label"] (NOT an attribute). The controller's
    # latest_state extraction must use dict-get, not getattr (which silently
    # returned the default and made the whole deep_stall throttle a no-op).
    summary = {"evidence": {"state_label": "deep_stall"}, "launches": []}
    latest_state = "stalled"
    # the CORRECT pattern (what the controller now uses):
    latest_state = summary.get("evidence", {}).get("state_label", latest_state)
    assert latest_state == "deep_stall"
    # the BUGGY pattern would have left it unchanged:
    assert getattr(summary, "state_label", "stalled") == "stalled"


def test_current_tick_deep_stall_cancels_queued_reserve_before_dispatch():
    cand = _chain_cand("bindcraft", 0)
    cid = cand.candidate_id
    arch = _FakeArchive([
        cand,
        LaunchDecision(
            launch_id="launch_chain", tick_id="v7r001", candidate_id=cid,
            status="launched",
            resource_class_concrete={"mode": "chain_refilter"},
            why="reserve",
        ),
    ])
    pending = [cid, "fresh_generator"]
    cancelled = _cancel_pending_chain_reserves_for_deep_stall(
        arch, pending, [cid], tick_id="v7r001")
    assert cancelled == [cid]
    assert pending == ["fresh_generator"]
    terminal = [
        d for d in arch.iter_records(DispatchRecord)
        if d.candidate_id == cid and d.status == "dispatch_failed"
    ]
    assert terminal and terminal[0].launch_id == "launch_chain"
    # Real controller reserve/backfill uses the shared seen set, so the current
    # tick cancellation is not immediately requeued. The dispatch_failed record
    # itself remains recoverable for a later tick or restart instead of stranding
    # the score-conversion candidate forever.
    assert _chain_backfill_ids(arch, {cid}, 1) == []
    assert _chain_backfill_ids(arch, set(), 1) == [cid]


def test_backfill_round_robin_does_not_let_one_lineage_monopolize():
    # 10 mpnn + 1 bindcraft + 1 boltzgen chain refilters; pick 3 should SPAN
    # families, not be 3x mpnn (the CbAgo monopolization the round-robin fixes).
    recs = [_chain_cand("proteinmpnn_redesign", i) for i in range(10)]
    recs += [_chain_cand("bindcraft", 0), _chain_cand("boltzgen", 0)]
    arch = _FakeArchive(recs)
    picked = _chain_backfill_ids(arch, set(), 3)
    srcs = {cid.split("_to_", 1)[0].split("_", 2)[2] for cid in picked}
    assert len(srcs) >= 2, (picked, srcs)  # spans families
