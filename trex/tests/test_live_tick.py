"""Live tick orchestration: archive persistence + lifecycle update.

These tests stub the Planner and Supervisor LLM calls (no GPU/API
required) and verify that records flow correctly into the archive
and that hypothesis lifecycle transitions when descendants exist.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from trex.archive import Archive
from trex.candidate_builder import WARMSTART_FAMILIES
from trex.foldseek_clusterer import ClusteringResult
from trex.lifecycle import LifecycleConfig
from trex.live_tick import (
    FoldseekConfig, LiveTickConfig, SequenceDedupConfig,
    _archive_llm_health,
    _compact_selector_context_for_archive, _compact_selector_debug_for_archive,
    _supervisor_decision_record,
    run_live_tick,
)
from trex.refilter_roles import CANONICAL_SCORE_CONVERSION
from trex.schemas import (
    ActionCandidate,
    EvidenceSummary,
    FeasibilityCheck,
    HypothesisCard,
    LLMCallRecord,
    LaunchDecision,
    PlannerOutput,
    PredictedChange,
    PreserveConstraint,
    ResultRecord,
    SupervisorDecision,
    SupervisorOutput,
    TargetConstraint,
)


def _result(rid: str, *, parent_ids: list[str] | None = None,
            pLDDT: float | None = 85.0, iPAE: float | None = 0.3,
            scRMSD: float | None = 1.5, target: str = "t1",
            family: str = "complexa_beam") -> ResultRecord:
    m = {}
    if pLDDT is not None: m["pLDDT"] = pLDDT
    if iPAE is not None: m["iPAE"] = iPAE
    if scRMSD is not None: m["binder_scRMSD"] = scRMSD
    return ResultRecord(
        result_id=rid, parent_ids=parent_ids or [], target_id=target,
        backend_family=family, runtime_bucket_id="rb1",
        metrics=m, metrics_calibrated=dict(m), route_lineage=[],
        gpu_h=0.4, exit_status="ok",
    )


def _hyp(hid: str = "h1", target: str = "t1", tick: int = 0) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid, target_id=target, tick_created=tick,
        claim="iPAE improves with MPNN",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["r_baseline"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
    )


def _action(cid: str = "c1", hid: str = "h1", parent_result_id: str = "r_baseline") -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=[hid], parent_result_id=parent_result_id,
        method_family="proteinmpnn_redesign", operator_id="interface_redesign",
        lane_id="proteinmpnn_redesign", config_delta={},
        downstream_route_plan=[], estimated_cost_class="low",
        expected_signal="iPAE down", evidence_refs=["e1"], feasibility=feas,
    )


def _planner_ok(cards: list[HypothesisCard] | None = None) -> PlannerOutput:
    return PlannerOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        cards=cards or [], rationale="x",
        raw_text='{"cards": []}', usage={},
    )


def _sup_ok(decisions=None) -> SupervisorOutput:
    return SupervisorOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        mode_mixture={"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        candidate_decisions=decisions or [], rationale="y",
        raw_text='{"mode_mixture": {}}', usage={},
    )


# ---- tests -----------------------------------------------------------------


def test_supervisor_decision_backfills_fallback_mode_mixture_from_selector_debug():
    sup = dataclasses.replace(_sup_ok(), mode_mixture={})
    raw = {"exploit": 0.25, "rescue": 0.5, "explore": 0.25}
    rec = _supervisor_decision_record(
        sup, tick_id="t_fallback", fallback_used=True, clamps=[],
        selector_debug={"raw_mixture": raw, "clamped_mixture": {"exploit": 0.2, "rescue": 0.6, "explore": 0.2}},
    )
    assert rec.fallback_used
    assert rec.mode_mixture == raw


def test_supervisor_decision_persists_compact_selector_context():
    sup = _sup_ok()
    context = {
        "quota_realization": "largest_remainder",
        "recent_realized_modes": {"window_disabled": True, "recent_modes": ["exploit"]},
        "cost_admission": {
            "capacity_pressure_families": ["bindcraft"],
            "high_cost_caps": {"bindcraft": {"cap_source": "dry_pivot", "running": 2}},
        },
        "score_conversion_backlog": {"total_unscored_diagnostic_artifacts": 12},
        "pending_family_load": {"by_family": {"bindcraft": {"running": 2, "queued": 1}}},
        "execution_realization": {"by_family": {"bindcraft": {"proposed": 4, "started": 2}}},
        "oversized": list(range(100)),
    }
    rec = _supervisor_decision_record(
        sup, tick_id="t1", fallback_used=False, clamps=[],
        selector_debug={"capacity_pressure_families": ["bindcraft"]},
        selector_context=context,
    )
    assert rec.selector_context["quota_realization"] == "largest_remainder"
    assert rec.selector_context["cost_admission"]["capacity_pressure_families"] == ["bindcraft"]
    assert rec.selector_context["score_conversion_backlog"]["total_unscored_diagnostic_artifacts"] == 12
    assert "oversized" not in rec.selector_context
    assert rec.selector_debug["capacity_pressure_families"] == ["bindcraft"]


def test_selector_context_compaction_is_bounded():
    context = {"cost_admission": {f"k{i}": i for i in range(40)}}
    compact = _compact_selector_context_for_archive(context)
    assert compact["cost_admission"]["_truncated_keys"] > 0


def test_live_tick_persists_evidence_and_llm_calls(tmp_path: Path):
    arc = Archive(tmp_path / "arc1")
    # seed with a few results so reducer can run
    for i in range(5):
        arc.append(_result(f"r{i}"))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )
    s = arc.summary()
    assert s["evidence_summaries.jsonl"] == 1
    assert s["llm_call_records.jsonl"] == 2  # planner + supervisor
    assert s["supervisor_decisions.jsonl"] == 1
    assert summary["target_id"] == "t1"


def test_archive_llm_health_reflects_recent_call_records(tmp_path: Path):
    arc = Archive(tmp_path / "arc_llm_health")
    statuses = [
        ("ok", None, False, 1.0),
        ("parse_fail", None, False, 2.0),
        ("schema_fail", None, False, 3.0),
        ("timeout", None, False, 4.0),
        ("call_error", None, False, 5.0),
        ("no_hypotheses_or_candidates", None, False, 6.0),
        ("ok", 0.2, False, 7.0),
        ("ok", None, True, 8.0),
    ]
    for i, (status, confidence, abstain, latency) in enumerate(statuses):
        arc.append(LLMCallRecord(
            call_id=f"llm{i}",
            tick_id=f"t{i}",
            role="planner",
            model="m",
            model_digest=None,
            prompt_hash=f"h{i}",
            schema_version="v",
            latency_s=latency,
            tokens_in=1,
            tokens_out=1,
            parse_status=status,  # type: ignore[arg-type]
            confidence=confidence,
            abstain=abstain,
            fallback_triggered=status != "ok",
        ))

    health = _archive_llm_health(arc, "m", window=8)
    assert health.last_calls_window == [
        "ok", "parse_fail", "schema_fail", "timeout", "call_error",
        "no_hypotheses_or_candidates", "low_conf", "abstain",
    ]
    assert health.parse_fail_rate == 0.125
    assert health.schema_fail_rate == 0.125
    assert health.timeout_count == 1
    assert health.median_latency_s == 4.5


def test_live_tick_evidence_contains_archive_derived_llm_health(tmp_path: Path):
    arc = Archive(tmp_path / "arc_live_llm_health")
    arc.append(_result("r0"))
    arc.append(LLMCallRecord(
        call_id="prior_schema_fail",
        tick_id="t0",
        role="planner",
        model="m",
        model_digest=None,
        prompt_hash="h0",
        schema_version="v",
        latency_s=2.5,
        tokens_in=1,
        tokens_out=1,
        parse_status="schema_fail",
        confidence=None,
        abstain=False,
        fallback_triggered=True,
    ))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.llm_health.last_calls_window == ["schema_fail"]
    assert ev.llm_health.schema_fail_rate == 1.0
    assert ev.llm_health.median_latency_s == 2.5


def test_worker_wall_gpu_count_is_not_free_slot_count(tmp_path: Path):
    arc = Archive(tmp_path / "arc_wall_gpu_count")
    arc.append(_result("r0"))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=2.0, remaining_wall_h=46.0,
            cfg=LiveTickConfig(worker_wall_gpu_count=3),
            available_slots_override=1,
        )

    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.worker_wall_gpu_count == 3.0
    assert ev.worker_wall_gpu_h_total == 6.0


def test_worker_wall_gpu_count_scales_beyond_reference_pool(tmp_path: Path):
    arc = Archive(tmp_path / "arc_wall_gpu_count_5")
    arc.append(_result("r0"))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=2.0, remaining_wall_h=46.0,
            cfg=LiveTickConfig(worker_wall_gpu_count=5),
            available_slots_override=5,
        )

    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.worker_wall_gpu_count == 5.0
    assert ev.worker_wall_gpu_h_total == 10.0
    supervisor = next(c for c in arc.iter_records(LLMCallRecord) if c.role == "supervisor")
    assert supervisor.prompt_audit["selector_context"]["available_slots_now"] == 5


def test_live_tick_persists_diagnostic_tldr_and_prompt_audit(tmp_path: Path):
    arc = Archive(tmp_path / "arc_diag")
    for i in range(3):
        arc.append(ResultRecord(
            result_id=f"cx{i}", parent_ids=[], target_id="t1",
            backend_family="complexa_beam", runtime_bucket_id="rb1",
            metrics={
                "pLDDT": 93.0, "iPAE": 0.18, "binder_scRMSD": 1.0,
                "ipTM": 0.60, "min_ipae": 0.10, "avg_ipsae": 0.50, "max_ipsae": 0.60,
            },
            metrics_calibrated={}, route_lineage=[], gpu_h=0.2,
            exit_status="ok", bins={"foldseek_su": f"b{i}"},
        ))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.diagnostic_axis_stats
    assert ev.diagnostic_driver_tldr != "none"
    calls = list(arc.iter_records(LLMCallRecord))
    assert calls and all(c.prompt_audit for c in calls)
    planner = next(c for c in calls if c.role == "planner")
    assert planner.prompt_audit["diagnostic_driver_tldr"] == ev.diagnostic_driver_tldr
    assert "evidence_tldr" in planner.prompt_audit
    supervisor = next(c for c in calls if c.role == "supervisor")
    assert supervisor.prompt_audit["selector_context"]["score_conversion_backlog"] is not None


def test_live_tick_dry_timer_ignores_strict_records_without_foldseek_su(tmp_path: Path):
    arc = Archive(tmp_path / "arc_no_su_bin")
    for i in range(3):
        arc.append(ResultRecord(
            result_id=f"strict_no_bin_{i}", parent_ids=[], target_id="t1",
            backend_family="complexa_beam", runtime_bucket_id="rb1",
            metrics={"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0},
            metrics_calibrated={}, route_lineage=[], gpu_h=0.2,
            exit_status="ok", bins={},
        ))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.run_su_count == 0
    assert ev.gpu_h_since_last_su and ev.gpu_h_since_last_su >= 0.59


def test_live_tick_records_bounded_tm08_fine_diversity_signal(tmp_path: Path):
    arc = Archive(tmp_path / "arc_tm08")
    pdb_seed = tmp_path / "seed.pdb"
    pdb_seed.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    for i in range(4):
        r = _result(
            f"r{i}", parent_ids=[f"chain{i}", f"src{i}"],
            pLDDT=92.0, iPAE=0.10, scRMSD=1.0, family="structure_refilter",
        )
        arc.append(dataclasses.replace(
            r,
            artifacts={"pdb_path": str(pdb_seed)},
            bins={
                "refilter_role": CANONICAL_SCORE_CONVERSION,
                "refilter_source": f"src{i}",
            },
        ))

    def fake_cluster(_results, *, min_tm_score, only_result_ids=None, **_kw):
        ids = sorted(only_result_ids or [])
        if abs(float(min_tm_score) - 0.60) < 1e-9:
            cmap = {
                rid: ("tm06_a" if rid in {"r0", "r1"} else "tm06_b")
                for rid in ids
            }
        else:
            cmap = {rid: f"tm08_{rid}" for rid in ids}
        return ClusteringResult(
            cluster_by_result_id=cmap,
            n_structures=len(ids),
            n_clusters=len(set(cmap.values())),
            status="ok",
            structure_scope="binder_chain",
        )

    cfg = LiveTickConfig(
        foldseek=FoldseekConfig(
            strict_tm08_diagnostic_enabled=True,
            fine_min_strict=4,
            fine_strict_window=8,
        ),
        sequence_dedup=SequenceDedupConfig(enabled=False),
    )
    target = TargetConstraint(target_id="t1", target_class="test")
    with patch(
        "trex.clustering_cache.cluster_archive_pdbs_cached",
        side_effect=fake_cluster,
    ), patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0, cfg=cfg,
        )

    ev = next(arc.iter_records(EvidenceSummary))
    assert ev.run_su_count == 2
    assert ev.strict_su_tm05_recent_count == 2
    assert ev.strict_su_tm08_recent_count == 4
    assert ev.strict_su_live_recent_count == 2
    assert ev.strict_su_tm08_delta_vs_live == 2
    assert ev.strict_su_tm08_live_split_ratio == 2.0
    assert ev.strict_su_tm08_delta_vs_tm05 == 2
    assert ev.strict_su_tm08_split_ratio == 2.0


def test_live_tick_emits_launches_when_planner_cards_exist(tmp_path: Path):
    arc = Archive(tmp_path / "arc2")
    # fix20 #2: parent-PDB precondition gates `proteinmpnn_redesign`.
    # Seed at least one ResultRecord with a usable pdb_path so the
    # feasibility check passes and the MPNN candidate gets emitted.
    pdb_seed = tmp_path / "seed.pdb"
    pdb_seed.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    for i in range(4):
        r = _result(f"r{i}")
        # Frozen dataclass — rebuild with artifacts populated
        r = dataclasses.replace(r, artifacts={"pdb_path": str(pdb_seed)})
        arc.append(r)
    target = TargetConstraint(target_id="t1", target_class="test")

    card = dataclasses.replace(
        _hyp(hid="h_new", target="t1", tick=1),
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["r0"], 0.20, None)
        ],
    )
    with patch("trex.live_tick.call_planner", return_value=_planner_ok([card])), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )
    # archive should now have launch decisions
    assert arc.summary()["launch_decisions.jsonl"] >= 1
    # the hypothesis card was persisted
    assert arc.summary()["hypothesis_cards.jsonl"] == 1
    # candidates derived from the card were persisted
    assert arc.summary()["action_candidates.jsonl"] >= 1


def test_live_tick_lifecycle_update_supported_when_descendants_pass(tmp_path: Path):
    """Pre-populate archive with hypothesis + action + 2 supporting descendants.
    Verify the live tick promotes the hypothesis to 'supported'."""
    arc = Archive(tmp_path / "arc3")
    # Complexa thresholds: keep pLDDT passing on baseline + descendants so
    # preserve_constraint(pLDDT) holds while iPAE improves materially.
    baseline = _result("r_baseline", parent_ids=[], iPAE=0.7, pLDDT=92.0)
    arc.append(baseline)

    h = _hyp(hid="h_lifecycle", target="t1", tick=0)
    arc.append(h)
    a = _action(cid="c_l1", hid="h_lifecycle", parent_result_id="r_baseline")
    arc.append(a)
    # two supporting descendants (iPAE deficit halved from baseline)
    arc.append(_result("d1", parent_ids=["c_l1"], iPAE=0.3, pLDDT=92.0))
    arc.append(_result("d2", parent_ids=["c_l1"], iPAE=0.25, pLDDT=91.0))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_002", tick_id_int=2,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    # Lifecycle update should have promoted the hypothesis
    statuses = [h.status for h in arc.iter_records(HypothesisCard)]
    assert "supported" in statuses
    assert summary["lifecycle_updates"]
    assert summary["lifecycle_updates"][0]["new_status"] == "supported"


def test_live_tick_lifecycle_uses_baseline_result_id_for_denovo_candidate(tmp_path: Path):
    """A de-novo generator has no execution parent, but can still be evaluated
    against the concrete baseline it cited in the hypothesis."""
    arc = Archive(tmp_path / "arc_denovo_lifecycle")
    baseline = _result("r_baseline", parent_ids=[], iPAE=0.7, pLDDT=92.0)
    arc.append(baseline)

    h = _hyp(hid="h_denovo", target="t1", tick=0)
    arc.append(h)
    a = dataclasses.replace(
        _action(cid="c_denovo", hid="h_denovo", parent_result_id=None),
        method_family="complexa_beam",
        operator_id="complexa_beam_default",
        lane_id="complexa",
        baseline_result_id="r_baseline",
    )
    arc.append(a)
    arc.append(_result("d1", parent_ids=["c_denovo"], iPAE=0.3, pLDDT=92.0))
    arc.append(_result("d2", parent_ids=["c_denovo"], iPAE=0.25, pLDDT=91.0))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_003", tick_id_int=2,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    statuses = [h.status for h in arc.iter_records(HypothesisCard)]
    assert "supported" in statuses
    assert summary["lifecycle_updates"]
    assert summary["lifecycle_updates"][0]["hypothesis_id"] == "h_denovo"



def test_live_tick_retires_expired_descendant_free_hypothesis_before_planner(tmp_path: Path):
    arc = Archive(tmp_path / "arc_ttl_no_desc")
    for i in range(3):
        arc.append(_result(f"r{i}"))
    expired = dataclasses.replace(_hyp(hid="h_expired", target="t1", tick=0), ttl_ticks=1)
    arc.append(expired)

    seen: dict[str, list[dict]] = {}

    def fake_planner(*_args, **kwargs):
        seen["active_hypotheses"] = list(kwargs.get("active_hypotheses") or [])
        return _planner_ok()

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", side_effect=fake_planner), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, target, tick_id="t_002", tick_id_int=2,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )

    assert not seen["active_hypotheses"]
    latest = [h for h in arc.iter_records(HypothesisCard) if h.hypothesis_id == "h_expired"][-1]
    assert latest.status == "retired"
    assert latest.last_evaluated_tick == 2
    assert not any(h.hypothesis_id == "h_expired" for h in arc.retrieve_active_hypotheses(target_id="t1"))


def test_evidence_only_refresh_does_not_mutate_hypothesis_lifecycle(tmp_path: Path):
    arc = Archive(tmp_path / "arc_evidence_only_lifecycle")
    expired = dataclasses.replace(
        _hyp(hid="h_expired", target="t1", tick=0), ttl_ticks=1,
    )
    arc.append(expired)
    before = list(arc.iter_records(HypothesisCard))

    summary = run_live_tick(
        arc,
        TargetConstraint(target_id="t1", target_class="test"),
        tick_id="t_005",
        tick_id_int=5,
        elapsed_wall_h=1.0,
        remaining_wall_h=47.0,
        cfg=LiveTickConfig(),
        evidence_only=True,
    )

    after = list(arc.iter_records(HypothesisCard))
    assert after == before
    assert after[-1].status == "active"
    assert list(arc.iter_records(EvidenceSummary)) == []
    assert summary["evidence_only"] is True
    assert summary["evidence"]["elapsed_wall_h"] == pytest.approx(1.0)
    assert summary["evidence"]["remaining_wall_h"] == pytest.approx(47.0)
    assert "worker_wall_gpu_h_total" in summary["evidence"]
    assert "production_panel_status" in summary["evidence"]


def test_one_slot_ticks_complete_each_warmstart_family_once(tmp_path: Path):
    arc = Archive(tmp_path / "arc_partial_warmstart")
    target = TargetConstraint(target_id="t1", target_class="test")
    selected_families: list[str] = []

    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        for tick in range(1, len(WARMSTART_FAMILIES) + 1):
            run_live_tick(
                arc, target, tick_id=f"t_{tick:03d}", tick_id_int=tick,
                elapsed_wall_h=float(tick - 1), remaining_wall_h=48.0 - tick,
                cfg=LiveTickConfig(), available_slots_override=1,
            )
            actions = {a.candidate_id: a for a in arc.iter_records(ActionCandidate)}
            launched = [
                decision
                for decision in arc.iter_records(LaunchDecision)
                if decision.tick_id == f"t_{tick:03d}"
                and decision.status == "launched"
                and decision.candidate_id.startswith("warmstart_")
            ]
            assert len(launched) == 1
            selected_families.append(actions[launched[0].candidate_id].method_family)

    assert len(selected_families) == len(set(selected_families))
    assert set(selected_families) == {family for family, _ in WARMSTART_FAMILIES}


def test_live_tick_lifecycle_recovers_chain_refilter_generator_baseline(tmp_path: Path):
    arc = Archive(tmp_path / "arc_chain_lifecycle")
    baseline = _result("r_baseline", parent_ids=[], iPAE=0.7, pLDDT=92.0)
    arc.append(baseline)

    h = _hyp(hid="h_chain", target="t1", tick=0)
    arc.append(h)
    gen_action = dataclasses.replace(
        _action(cid="c_gen", hid="h_chain", parent_result_id=None),
        method_family="bindcraft",
        operator_id="bindcraft_default",
        lane_id="bindcraft",
        downstream_route_plan=["structure_refilter"],
        estimated_cost_class="diagnostic",
        baseline_result_id="r_baseline",
    )
    arc.append(gen_action)
    arc.append(ResultRecord(
        result_id="g1", parent_ids=["c_gen"], target_id="t1",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
        exit_status="ok", artifacts={"pdb_path": str(tmp_path / "g1.pdb")},
    ))
    chain_action = ActionCandidate(
        candidate_id="chain_t_bindcraft_to_structure_refilter_001",
        hypothesis_ids=["h_chain"], parent_result_id="g1",
        method_family="structure_refilter", operator_id="af2_refilter",
        lane_id="structure_refilter", config_delta={}, downstream_route_plan=[],
        estimated_cost_class="low", expected_signal="auto_chain:bindcraft->structure_refilter parent=g1",
        evidence_refs=["g1"], feasibility=FeasibilityCheck(True, "rb1", True, True, True, True),
    )
    arc.append(chain_action)
    arc.append(_result(
        "rf1", parent_ids=["chain_t_bindcraft_to_structure_refilter_001", "g1"],
        iPAE=0.25, pLDDT=92.0, family="structure_refilter",
    ))

    target = TargetConstraint(target_id="t1", target_class="test")
    cfg = dataclasses.replace(
        LiveTickConfig(),
        lifecycle=LifecycleConfig(min_supporting_descendants=1),
    )
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()),          patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_chain", tick_id_int=2,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=cfg,
        )

    statuses = [h.status for h in arc.iter_records(HypothesisCard)]
    assert "supported" in statuses
    assert summary["lifecycle_updates"]
    assert summary["lifecycle_updates"][0]["hypothesis_id"] == "h_chain"

def test_live_tick_fallback_when_planner_invalid(tmp_path: Path):
    arc = Archive(tmp_path / "arc4")
    for i in range(4):
        arc.append(_result(f"r{i}"))
    target = TargetConstraint(target_id="t1", target_class="test")
    bad_planner = PlannerOutput(
        valid=False, abstain=False, confidence=0.0, fail_reason="parse_fail",
        cards=[], rationale="",
        raw_text="garbage", usage={},
    )
    with patch("trex.live_tick.call_planner", return_value=bad_planner), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )
    assert summary["fallback_used"]
    # planner LLMCallRecord still persisted (audit trail)
    assert arc.summary()["llm_call_records.jsonl"] >= 1


def test_live_tick_filters_by_target_id(tmp_path: Path):
    """ResultRecords for other targets must not bleed into this tick's evidence."""
    arc = Archive(tmp_path / "arc5")
    for i in range(3):
        arc.append(_result(f"a{i}", target="t1"))
    for i in range(10):
        arc.append(_result(f"b{i}", target="t2"))
    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        summary = run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=0.0, remaining_wall_h=48.0,
            cfg=LiveTickConfig(),
        )
    # evidence n_all_results should reflect ONLY t1 records (3), not t2
    assert summary["evidence"]["n_all_results"] == 3


def test_new_su_resets_dry_timer_even_with_inflight_gpu_h(tmp_path: Path):
    """A tick that raises the SU high-water mark is not dry, even if other workers
    are still running in parallel."""
    arc = Archive(tmp_path / "arc_new_su_dry_reset")
    pdb = tmp_path / "strict_new.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    arc.append(dataclasses.replace(
        _result(
            "strict_new", parent_ids=["chain0", "src0"],
            pLDDT=95.0, iPAE=0.15, scRMSD=0.8,
            family="structure_refilter",
        ),
        artifacts={"pdb_path": str(pdb)},
        bins={"refilter_role": CANONICAL_SCORE_CONVERSION, "refilter_source": "src0"},
    ))

    target = TargetConstraint(target_id="t1", target_class="test")
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()), \
         patch("trex.foldseek_clusterer.shutil.which", return_value="/fake/foldseek"):
        run_live_tick(
            arc, target, tick_id="t_001", tick_id_int=1,
            elapsed_wall_h=1.0, remaining_wall_h=47.0,
            cfg=LiveTickConfig(),
            inflight_gpu_h=40.0,
        )

    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.run_su_count >= 1
    assert ev.gpu_h_since_last_su == 0.0


def _canonical_refilter(rid: str, *, su_bin: str, gpu_h: float,
                        tick: str, source: str = "g1") -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[f"chain_t_bindcraft_to_structure_refilter_{rid}", source],
        target_id="t1",
        backend_family="structure_refilter",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=gpu_h,
        exit_status="ok",
        bins={"foldseek_su": su_bin, "refilter_source": source,
              "refilter_role": CANONICAL_SCORE_CONVERSION},
        tick_id=tick,
    )


def test_hwm_delta_resets_dry_timer_for_late_observed_old_tick_su(tmp_path: Path):
    """If official SU HWM rises in this tick, the adaptive dry timer resets.

    Whole-archive Foldseek reclustering and delayed result parsing can make the
    new SU bin belong to an older worker tick. The controller still OBSERVES the
    HWM increase now, so reporting "new SU + long dry plateau" in the same tick
    would mislead state classification and the LLM prompt.
    """
    arc = Archive(tmp_path / "arc_dry_timer_hwm_delta")
    arc.append(_canonical_refilter("rf1", su_bin="suA", gpu_h=0.2, tick="v7r001"))
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, TargetConstraint(target_id="t1", target_class="test"),
            tick_id="v7r001", tick_id_int=1, elapsed_wall_h=0.0,
            remaining_wall_h=48.0, cfg=LiveTickConfig(),
        )
    assert list(arc.iter_records(EvidenceSummary))[-1].run_su_hwm == 1

    # Late-observed old-tick SU: previous evidence did not know about suB, but
    # the result's original worker tick is older than the current controller tick.
    arc.append(_canonical_refilter("rf2", su_bin="suB", gpu_h=0.2, tick="v7r001"))
    with patch("trex.live_tick.call_planner", return_value=_planner_ok()), \
         patch("trex.live_tick.call_supervisor", return_value=_sup_ok()):
        run_live_tick(
            arc, TargetConstraint(target_id="t1", target_class="test"),
            tick_id="v7r002", tick_id_int=2, elapsed_wall_h=0.0,
            remaining_wall_h=48.0, cfg=LiveTickConfig(),
            inflight_gpu_h=9.0,
        )
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.run_su_hwm_delta == 1
    assert ev.gpu_h_since_last_su == pytest.approx(0.0)
    assert ev.ticks_since_last_su == 0


def test_canonical_chain_new_cluster_su_resets_dry_timer(tmp_path: Path):
    """Dry-timer fix (2026-06-12): a canonical conversion that mints a NEW
    Foldseek cluster IS real structural progress and MUST reset the dry timer.
    Diagnostic-generator targets (BindCraft/BoltzGen/MPNN) earn 100% of their SU
    this way; excluding it zeroed the high-water mark forever and forced premature
    deep_stall on every dry window (over-correction, now fixed)."""
    arc = Archive(tmp_path / "arc_dry_timer_new")
    arc.append(_result("g1", parent_ids=["c_gen"], pLDDT=None, iPAE=None,
                       scRMSD=None, family="bindcraft"))
    arc.append(_canonical_refilter("rf1", su_bin="suA", gpu_h=0.2, tick="v7r001"))
    summary = run_live_tick(
        arc, TargetConstraint(target_id="t1", target_class="test"),
        tick_id="v7r002", tick_id_int=2, elapsed_wall_h=0.0,
        remaining_wall_h=48.0, cfg=LiveTickConfig(), evidence_only=True,
    )
    assert summary["evidence"]["run_su_count"] == 1
    assert summary["evidence"]["gpu_h_since_last_su"] == pytest.approx(0.0)


def test_duplicate_canonical_rescore_does_not_reset_dry_timer(tmp_path: Path):
    """A canonical RE-score of an ALREADY-seen backbone (same Foldseek cluster)
    is plumbing, not progress, and must NOT reset the dry timer. This is the real
    'backlog re-score hides the stall' guard — the cluster-dedup, not a blanket
    canonical exclusion."""
    arc = Archive(tmp_path / "arc_dry_timer_dup")
    arc.append(_result("g1", parent_ids=["c_gen"], pLDDT=None, iPAE=None,
                       scRMSD=None, family="bindcraft"))
    arc.append(_canonical_refilter("rf1", su_bin="suA", gpu_h=0.2, tick="v7r001"))
    arc.append(_canonical_refilter("rf2", su_bin="suA", gpu_h=0.5, tick="v7r002"))
    summary = run_live_tick(
        arc, TargetConstraint(target_id="t1", target_class="test"),
        tick_id="v7r003", tick_id_int=3, elapsed_wall_h=0.0,
        remaining_wall_h=48.0, cfg=LiveTickConfig(), evidence_only=True,
    )
    assert summary["evidence"]["run_su_count"] == 1  # suA dedups to one SU
    assert summary["evidence"]["gpu_h_since_last_su"] == pytest.approx(0.5)


def test_selector_override_reasons_survive_archive_compaction():
    debug = {
        "quotas_raw": {"exploit": 0, "rescue": 1, "explore": 0},
        "quotas_final": {"exploit": 1, "rescue": 0, "explore": 0},
        "redistribute_log": [f"override_{i}" for i in range(20)],
        "clamp_log": ["category_a"],
        "forced_repeated_support_probe": {"candidate_id": "c1"},
    }

    compact = _compact_selector_debug_for_archive(debug)

    assert compact["quotas_raw"] != compact["quotas_final"]
    assert compact["redistribute_log"] == [f"override_{i}" for i in range(12)]
    assert compact["clamp_log"] == ["category_a"]
    assert compact["forced_repeated_support_probe"] == {"candidate_id": "c1"}
