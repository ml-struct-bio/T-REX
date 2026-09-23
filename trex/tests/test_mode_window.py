"""Tests for allocation over a moving window of confirmed starts, including small
fractional shares.
"""

from __future__ import annotations

from trex.fallback import largest_remainder
from trex.selector import SelectorConfig, _windowed_quotas

FEAS = {"exploit", "rescue", "explore"}


def _simulate(mixture, ticks, k, n_free=1):
    """Single-slot ticks; window carries realised modes across ticks."""
    counts = {"exploit": 0, "rescue": 0, "explore": 0}
    window: list[str] = []
    for _ in range(ticks):
        q = _windowed_quotas(mixture, n_free, window, k, FEAS)
        # realise the slot(s): one launch per quota unit
        for m in ("exploit", "rescue", "explore"):
            for _ in range(q[m]):
                counts[m] += 1
                window.append(m)
    return counts


def test_old_largest_remainder_drops_explore_on_single_slot():
    """Documents the no-window tradeoff: largest_remainder zeroes a 0.10
    explore share on an individual single-slot tick."""
    q = largest_remainder({"exploit": 0.65, "rescue": 0.25, "explore": 0.10}, 1)
    assert q["explore"] == 0
    assert q["rescue"] == 0


def test_window_preserves_explore_floor_single_slot():
    counts = _simulate(
        {"exploit": 0.65, "rescue": 0.25, "explore": 0.10}, ticks=40, k=10
    )
    # explore must actually launch, not round to 0
    assert counts["explore"] > 0
    # realised fractions track the target within window granularity
    assert 0.05 <= counts["explore"] / 40 <= 0.20
    assert 0.18 <= counts["rescue"] / 40 <= 0.35


def test_window_realises_small_005_floor():
    """The Cat-A productive explore floor (0.05) is reachable, even when
    rescue holds the same 0.05 share (fair recency tiebreak)."""
    counts = _simulate(
        {"exploit": 0.90, "rescue": 0.05, "explore": 0.05}, ticks=60, k=10
    )
    assert counts["explore"] > 0


def test_window_respects_feasibility():
    """Infeasible modes are never allocated a slot."""
    q = _windowed_quotas(
        {"exploit": 0.34, "rescue": 0.33, "explore": 0.33},
        n_slots=3,
        recent_modes=[],
        k=10,
        feasible_modes={"exploit"},  # only exploit feasible
    )
    assert q["rescue"] == 0 and q["explore"] == 0
    assert q["exploit"] == 3
    assert sum(q.values()) == 3


def test_window_quotas_sum_to_slots():
    for n in (1, 2, 3):
        q = _windowed_quotas(
            {"exploit": 0.5, "rescue": 0.3, "explore": 0.2},
            n_slots=n, recent_modes=["exploit", "exploit"], k=10, feasible_modes=FEAS,
        )
        assert sum(q.values()) == n


def test_default_k_is_resolvable():
    """The default K must be large enough to resolve the explore floor."""
    assert SelectorConfig().mode_window_k >= 10


# --- end-to-end: explore survives single-slot ticks through run_live_tick ----

def test_explore_not_starved_across_single_slot_ticks(tmp_path):
    """Integration for the K-window ablation: 1 free slot/tick plus an
    exploit-heavy mixture (explore=0.1). The deterministic_deficit path uses
    persisted LaunchDecision modes to realize small shares over many ticks."""
    from pathlib import Path
    from unittest.mock import patch

    from trex.archive import Archive
    from trex.live_tick import FoldseekConfig, LiveTickConfig, run_live_tick
    from trex.schemas import (
        EvidenceSummary, HypothesisCard, LaunchDecision, PlannerOutput,
        PredictedChange, PreserveConstraint, ResultRecord, SupervisorOutput,
        TargetConstraint,
    )

    def _card(hid, fam, aff):
        return HypothesisCard(
            hypothesis_id=hid, target_id="t1", tick_created=1, claim="c",
            mode_affinity=aff, evidence_refs=["e"],
            predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.2, None)],
            preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
            recommended_action_families=[fam],
        )

    cards = [
        _card("hx", "complexa_beam", {"exploit": 0.9, "rescue": 0.05, "explore": 0.05}),
        _card("he", "bindcraft", {"exploit": 0.05, "rescue": 0.05, "explore": 0.9}),
    ]
    planner = PlannerOutput(valid=True, abstain=False, confidence=0.8, fail_reason=None,
                            cards=cards, rationale="x", raw_text="{}", usage={})
    sup = SupervisorOutput(valid=True, abstain=False, confidence=0.8, fail_reason=None,
                           mode_mixture={"exploit": 0.9, "rescue": 0.0, "explore": 0.1},
                           candidate_decisions=[], rationale="y", raw_text="{}", usage={})

    arc = Archive(tmp_path / "arc_e2e")
    # 2 results → completed_children_window < 3 → state stays "low_evidence",
    # where the Supervisor's exploit-heavy mixture is honored (not clamped to
    # the stalled pivot response). This isolates the K-window's job: realise a
    # small explore floor under an exploit-dominant mixture.
    # Seed an UNRELATED family so neither card's config matches a joint_fail
    # recipe (otherwise the builder correctly dedups the exploit candidate and
    # only explore remains feasible — not a K-window test).
    for i in range(2):
        m = {"pLDDT": 70.0, "iPAE": 0.4, "binder_scRMSD": 2.0}
        arc.append(ResultRecord(
            result_id=f"r{i}", parent_ids=[], target_id="t1",
            backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
            metrics=m, metrics_calibrated=dict(m), route_lineage=[],
            gpu_h=0.4, exit_status="ok",
        ))
    target = TargetConstraint(target_id="t1", target_class="test")
    cfg = LiveTickConfig(
        # This test is specifically for the K-window ablation. Production uses
        # no-window largest-remainder by default.
        selector=SelectorConfig(available_slots=1, quota_realization="deterministic_deficit"),
        foldseek=FoldseekConfig(enabled=False),
    )
    with patch("trex.live_tick.call_planner", return_value=planner), \
         patch("trex.live_tick.call_supervisor", return_value=sup):
        for i in range(15):
            run_live_tick(arc, target, tick_id=f"v7r{i:03d}", tick_id_int=i,
                          elapsed_wall_h=0.0, remaining_wall_h=48.0, cfg=cfg)

    launched = [L for L in arc.iter_records(LaunchDecision) if L.status == "launched"]
    modes = [(L.resource_class_concrete or {}).get("mode") for L in launched]
    n_explore = modes.count("explore")
    n_exploit = modes.count("exploit")
    assert n_explore >= 1, f"explore starved over 15 single-slot ticks: modes={modes}"
    assert n_exploit > n_explore, "exploit should still dominate the exploit-heavy mixture"
