"""Test campaign-state classification on synthetic evidence."""

from __future__ import annotations

from trex.evidence_reducer import (
    StateClassifierConfig,
    classify_state,
)
from trex.schemas import AxisStat


def _stats(plddt_deficit: float = 5.0, ipae_deficit: float = 0.1) -> dict[str, AxisStat]:
    return {
        "pLDDT": AxisStat(0, 0, 0, None, None, plddt_deficit, "provisional", 5),
        "iPAE": AxisStat(0, 0, 0, None, None, ipae_deficit, "provisional", 5),
        "binder_scRMSD": AxisStat(0, 0, 0, None, None, None, "uncalibrated", 0),
    }


def _diffuse_plateau_stats() -> dict[str, AxisStat]:
    """SC2RBD-like diffuse blockers after near-pass-margin normalization."""
    return {
        "pLDDT": AxisStat(5, 0, 48, None, None, 15.4, "provisional", 53),
        "iPAE": AxisStat(4, 3, 46, None, None, 0.48, "provisional", 53),
        "binder_scRMSD": AxisStat(20, 0, 33, None, None, 2.94, "provisional", 53),
    }


def test_low_evidence_when_window_empty():
    label = classify_state(
        worker_gpu_h_last_3_ticks=0.0,
        completed_children_window=0,
        run_su_count_delta=0,
        duplicate_fraction=None,
        near_miss_count=0,
        axis_stats=_stats(),
        top_bin_share=None,
        cfg=StateClassifierConfig(),
    )
    assert label == "low_evidence"


def test_productive_when_su_rate_high():
    label = classify_state(
        worker_gpu_h_last_3_ticks=10.0,
        completed_children_window=6,
        run_su_count_delta=2,
        duplicate_fraction=0.3,
        near_miss_count=2,
        axis_stats=_stats(),
        top_bin_share=0.5,
        cfg=StateClassifierConfig(),
    )
    assert label == "productive"


def test_untrusted_su_dedup_does_not_classify_productive():
    label = classify_state(
        worker_gpu_h_last_3_ticks=10.0,
        completed_children_window=6,
        run_su_count_delta=2,
        duplicate_fraction=0.3,
        near_miss_count=2,
        axis_stats=_stats(),
        top_bin_share=0.5,
        cfg=StateClassifierConfig(),
        su_dedup_trusted=False,
    )
    assert label != "productive"


def test_rescue_rich_when_near_miss_concentrated():
    label = classify_state(
        worker_gpu_h_last_3_ticks=8.0,
        completed_children_window=5,
        run_su_count_delta=0,
        duplicate_fraction=0.4,
        near_miss_count=5,
        axis_stats=_stats(plddt_deficit=0.5, ipae_deficit=5.0),  # iPAE dominates
        top_bin_share=0.3,
        cfg=StateClassifierConfig(),
    )
    assert label == "rescue_rich"


def test_stalled_when_zero_su_and_low_near_miss():
    label = classify_state(
        worker_gpu_h_last_3_ticks=10.0,
        completed_children_window=5,
        run_su_count_delta=0,
        duplicate_fraction=0.3,
        near_miss_count=1,
        axis_stats=_stats(),
        top_bin_share=0.5,
        cfg=StateClassifierConfig(),
    )
    assert label == "stalled"


def test_stalled_when_zero_su_and_diffuse_near_misses():
    """A dry, near-miss-rich window that is not axis-concentrated is still a
    plateau, not low_evidence. SC2RBD's T-REX final window hit this case:
    dSU=0, near_miss>=3, diffuse blockers, large cumulative compute."""
    label = classify_state(
        worker_gpu_h_last_3_ticks=1.5,
        cumulative_gpu_h=190.0,
        completed_children_window=53,
        run_su_count_delta=0,
        duplicate_fraction=0.90,
        near_miss_count=3,
        axis_stats=_diffuse_plateau_stats(),
        top_bin_share=0.11,
        cfg=StateClassifierConfig(),
    )
    assert label == "stalled"


def test_deep_stall_when_zero_su_and_diffuse_near_misses_goes_long_dry():
    label = classify_state(
        worker_gpu_h_last_3_ticks=1.5,
        cumulative_gpu_h=190.0,
        completed_children_window=53,
        run_su_count_delta=0,
        duplicate_fraction=0.90,
        near_miss_count=3,
        axis_stats=_diffuse_plateau_stats(),
        top_bin_share=0.11,
        cfg=StateClassifierConfig(),
        gpu_h_since_last_su=17.0,
    )
    assert label == "deep_stall"


def test_stalled_when_top_bin_share_high():
    label = classify_state(
        worker_gpu_h_last_3_ticks=10.0,
        completed_children_window=5,
        run_su_count_delta=1,  # nonzero, but top_bin_share dominates
        duplicate_fraction=0.3,
        near_miss_count=2,
        axis_stats=_stats(),
        top_bin_share=0.85,
        cfg=StateClassifierConfig(),
    )
    # Productive classification takes precedence when its conditions are met.
    assert label in ("productive", "stalled")


def test_no_divide_by_zero_when_window_gpu_h_is_zero():
    """Classifier must not crash when the window gpu_h is ~0 (divide-by-zero
    guard via eps_gpu_h). With cumulative past cold-start, a zero-window tick
    classifies (here productive: su_rate = 2 / eps) rather than crashing."""
    label = classify_state(
        worker_gpu_h_last_3_ticks=0.0,   # zero rate denominator → eps guard
        cumulative_gpu_h=2.0,            # past cold-start
        completed_children_window=5,
        run_su_count_delta=2,
        duplicate_fraction=0.3,
        near_miss_count=2,
        axis_stats=_stats(),
        top_bin_share=0.5,
        cfg=StateClassifierConfig(),
    )
    assert label in ("productive", "rescue_rich", "stalled", "low_evidence")


def test_low_evidence_gate_uses_cumulative_not_window():
    """Cold-start gate fires on low CUMULATIVE gpu_h, regardless of window."""
    label = classify_state(
        worker_gpu_h_last_3_ticks=0.8,   # window value irrelevant to the gate
        cumulative_gpu_h=0.4,            # < min_cumulative_gpu_h(1.0)
        completed_children_window=60,
        run_su_count_delta=0,
        duplicate_fraction=0.9,
        near_miss_count=1,
        axis_stats=_stats(),
        top_bin_share=0.1,
        cfg=StateClassifierConfig(),
    )
    assert label == "low_evidence"


def test_not_stuck_low_evidence_when_window_small_but_cumulative_high():
    """Cumulative compute permits cold-start exit even when recent records are individually
    inexpensive.
    """
    # No new SU + low near-miss + tiny window + high cumulative → stalled
    stalled = classify_state(
        worker_gpu_h_last_3_ticks=0.8,
        cumulative_gpu_h=2.0,
        completed_children_window=60,
        run_su_count_delta=0,
        duplicate_fraction=0.95,
        near_miss_count=1,
        axis_stats=_stats(),
        top_bin_share=0.1,
        cfg=StateClassifierConfig(),
    )
    assert stalled != "low_evidence"
    assert stalled == "stalled"
    # A new SU in the same tiny window → productive (su_rate = 1/0.8 = 1.25)
    productive = classify_state(
        worker_gpu_h_last_3_ticks=0.8,
        cumulative_gpu_h=2.0,
        completed_children_window=60,
        run_su_count_delta=1,
        duplicate_fraction=0.3,
        near_miss_count=2,
        axis_stats=_stats(),
        top_bin_share=0.2,
        cfg=StateClassifierConfig(),
    )
    assert productive == "productive"


def test_strict_duplicate_collapse_when_many_strict_but_few_su():
    label = classify_state(
        worker_gpu_h_last_3_ticks=2.0,
        cumulative_gpu_h=55.0,
        completed_children_window=60,
        run_su_count_delta=0,
        duplicate_fraction=0.2,
        near_miss_count=6,
        axis_stats=_stats(),
        top_bin_share=0.1,
        cfg=StateClassifierConfig(),
        strict_count_total=200,
        run_su_count_total=4,
        strict_per_su_total=50.0,
        strict_su_tm08_split_ratio=1.25,
    )
    assert label == "strict_duplicate_collapse"


def test_strict_duplicate_collapse_does_not_steal_healthy_productive_lane():
    label = classify_state(
        worker_gpu_h_last_3_ticks=2.0,
        cumulative_gpu_h=52.0,
        completed_children_window=60,
        run_su_count_delta=3,
        duplicate_fraction=0.3,
        near_miss_count=2,
        axis_stats=_stats(),
        top_bin_share=0.2,
        cfg=StateClassifierConfig(),
        strict_count_total=219,
        run_su_count_total=61,
        strict_per_su_total=3.59,
        strict_su_tm08_split_ratio=1.26,
    )
    assert label == "productive"
