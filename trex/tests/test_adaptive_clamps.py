"""Diversity-aware adaptive clamps.

User Q: pure state→clamp lookup is too rigid for an LLM agent. When
diversity signals indicate panel collapse (e.g. one Foldseek bin
dominates a productive run), clamps should loosen so the LLM has
headroom to pivot toward diversity.
"""

from __future__ import annotations

from trex.fallback import (
    DEFAULT_CLAMPS,
    clamp_mixture,
    diversity_adjusted_clamp,
)


def test_productive_clamp_unchanged_when_no_diversity_signal():
    """No top_bin_share info → no loosening, original clamps apply."""
    out, log = clamp_mixture(
        {"exploit": 0.8, "rescue": 0.1, "explore": 0.1},
        "productive",
        top_bin_share=None,
    )
    # explore stays bounded at 0.10 (LLM proposal, within 0.20 cap)
    assert out["explore"] <= 0.20
    assert not any("diversity_loosen" in s for s in log)


def test_productive_clamp_loosens_on_structural_collapse():
    """top_bin_share=0.85 means one structure cluster dominates →
    loosen exploit cap (0.80→0.65) and explore cap (0.20→0.40)."""
    out, log = clamp_mixture(
        {"exploit": 0.7, "rescue": 0.0, "explore": 0.3},
        "productive",
        top_bin_share=0.85,
        panel_ready_bins_covered=4,
        panel_size_K=8,
    )
    assert any("diversity_loosen" in s for s in log)
    # explore was 0.3 — under loosened cap 0.40 → allowed
    assert out["explore"] >= 0.20  # would be capped at 0.20 normally
    # Clamp exploitation to the adjusted upper bound.
    assert out["exploit"] <= 0.65


def test_productive_clamp_loosens_on_shallow_panel_coverage():
    """Use shallow panel coverage only when live panel selection is enabled."""
    out, log = clamp_mixture(
        {"exploit": 0.6, "rescue": 0.1, "explore": 0.3},
        "productive",
        top_bin_share=0.3,        # below the 0.50 collapse trigger → isolate panel path
        panel_ready_bins_covered=2,  # only 2 of K=8 bins covered
        panel_size_K=8,
        panel_live=True,          # panel selector wired → trigger active
    )
    assert any("diversity_loosen" in s for s in log)


def test_productive_clamp_does_not_loosen_on_panel_coverage_when_panel_not_live():
    """Default (panel_live=False, production): shallow panel coverage must NOT
    loosen, since panel_ready_bins_covered is a constant-0 stub."""
    out, log = clamp_mixture(
        {"exploit": 0.6, "rescue": 0.1, "explore": 0.3},
        "productive",
        top_bin_share=0.3,        # below the 0.50 collapse trigger
        panel_ready_bins_covered=0,
        panel_size_K=8,
    )
    assert not any("diversity_loosen" in s for s in log)


def test_productive_duplicate_forces_explore_floor_on_collapse():
    """Structural concentration raises the exploration floor."""
    out, log = clamp_mixture(
        {"exploit": 0.80, "rescue": 0.10, "explore": 0.10},  # LLM held explore low
        "productive_duplicate",
        top_bin_share=0.55,       # >= 0.50 collapse trigger
        panel_ready_bins_covered=0,
        panel_size_K=8,
    )
    assert any("diversity_loosen" in s for s in log)
    assert out["explore"] >= 0.20 - 1e-9


def test_productive_duplicate_no_floor_below_threshold():
    """Below 0.50 top_bin: no forced explore floor (honor the LLM's mixture)."""
    out, log = clamp_mixture(
        {"exploit": 0.80, "rescue": 0.10, "explore": 0.10},
        "productive_duplicate",
        top_bin_share=0.40,
        panel_ready_bins_covered=0,
        panel_size_K=8,
    )
    assert not any("diversity_loosen" in s for s in log)


def test_productive_duplicate_loosens_on_strict_su_collapse_even_if_all_scored_not_collapsed():
    out, log = clamp_mixture(
        {"exploit": 0.80, "rescue": 0.10, "explore": 0.10},
        "productive_duplicate",
        top_bin_share=0.30,
        strict_su_top_bin_share=0.80,
        panel_ready_bins_covered=0,
        panel_size_K=8,
    )
    assert any("strict_su_top" in s for s in log)
    assert out["explore"] >= 0.20 - 1e-9


def test_rescue_rich_loosens_explore_on_collapse():
    out, log = clamp_mixture(
        {"exploit": 0.1, "rescue": 0.4, "explore": 0.5},
        "rescue_rich",
        top_bin_share=0.80,
    )
    assert any("diversity_loosen_rescue" in s for s in log)
    # explore was 0.5 — under loosened cap 0.50 → allowed
    assert out["explore"] >= 0.40


def test_stalled_clamp_unaffected_by_diversity():
    """Stalled already encourages explore; diversity signal not needed."""
    out, log = clamp_mixture(
        {"exploit": 0.1, "rescue": 0.2, "explore": 0.7},
        "stalled",
        top_bin_share=0.90,
    )
    # No diversity loosening on stalled — already explore-heavy
    assert not any("diversity_loosen" in s for s in log)


def test_diversity_loosen_preserves_safety_floors():
    """Loosened explore_max doesn't change exploit_min/rescue_min safety floors."""
    out, log = clamp_mixture(
        {"exploit": 0.0, "rescue": 0.0, "explore": 1.0},
        "productive",
        top_bin_share=0.85,
        panel_ready_bins_covered=2,
    )
    # exploit_min=0.45 still enforced (safety)
    assert out["exploit"] >= 0.45
    # rescue_min=0.10 still enforced
    assert out["rescue"] >= 0.10
    # explore goes high but capped at loosened 0.40
    assert out["explore"] <= 0.40
    # diversity loosening logged
    assert any("diversity_loosen" in s for s in log)
