"""Test rotating quota tiebreaks so equal small shares do not permanently starve one mode."""
from __future__ import annotations

from trex.selector import _windowed_quotas


def _simulate(mixture, ticks=300, k=10):
    recent: list[str] = []
    counts = {"exploit": 0, "rescue": 0, "explore": 0}
    feasible = {"exploit", "rescue", "explore"}
    for _ in range(ticks):
        q = _windowed_quotas(mixture, 1, recent, k=k, feasible_modes=feasible)
        pick = next(m for m, c in q.items() if c > 0)
        counts[pick] += 1
        recent.append(pick)
    return counts


def test_small_equal_shares_do_not_starve_explore():
    # Both equal small shares must receive starts.
    counts = _simulate({"exploit": 0.94, "rescue": 0.03, "explore": 0.03})
    assert counts["explore"] > 0, counts
    assert counts["rescue"] > 0, counts
    assert counts["exploit"] > counts["rescue"] + counts["explore"]   # still dominant


def test_explore_at_floor_realizes_reliably():
    # Shares below the moving-window resolution can remain quantization-limited.
    for mix in ({"exploit": 0.90, "rescue": 0.05, "explore": 0.05},
                {"exploit": 0.85, "rescue": 0.10, "explore": 0.05}):
        counts = _simulate(mix)
        assert counts["explore"] > 0 and counts["rescue"] > 0, (mix, counts)


def test_low_evidence_explore_floor_prevents_starvation():
    from trex.fallback import CATEGORY_A_CLAMPS, clamp_mixture
    assert CATEGORY_A_CLAMPS["low_evidence"].explore_min == 0.05
    mix, _log = clamp_mixture({"exploit": 0.96, "rescue": 0.02, "explore": 0.02},
                              "low_evidence")
    assert mix["explore"] >= 0.05, mix
    assert abs(sum(mix.values()) - 1.0) < 1e-9, mix
    # and the floored share now realizes nonzero launches over the window
    counts = _simulate(mix)
    assert counts["explore"] > 0, counts


def test_quotas_still_respect_slot_count_and_feasibility():
    # determinism + budget: exactly n_slots allocated, only feasible modes
    q = _windowed_quotas(
        {"exploit": 0.5, "rescue": 0.3, "explore": 0.2}, 3, ["exploit"], k=10,
        feasible_modes={"exploit", "rescue"},
    )
    assert sum(q.values()) == 3
    assert q["explore"] == 0   # explore infeasible → no slot wasted on it
