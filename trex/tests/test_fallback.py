"""Unit tests for fallback mixtures, clamps, largest-remainder rounding."""

from __future__ import annotations

import pytest

from trex.fallback import (
    DEFAULT_MIXTURES,
    clamp_mixture,
    fallback_mixture,
    largest_remainder,
    normalize,
    redistribute_empty_modes,
)


def test_normalize_handles_zero_sum():
    out = normalize({"exploit": 0.0, "rescue": 0.0, "explore": 0.0})
    assert all(abs(v - 1 / 3) < 1e-6 for v in out.values())


def test_normalize_clamps_negatives():
    out = normalize({"exploit": -0.5, "rescue": 0.4, "explore": 0.6})
    assert out["exploit"] == 0.0
    assert sum(out.values()) == pytest.approx(1.0)


def test_clamp_productive_floor():
    mixture = {"exploit": 0.30, "rescue": 0.50, "explore": 0.20}
    out, log = clamp_mixture(mixture, "productive")
    assert out["exploit"] >= 0.45
    assert "clamp:exploit_min" in "\n".join(log)


def test_clamp_productive_ceiling():
    mixture = {"exploit": 0.90, "rescue": 0.05, "explore": 0.05}
    out, log = clamp_mixture(mixture, "productive")
    assert out["exploit"] <= 0.80
    assert out["rescue"] >= 0.10  # after renorm floor satisfied
    assert any("exploit_max" in s for s in log)


def test_stalled_caps_exploit():
    mixture = {"exploit": 0.80, "rescue": 0.10, "explore": 0.10}
    out, _ = clamp_mixture(mixture, "stalled")
    assert out["exploit"] <= 0.35
    assert out["rescue"] >= 0.25
    assert out["explore"] >= 0.25


def test_rescue_rich_floor():
    mixture = {"exploit": 0.50, "rescue": 0.20, "explore": 0.30}
    out, _ = clamp_mixture(mixture, "rescue_rich")
    assert out["rescue"] >= 0.40


def test_productive_caps_explore_v6_3_lesson():
    """A productive campaign caps exploration at 0.20 when the proposed share is higher."""
    mixture = {"exploit": 0.60, "rescue": 0.00, "explore": 0.40}
    out, log = clamp_mixture(mixture, "productive")
    assert out["explore"] <= 0.20
    assert any("explore_max" in s for s in log)
    # remaining mass goes to exploit and rescue floor
    assert out["exploit"] >= 0.45
    assert out["rescue"] >= 0.10


def test_productive_explore_floor_still_enforced():
    """When LLM tries to zero out explore on productive, the 0.05 floor
    must still kick in (preserves diversity insurance)."""
    mixture = {"exploit": 0.95, "rescue": 0.05, "explore": 0.00}
    out, _ = clamp_mixture(mixture, "productive")
    assert out["explore"] >= 0.05


def test_rescue_rich_caps_explore():
    """When state is rescue_rich, LLM should not pivot to high explore."""
    mixture = {"exploit": 0.20, "rescue": 0.30, "explore": 0.50}
    out, log = clamp_mixture(mixture, "rescue_rich")
    assert out["explore"] <= 0.30
    assert out["rescue"] >= 0.40


def test_stalled_has_no_explore_cap():
    """Stalled targets SHOULD be allowed high explore — no upper bound."""
    mixture = {"exploit": 0.10, "rescue": 0.20, "explore": 0.70}
    out, _ = clamp_mixture(mixture, "stalled")
    # explore can stay high
    assert out["explore"] >= 0.50


def test_all_zero_mixture_normalizes_to_sum_1():
    """Edge case caught in selector probe: LLM returns all-zero mixture.
    Old behavior: all three modes pinned at their MIN floors, leaving
    residual mass unnormalized (sum=0.60 for productive). New behavior:
    scale pinned proportionally to reach sum=1.0."""
    mixture = {"exploit": 0.0, "rescue": 0.0, "explore": 0.0}
    out, log = clamp_mixture(mixture, "productive")
    s = sum(out.values())
    assert abs(s - 1.0) < 1e-6, f"sum={s}, out={out}"
    # all pinned at mins originally, so log should record the scaling
    assert any("all_pinned_scale" in s for s in log)


def test_all_zero_mixture_stalled_still_normalizes():
    mixture = {"exploit": 0.0, "rescue": 0.0, "explore": 0.0}
    out, _ = clamp_mixture(mixture, "stalled")
    s = sum(out.values())
    assert abs(s - 1.0) < 1e-6


def test_normal_mixture_still_unaffected():
    """The new all-pinned branch should not interfere with normal flows."""
    mixture = {"exploit": 0.5, "rescue": 0.3, "explore": 0.2}
    out, log = clamp_mixture(mixture, "productive")
    assert abs(sum(out.values()) - 1.0) < 1e-6
    # no clamps fire on this mixture (all within productive bounds)
    assert not any("all_pinned_scale" in s for s in log)


def test_rescue_rich_waived_when_backlog_saturated():
    mixture = {"exploit": 0.50, "rescue": 0.20, "explore": 0.30}
    out, _ = clamp_mixture(mixture, "rescue_rich", route_backlog_saturated=True)
    # waiver: rescue minimum not enforced
    assert out["rescue"] == pytest.approx(0.20)


def test_fallback_mixture_matches_defaults_after_clamp():
    for state, default in DEFAULT_MIXTURES.items():
        m, _ = fallback_mixture(state)
        # all components nonneg and sum to ~1
        assert sum(m.values()) == pytest.approx(1.0)
        for k in default:
            assert m[k] >= 0


def test_largest_remainder_sum_invariant():
    mixture = {"exploit": 0.55, "rescue": 0.25, "explore": 0.20}
    for n in (0, 1, 3, 7, 10, 50):
        out = largest_remainder(mixture, n)
        assert sum(out.values()) == n


def test_largest_remainder_deterministic_tiebreak():
    mixture = {"a": 0.33, "b": 0.33, "c": 0.34}
    out1 = largest_remainder(mixture, 10)
    out2 = largest_remainder(mixture, 10)
    assert out1 == out2


def test_redistribute_when_explore_infeasible():
    quotas = {"exploit": 4, "rescue": 3, "explore": 3}
    feasibility = {"exploit": True, "rescue": True, "explore": False}
    mixture = {"exploit": 0.40, "rescue": 0.30, "explore": 0.30}
    out, log = redistribute_empty_modes(quotas, feasibility, mixture)
    assert out["explore"] == 0
    assert sum(out.values()) == 10
    assert any("redistribute" in s for s in log)


def test_redistribute_no_op_when_all_feasible():
    quotas = {"exploit": 4, "rescue": 3, "explore": 3}
    feasibility = {"exploit": True, "rescue": True, "explore": True}
    out, log = redistribute_empty_modes(quotas, feasibility, {"exploit": 0.4, "rescue": 0.3, "explore": 0.3})
    assert out == quotas
    assert log == []
