"""Unit tests for the SSOT `success_criteria` module (Plan §2.5).

Every consumer of strict_success / near_miss / joint_fail / NEAR_PASS_MARGINS
imports from this file; a silent change here would propagate to lifecycle
TTL retire decisions, EvidenceSummary axis_stats, recipe classifications,
state classifier triggers, and the §18.1 acceptance gates. Hence the
boundary-condition coverage here is load-bearing.
"""

from __future__ import annotations

import pytest

from trex.success_criteria import (
    NEAR_MISS_MAX_HARD_FAIL_MARGIN_MULTIPLIER,
    NEAR_PASS_MARGINS,
    STRICT_SUCCESS,
    is_joint_fail,
    is_near_miss,
    is_strict_success,
)


# ---------------------------------------------------------------------------
# Threshold invariants (Plan §2.5 SSOT)
# ---------------------------------------------------------------------------


def test_strict_thresholds_match_plan_2_5():
    """Official Complexa: pLDDT >= 90, iPAE <= 7/31, scRMSD < 1.5 Å (strict)."""
    assert STRICT_SUCCESS["pLDDT"] == (90.0, "increase")
    assert abs(STRICT_SUCCESS["iPAE"][0] - 7 / 31) < 1e-12
    assert STRICT_SUCCESS["iPAE"][1] == "decrease"
    assert STRICT_SUCCESS["binder_scRMSD"] == (1.5, "decrease")


def test_near_pass_margins_match_plan_2_5():
    """Plan §2.5: pLDDT 5.0, iPAE 0.05, scRMSD 0.3 Å."""
    assert NEAR_PASS_MARGINS == {
        "pLDDT": 5.0,
        "iPAE": 0.05,
        "binder_scRMSD": 0.3,
    }


# ---------------------------------------------------------------------------
# is_strict_success — boundary cases
# ---------------------------------------------------------------------------


def test_strict_passes_at_exact_thresholds():
    """pLDDT (>=) and iPAE (<=) pass AT the boundary; scRMSD is STRICT (<) per
    official Complexa, so it must be just below 1.5."""
    assert is_strict_success(
        {"pLDDT": 90.0, "iPAE": 7 / 31, "binder_scRMSD": 1.4999}
    )


def test_strict_fails_scrmsd_at_exact_threshold():
    """Official Complexa scRMSD_ca op is strict '<': scRMSD == 1.5 A FAILS."""
    assert not is_strict_success(
        {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.5}
    )


def test_strict_fails_one_axis_just_below():
    """Any single axis just below the threshold → not strict."""
    for axis, val in [
        ("pLDDT", 89.99),  # < 90
        ("iPAE", 7 / 31 + 1e-6),  # > 7/31
        ("binder_scRMSD", 1.501),  # > 1.5
    ]:
        m = {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0}
        m[axis] = val
        assert not is_strict_success(m), f"{axis}={val} should fail"


def test_strict_fails_when_any_axis_missing():
    """Missing axis is treated as failure (cannot prove success)."""
    for missing in ("pLDDT", "iPAE", "binder_scRMSD"):
        m = {"pLDDT": 95.0, "iPAE": 0.10, "binder_scRMSD": 1.0}
        del m[missing]
        assert not is_strict_success(m), f"missing {missing} should fail"


def test_strict_fails_on_empty_metrics():
    assert not is_strict_success({})


def test_strict_pass_realistic_high_quality():
    """Realistic high-quality result."""
    assert is_strict_success(
        {"pLDDT": 92.3, "iPAE": 0.18, "binder_scRMSD": 1.1, "ipTM": 0.85}
    )


# ---------------------------------------------------------------------------
# is_near_miss — boundary cases
# ---------------------------------------------------------------------------


def test_near_miss_zero_axes_failing_one_near_pass():
    """0 fails, 2 pass + 1 near_pass → near_miss (barely-not-strict)."""
    # pLDDT 87 (near_pass: 85-90 in margin) — others pass
    assert is_near_miss(
        {"pLDDT": 87.0, "iPAE": 0.18, "binder_scRMSD": 1.0}
    )


def test_near_miss_exactly_one_axis_fail_others_pass():
    """1 fail, 2 pass → near_miss (single-blocker)."""
    # iPAE 0.35 way above 0.276 fail line; others pass
    assert is_near_miss(
        {"pLDDT": 92.0, "iPAE": 0.35, "binder_scRMSD": 1.0}
    )


def test_catastrophic_single_axis_failure_is_not_near_miss():
    """A detached 45 A binder remains evidence, but is not rescue-proximal."""
    assert NEAR_MISS_MAX_HARD_FAIL_MARGIN_MULTIPLIER == 10.0
    assert not is_near_miss(
        {"pLDDT": 92.0, "iPAE": 0.18, "binder_scRMSD": 45.0}
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_near_miss_rejects_nonfinite_axes(bad: float):
    assert not is_near_miss(
        {"pLDDT": 92.0, "iPAE": 0.18, "binder_scRMSD": bad}
    )


def test_not_near_miss_when_two_axes_fail():
    """2 fails → not near_miss (joint_fail territory)."""
    assert not is_near_miss(
        {"pLDDT": 60.0, "iPAE": 0.40, "binder_scRMSD": 1.0}
    )


def test_strict_success_is_not_near_miss():
    """If strict_success holds, near_miss must be False."""
    assert not is_near_miss(
        {"pLDDT": 92.0, "iPAE": 0.18, "binder_scRMSD": 1.0}
    )


def test_near_miss_requires_all_three_axes_present():
    """Partial verifier records cannot be near-misses; missing axis is unknown."""
    assert not is_near_miss({"pLDDT": 89.0})
    assert not is_near_miss({"pLDDT": 92.0, "iPAE": 0.35})


def test_near_miss_exactly_one_axis_fail_requires_other_axes_present():
    """1 fail, 2 present pass axes → near_miss."""
    assert is_near_miss({"pLDDT": 92.0, "iPAE": 0.35, "binder_scRMSD": 1.0})


# ---------------------------------------------------------------------------
# is_joint_fail — boundary cases
# ---------------------------------------------------------------------------


def test_joint_fail_both_pLDDT_and_iPAE_clearly_fail():
    """pLDDT < 85 AND iPAE > 0.276 -> joint_fail pivot/caution signal."""
    assert is_joint_fail({"pLDDT": 70.0, "iPAE": 0.40, "binder_scRMSD": 1.0})


def test_joint_fail_requires_both_pLDDT_and_iPAE_to_fail_hard():
    """Single-axis near-miss is NOT joint_fail."""
    assert not is_joint_fail(
        {"pLDDT": 92.0, "iPAE": 0.40, "binder_scRMSD": 1.0}
    )
    assert not is_joint_fail(
        {"pLDDT": 70.0, "iPAE": 0.18, "binder_scRMSD": 1.0}
    )


def test_joint_fail_scRMSD_doesnt_matter():
    """scRMSD is not part of the joint_fail signal."""
    assert is_joint_fail({"pLDDT": 60.0, "iPAE": 0.45, "binder_scRMSD": 5.0})
    assert is_joint_fail({"pLDDT": 60.0, "iPAE": 0.45})  # scRMSD missing too


def test_joint_fail_false_when_any_axis_missing():
    """Missing pLDDT or iPAE → joint_fail can't be decided."""
    assert not is_joint_fail({"iPAE": 0.45})
    assert not is_joint_fail({"pLDDT": 60.0})


def test_joint_fail_boundary_at_near_pass_margin():
    """Exactly at the near_pass boundary → NOT joint_fail (uses >, not >=)."""
    # pLDDT = 85.0 (exactly at 90 - 5 margin) → p_fail uses '<' so False
    # iPAE = 0.276 (exactly at 7/31 + 0.05) → i_fail uses '>' so False
    p_boundary = STRICT_SUCCESS["pLDDT"][0] - NEAR_PASS_MARGINS["pLDDT"]
    i_boundary = STRICT_SUCCESS["iPAE"][0] + NEAR_PASS_MARGINS["iPAE"]
    assert not is_joint_fail({"pLDDT": p_boundary, "iPAE": i_boundary})
