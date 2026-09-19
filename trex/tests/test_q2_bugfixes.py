"""Q2 bug fixes (smoke 8715062 findings):

  (a) Supervisor validator rejects duplicate candidate_id across modes.
  (b) Redistribute protects small-mixture modes from quota starvation.
"""

from __future__ import annotations

import pytest

from trex.fallback import redistribute_empty_modes
from trex.supervisor import _validate_schema


def _decision(cid: str, mode: str, rank: int = 1, global_rank: int | None = None):
    return {
        "candidate_id": cid,
        "mode": mode,
        "rank_in_mode": rank,
        "global_rank": global_rank if global_rank is not None else rank,
        "resource_class": "standard",
        "what": "x",
        "why": "y",
        "evidence_refs": ["e1"],
        "expected_signal": "z",
        "stop_or_downgrade_if": "w",
    }


# ---- Q2a: duplicate-candidate validator -----------------------------------


def test_supervisor_rejects_same_candidate_across_modes():
    obj = {
        "abstain": False,
        "confidence": 0.7,
        "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            _decision("c1", "exploit", rank=1),
            _decision("c2", "rescue", rank=1),
            _decision("c1", "explore", rank=1),  # c1 appears twice
        ],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1", "c2"})
    assert not ok
    assert "duplicate_candidate_across_modes" in why


def test_supervisor_accepts_unique_candidates():
    obj = {
        "abstain": False,
        "confidence": 0.7,
        "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [
            _decision("c1", "exploit", rank=1, global_rank=1),
            _decision("c2", "rescue", rank=1, global_rank=2),
            _decision("c3", "explore", rank=1, global_rank=3),
        ],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1", "c2", "c3"})
    assert ok, why


def test_supervisor_same_candidate_same_mode_caught_by_rank_check():
    """Two decisions for c1 both in exploit with same rank is caught by
    the existing dup_rank check. Different ranks within same mode is
    weird but not strictly invalid (single mode, single candidate could
    appear with rank 1 then rank 2 — but the dup_rank check actually
    requires distinct ranks per mode, so duplicate candidate same mode
    with same rank also fails)."""
    obj = {
        "abstain": False,
        "confidence": 0.7,
        "rationale": "r",
        "mode_mixture": {"exploit": 1.0, "rescue": 0.0, "explore": 0.0},
        "candidate_decisions": [
            _decision("c1", "exploit", rank=1),
            _decision("c1", "exploit", rank=2),  # same mode, distinct rank
        ],
    }
    ok, why = _validate_schema(obj, known_candidate_ids={"c1"})
    # The duplicate-candidate check fires first regardless of mode/rank
    assert not ok
    assert "duplicate_candidate_across_modes" in why


# ---- Q2b: redistribute min-representation guarantee -----------------------


def test_redistribute_protects_small_explore_mixture():
    """Smoke 8715062 case: exploit infeasible, mixture {0.5/0.4/0.1},
    raw quotas {2/1/0}, redistribute moved 2 slots → naively rescue:3,
    explore:0. With protection, explore (mixture 0.1) should steal 1
    from rescue when feasible."""
    quotas = {"exploit": 2, "rescue": 1, "explore": 0}
    feasibility = {"exploit": False, "rescue": True, "explore": True}
    mixture = {"exploit": 0.5, "rescue": 0.4, "explore": 0.1}
    out, log = redistribute_empty_modes(quotas, feasibility, mixture)
    assert out["explore"] >= 1, f"explore starved: {out}"
    assert any("min_representation" in s for s in log)
    # rescue is the donor — was 3, becomes 2
    assert out["rescue"] == 2


def test_redistribute_no_protection_when_mixture_too_small():
    """A feasible mode with mixture < MIN_REPRESENTATION_FLOOR (0.08)
    is NOT protected — that's intentional (very small mixture means LLM
    didn't request meaningful explore)."""
    quotas = {"exploit": 2, "rescue": 1, "explore": 0}
    feasibility = {"exploit": False, "rescue": True, "explore": True}
    mixture = {"exploit": 0.55, "rescue": 0.40, "explore": 0.05}  # explore below floor
    out, log = redistribute_empty_modes(quotas, feasibility, mixture)
    # explore mixture 0.05 < 0.08 floor → not protected
    assert out["explore"] == 0
    assert not any("min_representation" in s for s in log)


def test_redistribute_no_op_when_all_feasible():
    quotas = {"exploit": 1, "rescue": 1, "explore": 1}
    feasibility = {"exploit": True, "rescue": True, "explore": True}
    mixture = {"exploit": 0.4, "rescue": 0.3, "explore": 0.3}
    out, log = redistribute_empty_modes(quotas, feasibility, mixture)
    assert out == quotas
    assert log == []


def test_redistribute_does_not_steal_from_mode_with_just_one_slot():
    """If donor would drop to 0, abort (avoid the infinite swap)."""
    quotas = {"exploit": 1, "rescue": 0, "explore": 0}
    feasibility = {"exploit": True, "rescue": True, "explore": True}
    # both rescue and explore want protection but donor only has 1
    mixture = {"exploit": 0.4, "rescue": 0.3, "explore": 0.3}
    out, log = redistribute_empty_modes(quotas, feasibility, mixture)
    # With only 1 slot in exploit, can't steal — exploit stays 1, neither rescue
    # nor explore protected (since donor requires >1).
    assert out["exploit"] == 1
    assert out["rescue"] == 0
    assert out["explore"] == 0


def test_supervisor_rejects_nonfinite_confidence_and_mixture():
    base = {
        "abstain": False,
        "confidence": 0.7,
        "rationale": "r",
        "mode_mixture": {"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        "candidate_decisions": [_decision("c1", "exploit", rank=1)],
    }
    bad_conf = dict(base)
    bad_conf["confidence"] = float("nan")
    ok, why = _validate_schema(bad_conf, known_candidate_ids={"c1"})
    assert not ok
    assert why == "confidence_invalid"

    bad_mix = dict(base)
    bad_mix["mode_mixture"] = {"exploit": float("inf"), "rescue": 0.0, "explore": 0.0}
    ok, why = _validate_schema(bad_mix, known_candidate_ids={"c1"})
    assert not ok
    assert why == "mode_mixture_negative_or_non_numeric"
