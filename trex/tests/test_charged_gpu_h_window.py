"""Regression for raw reserved-allocation window metadata.

The recent reserved-GPU exposure must span the SU window's wall-span, be
tick-aligned/conservative, and be robust to empty/None/early-run edge cases.
Historical audit-only rates are retained, but are not the campaign objective.
"""
from __future__ import annotations

import types

from trex.live_tick import _charged_gpu_h_recent_for_window


def _ev(cc, ch):
    return types.SimpleNamespace(completed_children=cc, charged_gpu_h_total=ch)


def _recent(all_len, window_size, prior, total):
    return _charged_gpu_h_recent_for_window(
        prior, max(0, all_len - window_size), total
    )


def test_empty_prior_returns_total():
    assert _recent(100, 60, [], 50.0) == 50.0          # early run: whole run is window


def test_baseline_is_last_tick_at_or_below_window_start():
    pe = [_ev(10, 5.0), _ev(30, 15.0), _ev(50, 25.0), _ev(80, 40.0)]
    # window_start = 40 → baseline = tick cc=30 (15.0) → recent = 50-15 = 35
    assert _recent(100, 60, pe, 50.0) == 35.0


def test_none_completed_children_skipped_not_break():
    pe = [_ev(10, 5.0), _ev(None, 99.0), _ev(30, 15.0), _ev(80, 40.0)]
    assert _recent(100, 60, pe, 50.0) == 35.0


def test_none_charged_on_baseline_falls_back_to_earlier():
    pe = [_ev(10, 5.0), _ev(30, None), _ev(80, 40.0)]   # cc=30<=40 but charged None
    assert _recent(100, 60, pe, 50.0) == 45.0           # uses 5.0


def test_early_run_window_start_zero_returns_total():
    assert _recent(30, 60, [_ev(10, 5.0), _ev(20, 9.0)], 9.0) == 9.0


def test_never_negative():
    # baseline > total can't happen (monotonic), but the guard holds regardless
    assert _recent(100, 60, [_ev(30, 49.9)], 50.0) >= 0.0
    assert round(_recent(100, 60, [_ev(30, 49.9)], 50.0), 4) == 0.1
