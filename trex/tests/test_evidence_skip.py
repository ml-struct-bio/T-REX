"""Evidence-skip gate (over-calling reduction) — integration tests.

When the decision-relevant evidence signature is unchanged between ticks, the
Planner + Supervisor LLM calls are skipped and the prior plan reused. The
deterministic builder + selector still run, so launches are still produced.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from trex.archive import Archive
from trex.live_tick import LiveTickConfig, SkipConfig, run_live_tick
from trex.schemas import SupervisorDecision, TargetConstraint

from .test_live_tick import _hyp, _planner_ok, _result, _sup_ok


def _run(arc, target, *, tick_id, tick_id_int, cfg, planner_ret, sup_ret):
    with patch("trex.live_tick.call_planner", return_value=planner_ret) as pm, \
         patch("trex.live_tick.call_supervisor", return_value=sup_ret) as sm:
        run_live_tick(
            arc, target, tick_id=tick_id, tick_id_int=tick_id_int,
            elapsed_wall_h=0.0, remaining_wall_h=48.0, cfg=cfg,
        )
    return pm.call_count, sm.call_count


def test_skip_fires_when_signature_unchanged(tmp_path: Path):
    arc = Archive(tmp_path / "arc")
    for i in range(5):
        arc.append(_result(f"r{i}"))
    target = TargetConstraint(target_id="t1", target_class="test")
    cfg = LiveTickConfig(skip=SkipConfig(enabled=True))
    card = _hyp("h1", target="t1", tick=1)

    # Tick 1: full path (skip can't fire — no prior evidence/supervisor).
    p1, s1 = _run(arc, target, tick_id="t_001", tick_id_int=1, cfg=cfg,
                  planner_ret=_planner_ok([card]), sup_ret=_sup_ok())
    assert p1 == 1 and s1 == 1

    sup_before = len(list(arc.iter_records(SupervisorDecision)))

    # Tick 2: NO new results → identical signature → skip fires.
    p2, s2 = _run(arc, target, tick_id="t_002", tick_id_int=2, cfg=cfg,
                  planner_ret=_planner_ok([card]), sup_ret=_sup_ok())
    assert p2 == 0, "planner LLM should be skipped on unchanged evidence"
    assert s2 == 0, "supervisor LLM should be skipped on unchanged evidence"

    # A supervisor_decision is still written (reused mixture), so the selector
    # path ran normally.
    sup_after = len(list(arc.iter_records(SupervisorDecision)))
    assert sup_after == sup_before + 1


def test_no_skip_when_disabled(tmp_path: Path):
    arc = Archive(tmp_path / "arc")
    for i in range(5):
        arc.append(_result(f"r{i}"))
    target = TargetConstraint(target_id="t1", target_class="test")
    cfg = LiveTickConfig(skip=SkipConfig(enabled=False))  # default
    card = _hyp("h1", target="t1", tick=1)

    _run(arc, target, tick_id="t_001", tick_id_int=1, cfg=cfg,
         planner_ret=_planner_ok([card]), sup_ret=_sup_ok())
    # Tick 2 with skip disabled: LLMs MUST be called even on unchanged evidence.
    p2, s2 = _run(arc, target, tick_id="t_002", tick_id_int=2, cfg=cfg,
                  planner_ret=_planner_ok([card]), sup_ret=_sup_ok())
    assert p2 == 1 and s2 == 1


def test_no_skip_when_no_active_hyps(tmp_path: Path):
    arc = Archive(tmp_path / "arc")
    for i in range(5):
        arc.append(_result(f"r{i}"))
    target = TargetConstraint(target_id="t1", target_class="test")
    cfg = LiveTickConfig(skip=SkipConfig(enabled=True))

    # Planner returns NO cards → no active hyps persist → skip cannot reuse.
    _run(arc, target, tick_id="t_001", tick_id_int=1, cfg=cfg,
         planner_ret=_planner_ok([]), sup_ret=_sup_ok())
    p2, s2 = _run(arc, target, tick_id="t_002", tick_id_int=2, cfg=cfg,
                  planner_ret=_planner_ok([]), sup_ret=_sup_ok())
    assert p2 == 1, "no active hyps → must call planner (cannot reuse a plan)"
