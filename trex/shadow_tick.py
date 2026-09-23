"""Run a decision-only cycle and return its trace without launching workers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .candidate_builder import BuilderConfig, build_candidates
from .planner import PlannerCallConfig, call_planner
from .schemas import (
    EvidenceSummary,
    HypothesisCard,
    PlannerOutput,
    SupervisorOutput,
    to_jsonable,
)
from .selector import SelectorConfig, select_launches
from .supervisor import SupervisorCallConfig, call_supervisor


@dataclass(frozen=True)
class ShadowTickConfig:
    available_slots: int = 3
    planner: PlannerCallConfig = PlannerCallConfig()
    supervisor: SupervisorCallConfig = SupervisorCallConfig()
    selector: SelectorConfig = SelectorConfig()
    builder: BuilderConfig = BuilderConfig()


def run_shadow_tick(
    evidence: EvidenceSummary,
    *,
    active_hypotheses: list[HypothesisCard] | None = None,
    cfg: ShadowTickConfig | None = None,
    tick_id_int: int = 0,
    tick_id: str = "tick_shadow_000",
) -> dict[str, Any]:
    """Run a single complete tick in shadow mode (no launches).

    Returns a JSON-serializable trace dict.
    """
    cfg = cfg or ShadowTickConfig()
    active_hypotheses = active_hypotheses or []

    started = time.time()
    trace: dict[str, Any] = {
        "tick_id": tick_id,
        "evidence": to_jsonable(evidence),
        "active_hypotheses_in": [to_jsonable(h) for h in active_hypotheses],
        "stages": {},
    }

    # 1) Planner
    t0 = time.time()
    planner_out: PlannerOutput = call_planner(
        evidence,
        active_hypotheses=[to_jsonable(h) for h in active_hypotheses],
        seed_action_families=None,
        tick_id_int=tick_id_int,
        cfg=cfg.planner,
    )
    trace["stages"]["planner"] = {
        "elapsed_s": round(time.time() - t0, 2),
        "valid": planner_out.valid,
        "abstain": planner_out.abstain,
        "confidence": planner_out.confidence,
        "fail_reason": planner_out.fail_reason,
        "n_cards": len(planner_out.cards),
        "cards": [to_jsonable(c) for c in planner_out.cards],
        "rationale": planner_out.rationale[:400],
        "usage": planner_out.usage,
    }

    # Effective hypotheses for downstream stages: planner cards or fallback to active
    hyps = planner_out.cards if planner_out.valid and planner_out.cards else active_hypotheses
    fallback_reason = None
    if not planner_out.valid or planner_out.abstain or not planner_out.cards:
        fallback_reason = planner_out.fail_reason or "planner_no_cards"

    # 2) Candidate builder
    t0 = time.time()
    candidates = build_candidates(hyps, evidence, cfg=cfg.builder)
    trace["stages"]["candidate_builder"] = {
        "elapsed_s": round(time.time() - t0, 2),
        "n_candidates": len(candidates),
        "n_feasible": sum(1 for c in candidates if c.feasibility.all_ok()),
        "candidates": [
            {
                "id": c.candidate_id,
                "family": c.method_family,
                "hyp_ids": list(c.hypothesis_ids),
                "feas_ok": c.feasibility.all_ok(),
                "feas_reasons": list(c.feasibility.reasons),
            }
            for c in candidates
        ],
    }

    # 3) Supervisor
    t0 = time.time()
    sup_out: SupervisorOutput
    if not hyps or not candidates:
        sup_out = SupervisorOutput(
            valid=False,
            abstain=True,
            confidence=0.0,
            fail_reason="no_hypotheses_or_candidates",
            mode_mixture={},
            candidate_decisions=[],
            rationale="",
            raw_text="",
            usage={},
        )
    else:
        sup_out = call_supervisor(evidence, hyps, candidates, cfg=cfg.supervisor)
    trace["stages"]["supervisor"] = {
        "elapsed_s": round(time.time() - t0, 2),
        "valid": sup_out.valid,
        "abstain": sup_out.abstain,
        "confidence": sup_out.confidence,
        "fail_reason": sup_out.fail_reason,
        "mode_mixture": sup_out.mode_mixture,
        "n_decisions": len(sup_out.candidate_decisions),
        "decisions": [to_jsonable(d) for d in sup_out.candidate_decisions],
        "rationale": sup_out.rationale[:400],
        "usage": sup_out.usage,
    }

    if not sup_out.valid:
        fallback_reason = fallback_reason or (sup_out.fail_reason or "supervisor_invalid")

    # 4) Selector
    t0 = time.time()
    launches, sel_debug = select_launches(
        evidence,
        candidates,
        sup_out,
        cfg=cfg.selector,
        tick_id=tick_id,
        fallback_reason=fallback_reason,
    )
    trace["stages"]["selector"] = {
        "elapsed_s": round(time.time() - t0, 2),
        "fallback_reason": fallback_reason,
        "debug": sel_debug,
        "launches": [to_jsonable(l) for l in launches],
        "n_launched": sum(1 for l in launches if l.status == "launched"),
        "n_rejected": sum(1 for l in launches if l.status == "rejected"),
    }

    trace["wall_clock_s"] = round(time.time() - started, 2)
    trace["overall_status"] = (
        "fallback" if fallback_reason else "llm_driven"
    )
    return trace
