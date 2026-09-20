"""Supervisor output validation on four synthetic campaign-state cases.

Requires an existing LLM endpoint. No molecular backend is run.
Use python -m benchmarks.llm_validation.supervisor_validation --help for options.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from trex import SCHEMA_VERSION
from benchmarks.llm_validation.planner_validation import (
    case_productive,
    case_rescue_rich,
    case_stalled,
    case_panel_ready_diversity_short,
)
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    HypothesisCard,
    PredictedChange,
    PreserveConstraint,
    to_jsonable,
)
from trex.supervisor import SupervisorCallConfig, call_supervisor


def _feas_ok(bucket: str = "rb1") -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=True,
        runtime_bucket_id=bucket,
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )


def hypotheses_for_productive(target: str) -> list[HypothesisCard]:
    return [
        HypothesisCard(
            hypothesis_id="hyp_p1",
            target_id=target,
            tick_created=10,
            claim="Continue Complexa beam: SU velocity remains positive.",
            mode_affinity={"exploit": 0.8, "rescue": 0.1, "explore": 0.1},
            evidence_refs=["tick_010"],
            predicted_metric_changes=[
                PredictedChange("iPAE", "decrease", ["r_001"], 0.10, None),
            ],
            preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
            recommended_action_families=["complexa_beam"],
            status="supported",
            support_points=3.0,
        ),
        HypothesisCard(
            hypothesis_id="hyp_p2",
            target_id=target,
            tick_created=11,
            claim="MPNN redesign on near-miss parents may add diversity bins.",
            mode_affinity={"exploit": 0.2, "rescue": 0.6, "explore": 0.2},
            evidence_refs=["tick_010"],
            predicted_metric_changes=[
                PredictedChange("iPAE", "decrease", ["r_001"], 0.15, None),
            ],
            preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
            recommended_action_families=["proteinmpnn_redesign", "structure_refilter"],
        ),
    ]


def hypotheses_for_rescue_rich(target: str) -> list[HypothesisCard]:
    return [
        HypothesisCard(
            hypothesis_id="hyp_rr1",
            target_id=target,
            tick_created=12,
            claim="iPAE is the dominant failure axis; MPNN interface redesign should rescue.",
            mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
            evidence_refs=["tick_012"],
            predicted_metric_changes=[
                PredictedChange("iPAE", "decrease", ["r_betv_010"], 0.20, None),
            ],
            preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
            recommended_action_families=["proteinmpnn_redesign"],
        ),
    ]


def hypotheses_for_stalled(target: str) -> list[HypothesisCard]:
    return [
        HypothesisCard(
            hypothesis_id="hyp_s1",
            target_id=target,
            tick_created=15,
            claim="Local search exhausted; BoltzGen/BindCraft escape may unlock new scaffolds.",
            mode_affinity={"exploit": 0.1, "rescue": 0.2, "explore": 0.7},
            evidence_refs=["tick_015"],
            predicted_metric_changes=[
                PredictedChange("pLDDT", "increase", ["r_sc2_021"], 0.10, None),
                PredictedChange("iPAE", "decrease", ["r_sc2_021"], 0.20, None),
            ],
            preserve_constraints=[],
            recommended_action_families=["boltzgen", "bindcraft"],
        ),
    ]


def hypotheses_for_redundant(target: str) -> list[HypothesisCard]:
    return [
        HypothesisCard(
            hypothesis_id="hyp_d1",
            target_id=target,
            tick_created=18,
            claim="Foldseek top bin is saturated; explore new initializations.",
            mode_affinity={"exploit": 0.2, "rescue": 0.2, "explore": 0.6},
            evidence_refs=["tick_018"],
            predicted_metric_changes=[
                PredictedChange("binder_scRMSD", "decrease", ["r_pdl_080"], 0.15, None),
            ],
            preserve_constraints=[],
            recommended_action_families=["complexa_best_of_n"],
        ),
    ]


def candidates_for_hyps(hyps: list[HypothesisCard]) -> list[ActionCandidate]:
    out: list[ActionCandidate] = []
    cid = 0
    for h in hyps:
        for fam in h.recommended_action_families:
            cid += 1
            cost: str
            if fam in ("complexa_beam",):
                cost = "standard"
            elif fam in ("proteinmpnn_redesign", "structure_refilter"):
                cost = "low"
            elif fam in ("complexa_best_of_n",):
                cost = "standard"
            elif fam in ("boltzgen", "bindcraft"):
                cost = "diagnostic"
            else:
                cost = "standard"
            out.append(
                ActionCandidate(
                    candidate_id=f"cand_{h.hypothesis_id}_{cid:03d}",
                    hypothesis_ids=[h.hypothesis_id],
                    parent_result_id=h.evidence_refs[0] if h.evidence_refs else None,
                    method_family=fam,
                    operator_id=f"{fam}_default",
                    lane_id=fam,
                    config_delta={},
                    downstream_route_plan=(
                        ["structure_refilter"]
                        if fam in ("boltzgen", "bindcraft")
                        else []
                    ),
                    estimated_cost_class=cost,  # type: ignore[arg-type]
                    expected_signal=f"{fam} expected to improve {h.predicted_metric_changes[0].axis}",
                    evidence_refs=list(h.evidence_refs),
                    feasibility=_feas_ok(),
                )
            )
    return out


CASES = [
    ("productive_cd45_like", case_productive, hypotheses_for_productive,
     {"exploit_min": 0.45, "explore_max": 0.30}),
    ("rescue_rich_betv1_like", case_rescue_rich, hypotheses_for_rescue_rich,
     {"rescue_min": 0.40}),
    ("stalled_sc2rbd_like", case_stalled, hypotheses_for_stalled,
     {"explore_min": 0.25, "exploit_max": 0.45}),
    ("stalled_redundant_pdl1", case_panel_ready_diversity_short,
     hypotheses_for_redundant,
     {"explore_min": 0.25}),
]


def assess_mode_mixture(
    out_mixture: dict[str, float], expectations: dict[str, float]
) -> dict[str, Any]:
    checks = {}
    if not out_mixture:
        return {"all_ok": False, "reason": "empty_mixture"}
    s = sum(out_mixture.values())
    norm = {k: v / s for k, v in out_mixture.items()} if s > 0 else out_mixture
    for k, v in expectations.items():
        mode = k.split("_")[0]
        if k.endswith("_min"):
            checks[k] = norm.get(mode, 0.0) >= v
        elif k.endswith("_max"):
            checks[k] = norm.get(mode, 0.0) <= v
    return {
        "all_ok": all(checks.values()),
        "checks": checks,
        "normalized_mixture": norm,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cfg = SupervisorCallConfig(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        confidence_threshold=0.0,
    )

    records: list[dict[str, Any]] = []
    started = time.time()
    for repeat in range(args.repeats):
        for name, build_e, build_h, expectations in CASES:
            _, evidence = build_e(cfg.model)
            hyps = build_h(evidence.target_id)
            cands = candidates_for_hyps(hyps)

            t0 = time.time()
            out = call_supervisor(evidence, hyps, cands, cfg=cfg)
            elapsed = time.time() - t0

            assessment = (
                assess_mode_mixture(out.mode_mixture, expectations)
                if out.valid and out.mode_mixture
                else {"all_ok": False, "reason": "no_valid_mixture"}
            )

            rec = {
                "case": name,
                "repeat": repeat,
                "state": evidence.state_label,
                "n_candidates_in": len(cands),
                "elapsed_s": round(elapsed, 2),
                "valid": out.valid,
                "abstain": out.abstain,
                "confidence": out.confidence,
                "fail_reason": out.fail_reason,
                "mode_mixture": out.mode_mixture,
                "n_decisions": len(out.candidate_decisions),
                "decisions_per_mode": {
                    m: sum(1 for d in out.candidate_decisions if d.mode == m)
                    for m in ("exploit", "rescue", "explore")
                },
                "resources_used": [d.resource_class for d in out.candidate_decisions],
                "mixture_assessment": assessment,
                "usage": out.usage,
            }
            if not out.valid:
                rec["raw_excerpt"] = (out.raw_text or "")[:1200]
            records.append(rec)
            print(
                json.dumps(
                    {
                        k: rec[k]
                        for k in (
                            "case",
                            "valid",
                            "confidence",
                            "n_decisions",
                            "mode_mixture",
                            "fail_reason",
                            "elapsed_s",
                        )
                    }
                )
            )

    valid_records = [r for r in records if r["valid"]]
    mixture_ok = [r for r in valid_records if r["mixture_assessment"]["all_ok"]]
    confs = [r["confidence"] for r in valid_records]

    summary = {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "total_calls": len(records),
        "valid_calls": len(valid_records),
        "valid_rate": (len(valid_records) / len(records)) if records else 0.0,
        "mixture_aligned_with_state_rate": (
            len(mixture_ok) / len(valid_records) if valid_records else 0.0
        ),
        "confidence_min": min(confs) if confs else None,
        "confidence_max": max(confs) if confs else None,
        "confidence_median": statistics.median(confs) if confs else None,
        "median_latency_s": (
            statistics.median([r["elapsed_s"] for r in records]) if records else None
        ),
        "wall_clock_s": round(time.time() - started, 1),
        "schema_version": SCHEMA_VERSION,
    }
    return {"summary": summary, "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vllm/Qwen/Qwen3.6-27B-FP8")
    p.add_argument("--base-url", default="http://127.0.0.1:8500/v1")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-valid-rate", type=float, default=0.80)
    p.add_argument("--fail-under", action="store_true")
    args = p.parse_args()

    report = run_smoke(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))

    if args.fail_under and report["summary"]["valid_rate"] < args.min_valid_rate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
