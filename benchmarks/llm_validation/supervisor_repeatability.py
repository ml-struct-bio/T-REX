"""Supervisor ranking repeatability and sampling sensitivity on a fixed candidate set.

Requires an existing LLM endpoint. No molecular backend is run.
Use python -m benchmarks.llm_validation.supervisor_repeatability --help for options.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from trex import SCHEMA_VERSION
from benchmarks.llm_validation.planner_validation import case_productive
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    HypothesisCard,
    PredictedChange,
    PreserveConstraint,
)
from trex.supervisor import SupervisorCallConfig, call_supervisor


def _feas_ok() -> FeasibilityCheck:
    return FeasibilityCheck(True, "rb1", True, True, True, True)


def build_reliability_case() -> tuple[list[HypothesisCard], list[ActionCandidate]]:
    """5 candidates, 3 modes, with planted quality ordering.

    Rough expected supervisor ranking based on evidence quality:
      exploit: c_a (clear best, supported hypothesis) > c_b (also supported)
      rescue:  c_c (high-quality near-miss) > c_d (borderline)
      explore: c_e (sole external)
    """
    hyps = [
        HypothesisCard(
            hypothesis_id="hyp_A",
            target_id="t1",
            tick_created=10,
            claim="Strong exploit signal: complexa_beam is producing strict successes.",
            mode_affinity={"exploit": 0.8, "rescue": 0.1, "explore": 0.1},
            evidence_refs=["tick_010", "r_001"],
            predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["r_001"], 0.10, None)],
            preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
            recommended_action_families=["complexa_beam"],
            status="supported",
            support_points=4.0,
        ),
        HypothesisCard(
            hypothesis_id="hyp_B",
            target_id="t1",
            tick_created=11,
            claim="Secondary exploit: diversify on top hits to add panel coverage.",
            mode_affinity={"exploit": 0.6, "rescue": 0.2, "explore": 0.2},
            evidence_refs=["tick_010"],
            predicted_metric_changes=[PredictedChange("binder_scRMSD", "decrease", ["r_001"], 0.10, None)],
            preserve_constraints=[],
            recommended_action_families=["complexa_best_of_n"],
            status="supported",
            support_points=2.0,
        ),
        HypothesisCard(
            hypothesis_id="hyp_C",
            target_id="t1",
            tick_created=12,
            claim="High-quality near-miss: MPNN redesign should rescue iPAE on r_088.",
            mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
            evidence_refs=["tick_010", "r_088"],
            predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["r_088"], 0.25, None)],
            preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
            recommended_action_families=["proteinmpnn_redesign"],
        ),
        HypothesisCard(
            hypothesis_id="hyp_D",
            target_id="t1",
            tick_created=13,
            claim="Borderline rescue: structure_refilter retrying weaker candidates.",
            mode_affinity={"exploit": 0.1, "rescue": 0.6, "explore": 0.3},
            evidence_refs=["tick_010"],
            predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["r_088"], 0.10, None)],
            preserve_constraints=[],
            recommended_action_families=["structure_refilter"],
        ),
        HypothesisCard(
            hypothesis_id="hyp_E",
            target_id="t1",
            tick_created=14,
            claim="Single explore: BoltzGen for diversity beyond complexa lineage.",
            mode_affinity={"exploit": 0.05, "rescue": 0.10, "explore": 0.85},
            evidence_refs=["tick_010"],
            predicted_metric_changes=[PredictedChange("binder_scRMSD", "decrease", ["r_001"], 0.15, None)],
            preserve_constraints=[],
            recommended_action_families=["boltzgen"],
        ),
    ]

    candidates: list[ActionCandidate] = []
    family_to_cand_name = {
        "complexa_beam": "c_a",
        "complexa_best_of_n": "c_b",
        "proteinmpnn_redesign": "c_c",
        "structure_refilter": "c_d",
        "boltzgen": "c_e",
    }
    for h in hyps:
        fam = h.recommended_action_families[0]
        cid = family_to_cand_name[fam]
        cost = (
            "standard" if fam in ("complexa_beam", "complexa_best_of_n")
            else "low" if fam in ("proteinmpnn_redesign", "structure_refilter")
            else "diagnostic"
        )
        candidates.append(
            ActionCandidate(
                candidate_id=cid,
                hypothesis_ids=[h.hypothesis_id],
                parent_result_id=h.evidence_refs[-1] if h.evidence_refs else None,
                method_family=fam,
                operator_id=f"{fam}_default",
                lane_id=fam,
                config_delta={},
                downstream_route_plan=["structure_refilter"] if fam == "boltzgen" else [],
                estimated_cost_class=cost,  # type: ignore[arg-type]
                expected_signal=f"{fam} expected per {h.claim[:60]}",
                evidence_refs=list(h.evidence_refs),
                feasibility=_feas_ok(),
            )
        )

    return hyps, candidates


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float | None:
    """Kendall tau on two orderings of the same set of items.
    Returns None if orderings differ in items or are shorter than 2."""
    if set(rank_a) != set(rank_b) or len(rank_a) < 2:
        return None
    pos_a = {x: i for i, x in enumerate(rank_a)}
    pos_b = {x: i for i, x in enumerate(rank_b)}
    items = rank_a
    concordant = 0
    discordant = 0
    for x, y in combinations(items, 2):
        sign_a = (pos_a[x] - pos_a[y])
        sign_b = (pos_b[x] - pos_b[y])
        if sign_a * sign_b > 0:
            concordant += 1
        elif sign_a * sign_b < 0:
            discordant += 1
    n = concordant + discordant
    if n == 0:
        return None
    return (concordant - discordant) / n


def mixture_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _, evidence = case_productive(args.model)
    hyps, candidates = build_reliability_case()

    base_cfg = SupervisorCallConfig(
        model=args.model, base_url=args.base_url, max_tokens=args.max_tokens,
        confidence_threshold=0.0,
    )

    # Use varied temperatures to elicit any latent noise.
    temps = args.temperatures or [0.5, 0.6, 0.7, 0.8, 0.9]
    if len(temps) < args.repeats:
        # repeat the cycle to reach args.repeats
        temps = (temps * ((args.repeats // len(temps)) + 1))[: args.repeats]

    records = []
    started = time.time()
    for k in range(args.repeats):
        cfg = SupervisorCallConfig(
            model=base_cfg.model, base_url=base_cfg.base_url,
            max_tokens=base_cfg.max_tokens, enable_thinking=base_cfg.enable_thinking,
            confidence_threshold=base_cfg.confidence_threshold,
            timeout_s=base_cfg.timeout_s, temperature=temps[k],
        )
        t0 = time.time()
        out = call_supervisor(evidence, hyps, candidates, cfg=cfg)
        elapsed = time.time() - t0

        per_mode_rank: dict[str, list[str]] = {}
        for d in out.candidate_decisions:
            per_mode_rank.setdefault(d.mode, []).append((d.rank_in_mode, d.candidate_id))
        # Convert to ordered candidate_id lists per mode
        per_mode_order = {
            m: [cid for _, cid in sorted(pairs)] for m, pairs in per_mode_rank.items()
        }
        records.append({
            "repeat": k,
            "temperature": temps[k],
            "valid": out.valid,
            "confidence": out.confidence,
            "fail_reason": out.fail_reason,
            "mode_mixture": out.mode_mixture,
            "per_mode_order": per_mode_order,
            "n_decisions": len(out.candidate_decisions),
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  [run {k+1}/{args.repeats}] temp={temps[k]} valid={out.valid} "
              f"mix={out.mode_mixture} per_mode={per_mode_order}")

    # Pairwise comparisons
    valid_records = [r for r in records if r["valid"]]
    pairwise: list[dict[str, Any]] = []
    if len(valid_records) >= 2:
        for i, j in combinations(range(len(valid_records)), 2):
            a, b = valid_records[i], valid_records[j]
            mode_taus = {}
            for mode in set(a["per_mode_order"]) | set(b["per_mode_order"]):
                tau = kendall_tau(
                    a["per_mode_order"].get(mode, []),
                    b["per_mode_order"].get(mode, []),
                )
                if tau is not None:
                    mode_taus[mode] = round(tau, 3)
            pairwise.append({
                "i": a["repeat"], "j": b["repeat"],
                "mixture_l1": round(mixture_l1(a["mode_mixture"], b["mode_mixture"]), 3),
                "mode_kendall_taus": mode_taus,
                "top1_agreement": {
                    mode: (a["per_mode_order"].get(mode, [""])[:1]
                           == b["per_mode_order"].get(mode, [""])[:1])
                    for mode in set(a["per_mode_order"]) | set(b["per_mode_order"])
                },
            })

    # Aggregate
    all_taus = [tau for p in pairwise for tau in p["mode_kendall_taus"].values()]
    mixture_l1s = [p["mixture_l1"] for p in pairwise]
    confs = [r["confidence"] for r in valid_records]
    top1_agree_rate = (
        sum(1 for p in pairwise for v in p["top1_agreement"].values() if v)
        / max(1, sum(1 for p in pairwise for _ in p["top1_agreement"]))
    )

    summary = {
        "model": base_cfg.model,
        "n_repeats": args.repeats,
        "n_valid": len(valid_records),
        "kendall_tau_median": (statistics.median(all_taus) if all_taus else None),
        "kendall_tau_mean": (statistics.mean(all_taus) if all_taus else None),
        "kendall_tau_min": (min(all_taus) if all_taus else None),
        "mixture_l1_median": (statistics.median(mixture_l1s) if mixture_l1s else None),
        "mixture_l1_max": (max(mixture_l1s) if mixture_l1s else None),
        "top1_agreement_rate": top1_agree_rate,
        "confidence_median": (statistics.median(confs) if confs else None),
        "confidence_stdev": (statistics.pstdev(confs) if len(confs) > 1 else 0.0),
        "wall_clock_s": round(time.time() - started, 2),
        "schema_version": SCHEMA_VERSION,
        "interpretation": (
            "kendall_tau_median >= 0.6 → ranking is stable, BTL not needed for MVP. "
            "< 0.6 → BTL/Elo averaging would materially reduce ranking noise."
        ),
    }
    return {"summary": summary, "records": records, "pairwise": pairwise}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vllm/Qwen/Qwen3.6-27B-FP8")
    p.add_argument("--base-url", default="http://127.0.0.1:8500/v1")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--temperatures", type=float, nargs="*", default=None,
        help="Per-repeat temperature. If omitted, uses [0.5, 0.6, 0.7, 0.8, 0.9] cycled.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    report = run_smoke(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
