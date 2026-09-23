"""Run CPU regression tests and deterministic policy checks on synthetic evidence.

Write a JSON report covering state classification, hypothesis lifecycle, and
panel diversity. No LLM or GPU is required.

Usage:
  python -m trex.tests.policy_contracts --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from trex import SCHEMA_VERSION
from trex.evidence_reducer import (
    ReducerConfig,
    StateClassifierConfig,
    classify_state,
)
from trex.lifecycle import (
    LifecycleConfig,
    default_axis_thresholds,
    update_hypothesis,
)
from trex.panel import PanelConfig, diversity_at_K, select_panel
from trex.schemas import (
    AxisStat,
    HypothesisCard,
    PredictedChange,
    PreserveConstraint,
    ResultRecord,
)


def _axis(p: int, np: int, f: int, md: float | None) -> AxisStat:
    return AxisStat(p, np, f, None, None, md, "provisional", p + np + f)


def s2_state_classifier_cases() -> list[dict[str, Any]]:
    cfg = StateClassifierConfig()
    cases = [
        {
            "name": "CD45-like productive",
            "expected": "productive",
            "args": dict(
                worker_gpu_h_last_3_ticks=15.0,
                completed_children_window=6,
                run_su_count_delta=2,
                duplicate_fraction=0.35,
                near_miss_count=3,
                axis_stats={
                    "pLDDT": _axis(20, 4, 2, 1.0),
                    "iPAE": _axis(18, 4, 4, 0.05),
                    "binder_scRMSD": _axis(15, 8, 3, 0.3),
                },
                top_bin_share=0.40,
            ),
        },
        {
            "name": "BetV1-like rescue_rich",
            "expected": "rescue_rich",
            "args": dict(
                worker_gpu_h_last_3_ticks=10.0,
                completed_children_window=5,
                run_su_count_delta=0,
                duplicate_fraction=0.40,
                near_miss_count=6,
                axis_stats={
                    "pLDDT": _axis(20, 2, 3, 0.5),
                    "iPAE": _axis(2, 5, 18, 5.0),  # dominant iPAE deficit
                    "binder_scRMSD": _axis(20, 4, 1, 0.2),
                },
                top_bin_share=0.50,
            ),
        },
        {
            "name": "SC2RBD-like stalled",
            "expected": "stalled",
            "args": dict(
                worker_gpu_h_last_3_ticks=14.0,
                completed_children_window=5,
                run_su_count_delta=0,
                duplicate_fraction=0.60,
                near_miss_count=1,
                axis_stats={
                    "pLDDT": _axis(4, 4, 14, 8.0),
                    "iPAE": _axis(4, 4, 14, 0.30),
                    "binder_scRMSD": _axis(10, 6, 6, 0.5),
                },
                top_bin_share=0.55,
            ),
        },
        {
            "name": "CbAgo-like low_evidence (early)",
            "expected": "low_evidence",
            "args": dict(
                worker_gpu_h_last_3_ticks=1.5,  # below min_window_gpu_h
                completed_children_window=2,
                run_su_count_delta=0,
                duplicate_fraction=None,
                near_miss_count=0,
                axis_stats={
                    "pLDDT": _axis(0, 0, 0, None),
                    "iPAE": _axis(0, 0, 0, None),
                    "binder_scRMSD": _axis(0, 0, 0, None),
                },
                top_bin_share=None,
            ),
        },
        {
            # A productive but structurally repetitive route retains its
            # productive-duplicate state.
            "name": "top-bin-share collapse while still producing",
            "expected": "productive_duplicate",
            "args": dict(
                worker_gpu_h_last_3_ticks=12.0,
                completed_children_window=5,
                run_su_count_delta=1,
                duplicate_fraction=0.65,  # high dup -> not `productive`, but producing
                near_miss_count=2,
                axis_stats={
                    "pLDDT": _axis(10, 2, 2, 0.5),
                    "iPAE": _axis(10, 2, 2, 0.05),
                    "binder_scRMSD": _axis(10, 4, 1, 0.2),
                },
                top_bin_share=0.85,
            ),
        },
        {
            # Severe collapse with NO new SU (dSU==0) still abandons -> stalled.
            "name": "top-bin-share collapse, dry (no new SU)",
            "expected": "stalled",
            "args": dict(
                worker_gpu_h_last_3_ticks=12.0,
                completed_children_window=5,
                run_su_count_delta=0,
                duplicate_fraction=0.65,
                near_miss_count=2,
                axis_stats={
                    "pLDDT": _axis(10, 2, 2, 0.5),
                    "iPAE": _axis(10, 2, 2, 0.05),
                    "binder_scRMSD": _axis(10, 4, 1, 0.2),
                },
                top_bin_share=0.85,
            ),
        },
    ]
    results = []
    for c in cases:
        got = classify_state(cfg=cfg, **c["args"])
        ok = got == c["expected"]
        results.append({"name": c["name"], "expected": c["expected"], "got": got, "ok": ok})
    return results


def s6_lifecycle_replay() -> list[dict[str, Any]]:
    """Replay several known transitions and confirm determinism + correctness."""
    cfg = LifecycleConfig()
    thr = default_axis_thresholds()
    out: list[dict[str, Any]] = []

    def _r(rid: str, **m) -> ResultRecord:
        panel_ready = bool(m.pop("panel_ready", False))
        return ResultRecord(
            result_id=rid,
            parent_ids=[],
            target_id="t",
            backend_family="complexa_beam",
            runtime_bucket_id="rb1",
            metrics=m,
            metrics_calibrated=dict(m),
            route_lineage=[],
            gpu_h=1.0,
            exit_status="ok",
            panel_ready=panel_ready,
        )

    hyp = HypothesisCard(
        hypothesis_id="h_replay",
        target_id="t",
        tick_created=10,
        claim="iPAE improves with MPNN",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["e1"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["b"], 0.20, None),
        ],
        preserve_constraints=[
            PreserveConstraint("pLDDT", 0.05),
        ],
        recommended_action_families=["proteinmpnn_redesign"],
    )
    # Complexa thresholds (pLDDT>=90, iPAE<=0.226): keep pLDDT passing in
    # baseline + descendants so preserve_constraint(pLDDT) holds while iPAE
    # improves materially.
    baseline = _r("b", pLDDT=92.0, iPAE=0.7)

    # Test 1: 2 supporting → supported
    h1 = update_hypothesis(
        hyp,
        healthy_descendants=[
            (_r("d1", pLDDT=92.0, iPAE=0.50), baseline),
            (_r("d2", pLDDT=91.0, iPAE=0.45), baseline),
        ],
        current_tick=11,
        axis_thresholds=thr,
        cfg=cfg,
    )
    out.append(
        {
            "name": "two supporters → supported",
            "expected_status": "supported",
            "got_status": h1.status,
            "ok": h1.status == "supported",
        }
    )

    # Test 2: 3 contradicting → contradicted
    h2 = update_hypothesis(
        hyp,
        healthy_descendants=[
            (_r("d1", pLDDT=84.0, iPAE=0.695), baseline),  # RDR ~3%, contradicts
            (_r("d2", pLDDT=84.0, iPAE=0.69), baseline),
            (_r("d3", pLDDT=84.0, iPAE=0.691), baseline),
        ],
        current_tick=11,
        axis_thresholds=thr,
        cfg=cfg,
    )
    out.append(
        {
            "name": "three contradictors → contradicted",
            "expected_status": "contradicted",
            "got_status": h2.status,
            "ok": h2.status == "contradicted",
        }
    )

    # A card created at tick 10 with a lifetime of 10 ticks is expired at tick 21.
    h3 = update_hypothesis(
        hyp,
        healthy_descendants=[],
        current_tick=21,
        axis_thresholds=thr,
        cfg=cfg,
    )
    out.append(
        {
            "name": "TTL expire → retired",
            "expected_status": "retired",
            "got_status": h3.status,
            "ok": h3.status == "retired",
        }
    )

    # Test 4: a supporting panel-ready result receives the calibrated bonus.
    h4 = update_hypothesis(
        hyp,
        healthy_descendants=[(_r("d1", pLDDT=92.0, iPAE=0.50, panel_ready=True), baseline)],
        current_tick=11,
        axis_thresholds=thr,
        cfg=cfg,
    )
    out.append(
        {
            "name": "panel-ready bonus → supported",
            "expected_status": "supported",
            "got_status": h4.status,
            "ok": h4.status == "supported",
        }
    )

    return out


def s7_panel_selection() -> dict[str, Any]:
    def _c(rid, foldseek="FS_a", contact="C_a", epitope="E_a", sequence="S_a", q=0.7) -> ResultRecord:
        return ResultRecord(
            result_id=rid,
            parent_ids=[],
            target_id="t",
            backend_family="complexa_beam",
            runtime_bucket_id="rb1",
            metrics={"pLDDT": 85, "iPAE": 0.3},
            metrics_calibrated={
                "calibrated_structure_confidence": q,
                "calibrated_interface_confidence": q,
                "hotspot_contact_satisfaction": q,
                "clash_developability_score": q,
            },
            route_lineage=[],
            gpu_h=1.0,
            exit_status="ok",
            bins={"foldseek": foldseek, "contact": contact, "epitope": epitope, "sequence": sequence},
            panel_ready=True,
        )

    a = _c("a", foldseek="FS_a")
    b = _c("b", foldseek="FS_b")
    c_dup = _c("c", foldseek="FS_a", q=0.9)  # redundant FS but higher q

    p1 = select_panel([a, b, c_dup], cfg=PanelConfig(K=3))
    p2 = select_panel([c_dup, b, a], cfg=PanelConfig(K=3))  # input reorder

    same = p1.selected_ids == p2.selected_ids
    pv_positive = p1.panel_value > 0
    div = diversity_at_K(p1, {"foldseek": 4, "contact": 4, "epitope": 4, "sequence": 4})
    div_in_range = 0.0 < div <= 1.0

    return {
        "deterministic_replay": same,
        "panel_value_positive": pv_positive,
        "diversity_in_range": div_in_range,
        "panel_value": p1.panel_value,
        "diversity_at_K": div,
        "selected_ids_p1": p1.selected_ids,
        "selected_ids_p2": p2.selected_ids,
    }


def run_pytest_full(repo_root: Path) -> dict[str, Any]:
    py = os.environ.get("TREX_SMOKE_PYTHON", sys.executable)
    cmd = [py, "-m", "pytest", "trex/tests/", "-q", "--tb=line"]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env={**__import__("os").environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-15:],
        "stderr_tail": proc.stderr.splitlines()[-15:],
        "elapsed_s": round(time.time() - started, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = p.parse_args()

    started = time.time()
    s2 = s2_state_classifier_cases()
    s6 = s6_lifecycle_replay()
    s7 = s7_panel_selection()
    pytest_report = run_pytest_full(args.repo_root)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "s2_state_classifier_pass": sum(1 for r in s2 if r["ok"]),
        "s2_state_classifier_total": len(s2),
        "s6_lifecycle_pass": sum(1 for r in s6 if r["ok"]),
        "s6_lifecycle_total": len(s6),
        "s7_panel_deterministic": s7["deterministic_replay"],
        "s7_panel_value_positive": s7["panel_value_positive"],
        "s7_diversity_in_range": s7["diversity_in_range"],
        "pytest_returncode": pytest_report["returncode"],
        "wall_clock_s": round(time.time() - started, 2),
    }

    report = {
        "summary": summary,
        "s2": s2,
        "s6": s6,
        "s7": s7,
        "pytest": pytest_report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    overall_ok = (
        summary["s2_state_classifier_pass"] == summary["s2_state_classifier_total"]
        and summary["s6_lifecycle_pass"] == summary["s6_lifecycle_total"]
        and summary["s7_panel_deterministic"]
        and summary["s7_panel_value_positive"]
        and summary["s7_diversity_in_range"]
        and summary["pytest_returncode"] == 0
    )
    if not overall_ok:
        print("[trex_deterministic_smoke] FAILED", file=sys.stderr)
        raise SystemExit(2)
    print("[trex_deterministic_smoke] OK")


if __name__ == "__main__":
    main()
