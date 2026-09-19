#!/usr/bin/env python3
"""Verify the frozen seven-target benchmark and recompute headline results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
FIXED = ("Complexa-only", "BindCraft-only", "BoltzGen-only")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_unique(rows: Iterable[dict[str, str]], fields: tuple[str, ...], label: str) -> None:
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in seen:
            raise SystemExit(f"{label}: duplicate key {key}")
        seen.add(key)


def require_scope(
    rows: Iterable[dict[str, str]], targets: set[str], methods: set[str], label: str
) -> None:
    observed_targets = {row["target"] for row in rows}
    observed_methods = {row["method"] for row in rows}
    if observed_targets != targets:
        raise SystemExit(
            f"{label}: targets {sorted(observed_targets)} != {sorted(targets)}"
        )
    if observed_methods != methods:
        raise SystemExit(
            f"{label}: methods {sorted(observed_methods)} != {sorted(methods)}"
        )


def as_int(value: str, label: str) -> int:
    numeric = float(value)
    if not numeric.is_integer():
        raise SystemExit(f"{label}: expected integer-valued count, got {value}")
    return int(numeric)


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected_hash in manifest["files"].items():
        observed_hash = sha256(ROOT / name)
        if observed_hash != expected_hash:
            raise SystemExit(
                f"SHA256 mismatch for {name}: {observed_hash} != {expected_hash}"
            )

    target_order = list(manifest["scope"]["targets"])
    method_order = list(manifest["scope"]["methods"])
    targets = set(target_order)
    methods = set(method_order)
    expected_cells = len(targets) * len(methods)

    endpoints = read_rows("endpoints_tm06.csv")
    require_unique(endpoints, ("target", "method"), "endpoints")
    require_scope(endpoints, targets, methods, "endpoints")
    if len(endpoints) != expected_cells:
        raise SystemExit(f"endpoints: expected {expected_cells} rows, got {len(endpoints)}")
    endpoint_su = {
        (row["target"], row["method"]): as_int(row["su_n"], "endpoint su_n")
        for row in endpoints
    }

    expected = manifest["headline_recomputations"]
    puct_ratios: list[float] = []
    fixed_ratios: list[float] = []
    print("target,T-ReX,PUCT,best_fixed,T-ReX/PUCT,T-ReX/best_fixed")
    for target in target_order:
        trex = endpoint_su[(target, "T-ReX")]
        puct = endpoint_su[(target, "PUCT")]
        best_fixed = max(endpoint_su[(target, method)] for method in FIXED)
        if puct <= 0 or best_fixed <= 0:
            raise SystemExit(f"{target}: nonpositive comparator prevents a ratio")
        puct_ratio = trex / puct
        fixed_ratio = trex / best_fixed
        puct_ratios.append(puct_ratio)
        fixed_ratios.append(fixed_ratio)
        print(
            f"{target},{trex},{puct},{best_fixed},"
            f"{puct_ratio:.6f},{fixed_ratio:.6f}"
        )

    trex_total = sum(endpoint_su[(target, "T-ReX")] for target in targets)
    if trex_total != int(expected["trex_tm06_total_su"]):
        raise SystemExit(f"T-ReX total {trex_total} != manifest")
    gm_puct = math.exp(sum(math.log(value) for value in puct_ratios) / len(puct_ratios))
    gm_fixed = math.exp(sum(math.log(value) for value in fixed_ratios) / len(fixed_ratios))
    if not math.isclose(
        gm_puct, float(expected["geometric_mean_trex_over_puct_tm06"]), rel_tol=1e-12
    ):
        raise SystemExit("T-ReX/PUCT geometric mean does not match manifest")
    if not math.isclose(
        gm_fixed,
        float(expected["geometric_mean_trex_over_targetwise_best_fixed_tm06"]),
        rel_tol=1e-12,
    ):
        raise SystemExit("T-ReX/best-fixed geometric mean does not match manifest")

    sensitivity = read_rows("tm_sensitivity.csv")
    require_unique(sensitivity, ("target", "method", "tm"), "tm sensitivity")
    require_scope(sensitivity, targets, methods, "tm sensitivity")
    thresholds = {float(value) for value in manifest["scope"]["sensitivity_thresholds"]}
    observed_thresholds = {float(row["tm"]) for row in sensitivity}
    if observed_thresholds != thresholds:
        raise SystemExit(
            f"tm sensitivity: thresholds {sorted(observed_thresholds)} != {sorted(thresholds)}"
        )
    if len(sensitivity) != expected_cells * len(thresholds):
        raise SystemExit("tm sensitivity: incomplete target-method-threshold grid")
    sensitivity_su = {
        (row["target"], row["method"], float(row["tm"])): as_int(
            row["su_n"], "tm sensitivity su_n"
        )
        for row in sensitivity
    }
    for key, value in endpoint_su.items():
        if sensitivity_su[(key[0], key[1], 0.6)] != value:
            raise SystemExit(f"TM0.6 endpoint mismatch for {key}")
    threshold_leads = 0
    for target in targets:
        for threshold in thresholds:
            trex = sensitivity_su[(target, "T-ReX", threshold)]
            comparator = max(
                sensitivity_su[(target, method, threshold)]
                for method in methods
                if method != "T-ReX"
            )
            threshold_leads += int(trex > comparator)
    if threshold_leads != int(expected["trex_leading_target_tm_threshold_comparisons"]):
        raise SystemExit(f"T-ReX threshold leads {threshold_leads} != manifest")

    sequence = read_rows("mmseqs70_clusters.csv")
    require_unique(sequence, ("target", "method"), "MMseqs2")
    require_scope(sequence, targets, methods, "MMseqs2")
    if len(sequence) != expected_cells:
        raise SystemExit("MMseqs2: incomplete target-method grid")
    sequence_by_key = {(row["target"], row["method"]): row for row in sequence}
    for key, row in sequence_by_key.items():
        if as_int(row["su_n"], "MMseqs2 su_n") != endpoint_su[key]:
            raise SystemExit(f"MMseqs2 endpoint mismatch for {key}")
    sequence_leads = 0
    pool_leads = 0
    for target in targets:
        trex_row = sequence_by_key[(target, "T-ReX")]
        trex_clusters = as_int(trex_row["mmseqs_sequence_clusters"], "MMseqs2 clusters")
        trex_pool = as_int(trex_row["strict_n"], "qualified pool")
        other_rows = [
            sequence_by_key[(target, method)] for method in methods if method != "T-ReX"
        ]
        sequence_leads += int(
            trex_clusters
            > max(as_int(row["mmseqs_sequence_clusters"], "MMseqs2 clusters") for row in other_rows)
        )
        pool_leads += int(
            trex_pool > max(as_int(row["strict_n"], "qualified pool") for row in other_rows)
        )
    if sequence_leads != int(expected["trex_leading_sequence_cluster_throughput_targets"]):
        raise SystemExit(f"T-ReX sequence-cluster leads {sequence_leads} != manifest")
    if pool_leads != int(expected["trex_largest_qualified_pool_targets"]):
        raise SystemExit(f"T-ReX qualified-pool leads {pool_leads} != manifest")

    curves = read_rows("curves_tm06.csv")
    require_scope(curves, targets, methods, "curves")
    curve_keys = {(row["target"], row["method"]) for row in curves}
    if len(curve_keys) != expected_cells:
        raise SystemExit("curves: incomplete target-method coverage")
    nonblank_curve_tm = {float(row["tm"]) for row in curves if row["tm"]}
    if nonblank_curve_tm != {0.6}:
        raise SystemExit("curves: expected only TM0.6 records")
    if any(
        not row["tm"] and row.get("curve_role") != "endpoint_reconciled_terminal"
        for row in curves
    ):
        raise SystemExit("curves: blank TM allowed only on reconciled terminal rows")

    print()
    print(f"T-ReX_TM0.6_total_SU={trex_total}")
    print(f"geometric_mean_T-ReX_over_PUCT={gm_puct:.6f}")
    print(f"geometric_mean_T-ReX_over_best_fixed={gm_fixed:.6f}")
    print(f"T-ReX_target_by_TM_threshold_leads={threshold_leads}/21")
    print(f"T-ReX_sequence_cluster_throughput_leads={sequence_leads}/7")
    print(f"T-ReX_largest_qualified_pool_targets={pool_leads}/7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
