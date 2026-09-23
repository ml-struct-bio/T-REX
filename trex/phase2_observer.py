"""Observe independently running BindCraft jobs and append newly parsed results.

Periodically summarize the accumulated evidence without submitting new workers. This
utility supports the three target-specific output arguments exposed by its CLI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .archive import Archive
from .evidence_reducer import reduce_evidence
from .output_parsers.bindcraft import parse_bindcraft_output
from .output_parsers.types import ParseError, ParserContext
from .schemas import ResultRecord, TargetConstraint, to_jsonable
from .success_criteria import is_near_miss, is_strict_success


# Default TargetConstraints for the observer CLI targets.
TARGETS = {
    "cd45": TargetConstraint(
        target_id="05_CD45", target_class="receptor_tyrosine_phosphatase",
        hotspots=["A1"], chain_ids=["A"], panel_size_K=8,
    ),
    "betv1": TargetConstraint(
        target_id="23_BetV1", target_class="allergen_protein",
        hotspots=["A24"], chain_ids=["A"], panel_size_K=8,
    ),
    "sc2rbd": TargetConstraint(
        target_id="30_SC2RBD", target_class="viral_rbd",
        hotspots=["E485", "E489", "E494", "E500", "E505"],
        chain_ids=["E"], panel_size_K=8,
    ),
}


def ingest_target(
    target_key: str,
    output_dir: Path,
    archive: Archive,
    start_ts: float,
) -> tuple[int, int]:
    """Parse current BindCraft output + append NEW ResultRecords.

    Returns (n_new_records, n_total_in_archive_for_target).
    """
    target = TARGETS[target_key]
    if not output_dir.exists():
        return 0, 0  # job hasn't started writing yet
    designs_dir = output_dir / "designs"
    if not designs_dir.exists():
        return 0, 0

    ctx = ParserContext(
        target_id=target.target_id,
        runtime_bucket_id="bc_phase2",
        candidate_id=f"phase2_{target_key}",
        parent_ids=[f"phase2_{target_key}"],
    )

    # Snapshot what's already in archive for this target
    existing_ids = {r.result_id for r in archive.iter_records(ResultRecord)
                    if r.target_id == target.target_id}

    try:
        all_parsed = parse_bindcraft_output(output_dir, ctx)
    except ParseError:
        # Job hasn't produced parseable output yet (no CSV / no Accepted)
        return 0, len(existing_ids)
    except Exception as e:  # noqa: BLE001
        # Log and continue — observer must not crash mid-campaign
        print(f"  [WARN] parser exception for {target_key}: {type(e).__name__}: {e}")
        return 0, len(existing_ids)

    n_new = 0
    for r in all_parsed:
        if r.result_id not in existing_ids:
            archive.append(r)
            n_new += 1
            existing_ids.add(r.result_id)

    return n_new, len(existing_ids)


def run_tick_for_target(
    target_key: str,
    archive: Archive,
    tick_id: str,
    tick_id_int: int,
    elapsed_h: float,
    remaining_h: float,
) -> dict[str, Any]:
    """Run reduce_evidence + capture a summary snapshot. NO LaunchDecision
    emission here — observer is read-only on the T-REX archive contents.
    """
    target = TARGETS[target_key]
    target_results = [r for r in archive.iter_records(ResultRecord)
                       if r.target_id == target.target_id]
    if not target_results:
        return {
            "target_key": target_key,
            "target_id": target.target_id,
            "n_results": 0,
            "tick_id": tick_id,
        }

    # Use last 20 as window (≈ 3-ticks-equivalent in this observe loop)
    window = target_results[-20:]
    gpu_h_total = sum(r.gpu_h for r in target_results)

    near_miss_n = sum(1 for r in window if is_near_miss(r.metrics))
    strict_in_window = sum(1 for r in window if is_strict_success(r.metrics))
    strict_in_total = sum(1 for r in target_results if is_strict_success(r.metrics))

    # This observer uses raw qualified counts as unclustered proxies, not verified
    # structural SU counts.
    run_su_count = strict_in_total
    run_su_count_delta = strict_in_window

    # top_bin_share heuristic from BindCraft Design name prefix
    # (l<length>_s<seed>): designs sharing length+seed are from the same
    # trajectory ≈ same Foldseek bin proxy.
    from collections import Counter
    bin_counts = Counter()
    for r in window:
        design = r.bins.get("design") if r.bins else None
        if design:
            # extract "l<len>_s<seed>" prefix
            parts = design.split("_")
            bin_key = "_".join(parts[2:4]) if len(parts) >= 4 else "unknown"
            bin_counts[bin_key] += 1
    top_bin_share = (max(bin_counts.values()) / len(window)) if bin_counts and window else 0.0

    evidence = reduce_evidence(
        tick_id=tick_id,
        target_id=target.target_id,
        target_class=target.target_class,
        elapsed_wall_h=elapsed_h,
        remaining_wall_h=remaining_h,
        pending_children=0,
        worker_gpu_h_total=gpu_h_total,
        all_results=target_results,
        window_results=window,
        run_su_count=run_su_count,
        run_su_count_delta=run_su_count_delta,
        duplicate_fraction=0.0,  # without seq-clustering, leave 0
        near_miss_count=near_miss_n,
        top_bin_share=top_bin_share,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="(observer-no-llm)",
    )

    strict_count = evidence.strict_count
    return {
        "target_key": target_key,
        "target_id": target.target_id,
        "tick_id": tick_id,
        "tick_id_int": tick_id_int,
        "elapsed_h": round(elapsed_h, 2),
        "n_results": len(target_results),
        "n_window": len(window),
        "strict_count": strict_count,
        "state_label": evidence.state_label,
        "axis_stats": {
            k: {"p": v.pass_count, "n": v.near_pass_count, "f": v.fail_count, "median": v.median_raw}
            for k, v in evidence.axis_stats.items()
        },
        "diagnostic_axes_present": list(evidence.diagnostic_axis_stats.keys()),
        "diagnostic_axis_pass_counts": {
            k: v.pass_count for k, v in evidence.diagnostic_axis_stats.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Observe BindCraft campaign results")
    p.add_argument("--cd45-output-dir", type=Path, required=True)
    p.add_argument("--betv1-output-dir", type=Path, required=True)
    p.add_argument("--sc2rbd-output-dir", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--poll-interval-min", type=float, default=30.0)
    p.add_argument("--max-wall-h", type=float, default=25.0,
                    help="Observer should outlast the 24h BindCraft jobs slightly")
    args = p.parse_args()

    args.archive_root.mkdir(parents=True, exist_ok=True)
    archive = Archive(args.archive_root)
    log_path = args.archive_root / "phase2_observer.jsonl"
    targets = {
        "cd45": args.cd45_output_dir,
        "betv1": args.betv1_output_dir,
        "sc2rbd": args.sc2rbd_output_dir,
    }

    start_ts = time.time()
    tick_id_int = 0
    print(f"[phase2_observer] start at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[phase2_observer] poll every {args.poll_interval_min} min, max {args.max_wall_h} h")
    print(f"[phase2_observer] archive_root: {args.archive_root}")

    while True:
        elapsed_h = (time.time() - start_ts) / 3600.0
        if elapsed_h >= args.max_wall_h:
            print(f"[phase2_observer] max wall-h reached ({elapsed_h:.2f} h); exit.")
            break

        tick_id_int += 1
        tick_id = f"p2obs_t{tick_id_int:03d}"
        round_summary = {"tick_id": tick_id, "tick_id_int": tick_id_int,
                          "elapsed_h": round(elapsed_h, 2),
                          "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "per_target": {}}

        for target_key, output_dir in targets.items():
            n_new, n_total = ingest_target(target_key, output_dir, archive, start_ts)
            target_summary = run_tick_for_target(
                target_key, archive,
                tick_id=tick_id, tick_id_int=tick_id_int,
                elapsed_h=elapsed_h,
                remaining_h=max(0.0, args.max_wall_h - elapsed_h),
            )
            target_summary["n_new_this_round"] = n_new
            target_summary["n_total"] = n_total
            round_summary["per_target"][target_key] = target_summary
            state = target_summary.get("state_label") or "no_data"
            strict = target_summary.get("strict_count", 0)
            diag = target_summary.get("diagnostic_axes_present", [])
            print(f"  [{tick_id} {target_key:7s}] n_total={n_total:4d} new={n_new:3d} "
                  f"state={state:<13s} "
                  f"strict={strict:3d} "
                  f"diag={len(diag)}")

        with open(log_path, "a") as f:
            f.write(json.dumps(round_summary) + "\n")

        time.sleep(args.poll_interval_min * 60.0)

    print(f"[phase2_observer] done. log: {log_path}")


if __name__ == "__main__":
    main()
