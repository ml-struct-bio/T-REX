"""Finalize a production wet-lab panel from a T-ReX archive.

This is the end-of-run counterpart to the per-tick panel snapshot in
``live_tick``. It reruns strict-only Foldseek and MMseqs2 in memory before
selection because derived bins are not rewritten onto historical
``ResultRecord`` rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .archive import Archive
from .foldseek_clusterer import apply_clusters_to_bins, cluster_archive_pdbs
from .panel import (
    ProductionPanelConfig,
    best_near_miss_backups,
    select_production_panel,
)
from .schemas import PanelSelection, ResultRecord, to_jsonable
from .sequence_clusterer import (
    apply_sequence_clusters_to_bins,
    cluster_archive_sequences,
)
from .success_criteria import is_strict_success


FINAL_PANEL_OUTPUT_SCHEMA_VERSION = "trex.final-panel.v1"


def finalize_panel(
    archive: Archive,
    *,
    target_id: str,
    panel_size: int = 8,
    panel_id: str = "production_panel_final",
    foldseek_binary: str = "foldseek",
    mmseqs_binary: str = "mmseqs",
    run_dedup: bool = True,
    require_structure_artifact: bool = True,
    su_tm_score: float = 0.60,
    collapse_tm_score: float = 0.80,
    sequence_identity: float = 0.90,
    sequence_coverage: float = 0.80,
) -> tuple[PanelSelection, dict]:
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    for name, value in (
        ("su_tm_score", su_tm_score),
        ("collapse_tm_score", collapse_tm_score),
        ("sequence_identity", sequence_identity),
        ("sequence_coverage", sequence_coverage),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    results = [
        r for r in archive.iter_records(ResultRecord) if r.target_id == target_id
    ]
    strict_rids = {
        r.result_id
        for r in results
        if r.exit_status == "ok" and is_strict_success(r.metrics)
    }
    diagnostics: dict = {
        "target_id": target_id,
        "n_results": len(results),
        "n_strict": len(strict_rids),
        "foldseek_su_status": "disabled",
        "foldseek_su_coverage": None,
        "sequence_dedup_status": "disabled",
        "sequence_dedup_coverage": None,
    }

    if run_dedup and results:
        whole = cluster_archive_pdbs(
            results,
            target_id=target_id,
            foldseek_binary=foldseek_binary,
            # Preserve the historical post-hoc helper's 0.80 default. This is
            # separate from the live controller and the strict-only SU pass.
            # A different diagnostic threshold requires an explicit override.
            min_tm_score=collapse_tm_score,
        )
        apply_clusters_to_bins(results, whole.cluster_by_result_id)
        diagnostics.update(
            {
                "foldseek_archive_status": whole.status,
                "foldseek_archive_coverage": (
                    len(whole.cluster_by_result_id) / whole.n_structures
                    if whole.n_structures
                    else None
                ),
                "whole_archive_structure_dedup_scope": whole.structure_scope,
                "whole_archive_structure_dedup_fallback_count": whole.n_scope_fallback,
            }
        )
        if strict_rids:
            strict_struct = cluster_archive_pdbs(
                results,
                target_id=target_id,
                foldseek_binary=foldseek_binary,
                only_result_ids=strict_rids,
                min_tm_score=su_tm_score,  # match the live SU objective
            )
            n_su = apply_clusters_to_bins(
                results, strict_struct.cluster_by_result_id, bin_key="foldseek_su"
            )
            diagnostics.update(
                {
                    "foldseek_su_status": strict_struct.status,
                    "foldseek_su_coverage": n_su / len(strict_rids),
                    "structure_dedup_scope": strict_struct.structure_scope,
                    "structure_dedup_fallback_count": strict_struct.n_scope_fallback,
                }
            )
            seq = cluster_archive_sequences(
                results,
                target_id=target_id,
                mmseqs_binary=mmseqs_binary,
                only_result_ids=strict_rids,
                min_seq_id=sequence_identity,
                coverage=sequence_coverage,
            )
            n_seq = apply_sequence_clusters_to_bins(
                results, seq.cluster_by_result_id, bin_key="sequence_su"
            )
            diagnostics.update(
                {
                    "sequence_dedup_status": seq.status,
                    "sequence_dedup_coverage": n_seq / len(strict_rids),
                    "sequence_dedup_fallback_count": seq.n_sequence_fallback,
                }
            )
        else:
            diagnostics["foldseek_su_status"] = "no_strict"
            diagnostics["sequence_dedup_status"] = "no_strict"
            diagnostics["sequence_dedup_coverage"] = 1.0

    diagnostics["settings"] = {
        "panel_size": panel_size,
        "run_dedup": run_dedup,
        "require_structure_artifact": require_structure_artifact,
        "su_tm_score": su_tm_score,
        "collapse_tm_score": collapse_tm_score,
        "sequence_identity": sequence_identity,
        "sequence_coverage": sequence_coverage,
    }

    dedup_trusted = (
        not run_dedup
        or not strict_rids
        or (
            diagnostics.get("foldseek_su_status") == "ok"
            and diagnostics.get("foldseek_su_coverage") is not None
            and float(diagnostics["foldseek_su_coverage"]) >= 0.999
        )
    )
    panel = select_production_panel(
        results if dedup_trusted else [],
        cfg=ProductionPanelConfig(
            K=panel_size,
            require_structure_artifact=require_structure_artifact,
            require_trusted_structure_bin=True,
        ),
        panel_id=panel_id,
    )
    diagnostics["production_panel_status"] = (
        "ok"
        if panel.selected_ids
        else (
            "degraded_untrusted_structure_dedup"
            if strict_rids and not dedup_trusted
            else "no_eligible_strict"
        )
    )
    if strict_rids and not dedup_trusted:
        dedup_reason = (
            "untrusted_structure_dedup:"
            f"status={diagnostics.get('foldseek_su_status')},"
            f"coverage={diagnostics.get('foldseek_su_coverage')}"
        )
        panel = replace(
            panel,
            hard_gate_failures=[dedup_reason, *panel.hard_gate_failures],
        )
    diagnostics["near_miss_backup_ids"] = best_near_miss_backups(
        results,
        K=min(4, panel_size),
        require_structure_artifact=require_structure_artifact,
    )
    return panel, diagnostics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Finalize T-ReX production wet-lab panel")
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--target-id", required=True)
    p.add_argument("--panel-size", "--K", dest="panel_size", type=int, default=8)
    p.add_argument("--panel-id", default="production_panel_final")
    p.add_argument("--foldseek-binary", default="foldseek")
    p.add_argument("--mmseqs-binary", default="mmseqs")
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--allow-missing-structure", action="store_true")
    p.add_argument(
        "--su-tm-score",
        type=float,
        default=0.60,
        help="Strict-SU Foldseek TM (T-ReX live objective 0.60; "
        "pass the run's FOLDSEEK_SU_TM_SCORE to match the live SU).",
    )
    p.add_argument(
        "--collapse-tm-score",
        type=float,
        default=0.80,
        help="Whole-archive collapse TM threshold (historical post-hoc default "
        "0.80; distinct from the live controller's default).",
    )
    p.add_argument("--sequence-identity", type=float, default=0.90)
    p.add_argument("--sequence-coverage", type=float, default=0.80)
    p.add_argument("--append", action="store_true")
    args = p.parse_args(argv)

    try:
        archive_root = args.archive_root.expanduser().resolve()
        if not archive_root.is_dir():
            raise ValueError(f"archive root is not a directory: {archive_root}")
        archive = Archive(archive_root)
        if args.append and any(
            previous.panel_id == args.panel_id
            for previous in archive.iter_records(PanelSelection)
        ):
            raise ValueError(f"panel_id already exists in archive: {args.panel_id!r}")
        panel, diagnostics = finalize_panel(
            archive,
            target_id=args.target_id,
            panel_size=args.panel_size,
            panel_id=args.panel_id,
            foldseek_binary=args.foldseek_binary,
            mmseqs_binary=args.mmseqs_binary,
            run_dedup=not args.no_dedup,
            require_structure_artifact=not args.allow_missing_structure,
            su_tm_score=args.su_tm_score,
            collapse_tm_score=args.collapse_tm_score,
            sequence_identity=args.sequence_identity,
            sequence_coverage=args.sequence_coverage,
        )
        if args.append:
            archive.append(panel)
        print(
            json.dumps(
                {
                    "schema_version": FINAL_PANEL_OUTPUT_SCHEMA_VERSION,
                    "archive_root": str(archive_root),
                    "appended": args.append,
                    "panel": to_jsonable(panel),
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"trex-panel: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
