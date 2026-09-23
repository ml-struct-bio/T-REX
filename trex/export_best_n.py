"""Export the best-N strict-success binders by score for a wet-lab order.

This is a SEPARATE artifact from ``finalize_panel`` / ``select_production_panel``:
that selector returns a structurally-deduped DIVERSITY panel (at most one
candidate per Foldseek SU bin), so ``--panel-size 100`` there yields only as many designs
as there are distinct structures. The wet-lab "best-N by score" list instead
ranks every strict success by ``production_quality`` and takes the top N,
allowing structural near-duplicates (you may want several variants of a strong
basin). It then COPIES each binder PDB into a single, stable directory so the
panel survives even if the per-run scratch output dirs are later cleaned, and
writes a manifest (CSV + JSON) joining each design to its score, strict metrics,
and structural/sequence dedup bins.

Usage:
    python -m trex.export_best_n \
        --archive-root <ARCHIVE> --target-id 02_PDL1 --n 100 \
        --out-dir <ARCHIVE>/wetlab_panel_best100
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import json
import shutil
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .archive import Archive
from .output_identity import prepare_archive_results
from .foldseek_clusterer import apply_clusters_to_bins, cluster_archive_pdbs
from .panel import _has_structure_artifact, production_bins, production_quality
from .schemas import ResultRecord
from .sequence_clusterer import (
    apply_sequence_clusters_to_bins,
    cluster_archive_sequences,
)
from .success_criteria import STRICT_SUCCESS, is_strict_success

# Strict axes carried into the manifest for operator review (order = report order).
_STRICT_AXES = tuple(STRICT_SUCCESS.keys())
_DIAGNOSTIC_AXES = ("ipTM", "min_ipae", "avg_ipsae", "interface_contact_density")
BEST_N_MANIFEST_SCHEMA_VERSION = "trex.best-n-manifest.v1"
BEST_N_OUTPUT_SCHEMA_VERSION = "trex.best-n-export.v1"
_OUTPUT_ENTRY_NAMES = frozenset({"manifest.json", "manifest.csv", "pdbs"})
_MANIFEST_FIELD_ORDER = (
    "rank",
    "result_id",
    "target_id",
    "backend_family",
    "production_quality",
    *_STRICT_AXES,
    *_DIAGNOSTIC_AXES,
    "structure_bin",
    "sequence_bin",
    "binder_chain",
    "target_chains",
    "output_chain_identity",
    "src_pdb",
    "dst_pdb",
)


def _structure_source(rec: ResultRecord) -> str:
    """PDB/CIF path for this record, or '' if none on disk."""
    art = rec.artifacts or {}
    p = art.get("pdb_path")
    if p and Path(p).is_file():
        return p
    d = art.get("pdb_dir")
    if d and Path(d).is_dir():
        for pat in ("*.pdb", "*.cif", "*.mmcif"):
            hit = sorted(Path(d).glob(pat))
            if hit:
                return str(hit[0])
    return ""


def _validate_output_directory(out_dir: Path, *, overwrite: bool) -> Path:
    """Validate a destination without changing an existing export."""

    resolved = Path(out_dir).expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output path is not a directory: {resolved}")
    existing = list(resolved.iterdir()) if resolved.is_dir() else []
    if existing and not overwrite:
        raise ValueError(f"output directory is not empty: {resolved} (use --overwrite)")
    unknown = sorted(
        path.name for path in existing if path.name not in _OUTPUT_ENTRY_NAMES
    )
    if unknown:
        raise ValueError(
            f"refusing to overwrite directory with unrecognized entries: {unknown}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def _atomic_export_directory(destination: Path) -> Iterator[Path]:
    """Yield a staging root and publish it only after every write succeeds."""
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        yield staging
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.previous-{uuid.uuid4().hex}"
            )
            destination.rename(backup)
        try:
            staging.rename(destination)
        except OSError:
            if backup is not None and backup.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_export_artifacts(
    *,
    staging_dir: Path,
    archive_root: Path,
    target_id: str,
    diagnostics: dict,
    rows: list[dict],
    copy_pdbs: bool,
) -> None:
    """Write a complete export into an unpublished staging directory."""
    if copy_pdbs:
        staging_pdb_dir = staging_dir / "pdbs"
        staging_pdb_dir.mkdir()
        for row in rows:
            source = row["src_pdb"]
            if source:
                destination = staging_pdb_dir / Path(row["dst_pdb"]).name
                shutil.copy2(source, destination)
    with (staging_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELD_ORDER)
        writer.writeheader()
        writer.writerows(rows)
    (staging_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BEST_N_MANIFEST_SCHEMA_VERSION,
                "archive_root": str(archive_root),
                "target_id": target_id,
                "diagnostics": diagnostics,
                "designs": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def export_best_n(
    archive: Archive,
    *,
    target_id: str,
    n: int = 100,
    out_dir: Path,
    copy_pdbs: bool = True,
    annotate_dedup: bool = True,
    foldseek_binary: str = "foldseek",
    mmseqs_binary: str = "mmseqs",
    require_structure_artifact: bool = True,
    su_tm_score: float = 0.60,
    sequence_identity: float = 0.90,
    sequence_coverage: float = 0.80,
    overwrite: bool = False,
) -> dict:
    if n < 1:
        raise ValueError("n must be positive")
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    for name, value in (
        ("su_tm_score", su_tm_score),
        ("sequence_identity", sequence_identity),
        ("sequence_coverage", sequence_coverage),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    out_dir = _validate_output_directory(out_dir, overwrite=overwrite)

    results = prepare_archive_results(
        (r for r in archive.iter_records(ResultRecord) if r.target_id == target_id), archive.root,
    )
    strict = [
        r
        for r in results
        if r.exit_status == "ok"
        and is_strict_success(r.metrics)
        and r.bins.get("output_chain_identity") != "unresolved"
        and (not require_structure_artifact or _has_structure_artifact(r))
    ]

    diagnostics: dict = {
        "target_id": target_id,
        "n_results": len(results),
        "n_strict_with_structure": len(strict),
        "n_unresolved_output_identity": sum(r.bins.get("output_chain_identity") == "unresolved" for r in results),
        "foldseek_su_status": "disabled",
        "sequence_dedup_status": "disabled",
    }

    # Annotate strict records with strict-only structural (foldseek_su) and
    # sequence (sequence_su) bins so the manifest shows how much structural /
    # sequence redundancy is present among the top-N (the SAME bins the
    # objective's SU and the diversity panel use). Pure annotation — does not
    # change ranking or which designs are exported.
    if annotate_dedup and strict:
        strict_rids = {r.result_id for r in strict}
        struct = cluster_archive_pdbs(
            results,
            target_id=target_id,
            foldseek_binary=foldseek_binary,
            only_result_ids=strict_rids,
            min_tm_score=su_tm_score,
        )
        apply_clusters_to_bins(
            results, struct.cluster_by_result_id, bin_key="foldseek_su"
        )
        diagnostics["foldseek_su_status"] = struct.status
        seq = cluster_archive_sequences(
            results,
            target_id=target_id,
            mmseqs_binary=mmseqs_binary,
            only_result_ids=strict_rids,
            min_seq_id=sequence_identity,
            coverage=sequence_coverage,
        )
        apply_sequence_clusters_to_bins(
            results, seq.cluster_by_result_id, bin_key="sequence_su"
        )
        diagnostics["sequence_dedup_status"] = seq.status

    ranked = sorted(strict, key=lambda r: (-production_quality(r), r.result_id))[:n]

    pdb_dir = out_dir / "pdbs"

    rows: list[dict] = []
    n_copied = 0
    n_missing = 0
    family_counts: dict[str, int] = {}
    struct_bins: set[str] = set()
    for rank, rec in enumerate(ranked, start=1):
        bins = production_bins(rec)
        src = _structure_source(rec)
        ext = Path(src).suffix or ".pdb"
        dst = pdb_dir / f"rank{rank:03d}_{rec.backend_family}_{rec.result_id}{ext}"
        if copy_pdbs:
            if src:
                n_copied += 1
            else:
                n_missing += 1
        family_counts[rec.backend_family] = family_counts.get(rec.backend_family, 0) + 1
        struct_bins.add(bins["structure"])
        row = {
            "rank": rank,
            "result_id": rec.result_id,
            "target_id": rec.target_id,
            "backend_family": rec.backend_family,
            "production_quality": round(production_quality(rec), 6),
            "structure_bin": bins["structure"],
            "sequence_bin": bins.get("sequence", ""),
            "binder_chain": rec.artifacts.get("binder_chain", ""),
            "target_chains": rec.artifacts.get("target_chains", ""),
            "output_chain_identity": rec.bins.get("output_chain_identity", "legacy_unverified"),
            "src_pdb": src,
            "dst_pdb": str(dst) if (copy_pdbs and src) else "",
        }
        for axis in _STRICT_AXES:
            row[axis] = rec.metrics.get(axis)
        for axis in _DIAGNOSTIC_AXES:
            if axis in rec.metrics:
                row[axis] = rec.metrics.get(axis)
        rows.append(row)

    diagnostics.update(
        {
            "n_requested": n,
            "n_exported": len(rows),
            "n_pdbs_copied": n_copied,
            "n_pdbs_missing_source": n_missing,
            "distinct_structure_bins": len(struct_bins),
            "family_counts": family_counts,
            "settings": {
                "ranking": "production_quality_descending_then_result_id",
                "su_tm_score": su_tm_score,
                "sequence_identity": sequence_identity,
                "sequence_coverage": sequence_coverage,
                "annotate_dedup": annotate_dedup,
                "require_structure_artifact": require_structure_artifact,
                "copy_structures": copy_pdbs,
                "overwrite": overwrite,
            },
        }
    )

    # Publish only after every structure and manifest has been written.
    with _atomic_export_directory(out_dir) as staging_dir:
        _write_export_artifacts(
            staging_dir=staging_dir,
            archive_root=archive.root.resolve(),
            target_id=target_id,
            diagnostics=diagnostics,
            rows=rows,
            copy_pdbs=copy_pdbs,
        )
    return diagnostics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export best-N strict binders by score for a wet-lab order"
    )
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--target-id", required=True)
    p.add_argument("--n", "--N", dest="n", type=int, default=100)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <archive-root>/wetlab_panel_best<n>",
    )
    p.add_argument(
        "--no-copy", action="store_true", help="manifest only; do not copy PDB files"
    )
    p.add_argument(
        "--no-dedup", action="store_true", help="skip foldseek/mmseqs bin annotation"
    )
    p.add_argument("--foldseek-binary", default="foldseek")
    p.add_argument("--mmseqs-binary", default="mmseqs")
    p.add_argument("--su-tm-score", type=float, default=0.60)
    p.add_argument("--sequence-identity", type=float, default=0.90)
    p.add_argument("--sequence-coverage", type=float, default=0.80)
    p.add_argument("--allow-missing-structure", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only a prior T-REX export; reject unrelated files",
    )
    args = p.parse_args(argv)

    try:
        archive_root = args.archive_root.expanduser().resolve()
        if not archive_root.is_dir():
            raise ValueError(f"archive root is not a directory: {archive_root}")
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir is not None
            else archive_root / f"wetlab_panel_best{args.n}"
        )
        diagnostics = export_best_n(
            Archive(archive_root),
            target_id=args.target_id,
            n=args.n,
            out_dir=out_dir,
            copy_pdbs=not args.no_copy,
            annotate_dedup=not args.no_dedup,
            foldseek_binary=args.foldseek_binary,
            mmseqs_binary=args.mmseqs_binary,
            require_structure_artifact=not args.allow_missing_structure,
            su_tm_score=args.su_tm_score,
            sequence_identity=args.sequence_identity,
            sequence_coverage=args.sequence_coverage,
            overwrite=args.overwrite,
        )
        manifest_csv = out_dir / "manifest.csv"
        print(
            json.dumps(
                {
                    "schema_version": BEST_N_OUTPUT_SCHEMA_VERSION,
                    "archive_root": str(archive_root),
                    "out_dir": str(out_dir),
                    "manifest_json": str(out_dir / "manifest.json"),
                    "manifest_csv": str(manifest_csv),
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"trex-export: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
