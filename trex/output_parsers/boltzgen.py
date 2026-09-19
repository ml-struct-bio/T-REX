"""Parse BoltzGen de novo design output into T-ReX ResultRecord(s).

P2-B (2026-05-26): BoltzGen writes per-target aggregate CSVs (e.g.
``aggregate_metrics_<run>.csv``, ``final_designs_metrics_<budget>.csv``,
``all_designs_metrics.csv``) plus CIF outputs at
``final_ranked_designs/final_*_designs/*.cif``. Available metric columns
include ``design_iptm``, ``design_to_target_iptm``, ``design_iiptm``,
``design_ptm``, ``min_design_to_target_pae``, ``structure_confidence``,
``native_rmsd``, ``native_rmsd_refolded``, ``seq_recovery`` etc.

These are BOLTZGEN-NATIVE metrics, NOT T-ReX's AF2-calibrated strict-success
rule. So this parser emits ResultRecords with empty `metrics` (diagnostic-
only path) — the LLM is expected to chain a BoltzGen hit with
`structure_refilter` (AF2) in a follow-up tick to obtain T-ReX-calibrated
pLDDT/iPAE/scRMSD.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ..schemas import ResultRecord
from ..score_conversion import is_boltzgen_aggregate_artifact_path
from .types import ParseError, ParserContext

BOLTZGEN_GPU_H_PER_DESIGN = 0.50


def _safe_float(v):
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _find_metrics_csv(output_dir: Path) -> Path | None:
    """Locate the BoltzGen aggregate metrics CSV. Prefer
    `all_designs_metrics.csv` (full pool), fall back to
    `final_designs_metrics_*.csv` or `aggregate_metrics_*.csv`.

    fix25 (2026-05-26): BoltzGen actually writes its metric CSVs under
    ``final_ranked_designs/`` and ``intermediate_designs_inverse_folded/``,
    not the top-level output dir. The earlier flat lookup at
    ``output_dir/all_designs_metrics.csv`` always missed, returning 0
    records on every BoltzGen launch (fix24r3 CD45 r1). Use recursive
    glob to find them under any subdir.
    """
    # Prefer the full-pool CSV (recursive — usually in final_ranked_designs/)
    cands = sorted(output_dir.glob("**/all_designs_metrics.csv"))
    if cands:
        return cands[0]
    cands = sorted(output_dir.glob("**/final_designs_metrics_*.csv"))
    if cands:
        return cands[0]
    cands = sorted(output_dir.glob("**/aggregate_metrics_*.csv"))
    if cands:
        return cands[0]
    return None


def _index_cifs(output_dir: Path) -> dict[str, Path]:
    cif_by_id: dict[str, Path] = {}
    for cif in output_dir.glob("**/*.cif"):
        if is_boltzgen_aggregate_artifact_path(cif):
            continue
        stem = cif.stem
        if stem.startswith("rank"):
            parts = stem.split("_", 2)
            if len(parts) >= 3:
                stem = parts[1] + "_" + parts[2]
        if stem not in cif_by_id or "final_ranked" in str(cif):
            cif_by_id[stem] = cif
    return cif_by_id


def _resolve_artifact_path(raw: str, *, csv_path: Path | None, output_dir: Path) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [
        *(([csv_path.parent / path] if csv_path is not None else [])),
        output_dir / path,
        path,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return ""


def _boltzgen_record(
    *,
    ctx: ParserContext,
    fam: str,
    idx: int,
    design_id: str,
    bins: dict[str, str] | None,
    pdb_path: str,
) -> ResultRecord:
    b = {"refilter_source": "boltzgen", "sample_idx": str(idx)}
    if design_id:
        b["boltzgen_design_id"] = design_id
    if bins:
        b.update(bins)
    blob = f"{ctx.target_id}::{ctx.candidate_id}::boltzgen::{idx}::{design_id}".encode()
    rid = hashlib.sha256(blob).hexdigest()[:16]
    return ResultRecord(
        result_id=rid,
        parent_ids=list(ctx.parent_ids),
        backend_family=fam,
        runtime_bucket_id=ctx.runtime_bucket_id,
        target_id=ctx.target_id,
        metrics={},
        metrics_calibrated={},
        route_lineage=[fam],
        gpu_h=BOLTZGEN_GPU_H_PER_DESIGN,
        exit_status="ok",
        bins=b,
        artifacts={"pdb_path": pdb_path} if pdb_path else {},
        panel_ready=False,
        tick_id=(ctx.tick_id or None),
    )


def parse_boltzgen_output(
    output_dir: Path, ctx: ParserContext
) -> list[ResultRecord]:
    if not output_dir.exists():
        raise ParseError(f"BoltzGen output dir missing: {output_dir}")
    fam = ctx.method_family or "boltzgen"
    cif_by_id = _index_cifs(output_dir)
    csv_path = _find_metrics_csv(output_dir)
    if csv_path is None:
        if not cif_by_id:
            return []
        return [
            _boltzgen_record(
                ctx=ctx, fam=fam, idx=i, design_id=design_id,
                bins={"boltzgen_orphan_cif": "1"}, pdb_path=str(cif),
            )
            for i, (design_id, cif) in enumerate(sorted(cif_by_id.items()))
        ]

    records: list[ResultRecord] = []
    seen_design_ids: set[str] = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            bins: dict[str, str] = {}
            for k in ("design_iptm", "design_to_target_iptm", "design_iiptm",
                      "design_ptm", "min_design_to_target_pae",
                      "structure_confidence", "native_rmsd",
                      "native_rmsd_refolded", "seq_recovery"):
                v = _safe_float(row.get(k))
                if v is not None:
                    bins[f"boltzgen_{k}"] = f"{v:.4f}"
            for k in ("final_rank", "secondary_rank", "max_rank", "quality_score",
                      "num_filters_passed"):
                v = _safe_float(row.get(k))
                if v is not None:
                    bins[f"boltzgen_{k}"] = f"{v:.4f}"
            design_id = (row.get("id") or "").strip()
            pdb_path = _resolve_artifact_path(row.get("pdb_path") or row.get("path") or "", csv_path=csv_path, output_dir=output_dir)
            if not pdb_path and design_id in cif_by_id:
                pdb_path = str(cif_by_id[design_id])
            if is_boltzgen_aggregate_artifact_path(pdb_path):
                continue
            if design_id == "design" and not any(
                bins.get(k) for k in (
                    "boltzgen_final_rank",
                    "boltzgen_secondary_rank",
                    "boltzgen_max_rank",
                )
            ):
                continue
            if design_id:
                seen_design_ids.add(design_id)
            records.append(_boltzgen_record(
                ctx=ctx, fam=fam, idx=idx, design_id=design_id,
                bins=bins, pdb_path=pdb_path,
            ))
    for design_id, cif in sorted(cif_by_id.items()):
        if design_id in seen_design_ids:
            continue
        records.append(_boltzgen_record(
            ctx=ctx, fam=fam, idx=len(records), design_id=design_id,
            bins={"boltzgen_orphan_cif": "1"}, pdb_path=str(cif),
        ))
    # §22.8.8 (2026-05-27): tolerate empty/header-only CSV (e.g. when
    # BoltzGen was killed mid-pipeline by the controller timeout salvage
    # path). Returning `[]` lets the controller record a synthetic
    # `exit_status="timeout"` summary record with the elapsed gpu_h so
    # `method_health.cumulative_gpu_h` reflects actual GPU time spent.
    # The earlier ParseError caused the controller to log a parse_error
    # and credit 0 GPU-h, contributing to the "boltzgen looks under-
    # explored" mis-signal seen in 8813621/2.
    return records
