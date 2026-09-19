"""BindCraft output parser.

Reads:
    <output_dir>/designs/final_design_stats.csv
    <output_dir>/designs/mpnn_design_stats.csv
    <output_dir>/designs/Accepted/*.pdb
    <output_dir>/designs/Rejected/*.pdb

Output schema mapping (verified against real BindCraft job 8736558):

  Native proxy axes (3) — populate diagnostic metrics only:
    bindcraft_native_pLDDT       := row["modelN_pLDDT"] * 100.0
    bindcraft_native_iPAE        := row["modelN_i_pAE"]
    bindcraft_native_binder_RMSD := row["modelN_Binder_RMSD"]
    These never populate pLDDT/iPAE/binder_scRMSD strict-gate keys;
    canonical structure_refilter does that for official strict/SU.

  Diagnostic axes (Tier 1, §4.1) — populate via the "Average_*" columns,
  which are mean over the 5 AF2 prediction models per design:
    metrics["interface_dG"]               := row["Average_dG"]
    metrics["shape_complementarity"]      := row["Average_ShapeComplementarity"]
    metrics["interface_hbonds"]           := row["Average_n_InterfaceHbonds"]
    metrics["interface_unsat_hbonds"]     := row["Average_n_InterfaceUnsatHbonds"]
    metrics["buried_sasa"]                := row["Average_dSASA"]
    metrics["binder_pLDDT_avg"]           := row["Average_Binder_pLDDT"]
    metrics["hotspot_rmsd"]               := row["Average_Hotspot_RMSD"]

  Other ResultRecord fields:
    result_id          := hash(target_id, candidate_id, design_name)
    parent_ids         := ctx.parent_ids   (typically [candidate_id])
    backend_family     := "bindcraft"
    runtime_bucket_id  := ctx.runtime_bucket_id
    target_id          := ctx.target_id
    route_lineage      := ["bindcraft"]
    gpu_h              := parse DesignTime "X hours, Y minutes, Z seconds" → hours
    exit_status        := "ok" for parseable diagnostic artifacts
    bindcraft_filter_status := "accepted" or "rejected" in bins

Filter rules:
  - BindCraft's Accepted/Rejected split is NOT the T-ReX strict gate.
    Both directories emit diagnostic artifacts when a PDB exists.
  - BindCraft-native metrics remain diagnostic-only. A rejected artifact can
    still pass T-ReX after canonical AF2 score conversion.
  - Empty CSV with artifact PDBs → emit metric-light artifacts for downstream
    score conversion.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path
from typing import Any

from ..schemas import ResultRecord
from .types import ParseError, ParserContext


_DESIGN_TIME_RE = re.compile(
    r"(?:(\d+)\s*hours?)?[, ]*(?:(\d+)\s*minutes?)?[, ]*(?:(\d+)\s*seconds?)?",
    re.IGNORECASE,
)


def _parse_design_time(value: str | None) -> float:
    """Parse 'X hours, Y minutes, Z seconds' → hours.

    NOTE: BindCraft's DesignTime is the per-design CPU relaxation time
    (~1-2 min), NOT the GPU trajectory time. For accurate gpu_h we
    use BINDCRAFT_GPU_H_PER_DESIGN below; this helper remains for
    audit / cross-checking.
    """
    if not value:
        return 0.0
    m = _DESIGN_TIME_RE.search(value)
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    mn = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h + mn / 60.0 + s / 3600.0


# Calibrated from smoke 8736558: 12h budget produced ~12 accepted
# designs ⇒ ~1.0 GPU-h per accepted design (incl. failed trajectories).
# Used by parser as a stable proxy until BindCraft wrapper logs actual
# per-design GPU-h. State classifier reads this for budget tracking.
BINDCRAFT_GPU_H_PER_DESIGN = 1.0


def _safe_float(value: Any) -> float | None:
    """Convert to float, returning None on missing/NaN/empty."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _result_id(target_id: str, candidate_id: str, design_name: str) -> str:
    blob = f"{target_id}::{candidate_id}::{design_name}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]



def _plddt_to_100(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.5 else value


def _model_index_from_stem(stem: str) -> str:
    m = re.search(r"_model([1-5])$", stem)
    return m.group(1) if m else "1"


def _pref_for_stem(row: dict[str, Any], stem: str, axis: str, average_axis: str) -> float | None:
    """Prefer the metric matching this artifact's modelN suffix, then average."""
    idx = _model_index_from_stem(stem)
    v = _safe_float(row.get(f"{idx}_{axis}"))
    return v if v is not None else _safe_float(row.get(average_axis))


def _read_rows_by_design(csv_path: Path | None) -> dict[str, dict[str, Any]]:
    if csv_path is None or not csv_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            design = (row.get("Design") or "").strip()
            if design:
                rows[design] = row
    return rows


def _matching_stems(design: str, stems: set[str]) -> list[str]:
    # Require a token boundary so "mpnn1" cannot grab "mpnn10_model1".
    return sorted(s for s in stems if s == design or s.startswith(f"{design}_"))


def _metrics_from_row(row: dict[str, Any], stem: str) -> dict[str, float]:
    # Required success-axis proxies. These are BindCraft-native diagnostics only.
    plddt_01 = _pref_for_stem(row, stem, "pLDDT", "Average_pLDDT")
    ipae = _pref_for_stem(row, stem, "i_pAE", "Average_i_pAE")
    binder_rmsd = _pref_for_stem(row, stem, "Binder_RMSD", "Average_Binder_RMSD")

    metrics: dict[str, float] = {}
    plddt_100 = _plddt_to_100(plddt_01)
    if plddt_100 is not None:
        metrics["bindcraft_native_pLDDT"] = plddt_100
    if ipae is not None:
        metrics["bindcraft_native_iPAE"] = ipae
    if binder_rmsd is not None:
        metrics["bindcraft_native_binder_RMSD"] = binder_rmsd

    # Diagnostic axes (Tier 1) — populated only when available
    _maybe_add(metrics, "interface_dG", _safe_float(row.get("Average_dG")))
    _maybe_add(metrics, "shape_complementarity",
                _safe_float(row.get("Average_ShapeComplementarity")))
    _maybe_add(metrics, "interface_hbonds",
                _safe_float(row.get("Average_n_InterfaceHbonds")))
    _maybe_add(metrics, "interface_unsat_hbonds",
                _safe_float(row.get("Average_n_InterfaceUnsatHbonds")))
    _maybe_add(metrics, "buried_sasa",
                _safe_float(row.get("Average_dSASA")))
    _maybe_add(metrics, "binder_pLDDT_avg",
                _safe_float(row.get("Average_Binder_pLDDT")))
    _maybe_add(metrics, "hotspot_rmsd",
                _safe_float(row.get("Average_Hotspot_RMSD")))

    # Optional but very useful: interface area & confidence summary
    _maybe_add(metrics, "n_interface_residues",
                _safe_float(row.get("Average_n_InterfaceResidues")))
    _maybe_add(metrics, "binder_pTM_avg",
                _safe_float(row.get("Average_Binder_pTM")))
    return metrics


def _artifact_record(
    *,
    ctx: ParserContext,
    pdb_path: Path,
    stem: str,
    filter_status: str,
    bins: dict[str, str] | None = None,
    row: dict[str, Any] | None = None,
    result_key: str | None = None,
) -> ResultRecord:
    """BindCraft artifact that must still enter canonical AF2 scoring."""
    b = {
        "design": stem,
        "bindcraft_filter_status": filter_status,
        f"bindcraft_{filter_status}_artifact": "1",
    }
    if bins:
        b.update(bins)
    metrics = _metrics_from_row(row, stem) if row is not None else {}
    return ResultRecord(
        result_id=_result_id(
            ctx.target_id,
            ctx.candidate_id,
            result_key or f"{filter_status}::{stem}",
        ),
        parent_ids=list(ctx.parent_ids),
        backend_family="bindcraft",
        runtime_bucket_id=ctx.runtime_bucket_id,
        target_id=ctx.target_id,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        metrics_calibrated={},
        route_lineage=["bindcraft"],
        gpu_h=BINDCRAFT_GPU_H_PER_DESIGN,
        exit_status="ok",
        bins=b,
        artifacts={"pdb_path": str(pdb_path), "pdb_dir": str(pdb_path.parent)},
        panel_ready=False,
        tick_id=(ctx.tick_id or None),
    )


def parse_bindcraft_output(
    output_dir: Path, ctx: ParserContext
) -> list[ResultRecord]:
    """Convert a BindCraft output directory into T-ReX ResultRecords.

    Robustness invariants:
      - Missing designs/ → ParseError.
      - Missing Accepted/ is not enough to return []: Rejected/ artifacts can
        still pass T-ReX's canonical AF2 strict gate after score conversion.
      - CSV row without matching artifact PDB is ignored.
      - Artifact PDB without CSV row emits a metric-light record so canonical
        score conversion can still evaluate it.
      - BindCraft-native metrics are diagnostics only; strict-gate keys are
        never populated here.
    """
    designs_dir = output_dir / "designs"
    if not designs_dir.exists():
        raise ParseError(f"BindCraft output missing 'designs/' subdir: {output_dir}")

    accepted_dir = designs_dir / "Accepted"
    rejected_dir = designs_dir / "Rejected"
    accepted_stems = {p.stem for p in accepted_dir.glob("*.pdb")} if accepted_dir.exists() else set()
    rejected_stems = {p.stem for p in rejected_dir.glob("*.pdb")} if rejected_dir.exists() else set()
    if not accepted_stems and not rejected_stems:
        return []

    # final_design_stats has the final filter candidates; mpnn_design_stats
    # often contains rows for Rejected/ artifacts that never enter Accepted/.
    # Prefer final rows when both exist, but fall back to mpnn rows.
    rows_by_design = _read_rows_by_design(designs_dir / "mpnn_design_stats.csv")
    rows_by_design.update(_read_rows_by_design(designs_dir / "final_design_stats.csv"))

    records: list[ResultRecord] = []
    matched_accepted_stems: set[str] = set()
    matched_rejected_stems: set[str] = set()

    for design, row in rows_by_design.items():
        for filter_status, stems, artifact_dir, matched in (
            ("accepted", accepted_stems, accepted_dir, matched_accepted_stems),
            ("rejected", rejected_stems, rejected_dir, matched_rejected_stems),
        ):
            if not stems or not artifact_dir.exists():
                continue
            matched_stems = _matching_stems(design, stems)
            if not matched_stems:
                continue

            for chosen_stem in matched_stems:
                matched.add(chosen_stem)
                plddt_01 = _pref_for_stem(row, chosen_stem, "pLDDT", "Average_pLDDT")
                ipae = _pref_for_stem(row, chosen_stem, "i_pAE", "Average_i_pAE")
                binder_rmsd = _pref_for_stem(row, chosen_stem, "Binder_RMSD", "Average_Binder_RMSD")
                iptm = _pref_for_stem(row, chosen_stem, "i_pTM", "Average_i_pTM")
                bins = _bc_bins(
                    design,
                    plddt_01,
                    ipae,
                    binder_rmsd,
                    iptm,
                    filter_status=filter_status,
                    pdb_stem=chosen_stem,
                )
                result_key = (
                    design
                    if filter_status == "accepted" and len(matched_stems) == 1
                    else f"{filter_status}::{chosen_stem}"
                )
                records.append(_artifact_record(
                    ctx=ctx,
                    pdb_path=artifact_dir / f"{chosen_stem}.pdb",
                    stem=chosen_stem,
                    filter_status=filter_status,
                    bins=bins,
                    row=row,
                    result_key=result_key,
                ))
                _ = _parse_design_time(row.get("DesignTime"))  # parsed for audit; unused

    for stem in sorted(accepted_stems - matched_accepted_stems):
        records.append(_artifact_record(
            ctx=ctx,
            pdb_path=accepted_dir / f"{stem}.pdb",
            stem=stem,
            filter_status="accepted",
            bins={"bindcraft_orphan_accepted": "1"},
        ))
    for stem in sorted(rejected_stems - matched_rejected_stems):
        records.append(_artifact_record(
            ctx=ctx,
            pdb_path=rejected_dir / f"{stem}.pdb",
            stem=stem,
            filter_status="rejected",
            bins={"bindcraft_orphan_rejected": "1"},
        ))
    return records


def _bc_bins(
    design,
    plddt_01,
    ipae,
    binder_rmsd,
    iptm,
    *,
    filter_status: str = "accepted",
    pdb_stem: str | None = None,
) -> dict[str, str]:
    """Surface BindCraft's NATIVE scores to the Planner as bindcraft_* bins.

    All diagnostic — NOT the strict gate, which comes from canonical
    structure_refilter. The filter_status is preserved so the LLM can learn
    when BindCraft rejected a sample that T-ReX later validates.
    """
    b: dict[str, str] = {
        "design": design,
        "bindcraft_filter_status": filter_status,
        "bindcraft_native_model_index": _model_index_from_stem(pdb_stem or design),
    }
    if pdb_stem:
        b["bindcraft_pdb_stem"] = pdb_stem
    plddt_100 = _plddt_to_100(plddt_01)
    if plddt_100 is not None:
        b["bindcraft_native_pLDDT"] = f"{plddt_100:.2f}"
    if ipae is not None:
        b["bindcraft_native_iPAE"] = f"{ipae:.4f}"
    if binder_rmsd is not None:
        b["bindcraft_native_binder_RMSD"] = f"{binder_rmsd:.4f}"
    if iptm is not None:
        b["bindcraft_rank_iptm"] = f"{iptm:.4f}"
    return b


def _maybe_add(d: dict, key: str, value: float | None) -> None:
    """Add key->value to dict ONLY if value is not None.
    Absence != None — absent means "metric not reported by this pipeline",
    None would mean "we know this should be here but couldn't compute"."""
    if value is not None:
        d[key] = value
