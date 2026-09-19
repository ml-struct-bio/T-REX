"""Complexa output parser.

Reads:
    <output_dir>/rewards_<config>_<seed>.csv

Schema (verified against real CD45 fixture):
  Required columns:
    pdb_path                    — PDB output path on disk
    sample_type                 — "final" or "lookahead". Filter to "final".
    af2folding_plddt_log        — canonical model_1 pLDDT in 0-1 (use ×100)
    af2folding_i_pae            — canonical model_1 normalized iPAE [0-1]
    af2folding_rmsd             — canonical model_1 design-vs-prediction Cα RMSD in Å
  Optional (diagnostic):
    af2folding_i_ptm_log        — interface pTM
    af2folding_min_ipae         — min iPAE across models
    af2folding_avg_ipsae        — averaged interface pSAE
    af2folding_max_ipsae        — max iPAE-sub-error
    af2folding_con              — contact-density score
    af2folding_i_con            — interface contact density
    af2folding_fape             — FAPE loss

Filter rules:
  - Only emit ResultRecord for rows with `sample_type == "final"`
    (Plan §3 lifecycle: lookahead beam candidates are NOT final)
  - Rows with missing core metrics are emitted as diagnostic records; only
    rows with usable structure artifacts are marked for canonical scoring.
  - If `rewards_*.csv` missing → ParseError
  - If empty (no final samples) → return []

ResultRecord mapping:
  The controller forces the official Complexa AF2 gate (model_1,
  num_recycles=3, use_initial_guess=True, use_multimer=True), so these
  af2folding_* fields are trusted as direct strict/SU evidence when present.
  If the fields are absent, emit a diagnostic artifact only and mark it for
  canonical score conversion instead of fabricating an empty strict-scored
  record.
  result_id        := hash(target_id, candidate_id, launch namespace,
                           sample_index, pdb basename)
  parent_ids       := [ctx.candidate_id] (worker spawn point)
  backend_family   := from ctx.method_family (complexa_beam, complexa_best_of_n, complexa_fk_steering, complexa_mcts)
  target_id        := ctx.target_id
  metrics["pLDDT"]                         := af2folding_plddt_log × 100
  metrics["iPAE"]                          := af2folding_i_pae
  metrics["binder_scRMSD"]                 := af2folding_rmsd
  metrics["complexa_native_*"]             := same values, kept as provenance
  + diagnostic: i_ptm, min_ipae, ipsae, contact density when present
  gpu_h            := placeholder COMPLEXA_GPU_H_PER_FINAL (0.5). NOTE
                      (G-027, 2026-05-29): this parser does NOT read the
                      timing CSV; the authoritative per-record gpu_h is
                      reassigned post-parse by the controller (§22.8.9) to
                      (elapsed wall-clock GPU time / n_records), so this
                      constant never reaches method_health / su_per_gpu_h.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

from ..schemas import ResultRecord
from .types import ParseError, ParserContext


# Per-row gpu_h fallback: each Complexa final sample takes ~30 min on H100
# (calibrated from V5 SLURM elapsed time / N final samples).
COMPLEXA_GPU_H_PER_FINAL = 0.5


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _result_id(
    target_id: str,
    candidate_id: str,
    launch_namespace: str,
    sample_index: int,
    pdb_path: str,
) -> str:
    pdb_name = Path(pdb_path).name if pdb_path else ""
    blob = (
        f"{target_id}::{candidate_id}::{launch_namespace}::"
        f"{sample_index}::{pdb_name}"
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _find_rewards_csv(output_dir: Path) -> Path | None:
    """Find the rewards CSV (filename pattern: rewards_*.csv)."""
    candidates = list(output_dir.glob("rewards_*.csv"))
    # Prefer non-"all_rewards" file (which is the per-step accumulator)
    main_csvs = [p for p in candidates if not p.name.startswith("all_rewards")]
    return main_csvs[0] if main_csvs else (candidates[0] if candidates else None)


def _resolve_structure_path(raw_path: str, output_dir: Path) -> str | None:
    """Return an absolute existing structure path, or None if unusable."""
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = [p] if p.is_absolute() else [output_dir / p, p]
    for cand in candidates:
        try:
            if cand.exists() and cand.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                return str(cand.resolve())
        except OSError:
            continue
    return None


def parse_complexa_output(
    output_dir: Path, ctx: ParserContext
) -> list[ResultRecord]:
    """Convert a Complexa inference output directory into T-ReX ResultRecords."""
    if not output_dir.exists():
        raise ParseError(f"Complexa output dir missing: {output_dir}")

    csv_path = _find_rewards_csv(output_dir)
    if csv_path is None:
        raise ParseError(f"Complexa output dir has no rewards_*.csv: {output_dir}")

    records: list[ResultRecord] = []
    # B-009 fix (2026-05-26): prefer ctx.method_family (set by controller from
    # the spawning ActionCandidate); fall back to candidate_id-encoded family
    # then to "complexa_beam" if neither is present. Without this propagation,
    # MCTS / fk-steering / best-of-n launches would all be tagged as plain beam
    # in the archive.
    if ctx.method_family:
        family = ctx.method_family
    elif ":" in ctx.candidate_id:
        family = ctx.candidate_id.split(":")[-1]
    else:
        family = "complexa_beam"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if (row.get("sample_type") or "").strip() != "final":
                continue
            plddt_log = _safe_float(row.get("af2folding_plddt_log"))
            i_pae     = _safe_float(row.get("af2folding_i_pae"))
            rmsd      = _safe_float(row.get("af2folding_rmsd"))
            metrics: dict[str, float] = {}
            # The controller forces Complexa AF2RewardModel to the official
            # verifier settings: AF2-Multimer, model_1, num_recycles=3, and
            # use_initial_guess=True. Therefore these values are not merely
            # advisory for Complexa when present: they are the official
            # strict/SU axes. Keep complexa_native_* aliases as provenance for
            # prompt diagnostics.
            plddt = (plddt_log * 100.0) if plddt_log is not None else None
            has_core = (
                plddt is not None and i_pae is not None and rmsd is not None
            )
            if has_core:
                _maybe(metrics, "pLDDT", plddt)
                _maybe(metrics, "iPAE", i_pae)
                _maybe(metrics, "binder_scRMSD", rmsd)
                _maybe(metrics, "complexa_native_pLDDT", plddt)
                _maybe(metrics, "complexa_native_iPAE", i_pae)
                _maybe(metrics, "complexa_native_binder_scRMSD", rmsd)

            # Diagnostic axes (Tier 1 — Complexa provides ipTM, ipSAE, min_ipae natively)
            _maybe(metrics, "ipTM",         _safe_float(row.get("af2folding_i_ptm_log")))
            _maybe(metrics, "min_ipae",     _safe_float(row.get("af2folding_min_ipae")))
            _maybe(metrics, "avg_ipsae",    _safe_float(row.get("af2folding_avg_ipsae")))
            _maybe(metrics, "max_ipsae",    _safe_float(row.get("af2folding_max_ipsae")))
            _maybe(metrics, "contact_density",   _safe_float(row.get("af2folding_con")))
            _maybe(metrics, "interface_contact_density",
                                                _safe_float(row.get("af2folding_i_con")))
            _maybe(metrics, "fape",         _safe_float(row.get("af2folding_fape")))

            pdb_path_raw = (row.get("pdb_path") or "").strip()
            pdb_path = _resolve_structure_path(pdb_path_raw, output_dir)
            # `candidate_id` can intentionally repeat for warm-start Complexa
            # launches across ticks. Include a launch namespace so each
            # physical output row has a unique ResultRecord identity.
            launch_namespace = "::".join(
                part for part in (ctx.tick_id, output_dir.name) if part
            )
            rid = _result_id(
                ctx.target_id, ctx.candidate_id, launch_namespace, idx,
                pdb_path or pdb_path_raw,
            )

            bins = {"sample_index": str(idx)}
            if has_core and pdb_path:
                bins["strict_score_source"] = "complexa_af2folding_canonical"
            elif has_core:
                for key in ("pLDDT", "iPAE", "binder_scRMSD"):
                    metrics.pop(key, None)
                bins["complexa_af2folding_status"] = "missing_structure_artifact"
                bins["score_conversion_status"] = "missing_structure_artifact"
            else:
                bins["complexa_af2folding_status"] = "missing_core_metrics"
                if pdb_path:
                    bins["needs_canonical_score_conversion"] = "1"
                else:
                    bins["score_conversion_status"] = "missing_structure_artifact"
            if pdb_path_raw and not pdb_path:
                bins["artifact_status"] = "missing_or_unusable_pdb_path"

            records.append(ResultRecord(
                result_id=rid,
                parent_ids=list(ctx.parent_ids),
                backend_family=family,
                runtime_bucket_id=ctx.runtime_bucket_id,
                target_id=ctx.target_id,
                metrics=metrics,
                metrics_calibrated={},
                route_lineage=[family],
                gpu_h=COMPLEXA_GPU_H_PER_FINAL,
                exit_status="ok",
                bins=bins,
                artifacts={"pdb_path": pdb_path} if pdb_path else {},
                panel_ready=False,
                tick_id=(ctx.tick_id or None),
            ))
    return records


def _maybe(d: dict, key: str, value: float | None) -> None:
    if value is not None:
        d[key] = value
