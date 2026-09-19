"""Parse V5 AF2 refilter output (`af2_refilter_result.json`) into T-ReX
ResultRecord(s).

The runner (`trex.af2_refilter_runner`) writes a single JSON
with this schema:

    {
      "schema_version": "v5_af2_refilter_result.v1",
      "input_pdb": "...",
      "target_chain": "A",
      "binder_chain": "B",  # input chain; prediction labels may differ
      "prediction_chains": {
        "binder_chain": "B", "target_chains": ["A"],
        "sequence_sha256": {"A": "...", "B": "..."},
      },
      "metrics": {
        "i_pae": float,            # T-ReX iPAE (already 0..1)
        "plddt": float,             # T-ReX pLDDT in [0, 1] — multiply by 100
        "binder_scrmsd_ca": float,  # T-ReX binder_scRMSD (Å)
        "iptm": float,              # diagnostic
        "raw_losses": {...},
        "predicted_pdb": "...",
      }
    }

P2-A (2026-05-26): refilter outputs a SINGLE record per parent binder
PDB (the AF2 prediction of that exact complex). Multi-binder refiltering
is achieved by dispatching multiple Popen calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..af2_chain_identity import ChainIdentityError, resolve_prediction_chains
from ..refilter_roles import PARENT_MODEL_REFOLD
from ..schemas import ResultRecord
from .types import ParseError, ParserContext

AF2_REFILTER_GPU_H_PER_RUN = 0.05  # tiny — single AF2 multimer pass


def parse_af2_refilter_output(
    output_dir: Path, ctx: ParserContext
) -> list[ResultRecord]:
    """Read af2_refilter_result.json and emit one T-ReX ResultRecord."""
    if not output_dir.exists():
        raise ParseError(f"AF2 refilter output dir missing: {output_dir}")
    report_path = output_dir / "af2_refilter_result.json"
    if not report_path.exists():
        raise ParseError(f"af2_refilter_result.json missing in {output_dir}")
    report = json.loads(report_path.read_text())
    raw = report.get("metrics") or {}

    # SU-minting basis must be a FIXED function of the design (2026-06-12). The
    # strict gate (pLDDT/iPAE/scRMSD) is written ONLY from the CANONICAL AF2 score
    # — the fixed single-model, default-recycle, initial-guess config the auto-chain
    # conversion and the V5/V6.3 baselines use. AF2 structure_refilter has two uses:
    #   (1) canonical SCORE-CONVERSION — gate a Complexa/BoltzGen/BindCraft/MPNN
    #       design under the fixed config; MINTS SU.
    #   (2) parent_model_refold — re-fold under a DIFFERENT config (model ensemble,
    #       num_recycles, use_initial_guess) to probe "is the predicted STRUCTURE the
    #       bottleneck while the sequence is fine?". This is a different MEASUREMENT of
    #       the same design, so minting SU from it would make the strict gate
    #       config-dependent (re-fold a near-miss under a friendlier config until it
    #       passes = measurement shopping against a non-negotiable gate). It is
    #       therefore ADVISORY ONLY: its axis values are recorded as refold_* and
    #       never populate the canonical strict keys. (To actually fix a "good
    #       sequence / bad backbone" the correct ACTION is to re-DESIGN — ProteinMPNN
    #       or regeneration — not to re-measure the same design.)
    # A score is advisory iff the candidate is an intentional parent_model_refold OR
    # it ran a multi-model ensemble (ColabDesign mean-averages losses across models).
    _model_names = raw.get("model_names")
    if isinstance(_model_names, str):
        _model_names = [m.strip() for m in _model_names.split(",") if m.strip()]
    _n_models = len(_model_names) if isinstance(_model_names, (list, tuple)) else 1
    _is_advisory = ctx.refilter_role == PARENT_MODEL_REFOLD or _n_models > 1

    metrics: dict[str, float] = {}
    # All-or-none on the 3 strict axes (2026-05-31, strict_gate_nearmiss LOW):
    # emitting a PARTIAL subset (e.g. binder_scrmsd_ca dropped by the runner's
    # _float_dict when the ColabDesign rmsd loss is absent) would leave a 2-axis
    # record on which is_near_miss fires even though the blocking axis is
    # UNKNOWN. is_strict_success already requires all three; mirror that so a
    # partial refold is purely diagnostic (never strict, never near-miss).
    _p, _i, _r = raw.get("plddt"), raw.get("i_pae"), raw.get("binder_scrmsd_ca")
    if all(isinstance(x, (int, float)) for x in (_p, _i, _r)):
        if _is_advisory:
            metrics["refold_pLDDT"] = float(_p) * 100.0
            metrics["refold_iPAE"] = float(_i)
            metrics["refold_binder_scRMSD"] = float(_r)
        else:
            metrics["pLDDT"] = float(_p) * 100.0
            metrics["iPAE"] = float(_i)
            metrics["binder_scRMSD"] = float(_r)
    if isinstance(raw.get("iptm"), (int, float)):
        metrics["ipTM"] = float(raw["iptm"])

    fam = ctx.method_family or "structure_refilter"
    blob = f"{ctx.target_id}::{ctx.candidate_id}::af2_refilter".encode()
    rid = hashlib.sha256(blob).hexdigest()[:16]

    predicted_pdb = raw.get("predicted_pdb", "") or ""
    bins = {
        # Use the archive parent id when available. A basename-only input_pdb
        # fallback can collide across worker directories and under-count SU on
        # the degraded no-Foldseek path.
        "refilter_source": ctx.parent_result_id or report.get("input_pdb", "") or Path(report.get("input_pdb", "")).name,
        # Audit the SU-minting basis: only the canonical single-config score (a
        # canonical score-conversion) mints strict SU; a parent_model_refold or an
        # ensemble re-fold is advisory.
        "af2_strict_basis": "advisory_refold" if _is_advisory else "canonical",
        "af2_model_count": _n_models,
    }
    if ctx.refilter_role:
        bins["refilter_role"] = ctx.refilter_role
    if ctx.refilter_source_family:
        bins["refilter_source_family"] = ctx.refilter_source_family
    binder_chain = ""
    target_chains = ""
    chain_map = None
    try:
        prediction_path = Path(predicted_pdb)
        if not prediction_path.is_absolute():
            prediction_path = output_dir / prediction_path
        chain_map = resolve_prediction_chains(
            report, prediction_path, report_dir=output_dir,
            input_binder_chain=ctx.binder_chain or "",
            input_target_chains=ctx.target_chains_csv or "",
        )
        binder_chain = chain_map["binder_chain"]
        target_chains = ",".join(chain_map["target_chains"])
        predicted_pdb = str(prediction_path.resolve())
        bins["af2_chain_identity"] = "verified"
    except ChainIdentityError as exc:
        bins["af2_chain_identity"] = "unresolved"
        bins["af2_chain_identity_error"] = str(exc)
    if binder_chain:
        bins["binder_chain"] = binder_chain
    if target_chains:
        bins["target_chains"] = target_chains
    artifacts = {"pdb_path": predicted_pdb} if predicted_pdb else {}
    artifacts["af2_report_path"] = str(report_path.resolve())
    if chain_map is not None:
        artifacts["af2_prediction_chains"] = json.dumps(chain_map, sort_keys=True)
    if binder_chain:
        artifacts["binder_chain"] = binder_chain
    if target_chains:
        artifacts["target_chains"] = target_chains

    return [ResultRecord(
        result_id=rid,
        parent_ids=list(ctx.parent_ids),
        backend_family=fam,
        runtime_bucket_id=ctx.runtime_bucket_id,
        target_id=ctx.target_id,
        metrics=metrics,
        metrics_calibrated={},
        route_lineage=[fam],
        gpu_h=AF2_REFILTER_GPU_H_PER_RUN,
        exit_status="ok",
        bins=bins,
        artifacts=artifacts,
        panel_ready=False,
        tick_id=(ctx.tick_id or None),
    )]
