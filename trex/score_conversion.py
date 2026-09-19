"""Shared guards for canonical score-conversion candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .success_criteria import NEAR_PASS_MARGINS


_BOLTZGEN_AGGREGATE_NAMES = {
    "design.cif",
    "design.pdb",
    "design.converted.pdb",
}


def is_boltzgen_aggregate_artifact_path(path: str | Path | None) -> bool:
    """Return True for BoltzGen's top-level aggregate complex artifact.

    BoltzGen writes a root ``design.cif``/``design.converted.pdb`` next to the
    run metadata plus per-design ranked/intermediate CIFs such as
    ``design_08.cif`` or ``rank01_design_08.cif``. The root file is useful
    provenance, but it is not an individual binder candidate for official AF2
    score-conversion.
    """
    if path is None:
        return False
    try:
        return Path(path).name in _BOLTZGEN_AGGREGATE_NAMES
    except TypeError:
        return False


def is_boltzgen_aggregate_record(record: Any) -> bool:
    """Return True when a BoltzGen ResultRecord is aggregate provenance only."""
    if str(getattr(record, "backend_family", "") or "") != "boltzgen":
        return False
    bins = getattr(record, "bins", None) or {}
    artifacts = getattr(record, "artifacts", None) or {}
    path = (
        artifacts.get("pdb_path")
        or artifacts.get("cif_path")
        or artifacts.get("pdb_dir")
        or ""
    )
    has_rank = any(
        str(bins.get(k, "") or "").strip()
        for k in (
            "boltzgen_final_rank",
            "boltzgen_secondary_rank",
            "boltzgen_max_rank",
        )
    )
    if is_boltzgen_aggregate_artifact_path(path) and not has_rank:
        return True
    design_id = str(bins.get("boltzgen_design_id", "") or "").strip()
    if design_id == "design" and not has_rank:
        return True
    return False


def is_score_convertible_diagnostic_record(record: Any) -> bool:
    """Shared precondition for artifacts that may enter canonical AF2 scoring."""
    return not is_boltzgen_aggregate_record(record)


def _metric_float(record: Any, key: str) -> float | None:
    try:
        value = (getattr(record, "metrics", None) or {}).get(key)
        if value is None:
            return None
        out = float(value)
        return out if out == out and out not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _bin_float(record: Any, key: str) -> float | None:
    try:
        value = (getattr(record, "bins", None) or {}).get(key)
        if value is None:
            return None
        out = float(value)
        return out if out == out and out not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def bindcraft_native_strict_like(record: Any) -> bool:
    """Native BindCraft signal worth official canonical AF2 scoring.

    This is not strict success. It only protects score-conversion priority until
    the official AF2 gate has produced pLDDT/iPAE/scRMSD and Foldseek SU.
    """
    plddt = _metric_float(record, "bindcraft_native_pLDDT")
    ipae = _metric_float(record, "bindcraft_native_iPAE")
    rmsd = _metric_float(record, "bindcraft_native_binder_RMSD")
    return (
        plddt is not None and plddt >= 90.0
        and ipae is not None and ipae <= (7.0 / 31.0)
        and rmsd is not None and rmsd < 1.5
    )


def bindcraft_accepted_artifact(record: Any) -> bool:
    """BindCraft accepted output that still needs official AF2 scoring."""
    bins = getattr(record, "bins", None) or {}
    status = str(bins.get("bindcraft_filter_status", "") or "").strip().lower()
    if status == "accepted":
        return True
    accepted = str(bins.get("bindcraft_accepted_artifact", "") or "").strip().lower()
    if accepted in {"1", "true", "yes", "accepted"}:
        return True
    artifacts = getattr(record, "artifacts", None) or {}
    path = str(artifacts.get("pdb_path") or artifacts.get("pdb_dir") or "")
    try:
        parts = {p.lower() for p in Path(path).parts}
    except TypeError:
        parts = set()
    return "accepted" in parts


def bindcraft_native_near_pass(record: Any) -> bool:
    """Native BindCraft near-pass signal worth prompt score-conversion.

    This deliberately uses only BindCraft-native metrics as an ordering signal.
    It does not mint strict/SU credit; canonical AF2 scoring remains the only
    official success gate.
    """
    if bindcraft_native_strict_like(record):
        return True
    plddt = _metric_float(record, "bindcraft_native_pLDDT")
    ipae = _metric_float(record, "bindcraft_native_iPAE")
    rmsd = _metric_float(record, "bindcraft_native_binder_RMSD")
    if plddt is None or ipae is None or rmsd is None:
        return False
    p_def = max(0.0, 90.0 - plddt)
    i_def = max(0.0, ipae - (7.0 / 31.0))
    r_def = max(0.0, rmsd - 1.5)
    hard_fails = sum(
        1
        for axis, deficit in (
            ("pLDDT", p_def),
            ("iPAE", i_def),
            ("binder_scRMSD", r_def),
        )
        if deficit > NEAR_PASS_MARGINS[axis]
    )
    return hard_fails <= 1


def complexa_native_strict_like(record: Any) -> bool:
    """Native Complexa signal worth official canonical AF2 scoring."""
    plddt = _metric_float(record, "complexa_native_pLDDT")
    ipae = _metric_float(record, "complexa_native_iPAE")
    rmsd = _metric_float(record, "complexa_native_binder_scRMSD")
    return (
        plddt is not None and plddt >= 90.0
        and ipae is not None and ipae <= (7.0 / 31.0)
        and rmsd is not None and rmsd < 1.5
    )


def boltzgen_proxy_promising(record: Any) -> bool:
    """BoltzGen-native proxy signal for prioritizing official AF2 scoring.

    These metrics never mint SU. They only prevent a scientifically promising
    unscored BoltzGen artifact from being treated like bulk low-value backlog.
    """
    design_iptm = _bin_float(record, "boltzgen_design_iptm")
    target_iptm = _bin_float(record, "boltzgen_design_to_target_iptm")
    iiptm = _bin_float(record, "boltzgen_design_iiptm")
    min_pae = _bin_float(record, "boltzgen_min_design_to_target_pae")
    confidence = _bin_float(record, "boltzgen_structure_confidence")
    rmsd_refolded = _bin_float(record, "boltzgen_native_rmsd_refolded")

    signals = 0
    if design_iptm is not None and design_iptm >= 0.80:
        signals += 1
    if target_iptm is not None and target_iptm >= 0.65:
        signals += 1
    if iiptm is not None and iiptm >= 0.65:
        signals += 1
    if min_pae is not None and min_pae <= 7.0:
        signals += 1
    if confidence is not None and confidence >= 0.75:
        signals += 1
    if rmsd_refolded is not None and rmsd_refolded < 2.0:
        signals += 1
    return signals >= 2


def proteinmpnn_proxy_promising(record: Any) -> bool:
    """ProteinMPNN-native proxy signal for prioritizing official AF2 scoring."""
    global_score = _bin_float(record, "mpnn_global_score")
    local_score = _bin_float(record, "mpnn_score")
    seq_recovery = _bin_float(record, "mpnn_seq_recovery")
    score_ok = (
        (global_score is not None and global_score <= 1.20)
        or (local_score is not None and local_score <= 1.20)
    )
    recovery_ok = seq_recovery is None or 0.15 <= seq_recovery <= 0.85
    return bool(score_ok and recovery_ok)


def native_strict_like_pending_score_conversion(record: Any) -> bool:
    """Native/reward-gate near-official pass that still needs canonical scoring."""
    family = str(getattr(record, "backend_family", "") or "")
    if family == "bindcraft":
        return bindcraft_native_strict_like(record)
    if family.startswith("complexa_"):
        return complexa_native_strict_like(record)
    return False


def proxy_promising_pending_score_conversion(record: Any) -> bool:
    """Weaker native-model signal used for queue ordering, not strict credit."""
    family = str(getattr(record, "backend_family", "") or "")
    if family == "bindcraft":
        return bindcraft_accepted_artifact(record) or bindcraft_native_near_pass(record)
    if family == "boltzgen":
        return boltzgen_proxy_promising(record)
    if family == "proteinmpnn_redesign":
        return proteinmpnn_proxy_promising(record)
    return False


def high_value_pending_score_conversion(record: Any) -> bool:
    """Return True for unscored artifacts that should protect awaiting status."""
    return native_strict_like_pending_score_conversion(record) or proxy_promising_pending_score_conversion(record)
