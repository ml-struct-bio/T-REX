"""Greedy Pareto panel selection and PanelValue@K.

eps_quality_lift controls the minimum quality gain for partial credit within a redundant
bin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schemas import PanelSelection, ResultRecord
from .success_criteria import (
    NEAR_PASS_MARGINS,
    STRICT_SUCCESS,
    is_near_miss,
    is_strict_success,
)


@dataclass(frozen=True)
class PanelConfig:
    K: int = 8
    eps_quality_lift: float = 0.05
    eps_quality_component: float = 0.01
    bin_keys: tuple[str, ...] = ("foldseek", "contact", "epitope", "sequence")


@dataclass(frozen=True)
class ProductionPanelConfig:
    """Archive-level selector for a real wet-lab top-N panel.

    Unlike ``PanelConfig`` / ``select_panel`` this does not require parser-set
    ``panel_ready`` flags. Production parsers intentionally leave those false
    because the calibrated panel rubric is derived here from strict Complexa
    gates, dedup bins, and secondary panel-quality evidence.
    """

    K: int = 8
    near_miss_K: int = 4
    require_structure_artifact: bool = True
    require_trusted_structure_bin: bool = True
    allow_structural_duplicates_when_underfilled: bool = False
    calibration_ref: str = "complexa_strict_v7_panel_selector"
    bin_keys: tuple[str, ...] = (
        "structure",
        "sequence",
        "contact",
        "epitope",
        "family",
    )


def geometric_mean(values: list[float], eps: float = 0.01) -> float:
    if not values:
        return 0.0
    floored = [max(eps, v) for v in values]
    log_sum = sum(math.log(v) for v in floored)
    return math.exp(log_sum / len(floored))


def quality(
    candidate: ResultRecord,
    *,
    components: tuple[str, ...] = (
        "calibrated_structure_confidence",
        "calibrated_interface_confidence",
        "hotspot_contact_satisfaction",
        "clash_developability_score",
        "multi_verifier_agreement",
    ),
    eps: float = 0.01,
) -> float:
    if not candidate.panel_ready:
        return 0.0
    raw = candidate.metrics_calibrated
    values = [raw[k] for k in components if k in raw and raw[k] is not None]
    if not values:
        return 0.0
    return geometric_mean(values, eps=eps)


def _has_structure_artifact(candidate: ResultRecord) -> bool:
    art = candidate.artifacts or {}
    p = art.get("pdb_path")
    if p and Path(p).is_file():
        return True
    d = art.get("pdb_dir")
    if d and Path(d).is_dir():
        d_path = Path(d)
        return (
            any(d_path.glob("*.pdb"))
            or any(d_path.glob("*.cif"))
            or any(d_path.glob("*.mmcif"))
        )
    return False


def strict_margin_units(candidate: ResultRecord) -> dict[str, float]:
    """Positive strict-pass margin in near-pass units; missing axes score 0."""
    m = candidate.metrics or {}
    out: dict[str, float] = {}
    for axis, (thr, direction) in STRICT_SUCCESS.items():
        v = m.get(axis)
        if not isinstance(v, (int, float)):
            out[axis] = 0.0
            continue
        if direction == "increase":
            raw = float(v) - thr
        else:
            raw = thr - float(v)
        out[axis] = max(0.0, raw / (NEAR_PASS_MARGINS.get(axis, 1.0) or 1.0))
    return out


def strict_margin_quality(candidate: ResultRecord) -> float:
    """Comparable quality from the three canonical strict axes only."""
    margins = strict_margin_units(candidate)
    return sum(min(v, 3.0) for v in margins.values()) / (3.0 * len(STRICT_SUCCESS))


def production_quality(candidate: ResultRecord) -> float:
    """Transparent wet-lab panel rank score for strict-success records.

    The base is the strict-margin score on the same three Complexa axes used
    for success. Secondary terms are small bonuses only when already present;
    they must not convert a non-strict candidate into a selected success.
    """
    # /(3.0*len(STRICT_SUCCESS)) tracks the axis set automatically (each axis is
    # clamped to min(v,3.0)); hardcoding /9.0 silently mis-normalizes if a 4th
    # strict axis is ever added to success_criteria.STRICT_SUCCESS.
    base = strict_margin_quality(candidate)
    calibrated = candidate.metrics_calibrated or {}
    metrics = candidate.metrics or {}

    def _optional_score(*keys: str) -> float | None:
        for key in keys:
            v = calibrated.get(key, metrics.get(key))
            if isinstance(v, (int, float)):
                return max(0.0, min(1.0, float(v)))
        return None

    bonuses = []
    for keys in (
        ("clash_developability_score", "developability_score"),
        ("hotspot_contact_satisfaction", "contact_satisfaction"),
        ("multi_verifier_agreement",),
    ):
        v = _optional_score(*keys)
        if v is not None:
            bonuses.append(v)
    bonus = (sum(bonuses) / len(bonuses)) if bonuses else 0.0
    # Keep strict margins dominant. Optional panel metrics can reorder close
    # strict winners but cannot overwhelm weak strict margins.
    return 0.85 * base + 0.15 * bonus


def production_bins(candidate: ResultRecord) -> dict[str, str]:
    b = candidate.bins or {}
    out = {
        # `foldseek_su` is the only trusted production structural bin. The
        # fallback remains available for explicitly provisional/near-miss views,
        # but ProductionPanelConfig requires the trusted key by default.
        "structure": (
            b.get("foldseek_su") or b.get("refilter_source") or candidate.result_id
        ),
        # Resolve the generating family so evaluation records preserve family diversity.
        "family": b.get("refilter_source_family") or candidate.backend_family,
    }
    seq_bin = b.get("sequence_su") or b.get("sequence") or b.get("seq")
    if seq_bin:
        out["sequence"] = seq_bin
    for key in ("contact", "epitope"):
        if b.get(key):
            out[key] = b[key]
    return out


def _hard_gate_failure(
    candidate: ResultRecord, cfg: ProductionPanelConfig
) -> str | None:
    if candidate.exit_status != "ok":
        return f"exit_status:{candidate.exit_status}"
    if candidate.bins.get("output_chain_identity") == "unresolved":
        return "unresolved_output_chain_identity"
    if not is_strict_success(candidate.metrics):
        return "not_strict_success"
    if cfg.require_structure_artifact and not _has_structure_artifact(candidate):
        return "missing_structure_artifact"
    if cfg.require_trusted_structure_bin and not (candidate.bins or {}).get(
        "foldseek_su"
    ):
        return "missing_trusted_structure_bin"
    return None


def _summarize_failures(failures: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for reason in failures:
        counts[reason] = counts.get(reason, 0) + 1
    return [f"{reason}:{n}" for reason, n in sorted(counts.items())]


def _new_bin_count(
    candidate: ResultRecord,
    selected: list[ResultRecord],
    keys: tuple[str, ...],
) -> int:
    bins = production_bins(candidate)
    seen: dict[str, set[str]] = {k: set() for k in keys}
    for s in selected:
        for k, v in production_bins(s).items():
            if k in seen:
                seen[k].add(v)
    return sum(1 for k, v in bins.items() if k in seen and v not in seen[k])


def select_production_panel(
    pool: list[ResultRecord],
    *,
    cfg: ProductionPanelConfig | None = None,
    panel_id: str = "production_panel_0001",
) -> PanelSelection:
    """Select strict-success wet-lab candidates from the whole archive.

    Primary objective remains SU/GPU-h; this deterministic selector is the
    panel-facing artifact: high-quality strict candidates, structurally
    deduped first, with sequence/contact/epitope/source diversity as tie-breaks.
    """
    cfg = cfg or ProductionPanelConfig()
    failures: list[str] = []
    eligible: list[ResultRecord] = []
    for r in pool:
        reason = _hard_gate_failure(r, cfg)
        if reason is None:
            eligible.append(r)
        else:
            failures.append(reason)

    eligible.sort(key=lambda r: (-production_quality(r), r.result_id))
    selected: list[ResultRecord] = []
    history: list[tuple[str, float, float]] = []
    pareto_audit: list[dict] = []
    selected_structure_bins: set[str] = set()

    while len(selected) < cfg.K and eligible:
        best: ResultRecord | None = None
        best_score = float("-inf")
        best_dweight = 0.0
        for cand in eligible:
            bins = production_bins(cand)
            struct_bin = bins["structure"]
            structural_new = struct_bin not in selected_structure_bins
            if (
                not structural_new
                and not cfg.allow_structural_duplicates_when_underfilled
            ):
                continue
            q = production_quality(cand)
            new_bins = _new_bin_count(cand, selected, cfg.bin_keys)
            dweight = 1.0 if structural_new else 0.25
            score = q * dweight + 0.03 * new_bins
            if score > best_score:
                best = cand
                best_score = score
                best_dweight = dweight
        if best is None:
            break
        selected.append(best)
        selected_structure_bins.add(production_bins(best)["structure"])
        q = production_quality(best)
        history.append((best.result_id, best_dweight, q))
        pareto_audit.append(
            {
                "step": len(selected),
                "candidate_id": best.result_id,
                "quality": q,
                "quality_margins": strict_margin_units(best),
                "dweight": best_dweight,
                "new_bins_covered": _new_bin_count(best, selected[:-1], cfg.bin_keys),
                "bins": production_bins(best),
                "backend_family": best.backend_family,
            }
        )
        eligible = [r for r in eligible if r.result_id != best.result_id]

    diversity_bins = {
        k: len({production_bins(s)[k] for s in selected if k in production_bins(s)})
        for k in cfg.bin_keys
    }
    return PanelSelection(
        panel_id=panel_id,
        K=cfg.K,
        selected_ids=[s.result_id for s in selected],
        pareto_audit=pareto_audit,
        diversity_bins=diversity_bins,
        hard_gate_failures=_summarize_failures(failures),
        calibration_ref=cfg.calibration_ref,
        panel_value=sum(q * w for (_, w, q) in history),
    )


def best_near_miss_backups(
    pool: list[ResultRecord],
    *,
    K: int = 4,
    require_structure_artifact: bool = True,
) -> list[str]:
    """Return best near-miss backup candidates for operator review.

    These are not counted as strict successes or panel value; they are a
    separate rescue/back-up list when the strict panel has fewer than K entries.
    """
    cands = [
        r
        for r in pool
        if r.exit_status == "ok"
        and is_near_miss(r.metrics)
        and r.bins.get("output_chain_identity") != "unresolved"
        and (not require_structure_artifact or _has_structure_artifact(r))
    ]
    cands.sort(key=lambda r: (-production_quality(r), r.result_id))
    out: list[str] = []
    seen_struct: set[str] = set()
    for r in cands:
        b = production_bins(r)["structure"]
        if b in seen_struct:
            continue
        seen_struct.add(b)
        out.append(r.result_id)
        if len(out) >= K:
            break
    return out


def candidate_bins(candidate: ResultRecord, keys: tuple[str, ...]) -> dict[str, str]:
    return {k: candidate.bins[k] for k in keys if k in candidate.bins}


def new_bins_covered(
    candidate: ResultRecord, selected: list[ResultRecord], keys: tuple[str, ...]
) -> int:
    """Return how many bin-types this candidate uniquely covers."""
    if not selected:
        return sum(1 for k in keys if k in candidate.bins)
    seen: dict[str, set[str]] = {k: set() for k in keys}
    for s in selected:
        for k in keys:
            if k in s.bins:
                seen[k].add(s.bins[k])
    covered = 0
    for k in keys:
        if k in candidate.bins and candidate.bins[k] not in seen[k]:
            covered += 1
    return covered


def improves_best_in_existing_bin(
    candidate: ResultRecord,
    selected: list[ResultRecord],
    keys: tuple[str, ...],
    eps_quality_lift: float,
) -> bool:
    if not selected:
        return False
    cq = quality(candidate)
    for k in keys:
        if k not in candidate.bins:
            continue
        bin_val = candidate.bins[k]
        bucket = [s for s in selected if s.bins.get(k) == bin_val]
        if not bucket:
            continue
        best = max(quality(s) for s in bucket)
        if cq > best * (1.0 + eps_quality_lift):
            return True
    return False


def select_panel(
    pool: list[ResultRecord],
    *,
    cfg: PanelConfig | None = None,
    calibration_ref: str = "v6_3_replay",
    panel_id: str = "panel_0001",
) -> PanelSelection:
    cfg = cfg or PanelConfig()
    pool = [c for c in pool if c.panel_ready]
    # Deterministic ordering for tie-break stability
    pool = sorted(pool, key=lambda r: r.result_id)
    selected: list[ResultRecord] = []
    history: list[tuple[str, float, float]] = []  # (id, dweight, quality)
    pareto_audit: list[dict] = []

    while len(selected) < cfg.K and pool:
        best = None
        best_score = -1.0
        best_dweight = 0.0
        best_quality = 0.0
        for c in pool:
            cov = new_bins_covered(c, selected, cfg.bin_keys)
            if cov > 0:
                dw = 1.0
            elif improves_best_in_existing_bin(
                c, selected, cfg.bin_keys, cfg.eps_quality_lift
            ):
                dw = 0.3
            else:
                dw = 0.0
            q = quality(c, eps=cfg.eps_quality_component)
            score = q * dw
            if score > best_score:
                best = c
                best_score = score
                best_dweight = dw
                best_quality = q
        if best is None or best_score <= 0:
            break
        selected.append(best)
        history.append((best.result_id, best_dweight, best_quality))
        pareto_audit.append(
            {
                "step": len(selected),
                "candidate_id": best.result_id,
                "quality": best_quality,
                "dweight": best_dweight,
                "new_bins_covered": new_bins_covered(best, selected[:-1], cfg.bin_keys),
                "bins": candidate_bins(best, cfg.bin_keys),
            }
        )
        pool = [c for c in pool if c.result_id != best.result_id]

    panel_value = sum(q * w for (_, w, q) in history)
    diversity_bins = {
        k: len({s.bins[k] for s in selected if k in s.bins}) for k in cfg.bin_keys
    }
    return PanelSelection(
        panel_id=panel_id,
        K=cfg.K,
        selected_ids=[s.result_id for s in selected],
        pareto_audit=pareto_audit,
        diversity_bins=diversity_bins,
        hard_gate_failures=[],
        calibration_ref=calibration_ref,
        panel_value=panel_value,
    )


def diversity_at_K(
    panel: PanelSelection, available_bins_per_type: dict[str, int]
) -> float:
    """Mean of normalized unique bin counts across bin types.

    Each component: unique_count / min(K, available_bins_per_type[t]).
    """
    keys = list(panel.diversity_bins.keys())
    if not keys:
        return 0.0
    components = []
    for k in keys:
        denom = min(panel.K, available_bins_per_type.get(k, panel.K))
        if denom <= 0:
            continue
        components.append(panel.diversity_bins[k] / denom)
    return sum(components) / len(components) if components else 0.0
