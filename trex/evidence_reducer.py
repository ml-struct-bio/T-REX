"""Construct EvidenceSummary records and classify campaign state.

Deterministic rules summarize archived results, lineage, unfinished work, and
resources for the Planner and Supervisor.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION
from .dedup_trust import near_miss_dedup_trusted as _near_miss_dedup_trusted
from .refilter_roles import (
    CANONICAL_SCORE_CONVERSION,
    PARENT_MODEL_REFOLD,
    infer_refilter_role,
)
from .success_criteria import (
    NEAR_PASS_MARGINS,
    STRICT_SUCCESS,
    is_joint_fail,
    is_near_miss,
    is_strict_success,
)
from .score_conversion import high_value_pending_score_conversion
from .panel import strict_margin_quality, strict_margin_units
from .schemas import (
    ActionCandidate,
    AxisStat,
    EvidenceSummary,
    Example,
    Exemplar,
    JointPatternCount,
    LLMHealthSummary,
    MethodHealthSummary,
    Recipe,
    RouteValueSummary,
    RecipeClass,
    ResultRecord,
    RouteHealthSummary,
    StateLabel,
)

from .evidence.attribution import (
    _infer_refilter_role_for_record,
    is_canonical_su_record as _is_canonical_su_record,
    _lineage_parent_records,
    _refilter_role_for_record,
    _score_conversion_lineage_records,
    _score_conversion_parent_record,
    _su_key,
    resolve_generating_family,
    resolve_generating_record,
)
from .evidence.route_identity import (
    _route_identity_for_record,
    canonical_config_signature,
    route_component_key,
    route_role_label,
)
from .evidence.route_values import build_route_value_summaries


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateClassifierConfig:
    eps_gpu_h: float = 0.5
    # Use cumulative worker compute for the cold-start gate; record-count windows can
    # contain many inexpensive outputs.
    min_cumulative_gpu_h: float = 1.0
    # Minimum recent SU per worker GPU-hour for productive classification.
    productive_su_rate: float = 0.04
    productive_max_dup: float = 0.60
    rescue_min_near_miss: int = 3
    rescue_axis_concentration: float = 0.50
    stalled_max_near_miss: int = 2
    stalled_top_bin_share: float = 0.75
    # Worker GPU-hours without a new SU required for deep-stall classification.
    deep_stall_gpu_h: float = 12.0
    # Worker GPU-hours without a new SU required to escalate near-miss-rich campaigns to deep stall.
    rescue_deep_stall_gpu_h: float = 12.0
    # Detect abundant qualified designs with little structural novelty. Productive
    # duplication still yields new SUs; near-miss enrichment concerns failed qualification.
    strict_duplicate_min_strict_total: int = 40
    strict_duplicate_min_su_total: int = 2
    strict_duplicate_min_strict_per_su: float = 8.0
    strict_duplicate_max_total_su_per_gpu_h: float = 0.30
    strict_duplicate_max_recent_su_delta: int = 1
    strict_duplicate_max_tm08_split_ratio: float = 1.30
    # Exact route/config duplicate-pressure evidence. These numeric cutoffs are
    # NOT launch bans and are never meant to kill a route by themselves: the
    # route-status logic combines them with recent new-SU decay, low total
    # SU/GPU-h, or no pending score-conversion evidence. A cheap duplicate-heavy
    # route can keep "promote" only when it is still buying new official SU.
    route_duplicate_min_strict_count: int = 8
    route_duplicate_min_strict_per_su: float = 8.0
    route_duplicate_min_duplicate_fraction: float = 0.80
    route_diversify_min_strict_per_su: float = 4.0
    route_diversify_min_duplicate_fraction: float = 0.55
    route_promote_max_strict_per_su_without_recent_su: float = 8.0
    route_promote_max_duplicate_fraction_without_recent_su: float = 0.70
    route_zero_su_defer_gpu_h: float = 3.0
    # Cheap direct-scored routes can generate many fully scored samples before
    # accumulating much worker GPU-h. Do not leave an exact route/config marked
    # "under_tested" forever just because it is cheap: two Complexa batches with
    # no official SU and no trusted near-miss is enough to cool down that exact
    # config. This applies to exact route rows only; family rollups stay alive so
    # a bad reward-weight retry does not become "complexa_beam is bad".
    route_zero_su_defer_completions: int = 32
    # A lifetime-success route that is dry in the recent window should not stay
    # healthy/promoted forever. This is a soft label demotion only; the LLM can
    # still propose the route manually when other evidence supports it.
    route_stale_recent_gpu_h: float = 1.0
    # Cost-normalized recent route window, separate from the last-N-record view.
    # This window is the suffix of the archive spanning roughly
    # this many worker GPU-hours, so high-throughput cheap routes and slower
    # diagnostic routes are compared on a common budget scale.
    route_gpu_recent_window_h: float = 3.0
    # Secondary delayed-feedback window. It protects slow routes from premature
    # cooldown when the short window is empty, but it must not replace the short
    # gpu_recent signal as the primary exploit-rank score.
    route_gpu_medium_window_h: float = 6.0
    # If a duplicate-heavy route GPU-window marginal rate falls below this
    # fraction of its lifetime rate, mark it as dry_duplicate. Duplicate metrics
    # alone never trigger action changes.
    route_marginal_decay_fraction: float = 0.35


@dataclass(frozen=True)
class ReducerConfig:
    """All threshold + margin fields default to `success_criteria` constants.
    Override only at run-config level for ablations; do not change inline.
    """
    state: StateClassifierConfig = StateClassifierConfig()
    # near-pass margins (one source: success_criteria.NEAR_PASS_MARGINS)
    near_pass_margin_pLDDT: float = NEAR_PASS_MARGINS["pLDDT"]
    near_pass_margin_iPAE: float = NEAR_PASS_MARGINS["iPAE"]
    near_pass_margin_binder_scRMSD: float = NEAR_PASS_MARGINS["binder_scRMSD"]
    # strict_success thresholds (one source: success_criteria.STRICT_SUCCESS)
    pLDDT_threshold: float = STRICT_SUCCESS["pLDDT"][0]            # 90.0
    iPAE_threshold: float = STRICT_SUCCESS["iPAE"][0]              # 7/31 = 0.2258..
    binder_scRMSD_threshold: float = STRICT_SUCCESS["binder_scRMSD"][0]  # 1.5
    max_examples: int = 6
    max_exemplars_best: int = 5      # best-K proven binders (full setup + metrics)
    max_exemplars_near: int = 5  # Balance diagnostics across each active family recent records so a dominant family
    # does not erase other evidence.
    diagnostic_window_per_family: int = 40


# ---------------------------------------------------------------------------
# Per-axis labeling
# ---------------------------------------------------------------------------


def label_axis(
    value: float | None,
    threshold: float,
    direction: str,
    near_pass_margin: float,
) -> tuple[str, float]:
    """Return (label, deficit). label ∈ {"pass","near_pass","fail","missing"}.

    deficit is signed-positive: 0 if pass, otherwise distance from threshold.
    """
    # Treat None AND non-finite (NaN/±inf) as missing. Without the finite guard
    # a NaN metric slips through `max(0.0, thr - nan) -> 0.0 -> "pass"` and is
    # silently counted as a PASS, inflating pass_count. Production parsers
    # (complexa/bindcraft/boltzgen _safe_float) reject NaN today, so this is a
    # latent-defense guard against any future parser that forgets to sanitize.
    if value is None or not math.isfinite(value):
        return "missing", 0.0
    if direction == "increase":  # higher-is-better (pLDDT)
        d = max(0.0, threshold - value)
        if d <= 0:
            return "pass", 0.0
        if d <= near_pass_margin:
            return "near_pass", d
        return "fail", d
    # direction == "decrease" → lower-is-better (iPAE, scRMSD)
    d = max(0.0, value - threshold)
    if d <= 0:
        return "pass", 0.0
    if d <= near_pass_margin:
        return "near_pass", d
    return "fail", d


def _axis_metric_for(result: ResultRecord, axis: str) -> float | None:
    return result.metrics.get(axis)


def _axis_threshold(cfg: ReducerConfig, axis: str) -> tuple[float, str, float]:
    """Return (threshold, direction, near_pass_margin) for axis."""
    if axis == "pLDDT":
        return cfg.pLDDT_threshold, "increase", cfg.near_pass_margin_pLDDT
    if axis == "iPAE":
        return cfg.iPAE_threshold, "decrease", cfg.near_pass_margin_iPAE
    if axis == "binder_scRMSD":
        return (
            cfg.binder_scRMSD_threshold,
            "decrease",
            cfg.near_pass_margin_binder_scRMSD,
        )
    raise ValueError(f"unknown axis: {axis}")


def build_axis_stat(results: Iterable[ResultRecord], axis: str, cfg: ReducerConfig) -> AxisStat:
    thr, direction, margin = _axis_threshold(cfg, axis)
    p = nf = f = 0
    raws: list[float] = []
    deficits: list[float] = []
    for r in results:
        v = _axis_metric_for(r, axis)
        lbl, d = label_axis(v, thr, direction, margin)
        if lbl == "missing":
            continue
        raws.append(v)  # type: ignore[arg-type]
        deficits.append(d)
        if lbl == "pass":
            p += 1
        elif lbl == "near_pass":
            nf += 1
        else:
            f += 1
    n = p + nf + f
    return AxisStat(
        pass_count=p,
        near_pass_count=nf,
        fail_count=f,
        median_raw=statistics.median(raws) if raws else None,
        median_calibrated=statistics.median(raws) if raws else None,  # No additional calibration is applied here.
        median_deficit=statistics.median(deficits) if deficits else None,
        calibration_status="provisional" if n > 0 else "uncalibrated",
        n=n,
    )


# Each diagnostic threshold specifies tool acceptance, quality, direction, and near-pass
# margin. A missing acceptance threshold denotes a quality-only measurement. These
# advisory thresholds never determine qualification or SU credit.
DIAGNOSTIC_AXIS_THRESHOLDS: dict[str, tuple[float | None, float, str, float]] = {
    # ---- BindCraft-native (8); src: subgit/BindCraft/settings_filters/
    #      default_filters.json (pass) + 893-design pool p25 (quality) ----
    # interface_dG — Rosetta interface energy (REU), lower better.
    "interface_dG":            (0.0,   -56.0,  "decrease", 9.0),
    # shape_complementarity — Lawrence-Colman SC (0-1), higher better.
    # quality 0.65 = Lawrence & Colman 1993 well-packed band (≈ pool median).
    "shape_complementarity":   (0.60,   0.65,  "increase", 0.04),
    # interface_hbonds — count, higher better.
    "interface_hbonds":        (3.0,    5.0,   "increase", 2.0),
    # interface_unsat_hbonds — buried unsatisfied count, lower better.
    "interface_unsat_hbonds":  (4.0,    2.0,   "decrease", 1.0),
    # Interface dSASA in square angstroms; higher is better. This diagnostic uses a
    # quality threshold without a separate acceptance gate.
    "buried_sasa":             (None,   1650.0,"increase", 330.0),
    # binder_pLDDT_avg — BindCraft Average_Binder_pLDDT on a 0-1 scale (its own
    # key; NOT the 0-100 strict pLDDT). pass 0.80 = default_filters floor.
    "binder_pLDDT_avg":        (0.80,   0.90,  "increase", 0.03),
    # hotspot_rmsd — Å, lower better; coverage-gated to hotspot targets.
    "hotspot_rmsd":            (6.0,    2.0,   "decrease", 0.6),
    # Mean binder pTM; a quality-only diagnostic on a zero-to-one scale.
    "binder_pTM_avg":          (None,   0.79,  "increase", 0.05),

    # Complexa interface-confidence diagnostics use their own measurement
    # thresholds.
    "ipTM":                    (0.50,   0.75,  "increase", 0.27),
    # avg_ipsae / max_ipsae — Dunbrack ipSAE (0-1, distinct from ipTM), higher
    # better. Not in the strict gate → pass=None; quality = strict p25.
    "avg_ipsae":               (None,   0.55,  "increase", 0.25),
    "max_ipsae":               (None,   0.60,  "increase", 0.26),
    # Minimum normalized interface PAE uses the qualification scale and a
    # diagnostic-pool margin.
    "min_ipae":                (STRICT_SUCCESS["iPAE"][0], 0.07, "decrease", 0.21),

    # BoltzGen diagnostics come from bins and use backend-specific scales.
    "design_to_target_iptm":   (0.50,   0.60,  "increase", 0.09),
    "design_iiptm":            (0.50,   0.60,  "increase", 0.09),
    # design_ptm — global pTM (0-1), higher better; pass≈pool p25 (mis-fold guard).
    "design_ptm":              (0.70,   0.80,  "increase", 0.05),
    # Raw interface PAE uses angstroms and must not be compared with normalized iPAE.
    "min_design_to_target_pae":(15.0,   10.0,  "decrease", 5.0),
}
# Axes whose value lives in r.bins["boltzgen_<axis>"] rather than r.metrics.
_BOLTZGEN_BIN_AXES = frozenset({
    "design_to_target_iptm", "design_iiptm", "design_ptm", "min_design_to_target_pae",
})

# MARGIN_POLICY: near_pass_margin = ~1 std-dev of the metric across the
# DIAGNOSTIC POOL (all records carrying the axis), rounded to a metric-friendly
# value. The strict-success subset std collapses (e.g. min_ipae strict 0.011 vs
# pool 0.21) and would make near-miss never fire, so the pool std is the
# meaningful spread the LLM sees. Two bounded exceptions where the raw pool std
# is pathological: fape (fail-tail-dominated) and design_ptm (mis-fold tail) use
# a robust within-good-designs std instead — neither is currently an emitted axis.

DIAGNOSTIC_COVERAGE_MIN = 0.25  # Require both coverage and a minimum observation count before reporting an auxiliary
# diagnostic.
DIAGNOSTIC_MIN_N = 3


# Map diagnostic axes to available remediation settings, or None for corroborating
# evidence. Shared by prompt guidance and actionable-axis selection.
DIAGNOSTIC_AXIS_REMEDIATION: dict[str, str | None] = {
    # List only settings supported by both the registry and executor. Canonical score
    # conversion is automatic; parent_model_refold is an optional proposed action.
    "ipTM":                     "bindcraft.weights_iptm↑ OR complexa_beam.sc_scale_noise↑ OR complexa_beam.reward_i_ptm_weight↑ OR complexa_best_of_n.sc_scale_noise↑ OR complexa_best_of_n.reward_i_ptm_weight↑ OR complexa_fk_steering.sc_scale_noise↑ OR complexa_fk_steering.reward_i_ptm_weight↑ OR complexa_fk_steering.temperature↓ OR complexa_mcts.sc_scale_noise↑ OR complexa_mcts.reward_i_ptm_weight↑",
    "min_ipae":                 "bindcraft.weights_pae_inter↑ OR complexa_beam.sc_scale_noise↑ OR complexa_beam.refinement_algorithm=sequence_hallucination OR complexa_beam.reward_i_pae_weight↓ OR complexa_beam.reward_min_ipae_weight↓ OR complexa_best_of_n.sc_scale_noise↑ OR complexa_best_of_n.reward_i_pae_weight↓ OR complexa_best_of_n.reward_min_ipae_weight↓ OR complexa_fk_steering.sc_scale_noise↑ OR complexa_fk_steering.reward_i_pae_weight↓ OR complexa_fk_steering.reward_min_ipae_weight↓ OR complexa_mcts.sc_scale_noise↑ OR complexa_mcts.reward_i_pae_weight↓ OR complexa_mcts.reward_min_ipae_weight↓",
    "binder_pLDDT_avg":         "bindcraft.weights_plddt↑ OR complexa_beam.reward_plddt_weight↑ OR complexa_best_of_n.reward_plddt_weight↑ OR complexa_fk_steering.reward_plddt_weight↑ OR complexa_mcts.reward_plddt_weight↑",
    "buried_sasa":              "bindcraft.target_length_min↑ OR bindcraft.target_length_max↑",
    "design_to_target_iptm":    "boltzgen.noise_scale OR boltzgen.num_designs",
    "avg_ipsae":                "complexa_beam.reward_avg_ipsae_weight↑ OR complexa_best_of_n.reward_avg_ipsae_weight↑ OR complexa_fk_steering.reward_avg_ipsae_weight↑ OR complexa_mcts.reward_avg_ipsae_weight↑",
    "max_ipsae":                "complexa_beam.reward_max_ipsae_weight↑ OR complexa_best_of_n.reward_max_ipsae_weight↑ OR complexa_fk_steering.reward_max_ipsae_weight↑ OR complexa_mcts.reward_max_ipsae_weight↑",
    # corroboration-only: no direct config_delta lever any executor reads. Cite to
    # explain the failure mode, then act through a related real lever above. These
    # are NEVER surfaced as a per-design blocker (worst_actionable_diagnostic_axis
    # skips None-lever axes), so the LLM is never routed to a lever it cannot pull.
    "interface_dG":             None,  # Rosetta energy; bindcraft PAE weight is only indirect
    "design_iiptm":             None,  # BoltzGen-only confidence; the sole prior "lever" was the
    "design_ptm":               None,  # auto-chain conversion, which the LLM cannot itself propose
    "min_design_to_target_pae": None,  # — a dead lever; act via boltzgen.noise_scale/num_designs
    "shape_complementarity":    None,
    "interface_hbonds":         None,
    "interface_unsat_hbonds":   None,
    "hotspot_rmsd":             None,
    "binder_pTM_avg":           None,
}
CORROBORATION_ONLY_AXES = frozenset(
    a for a, lever in DIAGNOSTIC_AXIS_REMEDIATION.items() if lever is None
)


ADVISORY_SCORE_PREFIXES: dict[str, str] = {
    "boltzgen": "boltzgen_",
    "proteinmpnn_redesign": "mpnn_",
    "bindcraft": "bindcraft_",
}

ADVISORY_SCORE_DIRECTIONS: dict[str, str] = {
    "aggregate_score": "increase",
    "confidence_score": "increase",
    "ptm": "increase",
    "iptm": "increase",
    "complex_plddt": "increase",
    "complex_iplddt": "increase",
    "design_iptm": "increase",
    "seq_recovery": "increase",
    "native_pLDDT": "increase",
    "rank_iptm": "increase",
    "complex_pde": "decrease",
    "complex_ipde": "decrease",
    "native_iPAE": "decrease",
    "native_binder_RMSD": "decrease",
    "global_score": "decrease",
    "final_rank": "decrease",
    "secondary_rank": "decrease",
    "max_rank": "decrease",
}

ADVISORY_SCORE_INVERT_SCALES: dict[str, float] = {
    # Generator pDE/ipDE are PAE-like unbounded docking-error quantities. Preserve
    # raw values, but also expose an advisory 0-1 "quality-like" transform so
    # the LLM does not compare raw Å-scale errors directly against ipTM-style
    # confidence scores. This does NOT feed strict_success or SU.
    "complex_pde": 5.0,
    "complex_ipde": 5.0,
}


def _advisory_score_direction(score_key: str) -> str:
    direct = ADVISORY_SCORE_DIRECTIONS.get(score_key)
    if direct:
        return direct
    k = score_key.lower()
    if any(tok in k for tok in ("pde", "pae", "rmsd", "clash", "unsat")):
        return "decrease"
    return "increase"


def _advisory_quality_like(score_key: str, raw_value: float) -> float | None:
    scale = ADVISORY_SCORE_INVERT_SCALES.get(score_key)
    if scale is None or scale <= 0:
        return None
    return 1.0 / (1.0 + max(0.0, float(raw_value)) / scale)


def _record_advisory_scores(r: ResultRecord) -> dict[str, float]:
    prefix = ADVISORY_SCORE_PREFIXES.get(r.backend_family)
    if prefix is None or not r.bins:
        return {}
    out: dict[str, float] = {}
    for k, v in r.bins.items():
        if not k.startswith(prefix):
            continue
        try:
            out[k[len(prefix):]] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _summarize_advisory_scores(
    score_values: dict[str, list[float]],
    min_n: int = 1,
) -> dict[str, dict[str, Any]]:
    # Require the configured minimum count for advisory summaries; route feedback can
    # retain individual observations.
    out: dict[str, dict[str, Any]] = {}
    for key, vals in sorted(score_values.items()):
        if len(vals) < max(1, min_n):
            continue
        out[key] = {
            "median": float(statistics.median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "count": float(len(vals)),
            "direction": _advisory_score_direction(key),
        }
        q_vals = [
            q for q in (_advisory_quality_like(key, v) for v in vals)
            if q is not None
        ]
        if q_vals:
            scale = ADVISORY_SCORE_INVERT_SCALES[key]
            out[key].update({
                "quality_like_median": float(statistics.median(q_vals)),
                "quality_like_transform": f"1/(1+raw/{scale:g})",
                "quality_like_note": "advisory_only_not_success_score",
            })
    return out


def _diagnostic_axis_value(r: ResultRecord, axis: str) -> float | None:
    """Diagnostic-axis value for a record.

    BindCraft/Complexa axes live in ``r.metrics``; the 4 BoltzGen axes live in
    ``r.bins`` as ``"boltzgen_<axis>"`` strings — the boltzgen parser emits bins,
    not metrics, so its records stay ``metrics={}`` for the strict-success path.
    Reading both here gives BoltzGen a first-class diagnostic block WITHOUT
    touching the strict path (is_strict_success still sees empty metrics).
    """
    v = r.metrics.get(axis)
    if v is not None:
        return v
    if axis in _BOLTZGEN_BIN_AXES and r.bins:
        raw = r.bins.get(f"boltzgen_{axis}")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


def _diagnostic_axis_source_family(r: ResultRecord, axis: str) -> str | None:
    """Family that produced the diagnostic-axis carrier value.

    This is deliberately about the metric source, not SU credit attribution.
    For example, design_to_target_iptm is a BoltzGen-native bin, while min_ipae
    and ipTM are usually Complexa-native metrics. Surfacing this provenance
    prevents the LLM from reading a BoltzGen-only failure as a BindCraft knob.
    """
    v = r.metrics.get(axis)
    if v is not None:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = None
        if fv is not None and math.isfinite(fv):
            return str(r.backend_family or "unknown")
    if axis in _BOLTZGEN_BIN_AXES and r.bins:
        raw = r.bins.get(f"boltzgen_{axis}")
        try:
            fv = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            fv = None
        if fv is not None and math.isfinite(fv):
            return str(r.backend_family or "boltzgen")
    return None


def balanced_diagnostic_window(
    all_results: list[ResultRecord],
    recent: list[ResultRecord],
    k_per_family: int,
) -> list[ResultRecord]:
    """Records to aggregate DIAGNOSTICS over: each family's last `k_per_family`
    records UNION the recent state window.

    The flat last-N-records window makes a family's diagnostics VANISH whenever it
    is absent from the recent window (e.g. the controller specialized to another
    family for a few ticks, or one cheap family floods the window). This keeps
    every ACTIVE family's interface read present and recency-aware (last-K per
    family), well-balanced between past and recent, and bounded at
    ~n_families * k_per_family + |recent|. Order: per-family-recent first, then any
    remaining recent-window records, so de-dup is stable.
    """
    by_family: dict[str, list[ResultRecord]] = {}
    for r in all_results:
        by_family.setdefault(r.backend_family, []).append(r)
    seen: set[str] = set()
    out: list[ResultRecord] = []
    for rs in by_family.values():
        for r in rs[-max(1, k_per_family):]:
            if r.result_id not in seen:
                seen.add(r.result_id)
                out.append(r)
    for r in recent:
        if r.result_id not in seen:
            seen.add(r.result_id)
            out.append(r)
    return out


def worst_actionable_diagnostic_axis(r: ResultRecord) -> str | None:
    """The worst ACTIONABLE (levered) diagnostic axis for a record, vs its quality
    band, margin-normalized — or None if all levered diagnostics pass.

    Corroboration-only axes (DIAGNOSTIC_AXIS_REMEDIATION[axis] is None) are never
    returned, so a per-design exemplar's diagnostic blocker is always something the
    planner can actually remediate via the lever map. This is the diagnostic
    analogue of dominant_deficit_axis (which is strict-only), letting the LLM tie
    'this near-miss is blocked by ipTM=0.65' to a specific design + its lever.
    """
    worst_axis: str | None = None
    worst_norm = 0.0
    for axis, lever in DIAGNOSTIC_AXIS_REMEDIATION.items():
        if lever is None:
            continue
        v = _diagnostic_axis_value(r, axis)
        if v is None:
            continue
        _pass_thr, quality_thr, direction, margin = DIAGNOSTIC_AXIS_THRESHOLDS[axis]
        _lbl, d = label_axis(v, quality_thr, direction, margin)
        if d <= 0:  # passes its quality band → not a blocker
            continue
        norm = d / (margin or 1.0)
        if norm > worst_norm:
            worst_norm = norm
            worst_axis = axis
    return worst_axis


def _route_diagnostic_improvement(
    records: list[ResultRecord],
    recent_ids: set[str],
) -> tuple[float, list[str], int]:
    """Normalized auxiliary diagnostic progress for one route.

    This is evidence only. It cannot mint SU and it must not outrank route
    new-SU/GPU-h. It exists to prevent a no-SU route with clearly improving or
    near-pass scientific diagnostics from being misclassified as low-quality dry
    before canonical score conversion or a longer probe has had a chance to pay.
    """
    if not records or not recent_ids:
        return 0.0, [], 0
    axis_rows: list[tuple[float, int, str, str]] = []
    total_recent_obs = 0
    for axis, (_pass_thr, quality_thr, direction, margin) in DIAGNOSTIC_AXIS_THRESHOLDS.items():
        recent: list[tuple[float, str]] = []
        prior: list[tuple[float, str]] = []
        for r in records:
            v = _diagnostic_axis_value(r, axis)
            if v is None or not math.isfinite(v):
                continue
            lbl, deficit = label_axis(v, quality_thr, direction, margin)
            if lbl == "missing":
                continue
            norm_deficit = max(0.0, float(deficit) / max(float(margin or 1.0), 1e-9))
            item = (norm_deficit, lbl)
            if r.result_id in recent_ids:
                recent.append(item)
            else:
                prior.append(item)
        if not recent:
            continue
        # A single noisy advisory metric should not rescue an otherwise dry route
        # unless it actually hits the quality band. Under-tested routes are already
        # protected by marginal_status, so this guard is for mature dry routes.
        pass_n = sum(1 for _d, lbl in recent if lbl == "pass")
        near_n = sum(1 for _d, lbl in recent if lbl == "near_pass")
        if len(recent) < 2 and pass_n == 0:
            continue
        total_recent_obs += len(recent)
        quality_support = (pass_n + 0.5 * near_n) / max(1, len(recent))
        trend_support = 0.0
        if prior:
            prior_med = statistics.median(d for d, _lbl in prior)
            recent_med = statistics.median(d for d, _lbl in recent)
            trend_support = max(0.0, (prior_med - recent_med) / max(1.0, prior_med))
        score = max(float(quality_support), float(trend_support))
        if score <= 0.0:
            continue
        lever = DIAGNOSTIC_AXIS_REMEDIATION.get(axis)
        levered = lever is not None
        # Corroboration-only axes are useful scientific context, but weaker as an
        # automatic launch guard because no executor has a direct knob for them.
        weighted = min(1.0, score * (1.0 if levered else 0.6))
        label = (
            f"{axis}:{'levered' if levered else 'corroboration'}:"
            f"score={weighted:.2f}:quality={quality_support:.2f}:"
            f"trend={trend_support:.2f}:n={len(recent)}"
        )
        axis_rows.append((weighted, 1 if levered else 0, axis, label))
    if not axis_rows:
        return 0.0, [], total_recent_obs
    axis_rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
    top = axis_rows[:3]
    # Use the strongest axis as the route-level support score. Averaging can hide
    # a single actionable bottleneck, while max keeps the threshold interpretable.
    return round(float(top[0][0]), 3), [row[3] for row in top], total_recent_obs


def build_diagnostic_axis_stats(
    results: list[ResultRecord],
) -> dict[str, AxisStat]:
    """Aggregate diagnostic-axis statistics when observations satisfy coverage and count
    requirements.

    Classify pass, near-pass, and fail against quality_threshold; below_accept_count
    uses the tool acceptance threshold. These diagnostics do not determine qualification
    or SU credit.
    """
    if not results:
        return {}
    # Use each measurement source as its coverage denominator; unrelated backends need
    # not emit the same diagnostic keys.
    out: dict[str, AxisStat] = {}
    for axis, (pass_thr, quality_thr, direction, margin) in DIAGNOSTIC_AXIS_THRESHOLDS.items():
        # Carrier count uses the SAME finiteness criterion as label_axis (None or
        # non-finite → not a carrier), so a NaN/inf-valued record cannot inflate
        # the gate denominator while being excluded from `n` — keeping the latent
        # NaN defense consistent on both sides (production parsers already reject
        # NaN/inf, so this is defensive).
        carriers = sum(
            1 for r in results
            if (lambda v: v is not None and math.isfinite(v))(
                _diagnostic_axis_value(r, axis))
        )
        min_coverage_n = max(
            DIAGNOSTIC_MIN_N, int(round(DIAGNOSTIC_COVERAGE_MIN * carriers))
        )
        p = nf = f = below_accept = 0
        raws: list[float] = []
        deficits: list[float] = []
        source_families: dict[str, int] = {}
        for r in results:
            v = _diagnostic_axis_value(r, axis)
            lbl, d = label_axis(v, quality_thr, direction, margin)
            if lbl == "missing":
                continue
            src = _diagnostic_axis_source_family(r, axis)
            if src:
                source_families[src] = source_families.get(src, 0) + 1
            # A design below the tool's accept floor can NEVER be 'near' the
            # (stricter) quality band: a wide ~1-std near_margin can otherwise
            # overflow PAST the accept floor (e.g. min_ipae quality 0.07 + margin
            # 0.21 = 0.28 > accept 0.226; ipTM quality 0.75 - margin 0.27 = 0.48 <
            # accept 0.50) and mislabel a sub-accept design as near-good. Force it
            # to fail and count it below_accept (which is ⊆ fail by construction).
            below = (pass_thr is not None
                     and label_axis(v, pass_thr, direction, 0.0)[0] == "fail")
            if below:
                below_accept += 1
                lbl = "fail"
            raws.append(v)  # type: ignore[arg-type]
            deficits.append(d)
            if lbl == "pass":
                p += 1
            elif lbl == "near_pass":
                nf += 1
            else:
                f += 1
        n = p + nf + f
        if n < min_coverage_n:
            continue  # ≥25% coverage of the axis's OWN carrier set — omit if sparse
        out[axis] = AxisStat(
            pass_count=p,
            near_pass_count=nf,
            fail_count=f,
            median_raw=statistics.median(raws) if raws else None,
            median_calibrated=statistics.median(raws) if raws else None,
            median_deficit=statistics.median(deficits) if deficits else None,
            calibration_status="provisional" if n > 0 else "uncalibrated",
            n=n,
            pass_threshold=pass_thr,
            quality_threshold=quality_thr,
            below_accept_count=below_accept,
            source_families=source_families,
        )
    return out


def build_diagnostic_alt_model_scores(
    results: list[ResultRecord],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate source-prefixed diagnostic scores from result bins.

    Return backend-to-score summaries with median, count, minimum, maximum, and
    preferred direction. These are advisory measurements.
    """
    by_family: dict[str, dict[str, list[float]]] = {}
    for r in results:
        per_key = by_family.setdefault(r.backend_family, {})
        for key, fv in _record_advisory_scores(r).items():
            per_key.setdefault(key, []).append(fv)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for fam, per_key in by_family.items():
        # min_n=DIAGNOSTIC_MIN_N: this block ships a chain action and is the
        # more-cited diagnostic, so don't surface an n=1 advisory median.
        agg = _summarize_advisory_scores(per_key, min_n=DIAGNOSTIC_MIN_N)
        if agg:
            out[fam] = agg
    return out


def joint_patterns(
    results: Iterable[ResultRecord], cfg: ReducerConfig
) -> list[JointPatternCount]:
    """Currently only computes (pLDDT, iPAE) pair."""
    counts = {
        "both_pass": 0,
        "both_fail": 0,
        "A_pass_B_fail": 0,
        "A_fail_B_pass": 0,
        "both_near_pass": 0,
    }
    thr_a, dir_a, mar_a = _axis_threshold(cfg, "pLDDT")
    thr_b, dir_b, mar_b = _axis_threshold(cfg, "iPAE")
    for r in results:
        a = _axis_metric_for(r, "pLDDT")
        b = _axis_metric_for(r, "iPAE")
        la, _ = label_axis(a, thr_a, dir_a, mar_a)
        lb, _ = label_axis(b, thr_b, dir_b, mar_b)
        if la == "missing" or lb == "missing":
            continue
        if la == "pass" and lb == "pass":
            counts["both_pass"] += 1
        elif la == "fail" and lb == "fail":
            counts["both_fail"] += 1
        elif la == "pass" and lb == "fail":
            counts["A_pass_B_fail"] += 1
        elif la == "fail" and lb == "pass":
            counts["A_fail_B_pass"] += 1
        elif la == "near_pass" and lb == "near_pass":
            counts["both_near_pass"] += 1
    return [
        JointPatternCount(axes=("pLDDT", "iPAE"), pattern=p, count=c)  # type: ignore[arg-type]
        for p, c in counts.items()
    ]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def _has_usable_parent_artifact(r: ResultRecord) -> bool:
    artifacts = r.artifacts or {}
    for key in ("pdb_path", "cif_path"):
        p = artifacts.get(key)
        if p and Path(p).exists():
            return True
    d = artifacts.get("pdb_dir")
    if d and Path(d).exists():
        root = Path(d)
        return (
            any(root.glob("*.pdb"))
            or any(root.glob("*.cif"))
            or any(root.glob("*.mmcif"))
        )
    return False


def dominant_deficit_axis(axis_deficits: dict[str, float]) -> str | None:
    """Return the failing qualification axis with the largest normalized deficit.

    Normalize by the near-pass margins so measurement units do not determine the
    ranking. Return None when no positive deficit is found.
    """
    norm = {
        ax: d / (NEAR_PASS_MARGINS.get(ax, 1.0) or 1.0)
        for ax, d in axis_deficits.items()
        if d and d > 0
    }
    if not norm:
        return None
    return max(norm, key=norm.get)


def representative_examples(
    results: list[ResultRecord], cfg: ReducerConfig,
    *,
    by_result_id: dict[str, ResultRecord] | None = None,
    spawning_actions: dict[str, ActionCandidate] | None = None,
) -> list[Example]:
    """Select examples of distinct outcome patterns with recency tiebreaks. Resolve
    evaluated results to their generating families when lineage is available.
    """
    buckets: dict[str, list[ResultRecord]] = {}
    thr_a, dir_a, mar_a = _axis_threshold(cfg, "pLDDT")
    thr_b, dir_b, mar_b = _axis_threshold(cfg, "iPAE")
    for r in results:
        a = _axis_metric_for(r, "pLDDT")
        b = _axis_metric_for(r, "iPAE")
        la, _ = label_axis(a, thr_a, dir_a, mar_a)
        lb, _ = label_axis(b, thr_b, dir_b, mar_b)
        key = f"{la}/{lb}"
        buckets.setdefault(key, []).append(r)

    chosen: list[ResultRecord] = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        # most recent by tick_id (or result_id lex order as proxy)
        bucket.sort(key=lambda r: (r.tick_id or "", r.result_id), reverse=True)
        chosen.append(bucket[0])
        if len(chosen) >= cfg.max_examples:
            break

    examples = []
    for r in chosen:
        axis_values: dict[str, float] = {}
        axis_deficits: dict[str, float] = {}
        for axis in ("pLDDT", "iPAE", "binder_scRMSD"):
            v = _axis_metric_for(r, axis)
            if v is None:
                continue
            thr, direction, margin = _axis_threshold(cfg, axis)
            lbl, d = label_axis(v, thr, direction, margin)
            axis_values[axis] = v
            axis_deficits[axis] = d
        joint_label = None
        if "pLDDT" in axis_values and "iPAE" in axis_values:
            la, _ = label_axis(
                axis_values["pLDDT"], thr_a, dir_a, mar_a
            )
            lb, _ = label_axis(
                axis_values["iPAE"], thr_b, dir_b, mar_b
            )
            joint_label = f"pLDDT={la}/iPAE={lb}"
        ex_family = (
            resolve_generating_family(
                r, by_result_id=by_result_id, spawning_actions=spawning_actions)
            if by_result_id is not None and spawning_actions is not None
            else r.backend_family
        )
        examples.append(
            Example(
                result_id=r.result_id,
                family=ex_family,
                parent_id=r.parent_ids[0] if r.parent_ids else None,
                axis_values=axis_values,
                axis_deficits=axis_deficits,
                joint_pattern_label=joint_label,
                dominant_deficit_axis=dominant_deficit_axis(axis_deficits),
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Health summaries
# ---------------------------------------------------------------------------


def refilter_role_health(
    results: Iterable[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compact role-level accounting for refilter-like work.

    This is advisory evidence for the LLM/audit trail. It does not change the
    primary SU attribution ledger; canonical score-conversion SU continues to be
    credited to the upstream generating family through resolve_generating_family.
    """
    rs = list(results)
    sp = spawning_actions or {}
    by_id = {r.result_id: r for r in rs}
    out: dict[str, dict[str, Any]] = {}
    su_sets: dict[str, set[str]] = {}
    for r in rs:
        role = _refilter_role_for_record(r, spawning_actions=sp)
        if role is None:
            continue
        row = out.setdefault(role, {
            "attempts": 0,
            "ok": 0,
            "strict_yield": 0,
            "strict_yield_su": 0,
            "gpu_h": 0.0,
            "backend_families": {},
            "source_families": {},
            "credit_policy": (
                "strict_su_credited_to_upstream_generator"
                if role == CANONICAL_SCORE_CONVERSION else
                "intentional_refold_action_advisory_role_credit"
            ),
        })
        row["attempts"] += 1
        row["gpu_h"] = round(float(row["gpu_h"]) + float(r.gpu_h or 0.0), 6)
        row["backend_families"][r.backend_family] = row["backend_families"].get(r.backend_family, 0) + 1
        source_family = (r.bins or {}).get("refilter_source_family")
        if not source_family:
            source_id = ((r.bins or {}).get("refilter_source")
                         or (r.parent_ids[1] if len(r.parent_ids or []) >= 2 else None))
            if source_id and source_id in by_id:
                source_family = by_id[source_id].backend_family
        if source_family:
            row["source_families"][source_family] = row["source_families"].get(source_family, 0) + 1
        if r.exit_status == "ok":
            row["ok"] += 1
        if r.exit_status == "ok" and is_strict_success(r.metrics):
            row["strict_yield"] += 1
            if _is_canonical_su_record(r, sp):
                key = _su_key(r)
                if key is not None:
                    su_sets.setdefault(role, set()).add(key)
    for role, keys in su_sets.items():
        out[role]["strict_yield_su"] = len(keys)
    return out


def _diagnostic_parent_family(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> str | None:
    """Return upstream diagnostic generator family for canonical AF2 refilter.

    Non-canonical refilter records are advisory second opinions. They must
    remain their own strategy rows and must not charge canonical chained
    SU/GPU-h. Only structure_refilter is the AF2-calibrated path that can convert a
    diagnostic-only generator output into strict/SU evidence.
    """
    if r.backend_family != "structure_refilter":
        return None
    if _refilter_role_for_record(r, spawning_actions=spawning_actions) != CANONICAL_SCORE_CONVERSION:
        return None
    parent = _score_conversion_parent_record(
        r, by_result_id=by_result_id, spawning_actions=spawning_actions,
    )
    if parent is None:
        return None
    from .capability_registry import default_registry as _dr_chain
    cap = _dr_chain().get(parent.backend_family)
    if cap is None:
        return None
    # Include sequence redesign as well as generation when attributing evaluated
    # descendants.
    if getattr(cap, "outputs_diagnostic_only", False) and getattr(
        cap, "role", ""
    ) in ("generator", "seq_redesign"):
        return parent.backend_family
    return None


def method_health(
    results: Iterable[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None = None,
    near_miss_cluster_by_result_id: dict[str, str] | None = None,
    near_miss_dedup_trusted: bool = True,
) -> dict[str, MethodHealthSummary]:
    results = list(results)
    spawning_actions = spawning_actions or {}
    near_miss_cluster_by_result_id = near_miss_cluster_by_result_id or {}
    by_result_id = {r.result_id: r for r in results}
    by_family: dict[str, list[ResultRecord]] = {}
    for r in results:
        by_family.setdefault(r.backend_family, []).append(r)
    # Assign each strict structural cluster one owner so family counts sum to the
    # campaign SU count.
    _cluster_owner: dict[str, str] = {}
    _chain_credit: dict[str, set[str]] = {}
    # Credit the generating family through evaluation lineage. Prefer resolved
    # generation over advisory refolds so ownership does not depend on archive order.
    _strict_owned = [
        (r, resolve_generating_family(
            r, by_result_id=by_result_id, spawning_actions=spawning_actions))
        for r in results
        if _is_canonical_su_record(r, spawning_actions)
    ]
    for r, owner_fam in sorted(
        _strict_owned, key=lambda ro: ro[1] == "structure_refilter"
    ):
        su_key = _su_key(r)
        if su_key is None:
            continue
        _cluster_owner.setdefault(su_key, owner_fam)
        parent_fam = _diagnostic_parent_family(
            r, by_result_id=by_result_id, spawning_actions=spawning_actions,
        )
        if parent_fam is not None:
            _chain_credit.setdefault(parent_fam, set()).add(su_key)
    # Charge downstream evaluation, including failed evaluations, to the originating
    # route. Count each cost once so direct and separately evaluated routes are
    # comparable.
    _chain_downstream_gpu_h: dict[str, float] = {}
    for r in results:
        pf = _diagnostic_parent_family(
            r, by_result_id=by_result_id, spawning_actions=spawning_actions,
        )
        if pf is not None:
            _chain_downstream_gpu_h[pf] = _chain_downstream_gpu_h.get(pf, 0.0) + r.gpu_h
    _chain_upstream_parent_gpu_h: dict[str, float] = {}
    _chain_upstream_parent_ids: dict[str, set[str]] = {}
    from .capability_registry import default_registry as _dr_parent_cost
    _parent_cost_registry = _dr_parent_cost()
    for r in results:
        _cap_parent_cost = _parent_cost_registry.get(r.backend_family)
        if _cap_parent_cost is None:
            continue
        if not (
            getattr(_cap_parent_cost, "outputs_diagnostic_only", False)
            and getattr(_cap_parent_cost, "role", "") in ("generator", "seq_redesign")
        ):
            continue
        for parent in _lineage_parent_records(
            r, by_result_id=by_result_id, spawning_actions=spawning_actions,
        ):
            seen = _chain_upstream_parent_ids.setdefault(r.backend_family, set())
            if parent.result_id in seen:
                continue
            seen.add(parent.result_id)
            _chain_upstream_parent_gpu_h[r.backend_family] = (
                _chain_upstream_parent_gpu_h.get(r.backend_family, 0.0)
                + float(parent.gpu_h or 0.0)
            )
    out: dict[str, MethodHealthSummary] = {}
    for fam, rs in by_family.items():
        attempts = len(rs)
        completions = sum(1 for r in rs if r.exit_status in ("ok", "no_artifacts"))
        timeouts = sum(1 for r in rs if r.exit_status == "timeout")
        nonzero = sum(1 for r in rs if r.exit_status == "nonzero_exit")
        raw = sum(1 for r in rs if r.artifacts)
        scored = sum(1 for r in rs if any(k in r.metrics for k in ("pLDDT", "iPAE")))
        # Use the same successful-exit eligibility for strict and structure-unique
        # counts.
        strict = sum(
            1 for r in rs if _is_canonical_su_record(r, spawning_actions)
        )
        # Count strict-only clusters owned by this family.
        su_keys = {k for k, owner in _cluster_owner.items() if owner == fam}
        # Use the shared near-miss predicate and structural deduplication.
        _nm_bins: set[str] = set()
        if near_miss_dedup_trusted:
            for r in rs:
                if r.exit_status == "ok" and is_near_miss(r.metrics):
                    _b = r.bins or {}
                    # Near-miss is rescue evidence, not official SU. Include
                    # refilter_source only as a coarse same-parent dedup fallback
                    # when the near-miss dedup channel itself is trusted.
                    _nm_bins.add(
                        near_miss_cluster_by_result_id.get(r.result_id)
                        or _b.get("foldseek_near_miss")
                        or _b.get("foldseek")
                        or _b.get("refilter_source")
                        or r.result_id
                    )
        near_miss = len(_nm_bins)
        gpu_h = float(sum(r.gpu_h for r in rs))
        # Leave direct productivity unset for unevaluated diagnostic outputs and
        # evaluation-only families. Their contributions are represented through
        # route-level accounting.
        from .capability_registry import default_registry as _dr_role
        _cap = _dr_role().get(fam)
        _exclude_rate = _cap is not None and (
            getattr(_cap, "role", "generator") == "refilter"
            or getattr(_cap, "outputs_diagnostic_only", False)
        )
        su_per_gpu_h = (
            None if _exclude_rate
            else ((len(su_keys) / gpu_h) if gpu_h > 0 else None)
        )
        chained_su = len(_chain_credit.get(fam, set()))
        # Route cost includes generation and the downstream evaluation it requires.
        chained_route_gpu_h = (
            gpu_h
            + _chain_downstream_gpu_h.get(fam, 0.0)
            + _chain_upstream_parent_gpu_h.get(fam, 0.0)
        )
        chained_su_per_gpu_h = (
            (chained_su / chained_route_gpu_h)
            if chained_su and chained_route_gpu_h > 0 else None
        )
        out[fam] = MethodHealthSummary(
            family=fam,
            attempts=attempts,
            completions=completions,
            timeouts=timeouts,
            nonzero_exits=nonzero,
            raw_artifacts=raw,
            accepted_artifacts=raw,
            score_files=scored,
            strict_yield=strict,
            near_miss_yield=near_miss,
            routed_proxy=None,
            # Retain compute spent even when a family has produced no qualified design.
            cumulative_gpu_h=gpu_h,
            strict_yield_su=len(su_keys),
            su_per_gpu_h=su_per_gpu_h,
            chained_strict_yield_su=chained_su,
            chained_su_per_gpu_h=chained_su_per_gpu_h,
        )
    return out


def route_health(
    completed_route_count: int = 0,
    backlog_used: int = 0,
    backlog_cap: int = 96,
) -> RouteHealthSummary:
    """Stub; real producer reads RouteRecord archive."""
    return RouteHealthSummary(
        raw_routed=0,
        score_files_completed=completed_route_count,
        backlog_used=backlog_used,
        backlog_cap=backlog_cap,
        near_miss_conversion=None,
        strict_conversion=None,
        panel_ready_conversion=None,
    )


def _classify_result(r: ResultRecord, cfg: ReducerConfig) -> RecipeClass | None:
    """Return the recipe_class for a single ResultRecord, or None if not
    notable enough to record as a recipe.

    Uses the single-source success_criteria definitions (is_strict_success,
    is_near_miss, is_joint_fail) so classification is consistent between
    live_tick, recipe extraction, and state classification.
    """
    if r.exit_status != "ok":
        return None
    if r.panel_ready:
        return "panel_ready"
    if is_strict_success(r.metrics):
        return "strict_success"
    if is_near_miss(r.metrics):
        return "near_miss"
    # joint failure: pLDDT and iPAE both clearly fail
    from .success_criteria import is_joint_fail
    if is_joint_fail(r.metrics):
        return "joint_fail"
    return None


def _recipe_signature(operator_id: str, config_delta: dict[str, Any]) -> str:
    payload = json.dumps(
        {"op": operator_id, "cfg": config_delta},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _effective_provenance(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Resolve generating family, operator, and configuration through evaluation lineage.
    Evaluation settings must not replace the design settings in route evidence.
    """
    gen = resolve_generating_record(
        r, by_result_id=by_result_id, spawning_actions=spawning_actions,
    )
    gen_ac = spawning_actions.get(gen.result_id)
    family = gen.backend_family
    operator_id = gen_ac.operator_id if gen_ac is not None else f"{family}_default"
    config_delta = dict((gen_ac.config_delta or {}) if gen_ac is not None else {})
    if gen.result_id != r.result_id:
        # refilter → the generator record is the chain anchor
        parent_result_id = gen.result_id
    else:
        parent_result_id = (
            gen_ac.parent_result_id if gen_ac is not None and gen_ac.parent_result_id
            else (r.parent_ids[0] if r.parent_ids else None)
        )
    return family, operator_id, config_delta, parent_result_id


def build_route_values(
    results: list[ResultRecord],
    window_results: list[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None,
    cfg: StateClassifierConfig | None = None,
    near_miss_dedup_trusted: bool = True,
    gpu_h_since_last_su: float | None = None,
) -> list[RouteValueSummary]:
    """Build route summaries through the focused evidence component.

    This compatibility entry point preserves the historical public API while
    the implementation lives in :mod:`trex.evidence.route_values`.
    """
    return build_route_value_summaries(
        results=results,
        window_results=window_results,
        spawning_actions=spawning_actions,
        config=cfg or StateClassifierConfig(),
        route_diagnostic_improvement=_route_diagnostic_improvement,
        near_miss_dedup_trusted=near_miss_dedup_trusted,
        gpu_h_since_last_su=gpu_h_since_last_su,
    )


def extract_recipes(
    results: list[ResultRecord],
    *,
    spawning_action: dict[str, ActionCandidate] | None = None,
    target_class: str | None = None,
    current_tick: int = 0,
    cfg: ReducerConfig | None = None,
    top_strict: int = 3,
    top_panel_ready: int = 2,
    top_near_miss: int = 2,
    top_joint_fail: int = 4,
) -> list[Recipe]:
    """Aggregate results by (operator_id, config_delta) and return top-K recipes per class.

    spawning_action maps result IDs to their generating ActionCandidates. Without
    this mapping, use (backend_family, {}) for family-level evidence.
    """
    cfg = cfg or ReducerConfig()
    spawning_action = spawning_action or {}
    by_result_id = {r.result_id: r for r in results}
    lineage_cache: dict[str, tuple[ResultRecord, ...]] = {}

    def _cached_lineage(r: ResultRecord) -> tuple[ResultRecord, ...]:
        cached = lineage_cache.get(r.result_id)
        if cached is None:
            cached = tuple(_lineage_parent_records(
                r, by_result_id=by_result_id, spawning_actions=spawning_action,
            ))
            lineage_cache[r.result_id] = cached
        return cached

    by_class: dict[RecipeClass, dict[str, dict[str, Any]]] = {
        "strict_success": {},
        "panel_ready": {},
        "near_miss": {},
        "joint_fail": {},
    }

    def _sig_for(r: ResultRecord) -> tuple[str, str, str, dict]:
        family, operator_id, config_delta, _parent_rid = _effective_provenance(
            r, by_result_id=by_result_id, spawning_actions=spawning_action,
        )
        return _recipe_signature(operator_id, config_delta), family, operator_id, config_delta

    # Compute route productivity over all route costs, including unsuccessful work.
    # Evaluation-only routes do not receive independent generation productivity.
    from .capability_registry import default_registry as _dr_recipe
    _reg_recipe = _dr_recipe()
    route_gpu_h: dict[str, float] = {}
    route_su_bins: dict[str, set] = {}
    route_is_refilter: dict[str, bool] = {}
    route_lineage_charge_ids: dict[str, set[str]] = {}
    for r in results:
        sig0, fam0, _, _ = _sig_for(r)
        route_gpu_h[sig0] = route_gpu_h.get(sig0, 0.0) + float(r.gpu_h)
        for parent in _cached_lineage(r):
            parent_sig, _, _, _ = _sig_for(parent)
            if parent_sig == sig0:
                continue
            charged = route_lineage_charge_ids.setdefault(sig0, set())
            if parent.result_id in charged:
                continue
            charged.add(parent.result_id)
            route_gpu_h[sig0] = route_gpu_h.get(sig0, 0.0) + float(parent.gpu_h or 0.0)
        _cap0 = _reg_recipe.get(fam0)
        # Pure refilter routes are excluded from speed credit. Diagnostic-only
        # generator routes that later produce strict via structure_refilter are
        # intentionally credited to the upstream generator config.
        route_is_refilter[sig0] = _cap0 is not None and (
            getattr(_cap0, "role", "generator") == "refilter"
        )
        if r.exit_status == "ok" and is_strict_success(r.metrics):
            # Official SU credit requires a strict-only Foldseek bin.
            sb0 = _su_key(r)
            if sb0 is not None:
                route_su_bins.setdefault(sig0, set()).add(sb0)

    for r in results:
        cls = _classify_result(r, cfg)
        if cls is None:
            continue
        sig, family, operator_id, config_delta = _sig_for(r)
        bucket = by_class[cls].setdefault(
            sig,
            {
                "operator_id": operator_id,
                "method_family": family,
                "config_delta": config_delta,
                "descendants": [],
                "metrics_per_axis": {"pLDDT": [], "iPAE": [], "binder_scRMSD": []},
                "recency_tick": -1,
            },
        )
        bucket["descendants"].append(r.result_id)
        for ax in ("pLDDT", "iPAE", "binder_scRMSD"):
            v = r.metrics.get(ax)
            if isinstance(v, (int, float)):
                bucket["metrics_per_axis"][ax].append(float(v))
        # Support prefixed controller tick IDs and numeric IDs from older archives.
        if r.tick_id:
            t_int: int | None = None
            tid = r.tick_id
            if tid.startswith("v7r") and tid[3:].isdigit():
                t_int = int(tid[3:])
            elif tid.isdigit():
                t_int = int(tid)
            if t_int is not None and t_int > bucket["recency_tick"]:
                bucket["recency_tick"] = t_int

    limits = {
        "strict_success": top_strict,
        "panel_ready": top_panel_ready,
        "near_miss": top_near_miss,
        "joint_fail": top_joint_fail,
    }

    # Retain successful recipes by quality, support count, and recency so frequent
    # repeats do not displace the best observed settings.

    def _quality_score(metrics_per_axis: dict[str, list[float]]) -> float:
        """Strict-success quality = sum of normalized margins past thresholds.
        Higher = stronger pass. 0 if any axis missing (still a winner but
        less comparable). pLDDT margin scale = 5, iPAE = 0.05, scRMSD = 0.3."""
        p_vals = metrics_per_axis.get("pLDDT") or []
        i_vals = metrics_per_axis.get("iPAE") or []
        r_vals = metrics_per_axis.get("binder_scRMSD") or []
        if not (p_vals and i_vals and r_vals):
            return 0.0
        p = statistics.median(p_vals)
        i = statistics.median(i_vals)
        r = statistics.median(r_vals)
        # Derive normalized quality from the shared qualification thresholds and
        # margins.
        return (
            (p - STRICT_SUCCESS["pLDDT"][0]) / NEAR_PASS_MARGINS["pLDDT"]
            + (STRICT_SUCCESS["iPAE"][0] - i) / NEAR_PASS_MARGINS["iPAE"]
            + (STRICT_SUCCESS["binder_scRMSD"][0] - r) / NEAR_PASS_MARGINS["binder_scRMSD"]
        )

    out: list[Recipe] = []
    for cls, buckets in by_class.items():
        if not buckets:
            continue
        if cls == "strict_success":
            # Stratified: top-3 by quality + top-3 by count + top-2 by recency,
            # union-deduped by recipe_hash. Ensures the all-time best is never
            # silently pruned in favor of recent but-mediocre runs.
            by_quality = sorted(
                buckets.items(),
                key=lambda kv: (-_quality_score(kv[1]["metrics_per_axis"]),
                                  -kv[1]["recency_tick"]),
            )
            by_count = sorted(
                buckets.items(),
                key=lambda kv: (-len(kv[1]["descendants"]),
                                  -kv[1]["recency_tick"]),
            )
            by_recency = sorted(
                buckets.items(),
                key=lambda kv: (-kv[1]["recency_tick"],
                                  -len(kv[1]["descendants"])),
            )
            chosen: dict[str, dict] = {}  # sig → bucket, preserves first-seen order
            # 4+4+4 = up to 12 strict_success recipes (de-duped, typically
            # ~8-10 unique). Each ~200 tokens; total ~2K tokens of prompt
            # budget — easily within Qwen 64k context.
            for src, k in ((by_quality, 4), (by_count, 4), (by_recency, 4)):
                for sig, b in src[:k]:
                    if sig not in chosen:
                        chosen[sig] = b
            ranked = list(chosen.items())
        else:
            ranked = sorted(
                buckets.items(),
                key=lambda kv: (-len(kv[1]["descendants"]), -kv[1]["recency_tick"]),
            )[: limits[cls]]
        for sig, b in ranked:
            medians = {
                ax: statistics.median(vals)
                for ax, vals in b["metrics_per_axis"].items()
                if vals
            }
            # Use total route compute for successful recipes; evaluation-only routes
            # have no independent generation rate.
            g = route_gpu_h.get(sig, 0.0)
            su_bins = route_su_bins.get(sig, set())
            su_per_gpu_h = (
                (len(su_bins) / g)
                if (cls == "strict_success" and g > 0 and su_bins
                    and not route_is_refilter.get(sig, False))
                else None
            )
            out.append(
                Recipe(
                    recipe_hash=sig,
                    operator_id=b["operator_id"],
                    method_family=b["method_family"],
                    config_delta=b["config_delta"],
                    recipe_class=cls,
                    target_id=results[0].target_id if results else "?",
                    target_class=target_class,
                    descendant_count=len(b["descendants"]),
                    median_metrics=medians,
                    representative_result_ids=b["descendants"][:3],
                    recency_tick=b["recency_tick"] if b["recency_tick"] >= 0 else current_tick,
                    su_per_gpu_h=su_per_gpu_h,
                )
            )
    return out


STUCK_MIN_REFINEMENTS = 3         # >=K non-improving refinements of one backbone
STUCK_PLDDT_MIN_REFINEMENTS = 1   # a bad fold can't be fixed by fixed-backbone refine
STUCK_IMPROVE_EPS = 0.5           # margin-units a child must close to count as progress
STUCK_FAMILY_MIN_ATTEMPTS = 2  # Track exhaustion per parent and refinement family. Repeated strict duplicates can also
# exhaust a lineage.
STUCK_DUP_MIN_STRICT = 4          # >=N strict su-bearing children before judging
STUCK_DUP_REDUNDANCY = 3          # distinct bins <= n_strict // this == mined out


def _record_deficits(r: ResultRecord, cfg: ReducerConfig) -> dict[str, float]:
    """Signed-positive deficit per strict axis for one record (0 = passes)."""
    out: dict[str, float] = {}
    for axis in ("pLDDT", "iPAE", "binder_scRMSD"):
        v = _axis_metric_for(r, axis)
        if v is None:
            continue
        thr, direction, margin = _axis_threshold(cfg, axis)
        _, d = label_axis(v, thr, direction, margin)
        out[axis] = d
    return out


def stuck_lineage_roots(
    results: list[ResultRecord], cfg: ReducerConfig
) -> list[dict]:
    """Identify parent lineages with repeated non-improving refinements or structural
    duplicates.

    Only comparable qualification measurements count as refinement evidence. Follow
    sequence redesigns to their evaluated descendants, and recognize improvement on any
    originally failing measurement.
    """
    by_id = {r.result_id: r for r in results}
    children: dict[str, list[ResultRecord]] = {}
    for r in results:
        for pid in (r.parent_ids or []):
            if pid in by_id:
                children.setdefault(pid, []).append(r)
                break  # Resolve sequence-only children to their evaluated descendants before
                # judging refinement outcomes.
    refilter_grandkids: dict[str, list[ResultRecord]] = {}
    for r in results:
        src = (r.bins or {}).get("refilter_source")
        if src and _record_deficits(r, cfg):  # canonical refilter w/ strict metrics
            refilter_grandkids.setdefault(src, []).append(r)

    def _effective_deficits(c: ResultRecord) -> dict[str, float]:
        """Child's own strict deficits, or — for a sequence-only child — those of
        its BEST (lowest margin-normalized total) scored refilter grandchild."""
        d = _record_deficits(c, cfg)
        if d:
            return d
        best: dict[str, float] = {}
        best_score = None
        for g in refilter_grandkids.get(c.result_id, ()):
            gd = _record_deficits(g, cfg)
            sc = sum((gd.get(ax, 0.0) or 0.0) / (NEAR_PASS_MARGINS.get(ax, 1.0) or 1.0)
                     for ax in gd)
            if best_score is None or sc < best_score:
                best, best_score = gd, sc
        return best

    out: list[dict] = []
    for pid, kids_all in children.items():
        parent = by_id[pid]
        pdef = _record_deficits(parent, cfg)
        dom = dominant_deficit_axis(pdef)
        if dom is None:
            continue  # Count an attempt only when the child or its evaluated descendant has
            # comparable qualification measurements.
        kids = [(c, _effective_deficits(c)) for c in kids_all]
        kids = [(c, d) for c, d in kids if d]
        k = STUCK_PLDDT_MIN_REFINEMENTS if dom == "pLDDT" else STUCK_MIN_REFINEMENTS
        if len(kids) < k:
            continue
        # Improvement on any originally failing measurement counts as progress.
        failing = {ax: d for ax, d in pdef.items() if d and d > 0}
        improved = False
        for c, cdef in kids:
            for ax, pd in failing.items():
                m = NEAR_PASS_MARGINS.get(ax, 1.0) or 1.0
                if (cdef.get(ax, pd) / m) <= (pd / m) - STUCK_IMPROVE_EPS:
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        reason = (
            "structural_pLDDT" if dom == "pLDDT"
            else f"no_improvement_after_{len(kids)}"
        )
        # Track exhausted refinement families per parent; an empty set blocks all
        # refinement.
        exhausted_families: list[str] = []
        if dom != "pLDDT":
            _fc: dict[str, int] = {}
            for c, _ in kids:
                _fc[c.backend_family] = _fc.get(c.backend_family, 0) + 1
            exhausted_families = sorted(
                f for f, n in _fc.items() if n >= STUCK_FAMILY_MIN_ATTEMPTS
            )
        out.append({
            "root_result_id": pid,
            "family": parent.backend_family,
            "dominant_axis": dom,
            "attempts": len(kids),
            "reason": reason,
            # [] => block ALL refinement of this parent (regenerate); non-empty =>
            # block only these rescue families, allow an untried family to try.
            "exhausted_families": exhausted_families,
        })

    # Also detect qualified descendants concentrated in too few structural clusters.
    already = {e["root_result_id"] for e in out}
    for pid, kids_all in children.items():
        if pid in already:
            continue
        strict_bins = [
            key for c in kids_all
            if c.exit_status == "ok" and is_strict_success(c.metrics)
            for key in [_su_key(c)]
            if key is not None
        ]
        n_strict = len(strict_bins)
        distinct = len(set(strict_bins))
        if n_strict >= STUCK_DUP_MIN_STRICT and distinct <= max(
            1, n_strict // STUCK_DUP_REDUNDANCY
        ):
            out.append({
                "root_result_id": pid,
                "family": by_id[pid].backend_family,
                "dominant_axis": None,
                "attempts": n_strict,
                "reason": f"strict_duplicate_{distinct}of{n_strict}",
                # mined-out lineage -> regenerate from scratch: block ALL refinement.
                "exhausted_families": [],
            })
    return out


def build_exemplars(
    results: list[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None,
    cfg: ReducerConfig,
    *,
    k_best: int = 5,
    k_near: int = 5,
) -> list[Exemplar]:
    """Best-K strict + near-miss-K closest binders, each joined to the full setup
    (family / operator / config_delta) that produced it + its full metric vector
    (see the Exemplar docstring). Best = strict+ok ranked by margin past the strict
    thresholds, deduped by SU bin (distinct structures). Near-miss = closest-to-
    passing failures (smallest normalized dominant deficit), deduped by
    (operator, config) signature (diverse failing setups). Records missing the
    strict axes, and all-axes-fail records, are excluded (uninformative)."""
    sp = spawning_actions or {}
    by_result_id = {r.result_id: r for r in results}

    def _prov(r: ResultRecord):
        fam, op, cd, parent_rid = _effective_provenance(
            r, by_result_id=by_result_id, spawning_actions=sp,
        )
        return fam, op, cd, parent_rid

    def _quality(r: ResultRecord) -> float:
        p = _axis_metric_for(r, "pLDDT"); i = _axis_metric_for(r, "iPAE")
        s = _axis_metric_for(r, "binder_scRMSD")
        if p is None or i is None or s is None:
            return float("-inf")
        # Use the shared qualification thresholds and margins.
        return (
            (p - STRICT_SUCCESS["pLDDT"][0]) / NEAR_PASS_MARGINS["pLDDT"]
            + (STRICT_SUCCESS["iPAE"][0] - i) / NEAR_PASS_MARGINS["iPAE"]
            + (STRICT_SUCCESS["binder_scRMSD"][0] - s) / NEAR_PASS_MARGINS["binder_scRMSD"]
        )

    def _su_bin(r: ResultRecord):
        # Official SU exemplar bin; no fallback on raw/offline reducer replay.
        return _su_key(r)

    def _mk(r: ResultRecord, kind: str) -> Exemplar:
        fam, op, cd, parent_rid = _prov(r)
        defs = _record_deficits(r, cfg)
        # Read auxiliary diagnostics from the generating record and qualification
        # measurements from the evaluated record.
        gen = resolve_generating_record(
            r, by_result_id=by_result_id, spawning_actions=sp,
        )
        return Exemplar(
            kind=kind, result_id=r.result_id, family=fam,
            operator_id=op, config_delta=dict(cd), metrics=dict(r.metrics or {}),
            axis_deficits=defs, dominant_deficit_axis=dominant_deficit_axis(defs),
            parent_result_id=parent_rid,
            diagnostic_blocking_axis=worst_actionable_diagnostic_axis(gen),
            su_bin=_su_bin(r) if kind == "best" else None,
        )

    # BEST: strict+ok, deduped by SU bin (distinct structures), ranked by quality.
    strict = [r for r in results if r.exit_status == "ok" and is_strict_success(r.metrics)]
    strict.sort(key=lambda r: (-_quality(r), r.result_id))
    best: list[Exemplar] = []
    seen_bin: set = set()
    for r in strict:
        b = _su_bin(r)
        if b is None or b in seen_bin:
            continue
        seen_bin.add(b)
        best.append(_mk(r, "best"))
        if len(best) >= k_best:
            break

    # NEAR-MISS: closest-to-passing failures, deduped by (operator, config).
    def _closeness(r: ResultRecord) -> float:
        defs = _record_deficits(r, cfg)
        dom = dominant_deficit_axis(defs)
        if dom is None:
            return float("inf")
        return defs[dom] / (NEAR_PASS_MARGINS.get(dom, 1.0) or 1.0)

    nm = [r for r in results if r.exit_status == "ok" and is_near_miss(r.metrics)]
    nm.sort(key=lambda r: (_closeness(r), r.result_id))
    near: list[Exemplar] = []
    seen_sig: set = set()
    for r in nm:
        _fam, op, cd, _parent_rid = _prov(r)
        sig = _recipe_signature(op, cd)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        near.append(_mk(r, "near_miss"))
        if len(near) >= k_near:
            break

    return best + near


def build_strategy_feedback(
    results: list[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None,
    cfg: ReducerConfig,
    *,
    max_per_class: int = 4,
    near_miss_dedup_trusted: bool = True,
) -> list[dict[str, Any]]:
    """Compact outcome ledger by strategy.

    Planner feedback should not be success-only: it needs to know which
    strategies nearly worked, which failed on which axes, and which consumed
    budget without producing canonical signal. Diagnostic-only generator records
    that do not carry the strict AF2 axes are counted as diagnostic artifacts,
    not failures; their canonical pass/fail signal is attributed through the
    downstream structure_refilter record by _effective_provenance().
    """
    sp = spawning_actions or {}
    by_result_id = {r.result_id: r for r in results}
    lineage_cache: dict[str, tuple[ResultRecord, ...]] = {}

    def _cached_lineage(r: ResultRecord) -> tuple[ResultRecord, ...]:
        cached = lineage_cache.get(r.result_id)
        if cached is None:
            cached = tuple(_lineage_parent_records(
                r, by_result_id=by_result_id, spawning_actions=sp,
            ))
            lineage_cache[r.result_id] = cached
        return cached

    from .capability_registry import default_registry as _dr_strategy
    _strategy_registry = _dr_strategy()

    def _tick_int(tick_id: str | None) -> int:
        if not tick_id:
            return -1
        if tick_id.startswith("v7r") and tick_id[3:].isdigit():
            return int(tick_id[3:])
        if tick_id.isdigit():
            return int(tick_id)
        return -1

    def _has_strict_axes(r: ResultRecord) -> bool:
        return all(_axis_metric_for(r, ax) is not None for ax in ("pLDDT", "iPAE", "binder_scRMSD"))

    def _near_miss_key(r: ResultRecord) -> str:
        bins = r.bins or {}
        # Prefer full near-miss clustering over the recent scored-window clustering.
        return (
            bins.get("foldseek_near_miss")
            or bins.get("foldseek")
            or bins.get("foldseek_su")
            or bins.get("refilter_source")
            or r.result_id
        )

    def _rate_rankable_family(family: str) -> bool:
        cap = _strategy_registry.get(family)
        if cap is None:
            return True
        return (
            getattr(cap, "role", "generator") != "refilter"
            and not getattr(cap, "outputs_diagnostic_only", False)
        )

    def _closeness(r: ResultRecord) -> float:
        defs = _record_deficits(r, cfg)
        dom = dominant_deficit_axis(defs)
        if dom is None:
            return float("inf")
        return defs[dom] / (NEAR_PASS_MARGINS.get(dom, 1.0) or 1.0)

    buckets: dict[str, dict[str, Any]] = {}
    route_info_by_result_id: dict[str, dict[str, Any]] = {}
    for r in results:
        info = _route_identity_for_record(
            r, by_result_id=by_result_id, spawning_actions=sp,
        )
        route_info_by_result_id[r.result_id] = info
        fam = str(info["action_family"])
        op = str(info.get("operator_id") or f"{fam}_default")
        cd = dict(info.get("config_delta") or {})
        sig = str(info["strategy_key"])
        b = buckets.setdefault(
            sig,
            {
                "strategy_key": sig,
                "family": fam,
                "root_family": info.get("root_family"),
                "parent_strategy_key": info.get("parent_strategy_key"),
                "refilter_role": info.get("refilter_role"),
                "operator_id": op,
                "config_delta": dict(cd or {}),
                "attempts": 0,
                "attempt_keys": set(),
                "result_count": 0,
                "gpu_h": 0.0,
                "lineage_charge_ids": set(),
                "strict_count": 0,
                "strict_su_bins": set(),
                "panel_ready_count": 0,
                "near_miss_bins": set(),
                "joint_fail_count": 0,
                "other_failure_count": 0,
                "timeout_count": 0,
                "nonzero_exit_count": 0,
                "no_artifacts_count": 0,
                "diagnostic_only_count": 0,
                "dominant_failure_axes": {},
                "advisory_score_values": {},
                "near_miss_reps_by_bin": {},
                "failure_reps": [],
                "last_tick": -1,
            },
        )
        ac = sp.get(r.result_id)
        b["attempt_keys"].add(ac.candidate_id if ac is not None else r.result_id)
        b["result_count"] += 1
        b["gpu_h"] += float(r.gpu_h or 0.0)
        for parent in _cached_lineage(r):
            parent_info = route_info_by_result_id.get(parent.result_id)
            if parent_info is None:
                parent_info = _route_identity_for_record(
                    parent, by_result_id=by_result_id, spawning_actions=sp,
                )
                route_info_by_result_id[parent.result_id] = parent_info
            parent_sig = str(parent_info["strategy_key"])
            if parent_sig == sig:
                continue
            charged = b["lineage_charge_ids"]
            if parent.result_id in charged:
                continue
            charged.add(parent.result_id)
            b["gpu_h"] += float(parent.gpu_h or 0.0)
        b["last_tick"] = max(b["last_tick"], _tick_int(r.tick_id))
        if r.panel_ready:
            b["panel_ready_count"] += 1
        for score_key, score_value in _record_advisory_scores(r).items():
            b["advisory_score_values"].setdefault(score_key, []).append(score_value)

        ok = r.exit_status == "ok"
        has_axes = _has_strict_axes(r)
        if ok and is_strict_success(r.metrics):
            b["strict_count"] += 1
            key = _su_key(r)
            if key is not None:
                b["strict_su_bins"].add(key)
            continue

        defs = _record_deficits(r, cfg)
        dom = dominant_deficit_axis(defs)
        if dom is not None:
            axes = b["dominant_failure_axes"]
            axes[dom] = axes.get(dom, 0) + 1

        if ok and near_miss_dedup_trusted and is_near_miss(r.metrics):
            nm_key = _near_miss_key(r)
            b["near_miss_bins"].add(nm_key)
            rep = (_closeness(r), r.result_id)
            old = b["near_miss_reps_by_bin"].get(nm_key)
            if old is None or rep < old:
                b["near_miss_reps_by_bin"][nm_key] = rep
        elif ok and is_near_miss(r.metrics):
            # The design is scientifically close, but its structural near-miss
            # dedup is untrusted this tick. Do not count it as a near-miss basin,
            # and do not demote it to ordinary failure. The raw exemplar/metrics
            # remain available with dedup_trust=degraded in the prompt.
            b["diagnostic_only_count"] += 1
        elif ok and is_joint_fail(r.metrics):
            b["joint_fail_count"] += 1
            b["failure_reps"].append((b["last_tick"], r.result_id))
        elif not ok:
            if r.exit_status == "timeout":
                b["timeout_count"] += 1
            elif r.exit_status == "nonzero_exit":
                b["nonzero_exit_count"] += 1
            elif r.exit_status == "no_artifacts":
                b["no_artifacts_count"] += 1
            b["failure_reps"].append((b["last_tick"], r.result_id))
        elif has_axes:
            b["other_failure_count"] += 1
            b["failure_reps"].append((b["last_tick"], r.result_id))
        else:
            b["diagnostic_only_count"] += 1

    def _render(b: dict[str, Any]) -> dict[str, Any]:
        strict_su = len(b["strict_su_bins"])
        near_miss_count = len(b["near_miss_bins"])
        failure_count = (
            b["joint_fail_count"]
            + b["other_failure_count"]
            + b["timeout_count"]
            + b["nonzero_exit_count"]
            + b["no_artifacts_count"]
        )
        near_reps = [
            rid for _score, rid in sorted(
                b["near_miss_reps_by_bin"].values(),
                key=lambda x: (x[0], x[1]),
            )[:3]
        ]
        fail_reps = [
            rid for _tick, rid in sorted(b["failure_reps"], key=lambda x: (-x[0], x[1]))[:3]
        ]
        gpu_h = float(b["gpu_h"])
        rate_rankable = _rate_rankable_family(b["family"])
        return {
            "strategy_key": b["strategy_key"],
            "family": b["family"],
            "root_family": b.get("root_family"),
            "parent_strategy_key": b.get("parent_strategy_key"),
            "refilter_role": b.get("refilter_role"),
            "operator_id": b["operator_id"],
            "config_delta": b["config_delta"],
            "feedback_scope": "exact_route_operator_config",
            "repeat_policy": "not_a_ban_single_failure_is_insufficient",
            "attempts": len(b["attempt_keys"]),
            "result_count": b["result_count"],
            "gpu_h": round(gpu_h, 4),
            "strict_count": b["strict_count"],
            "strict_su": strict_su,
            "su_per_gpu_h": (
                round(strict_su / gpu_h, 4)
                if rate_rankable and strict_su and gpu_h > 0 else None
            ),
            "panel_ready_count": b["panel_ready_count"],
            "near_miss_count": near_miss_count,
            "failure_count": failure_count,
            "joint_fail_count": b["joint_fail_count"],
            "other_failure_count": b["other_failure_count"],
            "timeout_count": b["timeout_count"],
            "nonzero_exit_count": b["nonzero_exit_count"],
            "no_artifacts_count": b["no_artifacts_count"],
            "diagnostic_only_count": b["diagnostic_only_count"],
            "dominant_failure_axes": dict(sorted(b["dominant_failure_axes"].items())),
            "advisory_scores": _summarize_advisory_scores(b["advisory_score_values"]),
            "representative_near_miss_ids": near_reps,
            "representative_failure_ids": fail_reps,
            "last_tick": b["last_tick"],
        }

    rendered = [_render(b) for b in buckets.values()]
    success_pool = [x for x in rendered if x["strict_su"] > 0]
    primary_success = sorted(
        success_pool,
        key=lambda x: (-(x["su_per_gpu_h"] or 0.0), -x["strict_su"], -x["last_tick"]),
    )[:max_per_class]
    # Retain diagnostic-route successes even when direct productivity is unset.
    diagnostic_success = sorted(
        [
            x for x in success_pool
            if x["su_per_gpu_h"] is None and x.get("diagnostic_only_count", 0) > 0
        ],
        key=lambda x: (-x["strict_su"], -x["last_tick"]),
    )[:max_per_class]
    success_by_key: dict[str, dict[str, Any]] = {}
    for x in primary_success + diagnostic_success:
        success_by_key.setdefault(x["strategy_key"], x)
    success = list(success_by_key.values())
    near = sorted(
        [x for x in rendered if x["near_miss_count"] > 0],
        key=lambda x: (-x["near_miss_count"], x["failure_count"], -x["last_tick"]),
    )[:max_per_class]
    failed = sorted(
        [x for x in rendered if x["failure_count"] > 0 and x["strict_su"] == 0],
        key=lambda x: (-x["failure_count"], -x["gpu_h"], -x["last_tick"]),
    )[:max_per_class]
    diagnostic = sorted(
        [
            x for x in rendered
            if (x["diagnostic_only_count"] > 0 or x["advisory_scores"])
            and x["strict_su"] == 0
            and x["near_miss_count"] == 0
        ],
        key=lambda x: (
            -x["diagnostic_only_count"],
            -x["failure_count"],
            -x["gpu_h"],
            -x["last_tick"],
        ),
    )[:max(1, max_per_class // 2)]

    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in success + near + failed + diagnostic:
        key = (
            item["family"],
            item["operator_id"],
            json.dumps(item["config_delta"], sort_keys=True),
        )
        chosen.setdefault(key, item)
    return list(chosen.values())


def llm_health(model: str) -> LLMHealthSummary:
    """Default empty health; live producer updates from LLMCallRecord archive."""
    return LLMHealthSummary(
        model=model,
        last_calls_window=[],
        parse_fail_rate=0.0,
        schema_fail_rate=0.0,
        timeout_count=0,
        median_latency_s=0.0,
    )


# ---------------------------------------------------------------------------
# State classifier
# ---------------------------------------------------------------------------


def rescue_axis_concentration(axis_stats: dict[str, AxisStat]) -> float:
    """Return the largest normalized median deficit divided by their sum. Normalize by
    near-pass margins so measurement units do not dominate the comparison.
    """
    norm = [
        s.median_deficit / (NEAR_PASS_MARGINS.get(axis, 1.0) or 1.0)
        for axis, s in axis_stats.items()
        if s.median_deficit
    ]
    if not norm:
        return 0.0
    return max(norm) / sum(norm)


def classify_state(
    *,
    worker_gpu_h_last_3_ticks: float,
    completed_children_window: int,
    run_su_count_delta: int,
    duplicate_fraction: float | None,
    near_miss_count: int,
    axis_stats: dict[str, AxisStat],
    top_bin_share: float | None,
    cfg: StateClassifierConfig,
    cumulative_gpu_h: float | None = None,
    su_dedup_trusted: bool = True,
    charged_gpu_h_recent: float | None = None,
    gpu_h_since_last_su: float | None = None,
    strict_count_total: int | None = None,
    run_su_count_total: int | None = None,
    strict_per_su_total: float | None = None,
    strict_su_tm08_split_ratio: float | None = None,
) -> StateLabel:
    # Use cumulative compute for cold-start detection; older callers may supply only
    # window compute.
    cold_start_gpu_h = (
        cumulative_gpu_h if cumulative_gpu_h is not None
        else worker_gpu_h_last_3_ticks
    )
    if (
        cold_start_gpu_h < cfg.min_cumulative_gpu_h
        or completed_children_window < 3
    ):
        return "low_evidence"

    # Live classification uses worker compute. The optional charged denominator supports
    # offline comparisons.
    denom = max(
        charged_gpu_h_recent if charged_gpu_h_recent is not None
        else worker_gpu_h_last_3_ticks,
        cfg.eps_gpu_h,
    )
    su_rate = run_su_count_delta / denom
    # A lane is at a PRODUCTIVE rate when it is trustworthily producing new SU
    # fast enough — the shared condition behind both `productive` and
    # `productive_duplicate`. A productive-rate lane must NEVER be abandoned to
    # `stalled` just because it collapsed into one Foldseek basin (top_bin high);
    # it is still buying SU, so it belongs in productive_duplicate (keep exploit +
    # the diversity clamp loosens explore on top_bin>=0.70).
    productive_rate = (
        su_dedup_trusted
        and duplicate_fraction is not None
        and su_rate >= cfg.productive_su_rate
        and run_su_count_delta > 0
    )

    _strict_total = int(strict_count_total or 0)
    _su_total = int(run_su_count_total or 0)
    _strict_per_su = (
        float(strict_per_su_total)
        if strict_per_su_total is not None
        else ((_strict_total / _su_total) if _su_total > 0 else None)
    )
    _total_su_rate = (
        _su_total / max(float(cumulative_gpu_h or 0.0), cfg.eps_gpu_h)
        if _su_total > 0 and cumulative_gpu_h is not None
        else None
    )
    _tm08_not_splitting = (
        strict_su_tm08_split_ratio is None
        or strict_su_tm08_split_ratio <= cfg.strict_duplicate_max_tm08_split_ratio
    )
    strict_duplicate_collapse = (
        su_dedup_trusted
        and _strict_total >= cfg.strict_duplicate_min_strict_total
        and _su_total >= cfg.strict_duplicate_min_su_total
        and _strict_per_su is not None
        and _strict_per_su >= cfg.strict_duplicate_min_strict_per_su
        and _total_su_rate is not None
        and _total_su_rate <= cfg.strict_duplicate_max_total_su_per_gpu_h
        and run_su_count_delta <= cfg.strict_duplicate_max_recent_su_delta
        and _tm08_not_splitting
    )
    if strict_duplicate_collapse:
        return "strict_duplicate_collapse"

    if productive_rate and duplicate_fraction < cfg.productive_max_dup:
        return "productive"

    # A productive but duplicate-rich lane is still buying new SU. Keep it in
    # the productive_duplicate regime even if it also has concentrated near
    # misses; rescue can increase, but the classifier must not abandon exploit.
    if productive_rate:
        return "productive_duplicate"

    if (
        near_miss_count >= cfg.rescue_min_near_miss
        and rescue_axis_concentration(axis_stats) >= cfg.rescue_axis_concentration
    ):
        # Rescue-rich normally shadows deep_stall because it returns first. Do not
        # let a near-miss-rich but no-new-SU lane remain protected indefinitely:
        # once it reaches the same 12 worker-GPU-h stale interval as other stalls,
        # treat it as failed/stale rescue and let deep_stall exploration engage.
        if (
            gpu_h_since_last_su is not None
            and gpu_h_since_last_su >= cfg.rescue_deep_stall_gpu_h
        ):
            return "deep_stall"
        return "rescue_rich"

    # An UNTRUSTED SU dedup (degraded Foldseek) cannot prove new official SU.
    # Treat it as a dry plateau so the stalled/deep_stall escape can engage
    # instead of letting strict-but-unclustered records masquerade as progress.
    dry_non_rescue_plateau = (run_su_count_delta == 0) or (not su_dedup_trusted)
    if (
        dry_non_rescue_plateau
        or ((top_bin_share or 0) >= cfg.stalled_top_bin_share and not productive_rate)
    ):
        # Escalate prolonged absence of new SUs even when near misses are not
        # concentrated on one measurement.
        if (
            gpu_h_since_last_su is not None
            and gpu_h_since_last_su >= cfg.deep_stall_gpu_h
        ):
            return "deep_stall"
        return "stalled"

    # Distinguish productive but structurally repetitive routes from stalled ones.
    return "low_evidence"


# ---------------------------------------------------------------------------
# Top-level reducer
# ---------------------------------------------------------------------------


def _compute_diagnosis_outcomes_safe(
    all_results: list[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None = None,
    hypotheses: list[HypothesisCard] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute advisory diagnosis outcomes, returning an explicit _status row on error.

    Import lazily to avoid the evidence_reducer/diagnosis_outcome dependency cycle.
    """
    try:
        from .diagnosis_outcome import compute_diagnosis_outcomes
        return compute_diagnosis_outcomes(
            all_results, spawning_actions=spawning_actions, hypotheses=hypotheses)
    except Exception as exc:  # noqa: BLE001
        return {
            "_status": {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            }
        }


def build_refold_probe_outcomes(
    window_results: list[ResultRecord],
    all_results: list[ResultRecord],
) -> dict[str, Any]:
    """Summarize advisory refolds relative to their recorded parents.

    Join refold_pLDDT/iPAE/binder_scRMSD to the parent identified by
    bins["refilter_source"] and count four mutually exclusive outcomes:
      - structure_limited: the parent does not qualify, but the advisory refold
        measurements satisfy the same numeric thresholds.
      - confirmed_limited: neither the parent nor the advisory refold qualifies.
      - parent_already_passes: the parent's canonical measurements qualify.
      - unjoined: the parent identifier cannot be resolved.
    These labels describe measurement comparisons, not established causes.
    Their counts sum to `probed`; return {} when there are no advisory probes.
    Advisory refolds never change qualification or run_su_count, and parent
    qualification alone does not establish structural uniqueness.
    """
    by_rid = {r.result_id: r for r in all_results}
    probed = 0
    structure_limited = 0
    confirmed_limited = 0
    parent_already_passes = 0
    unjoined = 0
    examples: list[str] = []
    for r in window_results:
        m = r.metrics or {}
        if not all(
            k in m for k in ("refold_pLDDT", "refold_iPAE", "refold_binder_scRMSD")
        ):
            continue
        probed += 1
        parent_rid = (r.bins or {}).get("refilter_source")
        parent = by_rid.get(parent_rid) if parent_rid else None
        if parent is None:
            # refilter_source empty OR a basename/path (degraded no-Foldseek path):
            # the parent's canonical score is unknown, so we cannot classify this
            # probe as structure-limited. Count as unjoined and move on.
            unjoined += 1
            continue
        if parent.metrics and is_strict_success(parent.metrics):
            # The parent already qualifies under its canonical measurements.
            parent_already_passes += 1
            continue
        refold_pass = is_strict_success(
            {
                "pLDDT": m["refold_pLDDT"],
                "iPAE": m["refold_iPAE"],
                "binder_scRMSD": m["refold_binder_scRMSD"],
            }
        )
        if refold_pass:
            structure_limited += 1
            if parent_rid and parent_rid not in examples and len(examples) < 5:
                examples.append(parent_rid)
        else:
            confirmed_limited += 1
    if probed == 0:
        return {}
    out: dict[str, Any] = {
        "probed": probed,
        "structure_limited": structure_limited,
        "confirmed_limited": confirmed_limited,
    }
    if parent_already_passes:
        out["parent_already_passes"] = parent_already_passes
    if unjoined:
        out["unjoined"] = unjoined
    if examples:
        out["example_parent_ids"] = examples
    return out


def reduce_evidence(
    *,
    tick_id: str,
    target_id: str,
    target_class: str,
    elapsed_wall_h: float,
    remaining_wall_h: float,
    pending_children: int,
    worker_gpu_h_total: float,
    worker_wall_gpu_count: float | None = None,
    worker_wall_gpu_h_total: float | None = None,
    run_su_hwm: int | None = None,
    run_su_hwm_delta: int | None = None,
    gpu_h_since_last_su: float | None = None,
    ticks_since_last_su: int | None = None,
    all_results: list[ResultRecord],
    window_results: list[ResultRecord],
    run_su_count: int,
    run_su_count_delta: int,
    duplicate_fraction: float | None,
    near_miss_count: int,
    top_bin_share: float | None,
    panel_ready_count: int,
    panel_ready_bins_covered: int,
    llm_model: str,
    spawning_actions: dict[str, ActionCandidate] | None = None,
    route_health_summary: RouteHealthSummary | None = None,
    recent_fallback_rate: float = 0.0,
    enable_exemplars: bool = True,
    foldseek_su_status: str = "ok",
    foldseek_su_coverage: float | None = None,
    strict_su_top_bin_share: float | None = None,
    strict_su_tm08_status: str = "disabled",
    strict_su_tm08_coverage: float | None = None,
    strict_su_tm08_recent_count: int | None = None,
    strict_su_live_recent_count: int | None = None,
    strict_su_tm08_delta_vs_live: int | None = None,
    strict_su_tm08_live_split_ratio: float | None = None,
    strict_su_tm05_recent_count: int | None = None,
    strict_su_tm08_delta_vs_tm05: int | None = None,
    strict_su_tm08_split_ratio: float | None = None,
    strict_su_tm08_result_scope: str = "disabled",
    structure_dedup_scope: str = "legacy_or_unknown",
    structure_dedup_fallback_count: int = 0,
    foldseek_archive_status: str = "legacy_or_unknown",
    foldseek_archive_coverage: float | None = None,
    foldseek_archive_result_scope: str = "lifetime_or_unknown",
    whole_archive_structure_dedup_scope: str = "legacy_or_unknown",
    whole_archive_structure_dedup_fallback_count: int = 0,
    sequence_dedup_status: str = "disabled",
    sequence_dedup_coverage: float | None = None,
    near_miss_dedup_status: str = "legacy_or_unknown",
    near_miss_dedup_coverage: float | None = None,
    near_miss_cluster_by_result_id: dict[str, str] | None = None,
    seq_unique_strict_count: int | None = None,
    seq_unique_strict_delta: int | None = None,
    joint_struct_seq_unique_count: int | None = None,
    seq_duplicate_fraction: float | None = None,
    top_seq_bin_share: float | None = None,
    charged_gpu_count: float | None = None,
    charged_gpu_h_total: float | None = None,
    charged_gpu_h_recent: float | None = None,
    charged_gpu_h_scope: str = "unavailable",
    production_panel_status: str = "not_run",
    production_panel_value: float | None = None,
    production_panel_selected_ids: list[str] | None = None,
    production_panel_diversity_bins: dict[str, int] | None = None,
    production_panel_gap_reasons: list[str] | None = None,
    production_near_miss_ids: list[str] | None = None,
    dispatch_realization: dict[str, Any] | None = None,
    hypotheses: list[HypothesisCard] | None = None,
    cfg: ReducerConfig | None = None,
) -> EvidenceSummary:
    cfg = cfg or ReducerConfig()
    near_miss_signal_trusted = _near_miss_dedup_trusted(
        near_miss_dedup_status,
        near_miss_dedup_coverage,
        near_miss_count=near_miss_count,
    )

    axis_stats = {
        a: build_axis_stat(window_results, a, cfg)
        for a in ("pLDDT", "iPAE", "binder_scRMSD")
    }
    # Combine recent observations from every active family for diagnostic summaries.
    _diag_window = balanced_diagnostic_window(
        all_results, window_results, cfg.diagnostic_window_per_family
    )
    diagnostic_axis_stats = build_diagnostic_axis_stats(_diag_window)
    # Roll up native-generator scores from bins so the Planner can use them as
    # advisory signal for cross-paradigm score-conversion decisions. These are
    # not AF2-calibrated and therefore do not feed strict_success.
    diagnostic_alt_model_scores = build_diagnostic_alt_model_scores(
        _diag_window
    )
    jps = joint_patterns(window_results, cfg)
    examples = representative_examples(
        window_results, cfg,
        by_result_id={r.result_id: r for r in all_results},
        spawning_actions=spawning_actions,
    )
    mh = method_health(
        all_results,
        spawning_actions,
        near_miss_cluster_by_result_id=near_miss_cluster_by_result_id,
        near_miss_dedup_trusted=near_miss_signal_trusted,
    )
    # Overlay recent productivity on cumulative family evidence.
    mh_window = method_health(
        window_results,
        spawning_actions,
        near_miss_cluster_by_result_id=near_miss_cluster_by_result_id,
        near_miss_dedup_trusted=near_miss_signal_trusted,
    )
    # Credit only clusters first seen in the window; rediscovering an older cluster is
    # not a new SU.
    _window_ids = {r.result_id for r in window_results}
    _win_owner: dict[str, str] = {}
    _win_gpu_h: dict[str, float] = {}
    _by_rid_recent = {r.result_id: r for r in all_results}
    _spawn_recent = spawning_actions or {}
    _prewindow_su = {
        key for r in all_results
        if r.result_id not in _window_ids
        and _is_canonical_su_record(r, _spawn_recent)
        for key in [_su_key(r)]
        if key is not None
    }
    for r in window_results:
        _win_gpu_h[r.backend_family] = _win_gpu_h.get(r.backend_family, 0.0) + r.gpu_h
    # Match cumulative SU ownership when calculating recent rates: attribute canonical
    # evaluation outcomes to the generating family before considering advisory refolds.
    _win_strict_owned = [
        (r, resolve_generating_family(
            r, by_result_id=_by_rid_recent, spawning_actions=_spawn_recent))
        for r in window_results
        if _is_canonical_su_record(r, _spawn_recent)
    ]
    for r, _owner in sorted(
        _win_strict_owned, key=lambda ro: ro[1] == "structure_refilter"
    ):
        _key = _su_key(r)
        if _key is not None:
            _win_owner.setdefault(_key, _owner)
    _fam_marginal_su: dict[str, int] = {}
    for _k, _owner in _win_owner.items():
        if _k not in _prewindow_su:
            _fam_marginal_su[_owner] = _fam_marginal_su.get(_owner, 0) + 1
    def _near_miss_key_for_recent(r: ResultRecord) -> str:
        _b = r.bins or {}
        return (
            near_miss_cluster_by_result_id.get(r.result_id)
            or _b.get("foldseek_near_miss")
            or _b.get("foldseek")
            or _b.get("refilter_source")
            or r.result_id
        )
    _by_all = {r.result_id: r for r in all_results}
    _spawn = spawning_actions or {}
    _score_lineage_cache: dict[str, tuple[ResultRecord, ...]] = {}

    def _cached_score_lineage_all(r: ResultRecord) -> tuple[ResultRecord, ...]:
        cached = _score_lineage_cache.get(r.result_id)
        if cached is None:
            cached = tuple(_score_conversion_lineage_records(
                r, by_result_id=_by_all, spawning_actions=_spawn,
            ))
            _score_lineage_cache[r.result_id] = cached
        return cached
    _prewindow_nm = {
        _near_miss_key_for_recent(r) for r in all_results
        if near_miss_signal_trusted
        and r.result_id not in _window_ids
        and r.exit_status == "ok" and is_near_miss(r.metrics)
    }
    _win_nm_owner: dict[str, str] = {}
    if near_miss_signal_trusted:
        for r in window_results:
            if r.exit_status == "ok" and is_near_miss(r.metrics):
                _owner = resolve_generating_family(
                    r, by_result_id=_by_all, spawning_actions=_spawn,
                )
                _win_nm_owner.setdefault(_near_miss_key_for_recent(r), _owner)
    _fam_marginal_nm: dict[str, int] = {}
    for _k, _owner in _win_nm_owner.items():
        if _k not in _prewindow_nm:
            _fam_marginal_nm[_owner] = _fam_marginal_nm.get(_owner, 0) + 1
    # Resolve evaluated outputs against the full archive because their generating
    # parents may predate the recent window.
    _win_chain_downstream_gpu_h: dict[str, float] = {}
    _win_chain_parent_gpu_h: dict[str, float] = {}
    _win_chain_parent_ids: dict[str, set[str]] = {}
    _win_chain_owner: dict[str, str] = {}  # chained cluster key -> upstream family
    for r in window_results:
        _pf = _diagnostic_parent_family(r, by_result_id=_by_all, spawning_actions=_spawn)
        if _pf is None:
            continue
        # Include failed evaluations and charge an out-of-window parent once when its
        # evaluated child falls within the window.
        _win_chain_downstream_gpu_h[_pf] = _win_chain_downstream_gpu_h.get(_pf, 0.0) + r.gpu_h
        for _parent in _cached_score_lineage_all(r):
            if _parent.result_id in _window_ids:
                continue
            _seen = _win_chain_parent_ids.setdefault(_pf, set())
            if _parent.result_id in _seen:
                continue
            _seen.add(_parent.result_id)
            _win_chain_parent_gpu_h[_pf] = (
                _win_chain_parent_gpu_h.get(_pf, 0.0)
                + float(_parent.gpu_h or 0.0)
            )
        if _is_canonical_su_record(r, _spawn):
            _key = _su_key(r)
            if _key is not None:
                _win_chain_owner.setdefault(_key, _pf)
    _fam_marginal_chained: dict[str, int] = {}
    for _k, _pf in _win_chain_owner.items():
        if _k not in _prewindow_su:
            _fam_marginal_chained[_pf] = _fam_marginal_chained.get(_pf, 0) + 1
    for fam, s in list(mh.items()):
        w = mh_window.get(fam)
        g = _win_gpu_h.get(fam, 0.0)
        if w is not None and w.su_per_gpu_h is None:
            recent = None          # refilter / diagnostic-only: not rate-rankable
        elif g > 0:
            recent = _fam_marginal_su.get(fam, 0) / g
        else:
            recent = None
        # chained-recent: marginal chained SU / recent ROUTE gpu_h (upstream window
        # gpu_h + downstream refilter window gpu_h). None when the lane produced no
        # NEW chained SU this window — so a dry diagnostic lane reads as dry.
        chained_n = _fam_marginal_chained.get(fam, 0)
        chained_route_g = (
            g
            + _win_chain_downstream_gpu_h.get(fam, 0.0)
            + _win_chain_parent_gpu_h.get(fam, 0.0)
        )
        chained_recent = (chained_n / chained_route_g) if (chained_n and chained_route_g > 0) else None
        mh[fam] = replace(
            s,
            su_per_gpu_h_recent=recent,
            near_miss_yield_recent=_fam_marginal_nm.get(fam, 0),
            chained_strict_yield_su_recent=chained_n,
            chained_su_per_gpu_h_recent=chained_recent,
        )
    rh = route_health_summary or route_health()
    lh = llm_health(llm_model)

    gpu_h_window = sum(r.gpu_h for r in window_results)
    strict_count_total = sum(m.strict_yield for m in mh.values())
    strict_per_su_total = (
        strict_count_total / run_su_count if run_su_count > 0 else None
    )
    state = classify_state(
        worker_gpu_h_last_3_ticks=gpu_h_window,
        completed_children_window=len(window_results),
        run_su_count_delta=run_su_count_delta,
        duplicate_fraction=duplicate_fraction,
        near_miss_count=(near_miss_count if near_miss_signal_trusted else 0),
        axis_stats=axis_stats,
        top_bin_share=top_bin_share,
        cfg=cfg.state,
        # Cold-start gate uses cumulative GPU-h (family-mix-independent),
        # not the per-record window sum which is unreachable for Complexa.
        cumulative_gpu_h=worker_gpu_h_total,
        su_dedup_trusted=(
            foldseek_su_status == "ok"
            and foldseek_su_coverage is not None
            and foldseek_su_coverage >= 0.999
            and duplicate_fraction is not None
            # Incomplete binder-chain extraction makes structural deduplication
            # untrusted.
            and structure_dedup_fallback_count == 0
        ),
        # Use worker-window compute for the productive-state gate.
        charged_gpu_h_recent=None,
        gpu_h_since_last_su=gpu_h_since_last_su,
        strict_count_total=strict_count_total,
        run_su_count_total=run_su_count,
        strict_per_su_total=strict_per_su_total,
        strict_su_tm08_split_ratio=strict_su_tm08_split_ratio,
    )

    metric_avail: dict[str, dict[str, bool]] = {}
    for r in all_results:
        d = metric_avail.setdefault(r.backend_family, {})
        for k in ("pLDDT", "iPAE", "binder_scRMSD"):
            d[k] = d.get(k, False) or (k in r.metrics)

    su_per_gpu_h = run_su_count_delta / gpu_h_window if gpu_h_window > 0 else None
    # Cumulative route-attributed worker-compute rate.
    run_su_per_worker_gpu_h_total = (
        run_su_count / worker_gpu_h_total
        if worker_gpu_h_total and worker_gpu_h_total > 0
        else None
    )
    run_su_per_worker_wall_gpu_h_total = (
        run_su_count / worker_wall_gpu_h_total
        if worker_wall_gpu_h_total is not None and worker_wall_gpu_h_total > 0
        else None
    )
    run_su_hwm_per_worker_wall_gpu_h_total = (
        run_su_hwm / worker_wall_gpu_h_total
        if run_su_hwm is not None and worker_wall_gpu_h_total is not None and worker_wall_gpu_h_total > 0
        else None
    )
    run_su_per_charged_gpu_h_total = (
        run_su_count / charged_gpu_h_total
        if charged_gpu_h_total is not None and charged_gpu_h_total > 0
        else None
    )
    run_su_per_charged_gpu_h_recent = (
        run_su_count_delta / charged_gpu_h_recent
        if charged_gpu_h_recent is not None and charged_gpu_h_recent > 0
        else None
    )

    # Parse controller tick IDs consistently with recipe recency.
    current_tick_int: int = 0
    if tick_id.startswith("v7r") and tick_id[3:].isdigit():
        current_tick_int = int(tick_id[3:])
    elif tick_id.isdigit():
        current_tick_int = int(tick_id)
    recipes = extract_recipes(
        all_results,
        spawning_action=spawning_actions or {},
        target_class=target_class,
        current_tick=current_tick_int,
        cfg=cfg,
    )

    # Recent fallback frequency controls conditional allocation bounds.
    recent_fallback_high = recent_fallback_rate >= 0.30

    # Count refinement attempts over the full lineage, including results outside the recent window.
    stuck_roots = stuck_lineage_roots(all_results, cfg)
    parent_artifact_result_ids = sorted(
        r.result_id for r in all_results if _has_usable_parent_artifact(r)
    )

    # best-K / near-miss-K concrete binders WITH their full setup + metrics,
    # over the WHOLE run (best = all-time strongest to build on; near-miss =
    # closest failures to diagnose). spawning_actions supplies the config/operator.
    # enable_exemplars=False is the ablation OFF arm (the Planner sees no
    # best/near-miss exemplar block) — for the with/without A/B.
    exemplars = build_exemplars(
        all_results, spawning_actions, cfg,
        k_best=cfg.max_exemplars_best, k_near=cfg.max_exemplars_near,
    ) if enable_exemplars else []
    strategy_feedback = build_strategy_feedback(
        all_results, spawning_actions, cfg,
        near_miss_dedup_trusted=near_miss_signal_trusted,
    )
    route_values = build_route_values(
        all_results, window_results, spawning_actions, cfg.state,
        near_miss_dedup_trusted=near_miss_signal_trusted,
        gpu_h_since_last_su=gpu_h_since_last_su,
    )
    refilter_roles = refilter_role_health(all_results, spawning_actions)

    return EvidenceSummary(
        tick_id=tick_id,
        target_id=target_id,
        target_class=target_class,
        schema_version=SCHEMA_VERSION,
        elapsed_wall_h=elapsed_wall_h,
        remaining_wall_h=remaining_wall_h,
        completed_children=len(all_results),
        pending_children=pending_children,
        worker_gpu_h_total=worker_gpu_h_total,
        worker_gpu_h_last_3_ticks=gpu_h_window,
        gpu_h_since_last_su=gpu_h_since_last_su,
        ticks_since_last_su=ticks_since_last_su,
        strict_count=strict_count_total,
        # global_new stores the cumulative structurally unique qualified count.
        global_new_strict=run_su_count,
        run_su_count=run_su_count,
        run_su_count_delta=run_su_count_delta,
        strict_per_su_total=strict_per_su_total,
        strict_duplicate_collapse_signal=(state == "strict_duplicate_collapse"),
        foldseek_su_status=foldseek_su_status,
        foldseek_su_coverage=foldseek_su_coverage,
        strict_su_top_bin_share=strict_su_top_bin_share,
        strict_su_tm08_status=strict_su_tm08_status,
        strict_su_tm08_coverage=strict_su_tm08_coverage,
        strict_su_tm08_recent_count=strict_su_tm08_recent_count,
        strict_su_live_recent_count=strict_su_live_recent_count,
        strict_su_tm08_delta_vs_live=strict_su_tm08_delta_vs_live,
        strict_su_tm08_live_split_ratio=strict_su_tm08_live_split_ratio,
        strict_su_tm05_recent_count=strict_su_tm05_recent_count,
        strict_su_tm08_delta_vs_tm05=strict_su_tm08_delta_vs_tm05,
        strict_su_tm08_split_ratio=strict_su_tm08_split_ratio,
        strict_su_tm08_result_scope=strict_su_tm08_result_scope,
        structure_dedup_scope=structure_dedup_scope,
        structure_dedup_fallback_count=structure_dedup_fallback_count,
        foldseek_archive_status=foldseek_archive_status,
        foldseek_archive_coverage=foldseek_archive_coverage,
        foldseek_archive_result_scope=foldseek_archive_result_scope,
        whole_archive_structure_dedup_scope=whole_archive_structure_dedup_scope,
        whole_archive_structure_dedup_fallback_count=whole_archive_structure_dedup_fallback_count,
        near_miss_dedup_status=near_miss_dedup_status,
        near_miss_dedup_coverage=near_miss_dedup_coverage,
        sequence_dedup_status=sequence_dedup_status,
        sequence_dedup_coverage=sequence_dedup_coverage,
        seq_unique_strict_count=seq_unique_strict_count,
        seq_unique_strict_delta=seq_unique_strict_delta,
        joint_struct_seq_unique_count=joint_struct_seq_unique_count,
        seq_duplicate_fraction=seq_duplicate_fraction,
        top_seq_bin_share=top_seq_bin_share,
        run_su_per_worker_gpu_h_total=run_su_per_worker_gpu_h_total,
        worker_wall_gpu_count=worker_wall_gpu_count,
        worker_wall_gpu_h_total=worker_wall_gpu_h_total,
        run_su_per_worker_wall_gpu_h_total=run_su_per_worker_wall_gpu_h_total,
        run_su_hwm=run_su_hwm,
        run_su_hwm_delta=run_su_hwm_delta,
        run_su_hwm_per_worker_wall_gpu_h_total=run_su_hwm_per_worker_wall_gpu_h_total,
        charged_gpu_count=charged_gpu_count,
        charged_gpu_h_total=charged_gpu_h_total,
        charged_gpu_h_recent=charged_gpu_h_recent,
        charged_gpu_h_scope=charged_gpu_h_scope,
        run_su_per_charged_gpu_h_total=run_su_per_charged_gpu_h_total,
        run_su_per_charged_gpu_h_recent=run_su_per_charged_gpu_h_recent,
        production_panel_status=production_panel_status,
        production_panel_value=production_panel_value,
        production_panel_selected_ids=list(production_panel_selected_ids or []),
        production_panel_diversity_bins=dict(production_panel_diversity_bins or {}),
        production_panel_gap_reasons=list(production_panel_gap_reasons or []),
        production_near_miss_ids=list(production_near_miss_ids or []),
        parent_artifact_result_ids=parent_artifact_result_ids,
        su_per_gpu_h_recent=su_per_gpu_h,
        duplicate_fraction=duplicate_fraction,
        top_bin_share=top_bin_share,
        axis_stats=axis_stats,
        joint_patterns=jps,
        near_miss_count=near_miss_count,
        diagnostic_axis_stats=diagnostic_axis_stats,
        diagnostic_alt_model_scores=diagnostic_alt_model_scores,
        # Use a lazy import to avoid the evidence_reducer/diagnosis_outcome dependency cycle.
        diagnosis_outcomes=_compute_diagnosis_outcomes_safe(
            all_results, spawning_actions, hypotheses),
        # Join advisory refolds to canonical parent scores for subsequent planning.
        refold_probe_outcomes=build_refold_probe_outcomes(
            window_results, all_results),
        panel_ready_count=panel_ready_count,
        panel_ready_bins_covered=panel_ready_bins_covered,
        method_health=mh,
        refilter_role_health=refilter_roles,
        route_health=rh,
        llm_health=lh,
        state_label=state,
        examples=examples,
        metric_availability=metric_avail,
        recipes=recipes,
        strategy_feedback=strategy_feedback,
        route_values=route_values,
        exemplars=exemplars,
        stuck_lineage_roots=stuck_roots,
        recent_fallback_high=recent_fallback_high,
        dispatch_realization=dict(dispatch_realization or {}),
    )
