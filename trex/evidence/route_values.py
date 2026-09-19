"""Deterministic exact-route and family value summaries.

The reducer supplies immutable records plus an explicit configuration and
diagnostic evaluator. This module owns route cost attribution, unique-SU
ownership, status classification, and bounded presentation ordering.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..capability_registry import default_registry
from ..panel import strict_margin_quality, strict_margin_units
from ..refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from ..schemas import ActionCandidate, ResultRecord, RouteValueSummary
from ..score_conversion import high_value_pending_score_conversion
from ..success_criteria import STRICT_SUCCESS, is_near_miss
from .contracts import RouteDiagnosticEvaluator, RouteValueConfig
from .route_status import (
    classify_marginal_route_status,
    classify_route_status,
)
from .attribution import (
    is_canonical_su_record,
    _lineage_parent_records,
    _refilter_role_for_record,
    _score_conversion_lineage_records,
    _score_conversion_parent_record,
    _su_key,
)
from .route_identity import (
    RouteIdentity,
    _route_identity_for_record,
    route_role_label,
)


@dataclass
class RouteAccumulator:
    """Mutable internal ledger for one exact scientific route."""

    strategy_key: str
    root_family: str | None
    action_family: str
    scoring_family: str | None
    operator_id: str | None
    config_signature: str
    refilter_role: str | None
    config_delta: dict[str, Any]
    parent_strategy_key: str | None
    route_gpu_h: float = 0.0
    generator_gpu_h: float = 0.0
    canonical_refilter_gpu_h: float = 0.0
    canonical_score_conversion_count: int = 0
    attempts: int = 0
    completions: int = 0
    strict_count: int = 0
    near_miss_count: int = 0
    near_miss_recent: int = 0
    recent_route_gpu_h: float = 0.0
    gpu_recent_route_gpu_h: float = 0.0
    medium_recent_route_gpu_h: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)
    strict_bins: dict[str, int] = field(default_factory=dict)
    su_bins: set[str] = field(default_factory=set)
    recent_su_bins: set[str] = field(default_factory=set)
    gpu_recent_su_bins: set[str] = field(default_factory=set)
    medium_recent_su_bins: set[str] = field(default_factory=set)
    lineage_charge_ids: set[str] = field(default_factory=set)
    recent_parent_charge_ids: set[str] = field(default_factory=set)
    gpu_recent_parent_charge_ids: set[str] = field(default_factory=set)
    medium_recent_parent_charge_ids: set[str] = field(default_factory=set)
    diagnostic_record_ids: list[str] = field(default_factory=list)
    strict_quality_by_bin: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteAggregation:
    """Intermediate route ledger plus indexes needed for summarization."""

    routes: dict[str, RouteAccumulator]
    route_key_by_result_id: dict[str, str]
    result_by_id: dict[str, ResultRecord]
    medium_recent_result_ids: set[str]


@dataclass(frozen=True)
class PendingScoreConversionCounts:
    """Pending canonical-scoring counts at route and family scopes."""

    by_route: dict[str, int]
    by_family: dict[str, int]
    promising_by_route: dict[str, int]
    promising_by_family: dict[str, int]


def _aggregate_route_evidence(
    results: list[ResultRecord],
    window_results: list[ResultRecord],
    spawning_action_by_result_id: dict[str, ActionCandidate],
    config: RouteValueConfig,
    near_miss_dedup_trusted: bool,
) -> RouteAggregation:
    """Attribute compute, strict records, and unique SU to exact routes."""
    by_result_id = {result.result_id: result for result in results}
    lineage_cache: dict[str, tuple[ResultRecord, ...]] = {}
    score_lineage_cache: dict[str, tuple[ResultRecord, ...]] = {}

    def _cached_lineage(result: ResultRecord) -> tuple[ResultRecord, ...]:
        cached = lineage_cache.get(result.result_id)
        if cached is None:
            cached = tuple(_lineage_parent_records(
                result, by_result_id=by_result_id, spawning_actions=spawning_action_by_result_id,
            ))
            lineage_cache[result.result_id] = cached
        return cached

    def _cached_score_lineage(result: ResultRecord) -> tuple[ResultRecord, ...]:
        cached = score_lineage_cache.get(result.result_id)
        if cached is None:
            cached = tuple(_score_conversion_lineage_records(
                result, by_result_id=by_result_id, spawning_actions=spawning_action_by_result_id,
            ))
            score_lineage_cache[result.result_id] = cached
        return cached

    window_ids = {result.result_id for result in window_results}
    def _gpu_suffix_ids(target_h: float) -> set[str]:
        ids: set[str] = set()
        acc_h = 0.0
        target = max(0.0, float(target_h))
        for suffix_result in reversed(results):
            ids.add(suffix_result.result_id)
            acc_h += float(suffix_result.gpu_h or 0.0)
            if acc_h >= target:
                break
        return ids

    gpu_recent_ids = _gpu_suffix_ids(config.route_gpu_recent_window_h)
    medium_recent_ids = _gpu_suffix_ids(config.route_gpu_medium_window_h)
    prewindow_su = {
        key for result in results
        if result.result_id not in window_ids
        and is_canonical_su_record(result, spawning_action_by_result_id)
        for key in [_su_key(result)]
        if key is not None
    }
    pre_gpu_recent_su = {
        key for result in results
        if result.result_id not in gpu_recent_ids
        and is_canonical_su_record(result, spawning_action_by_result_id)
        for key in [_su_key(result)]
        if key is not None
    }
    pre_medium_recent_su = {
        key for result in results
        if result.result_id not in medium_recent_ids
        and is_canonical_su_record(result, spawning_action_by_result_id)
        for key in [_su_key(result)]
        if key is not None
    }

    routes: dict[str, RouteAccumulator] = {}

    def _get_or_create_route(info: RouteIdentity) -> RouteAccumulator:
        strategy_key = info["strategy_key"]
        route = routes.get(strategy_key)
        if route is None:
            route = RouteAccumulator(
                strategy_key=strategy_key,
                root_family=info["root_family"],
                action_family=info["action_family"],
                scoring_family=info["scoring_family"],
                operator_id=info["operator_id"],
                config_signature=info["config_signature"],
                refilter_role=info["refilter_role"],
                config_delta=dict(info["config_delta"]),
                parent_strategy_key=info["parent_strategy_key"],
            )
            routes[strategy_key] = route
        elif route.scoring_family is None and info["scoring_family"]:
            route.scoring_family = info["scoring_family"]

        # Prefer an explicit action role (parent-model refold / diagnostic score).
        # Canonical score conversion describes a plain generator route only when
        # no more specific action role has already been observed.
        info_role = info["refilter_role"]
        if info_role and (
            route.refilter_role is None
            or info_role != CANONICAL_SCORE_CONVERSION
        ):
            route.refilter_role = info_role
        return route

    route_key_by_result_id: dict[str, str] = {}
    route_info_by_result_id: dict[str, RouteIdentity] = {}
    for result in results:
        info = _route_identity_for_record(
            result, by_result_id=by_result_id, spawning_actions=spawning_action_by_result_id,
        )
        route_info_by_result_id[result.result_id] = info
        _get_or_create_route(info)
        route_key_by_result_id[result.result_id] = info["strategy_key"]
    for result in results:
        info = route_info_by_result_id[result.result_id]
        route = _get_or_create_route(info)
        result_gpu_h = float(result.gpu_h or 0.0)
        route.route_gpu_h += result_gpu_h
        route.attempts += 1
        if result.exit_status in ("ok", "no_artifacts"):
            route.completions += 1
        if result.backend_family == "structure_refilter" and info.get("refilter_role") == CANONICAL_SCORE_CONVERSION:
            route.canonical_refilter_gpu_h += result_gpu_h
            route.canonical_score_conversion_count += 1
        else:
            route.generator_gpu_h += result_gpu_h
        if result.result_id in window_ids:
            route.recent_route_gpu_h += result_gpu_h
        if result.result_id in gpu_recent_ids:
            route.gpu_recent_route_gpu_h += result_gpu_h
        if result.result_id in medium_recent_ids:
            route.medium_recent_route_gpu_h += result_gpu_h

        # Composite-route economics: parent-bound actions such as
        # Complexa->proteinmpnn_redesign need the parent generator GPU-h to be
        # charged to the route that consumes it. Charge each concrete ancestor
        # once per route bucket; direct parent records still keep their own
        # separate rows, so route rows are per-strategy costs, not a partition of
        # total worker GPU-h.
        for parent in _cached_lineage(result):
            if route_key_by_result_id.get(parent.result_id) == info["strategy_key"]:
                continue
            parent_gpu = float(parent.gpu_h or 0.0)
            charged_lifetime = route.lineage_charge_ids
            if parent.result_id not in charged_lifetime:
                route.route_gpu_h += parent_gpu
                route.generator_gpu_h += parent_gpu
                charged_lifetime.add(parent.result_id)
            if result.result_id in window_ids and parent.result_id not in window_ids:
                charged = route.recent_parent_charge_ids
                if parent.result_id not in charged:
                    route.recent_route_gpu_h += parent_gpu
                    charged.add(parent.result_id)
            if result.result_id in gpu_recent_ids and parent.result_id not in gpu_recent_ids:
                charged = route.gpu_recent_parent_charge_ids
                if parent.result_id not in charged:
                    route.gpu_recent_route_gpu_h += parent_gpu
                    charged.add(parent.result_id)
            if result.result_id in medium_recent_ids and parent.result_id not in medium_recent_ids:
                charged = route.medium_recent_parent_charge_ids
                if parent.result_id not in charged:
                    route.medium_recent_route_gpu_h += parent_gpu
                    charged.add(parent.result_id)

        if len(route.evidence_refs) < 64:
            route.evidence_refs.append(result.result_id)
        route.diagnostic_record_ids.append(result.result_id)
        if is_canonical_su_record(result, spawning_action_by_result_id):
            route.strict_count += 1
            su_key = _su_key(result)
            if su_key is not None:
                bins = route.strict_bins
                bins[su_key] = int(bins.get(su_key, 0)) + 1
                # Keep one coherent candidate per exact route/Foldseek bin.
                # Repeated structures cannot inflate the sample count, while a
                # later sequence/refinement improvement in the same structural
                # family remains visible to the secondary quality objective.
                quality_row = {
                    "quality": strict_margin_quality(result),
                    "margins": strict_margin_units(result),
                }
                prior_quality = route.strict_quality_by_bin.get(su_key)
                if (
                    prior_quality is None
                    or quality_row["quality"] > prior_quality["quality"]
                ):
                    route.strict_quality_by_bin[su_key] = quality_row
        elif (
            near_miss_dedup_trusted
            and result.exit_status == "ok"
            and is_near_miss(result.metrics)
        ):
            route.near_miss_count += 1
            if result.result_id in window_ids:
                route.near_miss_recent += 1

        # Delayed score-conversion guard: if a canonical AF2 score-conversion
        # child is recent but its generator/redesign parent chain fell outside
        # the record/GPU recent window, the route still paid that upstream
        # compute to buy the converted SU. Charge each concrete ancestor once
        # per window so recent route value cannot explode on refilter-only
        # denominators.
        if (
            result.backend_family == "structure_refilter"
            and info.get("refilter_role") == CANONICAL_SCORE_CONVERSION
        ):
            for parent in _cached_score_lineage(result):
                parent_gpu = float(parent.gpu_h or 0.0)
                if result.result_id in window_ids and parent.result_id not in window_ids:
                    charged = route.recent_parent_charge_ids
                    if parent.result_id not in charged:
                        route.recent_route_gpu_h += parent_gpu
                        charged.add(parent.result_id)
                if result.result_id in gpu_recent_ids and parent.result_id not in gpu_recent_ids:
                    charged = route.gpu_recent_parent_charge_ids
                    if parent.result_id not in charged:
                        route.gpu_recent_route_gpu_h += parent_gpu
                        charged.add(parent.result_id)
                if result.result_id in medium_recent_ids and parent.result_id not in medium_recent_ids:
                    charged = route.medium_recent_parent_charge_ids
                    if parent.result_id not in charged:
                        route.medium_recent_route_gpu_h += parent_gpu
                        charged.add(parent.result_id)

    # Assign each Foldseek SU cluster to exactly one route, mirroring the
    # run-level/per-family one-owner invariant.
    cluster_owner: dict[str, str] = {}
    for result in results:
        if not is_canonical_su_record(result, spawning_action_by_result_id):
            continue
        key = _su_key(result)
        route_key = route_key_by_result_id.get(result.result_id)
        if key is None or route_key is None:
            continue
        cluster_owner.setdefault(key, route_key)
    for su_key, route_key in cluster_owner.items():
        if route_key not in routes:
            continue
        routes[route_key].su_bins.add(su_key)
        if su_key not in prewindow_su:
            routes[route_key].recent_su_bins.add(su_key)
        if su_key not in pre_gpu_recent_su:
            routes[route_key].gpu_recent_su_bins.add(su_key)
        if su_key not in pre_medium_recent_su:
            routes[route_key].medium_recent_su_bins.add(su_key)

    return RouteAggregation(
        routes=routes,
        route_key_by_result_id=route_key_by_result_id,
        result_by_id=by_result_id,
        medium_recent_result_ids=medium_recent_ids,
    )


def _count_pending_score_conversions(
    results: list[ResultRecord],
    aggregation: RouteAggregation,
    spawning_action_by_result_id: dict[str, ActionCandidate],
) -> PendingScoreConversionCounts:
    """Count unscored artifacts without changing route-value semantics."""
    by_result_id = aggregation.result_by_id
    route_key_by_result_id = aggregation.route_key_by_result_id
    scored_parent_ids: set[str] = set()
    for result in results:
        if (
            result.backend_family == "structure_refilter"
            and _refilter_role_for_record(result, spawning_actions=spawning_action_by_result_id) == CANONICAL_SCORE_CONVERSION
        ):
            parent = _score_conversion_parent_record(
                result, by_result_id=by_result_id, spawning_actions=spawning_action_by_result_id,
            )
            if parent is not None:
                scored_parent_ids.add(parent.result_id)

    def _has_scoreable_artifact(result: ResultRecord) -> bool:
        artifacts = result.artifacts or {}
        artifact_path = artifacts.get("pdb_path") or artifacts.get("cif_path")
        if artifact_path and Path(str(artifact_path)).exists():
            return True
        artifact_dir = artifacts.get("pdb_dir")
        if artifact_dir and Path(str(artifact_dir)).exists():
            d_path = Path(str(artifact_dir))
            return any(d_path.glob("*.pdb")) or any(d_path.glob("*.cif")) or any(d_path.glob("*.mmcif"))
        return False

    def _record_has_canonical_axes(result: ResultRecord) -> bool:
        metrics = result.metrics or {}
        return all(isinstance(metrics.get(axis_name), (int, float)) for axis_name in ("pLDDT", "iPAE", "binder_scRMSD"))

    def _needs_score_conversion(result: ResultRecord) -> bool:
        capability = default_registry().get(result.backend_family)
        if capability is not None and getattr(capability, "outputs_diagnostic_only", False):
            return True
        if str(result.backend_family).startswith("complexa_") and not _record_has_canonical_axes(result):
            return True
        return False

    pending_score_route_count: dict[str, int] = {}
    pending_score_family_count: dict[str, int] = {}
    pending_promising_score_route_count: dict[str, int] = {}
    pending_promising_score_family_count: dict[str, int] = {}
    for result in results:
        if (
            result.exit_status == "ok"
            and result.result_id not in scored_parent_ids
            and _needs_score_conversion(result)
            and _has_scoreable_artifact(result)
        ):
            route_key = route_key_by_result_id.get(result.result_id)
            is_promising_pending = high_value_pending_score_conversion(result)
            if route_key:
                pending_score_route_count[route_key] = pending_score_route_count.get(route_key, 0) + 1
                if is_promising_pending:
                    pending_promising_score_route_count[route_key] = pending_promising_score_route_count.get(route_key, 0) + 1
            pending_score_family_count[result.backend_family] = pending_score_family_count.get(result.backend_family, 0) + 1
            if is_promising_pending:
                pending_promising_score_family_count[result.backend_family] = pending_promising_score_family_count.get(result.backend_family, 0) + 1

    return PendingScoreConversionCounts(
        by_route=pending_score_route_count,
        by_family=pending_score_family_count,
        promising_by_route=pending_promising_score_route_count,
        promising_by_family=pending_promising_score_family_count,
    )


def _build_exact_route_summaries(
    aggregation: RouteAggregation,
    pending_score_conversions: PendingScoreConversionCounts,
    config: RouteValueConfig,
    route_diagnostic_improvement: RouteDiagnosticEvaluator,
) -> tuple[list[RouteValueSummary], float]:
    """Project exact-route accumulators into immutable public summaries."""
    routes = aggregation.routes
    by_result_id = aggregation.result_by_id
    medium_recent_ids = aggregation.medium_recent_result_ids
    pending_score_route_count = pending_score_conversions.by_route
    pending_promising_score_route_count = pending_score_conversions.promising_by_route
    mature_rates: list[float] = []
    for route in routes.values():
        unique_su_count = len(route.su_bins)
        route_gpu_h = float(route.route_gpu_h or 0.0)
        if unique_su_count >= 2 and route_gpu_h >= 2.0:
            mature_rates.append(unique_su_count / route_gpu_h)
    best_rate = max(mature_rates, default=0.0)

    route_rows: list[RouteValueSummary] = []
    for route in routes.values():
        su_bins = set(route.su_bins)
        recent_su_bins = set(route.recent_su_bins)
        gpu_recent_su_bins = set(route.gpu_recent_su_bins)
        medium_recent_su_bins = set(route.medium_recent_su_bins)
        route_gpu = float(route.route_gpu_h or 0.0)
        recent_gpu = float(route.recent_route_gpu_h or 0.0)
        gpu_recent_gpu = float(route.gpu_recent_route_gpu_h or 0.0)
        medium_recent_gpu = float(route.medium_recent_route_gpu_h or 0.0)
        new_su = len(su_bins)
        recent_su = len(recent_su_bins)
        gpu_recent_su = len(gpu_recent_su_bins)
        medium_recent_su = len(medium_recent_su_bins)
        rate = (new_su / route_gpu) if new_su and route_gpu > 0 else None
        recent_rate = (recent_su / recent_gpu) if recent_su and recent_gpu > 0 else None
        gpu_recent_rate = (gpu_recent_su / gpu_recent_gpu) if gpu_recent_su and gpu_recent_gpu > 0 else None
        medium_recent_rate = (medium_recent_su / medium_recent_gpu) if medium_recent_su and medium_recent_gpu > 0 else None
        strict_count = int(route.strict_count or 0)
        strict_per_su = (strict_count / new_su) if new_su else None
        strict_bins = route.strict_bins or {}
        duplicate_fraction = (
            1.0 - (len(strict_bins) / max(1, sum(int(bin_count) for bin_count in strict_bins.values())))
            if strict_bins else None
        )
        action_family = str(route.action_family)
        pending_score_conversion_count = int(
            pending_score_route_count.get(str(route.strategy_key), 0)
        )
        pending_promising_score_conversion_count = int(
            pending_promising_score_route_count.get(str(route.strategy_key), 0)
        )
        capability = default_registry().get(action_family)
        status = classify_route_status(
            config=config,
            route_gpu_h=route_gpu, new_su=new_su, recent_su=recent_su,
            medium_recent_su=medium_recent_su,
            near_recent=int(route.near_miss_recent or 0),
            recent_route_gpu_h=recent_gpu, rate=rate,
            best_rate=best_rate, strict_per_su=strict_per_su,
            duplicate_bin_fraction=duplicate_fraction,
            role=(getattr(capability, "role", None) if capability is not None else None),
            family=action_family,
            refilter_role=route.refilter_role,
            pending_score_conversion_count=pending_score_conversion_count,
            pending_promising_score_conversion_count=pending_promising_score_conversion_count,
            completions=int(route.completions or 0),
            completion_dry_enabled=True,
        )
        marginal_status = classify_marginal_route_status(
            config=config,
            route_gpu_h=route_gpu, new_su=new_su, strict_per_su=strict_per_su,
            duplicate_bin_fraction=duplicate_fraction, lifetime_rate=rate,
            record_recent_su=recent_su, record_recent_rate=recent_rate,
            gpu_recent_su=gpu_recent_su, gpu_recent_gpu_h=gpu_recent_gpu,
            gpu_recent_rate=gpu_recent_rate,
            medium_recent_su=medium_recent_su,
            near_recent=int(route.near_miss_recent or 0),
            role=(getattr(capability, "role", None) if capability is not None else None),
            family=action_family, refilter_role=route.refilter_role,
            pending_score_conversion_count=pending_score_conversion_count,
            pending_promising_score_conversion_count=pending_promising_score_conversion_count,
            completions=int(route.completions or 0),
            completion_dry_enabled=True,
        )
        route_role = route_role_label(
            action_family,
            refilter_role=route.refilter_role,
            canonical_refilter_gpu_h=float(route.canonical_refilter_gpu_h or 0.0),
        )
        diagnostic_records = [by_result_id[result_id] for result_id in route.diagnostic_record_ids if result_id in by_result_id]
        diagnostic_score, diagnostic_axes, diagnostic_count = route_diagnostic_improvement(diagnostic_records, medium_recent_ids)
        quality_rows = list((route.strict_quality_by_bin or {}).values())
        quality_values = sorted(float(row["quality"]) for row in quality_rows)
        quality_n = len(quality_values)
        quality_median = statistics.median(quality_values) if quality_values else None
        quality_p25 = (
            statistics.quantiles(quality_values, n=4, method="inclusive")[0]
            if len(quality_values) >= 2
            else (quality_values[0] if quality_values else None)
        )
        quality_axis_margins = {
            axis: statistics.median([
                float(row["margins"].get(axis, 0.0)) for row in quality_rows
            ])
            for axis in STRICT_SUCCESS
        } if quality_rows else {}
        route_rows.append(RouteValueSummary(
            strategy_key=str(route.strategy_key),
            scope="route",
            family=action_family,
            root_family=route.root_family,
            action_family=action_family,
            scoring_family=route.scoring_family,
            operator_id=route.operator_id,
            config_signature=str(route.config_signature or "default"),
            route_role=route_role,
            refilter_role=route.refilter_role,
            config_delta=dict(route.config_delta or {}),
            parent_strategy_key=route.parent_strategy_key,
            route_gpu_h=route_gpu,
            generator_gpu_h=float(route.generator_gpu_h or 0.0),
            canonical_refilter_gpu_h=float(route.canonical_refilter_gpu_h or 0.0),
            canonical_score_conversion_count=int(
                route.canonical_score_conversion_count or 0
            ),
            attempts=int(route.attempts or 0),
            completions=int(route.completions or 0),
            strict_count=strict_count,
            new_su=new_su,
            record_recent_new_su=recent_su,
            new_su_recent=recent_su,
            new_su_per_route_gpu_h=rate,
            record_recent_route_gpu_h=recent_gpu,
            recent_route_gpu_h=recent_gpu,
            record_recent_new_su_per_route_gpu_h=recent_rate,
            recent_new_su_per_route_gpu_h=recent_rate,
            new_su_recent_gpu=gpu_recent_su,
            gpu_recent_route_gpu_h=gpu_recent_gpu,
            gpu_recent_new_su_per_route_gpu_h=gpu_recent_rate,
            medium_recent_new_su=medium_recent_su,
            medium_recent_route_gpu_h=medium_recent_gpu,
            medium_recent_new_su_per_route_gpu_h=medium_recent_rate,
            near_miss_count=int(route.near_miss_count or 0),
            near_miss_recent=int(route.near_miss_recent or 0),
            strict_per_su=strict_per_su,
            duplicate_bin_fraction=duplicate_fraction,
            status=status,
            marginal_status=marginal_status,
            pending_score_conversion_count=pending_score_conversion_count,
            pending_promising_score_conversion_count=pending_promising_score_conversion_count,
            strict_quality_n_unique_bins=quality_n,
            strict_quality_median=quality_median,
            strict_quality_p25=quality_p25,
            strict_quality_axis_margins=quality_axis_margins,
            diagnostic_improvement_score=diagnostic_score,
            diagnostic_improvement_axes=diagnostic_axes,
            diagnostic_improvement_n=diagnostic_count,
            evidence_refs=list(route.evidence_refs),
        ))

    return route_rows, best_rate


def _build_family_route_summaries(
    results: list[ResultRecord],
    route_rows: list[RouteValueSummary],
    best_rate: float,
    pending_score_conversions: PendingScoreConversionCounts,
    config: RouteValueConfig,
) -> list[RouteValueSummary]:
    """Roll exact routes up to every registry-defined family."""
    pending_score_family_count = pending_score_conversions.by_family
    pending_promising_score_family_count = pending_score_conversions.promising_by_family
    family_rows: list[RouteValueSummary] = []
    reg = default_registry()
    for family, capability in sorted(reg.capabilities.items()):
        family_routes = [r for r in route_rows if r.action_family == family]
        direct_gpu_h = sum(float(r.gpu_h or 0.0) for r in results if r.backend_family == family)
        route_gpu_h = sum(r.route_gpu_h for r in family_routes)
        # Refilter-only families own their direct GPU as plumbing/advisory rows;
        # generator family rows use route GPU including canonical scoring cost.
        if getattr(capability, "role", "generator") == "refilter":
            route_gpu_h = direct_gpu_h
        new_su = sum(r.new_su for r in family_routes)
        recent_su = sum(r.record_recent_new_su for r in family_routes)
        recent_gpu = sum(r.record_recent_route_gpu_h for r in family_routes)
        gpu_recent_su = sum(r.new_su_recent_gpu for r in family_routes)
        gpu_recent_gpu_h = sum(r.gpu_recent_route_gpu_h for r in family_routes)
        medium_recent_su = sum(r.medium_recent_new_su for r in family_routes)
        medium_recent_gpu_h = sum(r.medium_recent_route_gpu_h for r in family_routes)
        strict_count = sum(r.strict_count for r in family_routes)
        near_miss_count = sum(r.near_miss_count for r in family_routes)
        near_recent = sum(r.near_miss_recent for r in family_routes)
        rate = (new_su / route_gpu_h) if new_su and route_gpu_h > 0 else None
        recent_rate = (recent_su / recent_gpu) if recent_su and recent_gpu > 0 else None
        gpu_recent_rate = (gpu_recent_su / gpu_recent_gpu_h) if gpu_recent_su and gpu_recent_gpu_h > 0 else None
        medium_recent_rate = (medium_recent_su / medium_recent_gpu_h) if medium_recent_su and medium_recent_gpu_h > 0 else None
        strict_per_su = (strict_count / new_su) if new_su else None
        pending_score_conversion_count = int(pending_score_family_count.get(family, 0))
        pending_promising_score_conversion_count = int(pending_promising_score_family_count.get(family, 0))
        status = classify_route_status(
            config=config,
            route_gpu_h=route_gpu_h, new_su=new_su, recent_su=recent_su,
            medium_recent_su=medium_recent_su,
            near_recent=near_recent, recent_route_gpu_h=recent_gpu,
            rate=rate, best_rate=best_rate,
            strict_per_su=strict_per_su, duplicate_bin_fraction=None,
            role=capability.role, family=family,
            refilter_role=None,
            pending_score_conversion_count=pending_score_conversion_count,
            pending_promising_score_conversion_count=pending_promising_score_conversion_count,
            completions=0,
            completion_dry_enabled=False,
        )
        marginal_status = classify_marginal_route_status(
            config=config,
            route_gpu_h=route_gpu_h, new_su=new_su, strict_per_su=strict_per_su,
            duplicate_bin_fraction=None, lifetime_rate=rate,
            record_recent_su=recent_su, record_recent_rate=recent_rate,
            gpu_recent_su=gpu_recent_su, gpu_recent_gpu_h=gpu_recent_gpu_h,
            gpu_recent_rate=gpu_recent_rate,
            medium_recent_su=medium_recent_su, near_recent=near_recent,
            role=capability.role, family=family, refilter_role=None,
            pending_score_conversion_count=pending_score_conversion_count,
            pending_promising_score_conversion_count=pending_promising_score_conversion_count,
            completions=0,
            completion_dry_enabled=False,
        )
        family_route_role = route_role_label(family, scope="family")
        best_diagnostic_route = max(
            family_routes,
            key=lambda r: float(r.diagnostic_improvement_score or 0.0),
            default=None,
        )
        family_diagnostic_score = float(getattr(best_diagnostic_route, "diagnostic_improvement_score", 0.0) or 0.0)
        family_diagnostic_axes = list(getattr(best_diagnostic_route, "diagnostic_improvement_axes", []) or [])
        family_diagnostic_count = sum(int(getattr(r, "diagnostic_improvement_n", 0) or 0) for r in family_routes)
        best_quality_route = max(
            (r for r in family_routes if r.strict_quality_n_unique_bins > 0),
            key=lambda r: (
                float(r.strict_quality_p25 or 0.0),
                float(r.strict_quality_median or 0.0),
                r.strict_quality_n_unique_bins,
            ),
            default=None,
        )
        family_rows.append(RouteValueSummary(
            strategy_key=f"family::{family}", scope="family", family=family,
            root_family=None, action_family=family,
            scoring_family=(family if capability.role == "refilter" else None),
            operator_id=capability.default_operator_id,
            config_signature="family_rollup",
            route_role=family_route_role,
            refilter_role=None,
            route_gpu_h=route_gpu_h,
            generator_gpu_h=sum(r.generator_gpu_h for r in family_routes) if family_routes else direct_gpu_h,
            canonical_refilter_gpu_h=sum(r.canonical_refilter_gpu_h for r in family_routes),
            canonical_score_conversion_count=sum(
                r.canonical_score_conversion_count for r in family_routes
            ),
            attempts=sum(r.attempts for r in family_routes) if family_routes else sum(1 for r in results if r.backend_family == family),
            completions=sum(r.completions for r in family_routes) if family_routes else sum(1 for r in results if r.backend_family == family and r.exit_status in ("ok", "no_artifacts")),
            strict_count=strict_count,
            new_su=new_su,
            record_recent_new_su=recent_su,
            new_su_recent=recent_su,
            new_su_per_route_gpu_h=rate,
            record_recent_route_gpu_h=recent_gpu,
            recent_route_gpu_h=recent_gpu,
            record_recent_new_su_per_route_gpu_h=recent_rate,
            recent_new_su_per_route_gpu_h=recent_rate,
            new_su_recent_gpu=gpu_recent_su,
            gpu_recent_route_gpu_h=gpu_recent_gpu_h,
            gpu_recent_new_su_per_route_gpu_h=gpu_recent_rate,
            medium_recent_new_su=medium_recent_su,
            medium_recent_route_gpu_h=medium_recent_gpu_h,
            medium_recent_new_su_per_route_gpu_h=medium_recent_rate,
            near_miss_count=near_miss_count,
            near_miss_recent=near_recent,
            strict_per_su=strict_per_su,
            status=status,
            marginal_status=marginal_status,
            pending_score_conversion_count=pending_score_conversion_count,
            pending_promising_score_conversion_count=pending_promising_score_conversion_count,
            strict_quality_n_unique_bins=(
                best_quality_route.strict_quality_n_unique_bins if best_quality_route else 0
            ),
            strict_quality_median=(
                best_quality_route.strict_quality_median if best_quality_route else None
            ),
            strict_quality_p25=(
                best_quality_route.strict_quality_p25 if best_quality_route else None
            ),
            strict_quality_axis_margins=(
                dict(best_quality_route.strict_quality_axis_margins) if best_quality_route else {}
            ),
            diagnostic_improvement_score=family_diagnostic_score,
            diagnostic_improvement_axes=family_diagnostic_axes,
            diagnostic_improvement_n=family_diagnostic_count,
            evidence_refs=[rid for r in family_routes for rid in r.evidence_refs[:1]][:6],
        ))

    return family_rows


def _select_decision_relevant_routes(
    route_rows: list[RouteValueSummary],
    config: RouteValueConfig,
    gpu_h_since_last_su: float | None,
) -> list[RouteValueSummary]:
    """Return the bounded exact-route projection used by the Planner."""
    status_rank = {"promote": 0, "awaiting_score_conversion": 1, "healthy": 2, "diversify": 3, "collapse_risk": 4, "defer": 5, "observed": 6, "advisory": 7, "plumbing": 8, "untried": 9}

    def _decision_safe_route_rate(route_summary: RouteValueSummary) -> tuple[float | None, int]:
        if route_summary.gpu_recent_new_su_per_route_gpu_h is not None:
            return route_summary.gpu_recent_new_su_per_route_gpu_h, 0
        if route_summary.medium_recent_new_su_per_route_gpu_h is not None:
            return route_summary.medium_recent_new_su_per_route_gpu_h, 1
        role = str(route_summary.route_role or "")
        dry_gpu_h = float(gpu_h_since_last_su or 0.0)
        if (
            route_summary.record_recent_new_su_per_route_gpu_h is not None
            and float(route_summary.canonical_refilter_gpu_h or 0.0) <= 0.0
            and "score_conversion" not in role
            and dry_gpu_h < config.deep_stall_gpu_h
        ):
            return route_summary.record_recent_new_su_per_route_gpu_h, 2
        return None, 3

    def _route_row_priority(route_summary: RouteValueSummary) -> tuple[float, int, int, float, float, float, float, float, float, str]:
        dry_gpu_h = float(gpu_h_since_last_su or 0.0)
        role = str(route_summary.route_role or "")
        safe_record_recent = (
            float(route_summary.canonical_refilter_gpu_h or 0.0) <= 0.0
            and "score_conversion" not in role
            and dry_gpu_h < config.deep_stall_gpu_h
        )
        recent_su = max(
            float(route_summary.new_su_recent_gpu or 0.0),
            float(route_summary.medium_recent_new_su or 0.0),
            float(route_summary.record_recent_new_su or 0.0) if safe_record_recent else 0.0,
        )
        rate, source_rank = _decision_safe_route_rate(route_summary)
        current_value_rank = 0 if recent_su > 0 and float(rate or 0.0) > 0.0 else 1
        lifetime_rate = float(route_summary.new_su_per_route_gpu_h or 0.0)
        return (
            current_value_rank,
            source_rank,
            status_rank.get(route_summary.status, 9),
            -float(rate or 0.0),
            -float(recent_su),
            -lifetime_rate,
            -float(route_summary.new_su or 0.0),
            -float(route_summary.diagnostic_improvement_score or 0.0),
            -route_summary.route_gpu_h,
            route_summary.strategy_key,
        )

    route_rows.sort(key=_route_row_priority)

    return route_rows[:32]


def build_route_value_summaries(
    results: list[ResultRecord],
    window_results: list[ResultRecord],
    spawning_actions: dict[str, ActionCandidate] | None,
    config: RouteValueConfig,
    route_diagnostic_improvement: RouteDiagnosticEvaluator,
    near_miss_dedup_trusted: bool = True,
    gpu_h_since_last_su: float | None = None,
) -> list[RouteValueSummary]:
    """Build deterministic family and exact-route value summaries.

    The named phases make lineage attribution, pending-score accounting,
    immutable summary projection, and bounded presentation independently
    reviewable. Scientific thresholds and ordering remain configuration driven.
    """
    spawning_action_by_result_id = spawning_actions or {}
    aggregation = _aggregate_route_evidence(
        results=results,
        window_results=window_results,
        spawning_action_by_result_id=spawning_action_by_result_id,
        config=config,
        near_miss_dedup_trusted=near_miss_dedup_trusted,
    )
    pending_score_conversions = _count_pending_score_conversions(
        results=results,
        aggregation=aggregation,
        spawning_action_by_result_id=spawning_action_by_result_id,
    )
    route_rows, best_rate = _build_exact_route_summaries(
        aggregation=aggregation,
        pending_score_conversions=pending_score_conversions,
        config=config,
        route_diagnostic_improvement=route_diagnostic_improvement,
    )
    family_rows = _build_family_route_summaries(
        results=results,
        route_rows=route_rows,
        best_rate=best_rate,
        pending_score_conversions=pending_score_conversions,
        config=config,
    )
    decision_routes = _select_decision_relevant_routes(
        route_rows=route_rows,
        config=config,
        gpu_h_since_last_su=gpu_h_since_last_su,
    )
    return family_rows + decision_routes
