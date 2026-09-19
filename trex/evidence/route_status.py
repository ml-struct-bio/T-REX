"""Policy labels derived from route-value evidence."""

from __future__ import annotations

from ..refilter_roles import PARENT_MODEL_REFOLD
from .contracts import RouteValueConfig


def classify_route_status(
    *,
    config: RouteValueConfig,
    route_gpu_h: float,
    new_su: int,
    recent_su: int,
    medium_recent_su: int,
    near_recent: int,
    recent_route_gpu_h: float,
    rate: float | None,
    best_rate: float,
    strict_per_su: float | None,
    duplicate_bin_fraction: float | None,
    role: str | None = None,
    family: str | None = None,
    refilter_role: str | None = None,
    pending_score_conversion_count: int = 0,
    pending_promising_score_conversion_count: int = 0,
    completions: int = 0,
    completion_dry_enabled: bool = False,
) -> str:
    if family == "structure_refilter":
        return "advisory" if refilter_role == PARENT_MODEL_REFOLD else "plumbing"
    if route_gpu_h <= 0:
        return "untried"
    if pending_promising_score_conversion_count > 0 and new_su == 0:
        return "awaiting_score_conversion"
    duplicate_heavy = (
        strict_per_su is not None
        and strict_per_su >= config.route_duplicate_min_strict_per_su
    ) or (
        duplicate_bin_fraction is not None
        and duplicate_bin_fraction >= config.route_duplicate_min_duplicate_fraction
    )
    diversify_heavy = (
        strict_per_su is not None
        and strict_per_su >= config.route_diversify_min_strict_per_su
    ) or (
        duplicate_bin_fraction is not None
        and duplicate_bin_fraction >= config.route_diversify_min_duplicate_fraction
    )
    delayed_recent_su = medium_recent_su > 0 and recent_su == 0
    if (
        duplicate_heavy
        and new_su > 0
        and recent_su == 0
        and not delayed_recent_su
        and (
            strict_per_su is None
            or strict_per_su * max(1, new_su) >= config.route_duplicate_min_strict_count
        )
    ):
        return "collapse_risk"
    if (
        strict_per_su is not None
        and strict_per_su >= config.strict_duplicate_min_strict_per_su
        and (rate or 0.0) <= config.strict_duplicate_max_total_su_per_gpu_h
        and not delayed_recent_su
    ):
        return "collapse_risk"
    completion_dry = (
        completion_dry_enabled
        and completions >= config.route_zero_su_defer_completions
    )
    if (
        new_su == 0
        and near_recent == 0
        and (route_gpu_h >= config.route_zero_su_defer_gpu_h or completion_dry)
    ):
        return "defer"
    if best_rate >= 0.50 and route_gpu_h >= 3.0 and (rate or 0.0) < best_rate * 0.35 and recent_su == 0 and not delayed_recent_su and near_recent == 0:
        return "defer"
    if (
        new_su > 0
        and recent_su == 0
        and not delayed_recent_su
        and near_recent == 0
        and recent_route_gpu_h >= config.route_stale_recent_gpu_h
    ):
        return "observed"
    if new_su > 0 and diversify_heavy:
        return "diversify"
    promote_blocked_by_duplicates = (
        recent_su == 0
        and (
            (
                strict_per_su is not None
                and strict_per_su >= config.route_promote_max_strict_per_su_without_recent_su
            )
            or (
                duplicate_bin_fraction is not None
                and duplicate_bin_fraction >= config.route_promote_max_duplicate_fraction_without_recent_su
            )
        )
    )
    if (
        rate is not None
        and best_rate > 0
        and rate >= best_rate * 0.80
        and new_su > 0
        and not promote_blocked_by_duplicates
    ):
        return "promote"
    return "healthy" if new_su > 0 or near_recent > 0 else "observed"


def classify_marginal_route_status(
    *,
    config: RouteValueConfig,
    route_gpu_h: float,
    new_su: int,
    strict_per_su: float | None,
    duplicate_bin_fraction: float | None,
    lifetime_rate: float | None,
    record_recent_su: int,
    record_recent_rate: float | None,
    gpu_recent_su: int,
    gpu_recent_gpu_h: float,
    gpu_recent_rate: float | None,
    medium_recent_su: int,
    near_recent: int,
    role: str | None = None,
    family: str | None = None,
    refilter_role: str | None = None,
    pending_score_conversion_count: int = 0,
    pending_promising_score_conversion_count: int = 0,
    completions: int = 0,
    completion_dry_enabled: bool = False,
) -> str:
    if family == "structure_refilter":
        return "advisory" if refilter_role == PARENT_MODEL_REFOLD else "plumbing"
    if route_gpu_h <= 0:
        return "untried"
    if pending_promising_score_conversion_count > 0 and new_su == 0:
        return "awaiting_score_conversion"
    duplicate_warning = (
        (strict_per_su is not None and strict_per_su >= config.route_duplicate_min_strict_per_su)
        or (duplicate_bin_fraction is not None and duplicate_bin_fraction >= config.route_duplicate_min_duplicate_fraction)
    )
    any_recent_su = record_recent_su > 0 or gpu_recent_su > 0
    delayed_recent_su = medium_recent_su > 0 and not any_recent_su
    recent_rate = max(float(record_recent_rate or 0.0), float(gpu_recent_rate or 0.0))
    lifetime = float(lifetime_rate or 0.0)
    enough_recent_gpu = gpu_recent_gpu_h >= config.route_stale_recent_gpu_h
    decayed = (
        enough_recent_gpu
        and lifetime > 0.0
        and recent_rate < lifetime * config.route_marginal_decay_fraction
    )
    completion_dry = (
        completion_dry_enabled
        and completions >= config.route_zero_su_defer_completions
    )
    if new_su == 0:
        return (
            "under_tested"
            if (route_gpu_h < config.route_zero_su_defer_gpu_h and not completion_dry)
            or near_recent > 0
            else "dry_low_quality"
        )
    if duplicate_warning and any_recent_su:
        return "productive_but_duplicate"
    if delayed_recent_su:
        return "delayed_productive_duplicate" if duplicate_warning else "delayed_productive"
    if duplicate_warning and (gpu_recent_su == 0) and medium_recent_su == 0 and decayed:
        return "dry_duplicate"
    if any_recent_su:
        return "productive"
    if enough_recent_gpu and lifetime > 0.0 and medium_recent_su == 0 and decayed:
        return "dry"
    return "observed"
