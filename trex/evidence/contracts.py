"""Typed contracts shared by route-evidence components."""

from __future__ import annotations

from typing import Callable, Protocol

from ..schemas import ResultRecord


class RouteValueConfig(Protocol):
    """Configuration fields consumed by route-value construction."""

    deep_stall_gpu_h: float
    route_diversify_min_duplicate_fraction: float
    route_diversify_min_strict_per_su: float
    route_duplicate_min_duplicate_fraction: float
    route_duplicate_min_strict_count: int
    route_duplicate_min_strict_per_su: float
    route_gpu_medium_window_h: float
    route_gpu_recent_window_h: float
    route_marginal_decay_fraction: float
    route_promote_max_duplicate_fraction_without_recent_su: float
    route_promote_max_strict_per_su_without_recent_su: float
    route_stale_recent_gpu_h: float
    route_zero_su_defer_completions: int
    route_zero_su_defer_gpu_h: float
    strict_duplicate_max_total_su_per_gpu_h: float
    strict_duplicate_min_strict_per_su: float


RouteDiagnosticEvaluator = Callable[
    [list[ResultRecord], set[str]], tuple[float, list[str], int]
]
