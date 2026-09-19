"""Focused, deterministic evidence-reduction components.

Public compatibility imports remain available from :mod:`trex.evidence_reducer`.
"""

from .attribution import (
    is_canonical_su_record,
    resolve_generating_family,
    resolve_generating_record,
)
from .contracts import RouteDiagnosticEvaluator, RouteValueConfig
from .route_identity import canonical_config_signature, route_component_key
from .route_values import build_route_value_summaries

__all__ = [
    "RouteDiagnosticEvaluator",
    "RouteValueConfig",
    "build_route_value_summaries",
    "canonical_config_signature",
    "is_canonical_su_record",
    "resolve_generating_family",
    "resolve_generating_record",
    "route_component_key",
]
