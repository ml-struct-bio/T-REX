"""Refilter role labels shared across builder, selector, parser, and reducer.

The backend family name stays ``structure_refilter`` because the executor is the
same AF2 refold path. These role labels separate automatic score conversion
from an intentional LLM-selected refold/refinement action in the audit trail.
"""

from __future__ import annotations

from typing import Any

CANONICAL_SCORE_CONVERSION = "canonical_score_conversion"
PARENT_MODEL_REFOLD = "parent_model_refold"


def infer_refilter_role(obj: Any) -> str | None:
    """Best-effort role for an ActionCandidate-like object."""
    role = getattr(obj, "refilter_role", None)
    if role:
        return str(role)
    family = getattr(obj, "method_family", None) or getattr(obj, "backend_family", None)
    if family == "structure_refilter":
        cid = getattr(obj, "candidate_id", "") or ""
        if cid.startswith("chain_"):
            return CANONICAL_SCORE_CONVERSION
        return PARENT_MODEL_REFOLD
    return None


def is_canonical_score_conversion(obj: Any) -> bool:
    return infer_refilter_role(obj) == CANONICAL_SCORE_CONVERSION
