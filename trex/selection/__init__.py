"""Typed deterministic components used by :mod:`trex.selector`."""

from .admission import (
    CandidateAdmissionRequest,
    CandidateAdmissionResult,
    admit_candidates,
)
from .emission import (
    PlannedEvidenceSnapshot,
    SelectionEmissionRequest,
    SelectionEmissionResult,
    emit_selection_decisions,
)
from .policy import (
    SelectionPolicyRequest,
    SelectionPolicyResult,
    resolve_selection_policy,
)
from .quota import (
    ModeQuotaRequest,
    ModeQuotaResult,
    QuotaCandidateContext,
    realize_mode_quotas,
)
from .ranking import (
    BatchDiversityPolicy,
    RankingCandidateContext,
    SelectionRankingRequest,
    SelectionRankingResult,
    rank_candidates,
    resolve_batch_diversity_policy,
)

__all__ = [
    "BatchDiversityPolicy",
    "CandidateAdmissionRequest",
    "CandidateAdmissionResult",
    "ModeQuotaRequest",
    "ModeQuotaResult",
    "PlannedEvidenceSnapshot",
    "QuotaCandidateContext",
    "RankingCandidateContext",
    "SelectionEmissionRequest",
    "SelectionEmissionResult",
    "SelectionPolicyRequest",
    "SelectionPolicyResult",
    "SelectionRankingRequest",
    "SelectionRankingResult",
    "admit_candidates",
    "emit_selection_decisions",
    "rank_candidates",
    "realize_mode_quotas",
    "resolve_batch_diversity_policy",
    "resolve_selection_policy",
]
