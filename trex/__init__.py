"""T-ReX evidence-guided adaptive allocation controller.

See docs/architecture.md for the design.

MVP modules:
  schemas         — all record dataclasses (frozen)
  evidence_reducer — archive records → EvidenceSummary + state classifier
  lifecycle       — HypothesisCard update arithmetic
  planner         — Qwen Planner LLM
  supervisor      — Qwen Supervisor LLM
  candidate_builder — frontier-based seeds + FeasibilityCheck
  selector        — ModeBudgetedBatchSelector
  panel           — greedy Pareto panel selection
  fallback        — state-conditioned mixtures + clamps
  archive         — JSONL archive I/O
  shadow_tick     — end-to-end runner without launch
"""

__version__ = "0.1.1"

SCHEMA_VERSION = "v7.3.3-reward-guard"
