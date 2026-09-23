"""Shared types for output parsers."""

from __future__ import annotations

from dataclasses import dataclass


class ParseError(Exception):
    """Raised by a parser when output is malformed. Daemon catches and
    logs to archive/parse_errors.jsonl instead of crashing the loop."""


@dataclass(frozen=True)
class ParserContext:
    """Context passed alongside the output_dir into each parser.

    The parser uses this to populate ResultRecord fields that depend on
    the spawning LaunchDecision/ActionCandidate rather than the worker
    output itself."""

    target_id: str
    runtime_bucket_id: str
    candidate_id: str
    parent_ids: list[str]  # typically [candidate_id] + any scaffold result_ids
    method_family: str = ""  # Family of the spawning candidate.
    tick_id: str = ""  # Planning-cycle identifier for evidence recency.
    parent_pdb_path: str = ""  # Input parent backbone for chained jobs.
    parent_result_id: str = ""  # Parent result identifier for lineage attribution.
    target_chains_csv: str = ""
    binder_chain: str = ""
    refilter_role: str = ""
    refilter_source_family: str = ""
