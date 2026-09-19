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
    method_family: str = ""  # B-009 fix (2026-05-26): Complexa records need
    # to carry the actual algorithm used (complexa_mcts, complexa_fk_steering,
    # etc.) — previously parser defaulted to "complexa_beam" which biased the
    # LLM's perception of which algorithms had been tried. Empty default kept
    # for backward compatibility with bindcraft parser (hardcodes its family).
    tick_id: str = ""  # C-2 fix (2026-05-26): ResultRecord.tick_id was left
    # None because parsers had no round context. EvidenceReducer recipe-recency
    # ordering and "examples" sort key both read r.tick_id and silently fell
    # back to -1 for None, so the LLM saw stale evidence in arbitrary order.
    parent_pdb_path: str = ""  # HIGH 2 fix (2026-05-26): MPNN parser carries
    # the parent backbone path through artifacts["pdb_path"] so chained AF2
    # refilter can refold the redesigned sequence onto it. Also useful for
    # other chain targets.
    parent_result_id: str = ""  # MEDIUM 1 fix (2026-05-26): preserve originating
    # ResultRecord.result_id across chained stages so lineage attribution
    # (recipe / method health) survives multi-stage Complexa→MPNN→refilter
    # workflows.
    target_chains_csv: str = ""
    binder_chain: str = ""
    refilter_role: str = ""
    refilter_source_family: str = ""
