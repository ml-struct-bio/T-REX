"""Machine-readable layout and join guidance for T-REX archives."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from .archive import RECORD_FILES


ARCHIVE_LAYOUT_SCHEMA_VERSION = "trex.archive-layout.v1"

_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "result_records.jsonl": ("result_id",),
    "evidence_summaries.jsonl": ("tick_id",),
    "metric_calibrations.jsonl": ("metric", "source", "backend", "model_digest"),
    "runtime_buckets.jsonl": ("bucket_id",),
    "target_constraints.jsonl": ("target_id",),
    "hypothesis_cards.jsonl": ("hypothesis_id",),
    "action_candidates.jsonl": ("candidate_id",),
    "supervisor_decisions.jsonl": ("tick_id",),
    "llm_call_records.jsonl": ("call_id",),
    "launch_decisions.jsonl": ("launch_id",),
    "dispatch_records.jsonl": ("dispatch_id",),
    "route_records.jsonl": ("route_id",),
    "panel_selections.jsonl": ("panel_id",),
}

_JOIN_KEYS: dict[str, tuple[str, ...]] = {
    "result_records.jsonl": ("target_id", "tick_id", "parent_ids"),
    "evidence_summaries.jsonl": ("target_id", "tick_id"),
    "target_constraints.jsonl": ("target_id",),
    "hypothesis_cards.jsonl": ("target_id", "hypothesis_id", "tick_created"),
    "action_candidates.jsonl": (
        "candidate_id",
        "hypothesis_ids",
        "parent_result_id",
        "baseline_result_id",
    ),
    "supervisor_decisions.jsonl": ("tick_id",),
    "llm_call_records.jsonl": ("tick_id", "role"),
    "launch_decisions.jsonl": ("tick_id", "candidate_id", "launch_id"),
    "dispatch_records.jsonl": (
        "tick_id",
        "candidate_id",
        "launch_id",
        "parent_result_id",
    ),
    "panel_selections.jsonl": ("selected_ids",),
}

_STREAM_DESCRIPTIONS = {
    "result_records.jsonl": "Backend outcomes, canonical metrics, lineage, and artifacts.",
    "evidence_summaries.jsonl": "Per-tick aggregate scientific and execution evidence.",
    "metric_calibrations.jsonl": "Metric calibration contracts and thresholds.",
    "runtime_buckets.jsonl": "Immutable runtime and dependency digest groups.",
    "target_constraints.jsonl": "Target identity, geometry, and design constraints.",
    "hypothesis_cards.jsonl": "Evidence-cited scientific hypotheses.",
    "action_candidates.jsonl": "Validated candidate actions proposed for selection.",
    "supervisor_decisions.jsonl": "Per-tick resource mixture and action ranking.",
    "llm_call_records.jsonl": "Bounded LLM call provenance and health metadata.",
    "launch_decisions.jsonl": "Selector launch intent or rejection records.",
    "dispatch_records.jsonl": "Actual worker start or dispatch-failure records.",
    "route_records.jsonl": "Multi-stage conversion route outcomes and credit.",
    "panel_selections.jsonl": "Immutable selected-result panels and audits.",
}


_RELATIONSHIPS = (
    ("result_records.jsonl", "target_id", "target_constraints.jsonl", "target_id"),
    ("result_records.jsonl", "runtime_bucket_id", "runtime_buckets.jsonl", "bucket_id"),
    (
        "result_records.jsonl",
        "parent_ids",
        "action_candidates.jsonl",
        "candidate_id",
    ),
    ("result_records.jsonl", "parent_ids", "result_records.jsonl", "result_id"),
    ("result_records.jsonl", "tick_id", "evidence_summaries.jsonl", "tick_id"),
    ("hypothesis_cards.jsonl", "target_id", "target_constraints.jsonl", "target_id"),
    (
        "action_candidates.jsonl",
        "hypothesis_ids",
        "hypothesis_cards.jsonl",
        "hypothesis_id",
    ),
    (
        "action_candidates.jsonl",
        "parent_result_id",
        "result_records.jsonl",
        "result_id",
    ),
    (
        "action_candidates.jsonl",
        "baseline_result_id",
        "result_records.jsonl",
        "result_id",
    ),
    ("supervisor_decisions.jsonl", "tick_id", "evidence_summaries.jsonl", "tick_id"),
    ("llm_call_records.jsonl", "tick_id", "evidence_summaries.jsonl", "tick_id"),
    ("launch_decisions.jsonl", "tick_id", "evidence_summaries.jsonl", "tick_id"),
    (
        "launch_decisions.jsonl",
        "candidate_id",
        "action_candidates.jsonl",
        "candidate_id",
    ),
    ("dispatch_records.jsonl", "tick_id", "evidence_summaries.jsonl", "tick_id"),
    (
        "dispatch_records.jsonl",
        "candidate_id",
        "action_candidates.jsonl",
        "candidate_id",
    ),
    ("dispatch_records.jsonl", "launch_id", "launch_decisions.jsonl", "launch_id"),
    ("dispatch_records.jsonl", "parent_result_id", "result_records.jsonl", "result_id"),
    ("panel_selections.jsonl", "selected_ids", "result_records.jsonl", "result_id"),
)


def archive_layout() -> dict[str, Any]:
    """Return stream fields and stable keys without reading an archive."""

    streams = []
    for record_type, file_name in RECORD_FILES.items():
        stream_fields = []
        for field in fields(record_type):
            required = field.default is MISSING and field.default_factory is MISSING
            stream_fields.append(
                {
                    "name": field.name,
                    "type": str(field.type),
                    "required": required,
                }
            )
        streams.append(
            {
                "file_name": file_name,
                "record_type": record_type.__name__,
                "description": _STREAM_DESCRIPTIONS[file_name],
                "primary_key": list(_PRIMARY_KEYS.get(file_name, ())),
                "join_keys": list(_JOIN_KEYS.get(file_name, ())),
                "fields": stream_fields,
            }
        )
    return {
        "schema_version": ARCHIVE_LAYOUT_SCHEMA_VERSION,
        "format": "JSON Lines; one object per line; append-only",
        "key_semantics": (
            "primary_key is the intended logical record identifier; list-valued "
            "join fields must be exploded before a tabular join; relationships "
            "describe valid links, not mandatory foreign-key constraints"
        ),
        "lineage_semantics": (
            "ResultRecord.parent_ids is polymorphic: the spawning candidate_id is "
            "normally first and an optional concrete parent result_id is normally "
            "second. Legacy and manually imported rows can also contain external "
            "lineage tokens. Classify exploded values by matching candidate_id and "
            "result_id; do not rename every value to parent_result_id."
        ),
        "streams": streams,
        "relationships": [
            {
                "source_stream": source_stream,
                "source_field": source_field,
                "target_stream": target_stream,
                "target_field": target_field,
            }
            for source_stream, source_field, target_stream, target_field in _RELATIONSHIPS
        ],
    }
