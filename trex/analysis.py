"""Inspect, validate, and audit a T-ReX append-only campaign archive.

The commands in this module operate on archived records only.  They never
modify generated structures, scores, or EvidenceSummary rows.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from .archive import Archive, RECORD_FILES
from .archive_schema import archive_layout
from .dispatch_outcomes import classify_dispatch_outcome, count_dispatch_outcomes
from .provenance import (
    PROVENANCE_SCHEMA,
    SUPPORTED_PROVENANCE_SCHEMAS,
)


SUMMARY_SCHEMA_VERSION = "trex.analysis-summary.v1"
VALIDATION_SCHEMA_VERSION = "trex.archive-validation.v1"
TRACE_ENTRY_SCHEMA_VERSION = "trex.decision-trace-entry.v1"
TRACE_OUTPUT_SCHEMA_VERSION = "trex.decision-trace.v1"

# These values identify deterministic system-origin candidates rather than
# archived HypothesisCard rows. Keep the vocabulary explicit so a typo or a
# genuinely broken card reference still produces a validation warning.
SYSTEM_CANDIDATE_ORIGIN_IDS = frozenset(
    {
        "cross_family_escape",
        "diagnostic_i4_mcts",
        "evidence_fallback",
        "evidence_fallback_dead_probe",
        "route_value_replay",
        "warmstart",
    }
)


JSONL_FILES = tuple(RECORD_FILES.values())


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return rows, errors
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path.name}:{line_no}: expected a JSON object")
                continue
            rows.append(value)
    return rows, errors


def _records(root: Path, name: str) -> list[dict[str, Any]]:
    return _read_jsonl(root / name)[0]


def _require_archive_root(root: Path | str) -> Path:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"archive root is not a directory: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    values = collections.Counter(str(row.get(key, "<missing>")) for row in rows)
    return dict(sorted(values.items()))


def _string_values(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    return values


def _list_string_values(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str) and item)
    return values


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _matches_json_field_shape(value: Any, declared_type: str) -> bool:
    """Check the top-level JSON shape represented by a dataclass annotation."""

    if value is None:
        return "None" in declared_type or declared_type == "Any"
    base_type = declared_type.split("|", 1)[0].strip()
    if base_type == "str":
        return isinstance(value, str)
    if base_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if base_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if base_type == "bool":
        return isinstance(value, bool)
    if base_type.startswith("list["):
        return isinstance(value, list)
    if base_type.startswith("tuple["):
        return isinstance(value, list)
    if base_type.startswith("dict["):
        return isinstance(value, dict)
    return True


def campaign_summary(root: Path) -> dict[str, Any]:
    root = _require_archive_root(root)
    results = _records(root, "result_records.jsonl")
    evidence = _records(root, "evidence_summaries.jsonl")
    calls = _records(root, "llm_call_records.jsonl")
    launches = _records(root, "launch_decisions.jsonl")
    dispatches = _records(root, "dispatch_records.jsonl")
    candidates = _records(root, "action_candidates.jsonl")
    panels = _records(root, "panel_selections.jsonl")
    latest = evidence[-1] if evidence else {}

    by_role: dict[str, dict[str, Any]] = {}
    for role, rows in _group_by(calls, "role").items():
        by_role[role] = {
            "calls": len(rows),
            "tokens_in": sum(int(row.get("tokens_in") or 0) for row in rows),
            "tokens_out": sum(int(row.get("tokens_out") or 0) for row in rows),
            "latency_s": sum(float(row.get("latency_s") or 0.0) for row in rows),
            "parse_status": _counter(rows, "parse_status"),
            "fallback_calls": sum(bool(row.get("fallback_triggered")) for row in rows),
        }

    started = [row for row in dispatches if row.get("status") == "started"]
    target_ids = sorted(
        {
            str(row.get("target_id"))
            for row in results + evidence
            if row.get("target_id") is not None
        }
    )
    worker_wall = _finite(latest.get("worker_wall_gpu_h_total"))
    su_count = latest.get("run_su_count")
    su_rate = _finite(latest.get("run_su_per_worker_wall_gpu_h_total"))
    if su_rate is None and worker_wall and su_count is not None and worker_wall > 0:
        su_rate = float(su_count) / worker_wall
    provenance_files = sorted(path.name for path in root.glob("run_provenance*.json"))

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "archive_root": str(root.resolve()),
        "target_ids": target_ids,
        "latest_tick": latest.get("tick_id"),
        "state_label": latest.get("state_label"),
        "budget": {
            "elapsed_wall_h": _finite(latest.get("elapsed_wall_h")),
            "remaining_wall_h": _finite(latest.get("remaining_wall_h")),
            "worker_wall_gpu_count": _finite(latest.get("worker_wall_gpu_count")),
            "worker_wall_gpu_h": worker_wall,
            "completed_result_gpu_h": _finite(latest.get("worker_gpu_h_total")),
        },
        "endpoint": {
            "strict_count": latest.get("strict_count"),
            "structure_unique_successes_tm_live": su_count,
            "su_per_worker_wall_gpu_h": su_rate,
            "foldseek_su_status": latest.get("foldseek_su_status"),
            "foldseek_su_coverage": _finite(latest.get("foldseek_su_coverage")),
            "structure_dedup_scope": latest.get("structure_dedup_scope"),
            "sequence_dedup_status": latest.get("sequence_dedup_status"),
            "sequence_dedup_coverage": _finite(latest.get("sequence_dedup_coverage")),
            "sequence_unique_strict_count": latest.get("seq_unique_strict_count"),
        },
        "records": {
            "results": len(results),
            "evidence_summaries": len(evidence),
            "action_candidates": len(candidates),
            "launch_decisions": len(launches),
            "dispatch_records": len(dispatches),
            "panel_selections": len(panels),
        },
        "results_by_family": _counter(results, "backend_family"),
        "result_exit_status": _counter(results, "exit_status"),
        "launch_status": _counter(launches, "status"),
        "dispatch_status": _counter(dispatches, "status"),
        "dispatch_outcome": count_dispatch_outcomes(dispatches),
        "started_by_family": _counter(started, "method_family"),
        "started_by_mode": _counter(started, "supervisor_mode"),
        "llm_usage": {
            "calls": len(calls),
            "tokens_in": sum(int(row.get("tokens_in") or 0) for row in calls),
            "tokens_out": sum(int(row.get("tokens_out") or 0) for row in calls),
            "by_role": by_role,
        },
        "provenance_files": provenance_files,
    }


def _group_by(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "<missing>"))].append(row)
    return dict(grouped)


def _dispatch_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_id",
        "status",
        "method_family",
        "operator_id",
        "supervisor_mode",
        "worker_slot",
        "gpu_id",
        "score_credit_basis",
    )
    projected = {field: row.get(field) for field in fields}
    projected["outcome"] = classify_dispatch_outcome(row)
    return projected


def validate_archive(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    streams: dict[str, list[dict[str, Any]]] = {}
    if not root.is_dir():
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "ok": False,
            "errors": [f"archive root is not a directory: {root}"],
            "warnings": [],
            "record_counts": {},
            "typed_record_counts": {},
            "skipped_records": {},
            "target_ids": [],
            "provenance_files": [],
            "lineage_references": {
                "total": 0,
                "candidate_only": 0,
                "result_only": 0,
                "ambiguous": 0,
                "external_or_unknown": 0,
                "external_or_unknown_examples": [],
            },
        }

    for name in JSONL_FILES:
        rows, stream_errors = _read_jsonl(root / name)
        streams[name] = rows
        counts[name] = len(rows)
        errors.extend(stream_errors)

    typed_archive = Archive(root)
    layout_by_file = {
        stream["file_name"]: stream for stream in archive_layout()["streams"]
    }
    typed_counts: dict[str, int] = {}
    for record_type, file_name in RECORD_FILES.items():
        try:
            typed_counts[file_name] = sum(
                1 for _ in typed_archive.iter_records(record_type)
            )
        except (OSError, TypeError, ValueError) as exc:
            typed_counts[file_name] = 0
            errors.append(
                f"{file_name}: typed schema validation failed: "
                f"{type(exc).__name__}: {exc}"
            )
        stream_schema = layout_by_file[file_name]
        fields_by_name = {field["name"]: field for field in stream_schema["fields"]}
        for record_number, row in enumerate(streams[file_name], start=1):
            missing = sorted(
                field_name
                for field_name, field_schema in fields_by_name.items()
                if field_schema["required"] and field_name not in row
            )
            if missing:
                errors.append(
                    f"{file_name}:record {record_number}: missing required fields: {missing}"
                )
            unknown = sorted(set(row) - set(fields_by_name))
            if unknown:
                errors.append(
                    f"{file_name}:record {record_number}: unknown fields: {unknown}"
                )
            for field_name, field_schema in fields_by_name.items():
                if field_name in row and not _matches_json_field_shape(
                    row[field_name],
                    field_schema["type"],
                ):
                    errors.append(
                        f"{file_name}:record {record_number}: field {field_name!r} "
                        f"expected {field_schema['type']}; got {type(row[field_name]).__name__}"
                    )
    typed_skips = typed_archive.read_skip_counts()
    for key, count in sorted(typed_skips.items()):
        if key.endswith(":schema_drift"):
            errors.append(f"{key}: {count} record(s) failed typed reconstruction")

    results = streams["result_records.jsonl"]
    result_ids = _string_values(results, "result_id")
    duplicate_results = sorted(
        value for value, count in collections.Counter(result_ids).items() if count > 1
    )
    if duplicate_results:
        errors.append(f"duplicate result_id values: {duplicate_results[:20]}")
    for stream_name, identifier in (
        ("llm_call_records.jsonl", "call_id"),
        ("launch_decisions.jsonl", "launch_id"),
        ("dispatch_records.jsonl", "dispatch_id"),
        ("route_records.jsonl", "route_id"),
        ("panel_selections.jsonl", "panel_id"),
    ):
        duplicates = sorted(
            value
            for value, count in collections.Counter(
                _string_values(streams[stream_name], identifier)
            ).items()
            if count > 1
        )
        if duplicates:
            errors.append(f"duplicate {identifier} values: {duplicates[:20]}")

    candidates = streams["action_candidates.jsonl"]
    candidate_ids = set(_string_values(candidates, "candidate_id"))
    result_id_set = set(result_ids)
    duplicate_candidates = sorted(
        value
        for value, count in collections.Counter(
            _string_values(candidates, "candidate_id")
        ).items()
        if count > 1
    )
    if duplicate_candidates:
        warnings.append(
            "candidate_id values recur across ticks/resumes; joins use the archived "
            f"candidate rows: {duplicate_candidates[:20]}"
        )
    parent_tokens = _list_string_values(results, "parent_ids")
    candidate_only = [
        value
        for value in parent_tokens
        if value in candidate_ids and value not in result_id_set
    ]
    result_only = [
        value
        for value in parent_tokens
        if value in result_id_set and value not in candidate_ids
    ]
    ambiguous = [
        value
        for value in parent_tokens
        if value in candidate_ids and value in result_id_set
    ]
    external_or_unknown = [
        value
        for value in parent_tokens
        if value not in candidate_ids and value not in result_id_set
    ]
    lineage_references = {
        "total": len(parent_tokens),
        "candidate_only": len(candidate_only),
        "result_only": len(result_only),
        "ambiguous": len(ambiguous),
        "external_or_unknown": len(external_or_unknown),
        "external_or_unknown_examples": sorted(set(external_or_unknown))[:20],
    }

    for stream_name in ("launch_decisions.jsonl", "dispatch_records.jsonl"):
        unknown = sorted(
            set(_string_values(streams[stream_name], "candidate_id")) - candidate_ids
        )
        if unknown:
            warnings.append(
                f"{stream_name} references {len(unknown)} candidate(s) absent from "
                f"action_candidates.jsonl; first={unknown[:5]}"
            )

    launch_ids = set(_string_values(streams["launch_decisions.jsonl"], "launch_id"))
    unknown_launch_ids = sorted(
        set(_string_values(streams["dispatch_records.jsonl"], "launch_id")) - launch_ids
    )
    if unknown_launch_ids:
        warnings.append(
            f"dispatch_records.jsonl references unknown launch_id values: "
            f"{unknown_launch_ids[:20]}"
        )

    for field_name in ("parent_result_id", "baseline_result_id"):
        unknown_result_ids = sorted(
            set(_string_values(candidates, field_name)) - result_id_set
        )
        if unknown_result_ids:
            warnings.append(
                f"action_candidates.jsonl references unknown {field_name} values: "
                f"{unknown_result_ids[:20]}"
            )

    hypothesis_ids = set(
        _string_values(streams["hypothesis_cards.jsonl"], "hypothesis_id")
    )
    unknown_hypothesis_ids = sorted(
        set(_list_string_values(candidates, "hypothesis_ids"))
        - hypothesis_ids
        - SYSTEM_CANDIDATE_ORIGIN_IDS
    )
    if unknown_hypothesis_ids:
        warnings.append(
            "action_candidates.jsonl references unknown hypothesis_id values: "
            f"{unknown_hypothesis_ids[:20]}"
        )

    unknown_panel_result_ids = sorted(
        set(_list_string_values(streams["panel_selections.jsonl"], "selected_ids"))
        - result_id_set
    )
    if unknown_panel_result_ids:
        errors.append(
            "panel_selections.jsonl references unknown result_id values: "
            f"{unknown_panel_result_ids[:20]}"
        )

    target_ids = {
        str(row["target_id"])
        for name in (
            "result_records.jsonl",
            "evidence_summaries.jsonl",
            "target_constraints.jsonl",
        )
        for row in streams[name]
        if row.get("target_id") is not None
    }
    if len(target_ids) > 1:
        errors.append(f"archive contains multiple target_ids: {sorted(target_ids)}")
    if not streams["evidence_summaries.jsonl"]:
        warnings.append(
            "no EvidenceSummary rows; endpoint and controller state are unavailable"
        )
    if not streams["dispatch_records.jsonl"]:
        warnings.append(
            "no DispatchRecord rows; selected-to-started realization cannot be audited"
        )
    provenance = sorted(root.glob("run_provenance*.json"))
    if not provenance:
        warnings.append(
            "no run_provenance*.json; exact execution provenance is unavailable"
        )
    else:
        for path in provenance:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: invalid provenance JSON: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path.name}: provenance root must be a JSON object")
                continue
            for section, digest_key in (
                ("source", "tree_sha256"),
                ("target", "pdb_sha256"),
                ("model", "content_sha256"),
            ):
                component = value.get(section)
                if not isinstance(component, dict) or not component.get(digest_key):
                    errors.append(f"{path.name}: missing {section}.{digest_key}")
            schema_version = value.get("schema_version")
            if (
                schema_version is not None
                and schema_version not in SUPPORTED_PROVENANCE_SCHEMAS
            ):
                errors.append(
                    f"{path.name}: unsupported provenance schema: {schema_version!r}"
                )
            elif schema_version == PROVENANCE_SCHEMA:
                prompts = value.get("prompts")
                prompt_roles = (
                    prompts.get("roles", {}) if isinstance(prompts, dict) else {}
                )
                if not isinstance(prompt_roles, dict):
                    prompt_roles = {}
                for role in ("planner", "supervisor", "critic"):
                    role_payload = prompt_roles.get(role)
                    prompt_sha = (
                        role_payload.get("system_prompt_sha256")
                        if isinstance(role_payload, dict)
                        else None
                    )
                    if (
                        not isinstance(prompt_sha, str)
                        or len(prompt_sha) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in prompt_sha
                        )
                    ):
                        errors.append(
                            f"{path.name}: missing/invalid "
                            f"prompts.roles.{role}.system_prompt_sha256"
                        )
                campaign = value.get("campaign")
                if campaign is not None:
                    if not isinstance(campaign, dict):
                        errors.append(
                            f"{path.name}: campaign provenance must be an object"
                        )
                    else:
                        for artifact_key in (
                            "input_artifact",
                            "resolved_artifact",
                        ):
                            declared = campaign.get(artifact_key)
                            expected_sha = campaign.get(f"{artifact_key}_sha256")
                            if not declared or not expected_sha:
                                errors.append(
                                    f"{path.name}: missing "
                                    f"campaign.{artifact_key} linkage"
                                )
                                continue
                            artifact_path = root / Path(str(declared)).name
                            if not artifact_path.is_file():
                                errors.append(
                                    f"{path.name}: linked artifact not found: "
                                    f"{artifact_path.name}"
                                )
                            elif _sha256_file(artifact_path) != expected_sha:
                                errors.append(
                                    f"{path.name}: linked artifact SHA256 "
                                    f"mismatch: {artifact_path.name}"
                                )
                        source_sha = campaign.get("source_sha256")
                        input_sha = campaign.get("input_artifact_sha256")
                        if source_sha and input_sha and source_sha != input_sha:
                            errors.append(
                                f"{path.name}: campaign source/input SHA256 " "mismatch"
                            )

    latest = (
        streams["evidence_summaries.jsonl"][-1]
        if streams["evidence_summaries.jsonl"]
        else {}
    )
    if latest:
        if latest.get("foldseek_su_status") != "ok":
            warnings.append(
                f"latest Foldseek SU status is {latest.get('foldseek_su_status')!r}"
            )
        coverage = _finite(latest.get("foldseek_su_coverage"))
        if coverage is not None and coverage < 0.999:
            warnings.append(
                f"latest Foldseek SU coverage is {coverage:.3f}, below 0.999"
            )
        scope = latest.get("structure_dedup_scope")
        if scope not in (None, "binder_chain"):
            warnings.append(
                f"latest structure dedup scope is {scope!r}, not 'binder_chain'"
            )
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "record_counts": counts,
        "typed_record_counts": typed_counts,
        "skipped_records": typed_skips,
        "target_ids": sorted(target_ids),
        "provenance_files": [path.name for path in provenance],
        "lineage_references": lineage_references,
    }


def decision_trace(root: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    root = _require_archive_root(root)
    if limit < 1:
        raise ValueError("trace limit must be positive")
    evidence = _records(root, "evidence_summaries.jsonl")
    hypotheses = _records(root, "hypothesis_cards.jsonl")
    candidates = _records(root, "action_candidates.jsonl")
    supervisors = _records(root, "supervisor_decisions.jsonl")
    launches = _records(root, "launch_decisions.jsonl")
    dispatches = _records(root, "dispatch_records.jsonl")

    ticks = [str(row.get("tick_id")) for row in evidence if row.get("tick_id")]
    selected_ticks = ticks[-max(1, limit) :]
    evidence_by_tick = {str(row.get("tick_id")): row for row in evidence}
    supervisor_by_tick = {str(row.get("tick_id")): row for row in supervisors}
    hypotheses_by_tick = {
        tick: list({str(row.get("hypothesis_id")): row for row in rows}.values())
        for tick, rows in _group_by(hypotheses, "tick_created").items()
    }
    candidate_rows_by_id = {
        candidate_id: rows
        for candidate_id, rows in _group_by(candidates, "candidate_id").items()
        if candidate_id != "<missing>"
    }
    candidate_by_id = {
        candidate_id: rows[0]
        for candidate_id, rows in candidate_rows_by_id.items()
        if len(rows) == 1
    }
    launches_by_tick = _group_by(launches, "tick_id")
    dispatch_by_tick = _group_by(dispatches, "tick_id")

    hypothesis_ticks: list[str] = []
    action_ticks: list[str] = []
    for tick in ticks:
        trailing_tick = re.search(r"(\d+)$", tick)
        hypothesis_key = str(int(trailing_tick.group(1))) if trailing_tick else tick
        if hypotheses_by_tick.get(hypothesis_key):
            hypothesis_ticks.append(tick)
        if launches_by_tick.get(tick) or dispatch_by_tick.get(tick):
            action_ticks.append(tick)
    selected_ticks = (hypothesis_ticks or action_ticks or ticks)[-max(1, limit) :]

    output: list[dict[str, Any]] = []
    for tick in selected_ticks:
        ev = evidence_by_tick.get(tick, {})
        supervisor = supervisor_by_tick.get(tick, {})
        trailing_tick = re.search(r"(\d+)$", tick)
        hypothesis_key = str(int(trailing_tick.group(1))) if trailing_tick else tick
        cards = hypotheses_by_tick.get(hypothesis_key, [])
        launch_rows = launches_by_tick.get(tick, [])
        dispatch_rows = dispatch_by_tick.get(tick, [])
        launched_candidates = []
        for launch in launch_rows:
            candidate_id = str(launch.get("candidate_id"))
            candidate_matches = candidate_rows_by_id.get(candidate_id, [])
            candidate = candidate_by_id.get(candidate_id, {})
            launched_candidates.append(
                {
                    "candidate_id": launch.get("candidate_id"),
                    "candidate_join_status": (
                        "unique"
                        if len(candidate_matches) == 1
                        else "missing"
                        if not candidate_matches
                        else "ambiguous"
                    ),
                    "candidate_match_count": len(candidate_matches),
                    "launch_status": launch.get("status"),
                    "family": candidate.get("method_family"),
                    "operator": candidate.get("operator_id"),
                    "mode": candidate.get("supervisor_mode"),
                    "config_delta": candidate.get("config_delta", {}),
                    "expected_signal": candidate.get("expected_signal"),
                    "why": launch.get("why"),
                }
            )
        output.append(
            {
                "schema_version": TRACE_ENTRY_SCHEMA_VERSION,
                "tick_id": tick,
                "state_label": ev.get("state_label"),
                "diagnostic_driver": ev.get("diagnostic_driver_tldr"),
                "strict_count": ev.get("strict_count"),
                "run_su_count": ev.get("run_su_count"),
                "su_per_worker_wall_gpu_h": ev.get(
                    "run_su_per_worker_wall_gpu_h_total"
                ),
                "hypotheses": [
                    {
                        "hypothesis_id": card.get("hypothesis_id"),
                        "claim": card.get("claim"),
                        "mode_affinity": card.get("mode_affinity"),
                        "evidence_refs": card.get("evidence_refs"),
                        "recommended_action_families": card.get(
                            "recommended_action_families"
                        ),
                        "reasoning_trace": card.get("reasoning_trace"),
                    }
                    for card in cards
                ],
                "supervisor": {
                    "mode_mixture": supervisor.get("mode_mixture"),
                    "fallback_used": supervisor.get("fallback_used"),
                    "clamps_applied": supervisor.get("clamps_applied", []),
                    "rationale": supervisor.get("rationale"),
                },
                "launches": launched_candidates,
                "dispatches": [_dispatch_trace_row(row) for row in dispatch_rows],
            }
        )
    return output


def _print_summary(payload: dict[str, Any]) -> None:
    endpoint = payload["endpoint"]
    budget = payload["budget"]
    print(f"Archive: {payload['archive_root']}")
    print(f"Target:  {', '.join(payload['target_ids']) or '<unknown>'}")
    print(
        f"Tick/state: {payload['latest_tick'] or '<none>'} / "
        f"{payload['state_label'] or '<unknown>'}"
    )
    print(
        "Endpoint: "
        f"strict={endpoint['strict_count']} "
        f"SU={endpoint['structure_unique_successes_tm_live']} "
        f"worker-wall GPU-h={budget['worker_wall_gpu_h']} "
        f"SU/worker-wall-GPU-h={endpoint['su_per_worker_wall_gpu_h']}"
    )
    print(
        "Dedup:   "
        f"Foldseek={endpoint['foldseek_su_status']} "
        f"coverage={endpoint['foldseek_su_coverage']} "
        f"scope={endpoint['structure_dedup_scope']}"
    )
    print(
        "LLM records: "
        f"count={payload['llm_usage']['calls']} "
        f"input_tokens={payload['llm_usage']['tokens_in']} "
        f"output_tokens={payload['llm_usage']['tokens_out']}"
    )
    print(
        "  Counts are archive records, not API invocation counts "
        "(includes deterministic guards and skipped decisions)."
    )
    print(
        "Started by family: " + json.dumps(payload["started_by_family"], sort_keys=True)
    )


def _print_trace(payload: list[dict[str, Any]]) -> None:
    if not payload:
        print("No evidence-backed decision ticks found.")
        return
    for index, tick in enumerate(payload):
        if index:
            print()
        print(
            f"{tick['tick_id']}: state={tick['state_label']} "
            f"strict={tick['strict_count']} SU={tick['run_su_count']}"
        )
        print(f"  diagnostic: {tick['diagnostic_driver'] or '<not recorded>'}")
        print(
            "  hypotheses: "
            + (
                ", ".join(str(card.get("hypothesis_id")) for card in tick["hypotheses"])
                if tick["hypotheses"]
                else "none"
            )
        )
        for label, rows, status_key in (
            ("launches", tick["launches"], "launch_status"),
            ("dispatches", tick["dispatches"], "outcome"),
        ):
            rendered = ", ".join(
                f"{row.get('candidate_id')}:{row.get(status_key)}" for row in rows
            )
            print(f"  {label}: {rendered or 'none'}")


def _print_archive_layout(payload: dict[str, Any]) -> None:
    print(f"Archive layout: {payload['schema_version']}")
    print(f"Lineage: {payload['lineage_semantics']}")
    for stream in payload["streams"]:
        primary_key = ", ".join(stream["primary_key"]) or "<none>"
        join_keys = ", ".join(stream["join_keys"]) or "<none>"
        print(
            f"  {stream['file_name']:<30} {stream['record_type']:<22} "
            f"primary={primary_key}; joins={join_keys}"
        )
        print(f"    {stream['description']}")
    print("Relationships (explode list-valued source fields before joining):")
    for relationship in payload["relationships"]:
        print(
            "  "
            f"{relationship['source_stream']}.{relationship['source_field']} -> "
            f"{relationship['target_stream']}.{relationship['target_field']}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--archive-root", type=Path, required=True)
        command.add_argument("--json", action="store_true", dest="as_json")
    trace = sub.add_parser("trace", help="join recent evidence-to-dispatch decisions")
    trace.add_argument("--archive-root", type=Path, required=True)
    trace.add_argument("--limit", type=int, default=10)
    trace.add_argument("--json", action="store_true", dest="as_json")
    schema = sub.add_parser(
        "schema", help="describe archive streams, fields, and join keys"
    )
    schema.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "summary":
            payload = campaign_summary(args.archive_root)
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_summary(payload)
            return 0
        if args.command == "validate":
            payload = validate_archive(args.archive_root)
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("PASS" if payload["ok"] else "FAIL")
                for item in payload["errors"]:
                    print(f"ERROR: {item}")
                for item in payload["warnings"]:
                    print(f"WARN:  {item}")
                print(
                    "Records: " + json.dumps(payload["record_counts"], sort_keys=True)
                )
            return 0 if payload["ok"] else 1
        if args.command == "trace":
            payload = decision_trace(args.archive_root, limit=args.limit)
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": TRACE_OUTPUT_SCHEMA_VERSION,
                            "archive_root": str(
                                args.archive_root.expanduser().resolve()
                            ),
                            "limit": args.limit,
                            "entries": payload,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                _print_trace(payload)
            return 0
        if args.command == "schema":
            payload = archive_layout()
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_archive_layout(payload)
            return 0
    except (OSError, ValueError) as exc:
        print(f"trex-analyze: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
