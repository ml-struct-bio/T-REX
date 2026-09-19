"""Optional cross-campaign memory injection for Planner/Supervisor prompts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


ENV_NAME = "TREX_CROSS_CAMPAIGN_MEMORY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_path_from_env() -> Path | None:
    raw = os.environ.get(ENV_NAME, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@lru_cache(maxsize=4)
def _load_memory(path_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"{ENV_NAME} points to missing file: {path}")
    data = json.loads(path.read_text())
    if data.get("schema_version") != "cross-campaign-memory-v3":
        raise ValueError(
            f"{ENV_NAME} must point to cross-campaign-memory-v3 JSON, got "
            f"{data.get('schema_version')!r}"
        )
    for key in ("scope", "usage_contract", "stable_core", "contextual_design_memory"):
        if key not in data:
            raise ValueError(f"{ENV_NAME} missing required top-level key: {key}")
    digest = _sha256(path)
    meta = {
        "enabled": True,
        "path": str(path),
        "sha256": digest,
        "schema_version": data.get("schema_version"),
        "historical_campaign_count": (data.get("scope") or {}).get("historical_campaign_count"),
        "assembled_token_count": (data.get("retrieval_metadata") or {}).get("assembled_token_count"),
        "selected_contextual_entry_ids": (data.get("retrieval_metadata") or {}).get("selected_contextual_entry_ids", []),
    }
    return data, meta


def cross_campaign_memory_metadata(path: Path | str) -> dict[str, Any]:
    """Validate a memory artifact and return its content identity."""

    _, metadata = _load_memory(str(Path(path).expanduser().resolve()))
    return copy.deepcopy(metadata)


def cross_campaign_memory_for_prompt() -> dict[str, Any] | None:
    """Return a copy of the validated LLM-facing memory, or None when disabled.

    The prompt never receives the filesystem path. The path and SHA are kept in
    prompt audit only, so memory-enabled runs are traceable without leaking local
    paths into the model context.
    """
    path = _memory_path_from_env()
    if path is None:
        return None
    data, meta = _load_memory(str(path))
    payload = copy.deepcopy(data)
    payload.setdefault("source", {})
    payload["source"]["sha256"] = meta["sha256"]
    return payload


def cross_campaign_memory_prompt_audit() -> dict[str, Any] | None:
    path = _memory_path_from_env()
    if path is None:
        return None
    _, meta = _load_memory(str(path))
    return copy.deepcopy(meta)


def cross_campaign_memory_evidence_refs() -> set[str]:
    """Evidence-ref labels that the LLM may cite from cross-campaign memory."""
    path = _memory_path_from_env()
    if path is None:
        return set()
    data, _ = _load_memory(str(path))
    prefixes = ("caution:", "config:", "diagnostic:", "state_response:", "target_context:")
    refs = {
        "cross_campaign_memory",
        "stable_core",
        "contextual_design_memory",
        "usage_contract",
    }

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and key.strip():
                    refs.add(key)
                if key in {"entry_id", "id", "evidence_ref"} and isinstance(value, str):
                    refs.add(value)
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
        elif isinstance(obj, str) and obj.startswith(prefixes):
            refs.add(obj)

    visit(data)
    base_refs = set(refs)
    refs.update(f"cross_campaign_memory:{ref}" for ref in base_refs if ref)
    refs.update(f"cross_campaign_memory::{ref}" for ref in base_refs if ref)
    return refs


def cross_campaign_memory_system_appendix(role: str) -> str:
    if _memory_path_from_env() is None:
        return ""
    return (
        "\n\nCROSS-CAMPAIGN MEMORY CONTRACT:\n"
        f"- A validated cross_campaign_memory block may appear in the {role} user payload.\n"
        "- Treat it only as historical advisory prior experience from completed campaigns.\n"
        "- Current EvidenceSummary, available_families_this_cluster, parent availability, "
        "evaluation requirements, and deterministic safeguards remain authoritative.\n"
        "- Use memory entries only when their match_features are observable in the current "
        "evidence; do not use target name, hidden identity, or unsupported target-class priors.\n"
        "- Prefer allocation_prior entries with Grade A/B support. Grade C entries are "
        "hypothesis_only and may only motivate bounded exploration or falsifiable checks.\n"
    )
