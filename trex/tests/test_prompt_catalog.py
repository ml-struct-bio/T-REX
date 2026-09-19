"""Publication prompt-catalog and cross-campaign-memory contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trex.cross_campaign_memory import _load_memory
from trex.planner import planner_system_prompt
from trex.prompt_catalog import (
    build_prompt_catalog,
    render_prompt_catalog,
    runtime_prompt_metadata,
)
from trex.supervisor import supervisor_system_prompt


def _memory_payload() -> dict:
    return {
        "schema_version": "cross-campaign-memory-v3",
        "scope": {"historical_campaign_count": 2},
        "usage_contract": {"authority": "advisory"},
        "stable_core": {"entries": []},
        "contextual_design_memory": {"entries": []},
    }


def test_system_prompts_read_memory_from_current_run_environment(
    tmp_path: Path, monkeypatch,
) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(_memory_payload()))
    _load_memory.cache_clear()
    monkeypatch.delenv("TREX_CROSS_CAMPAIGN_MEMORY", raising=False)
    assert "CROSS-CAMPAIGN MEMORY CONTRACT" not in planner_system_prompt()
    assert "CROSS-CAMPAIGN MEMORY CONTRACT" not in supervisor_system_prompt()

    monkeypatch.setenv("TREX_CROSS_CAMPAIGN_MEMORY", str(memory_path))
    assert "CROSS-CAMPAIGN MEMORY CONTRACT" in planner_system_prompt()
    assert "CROSS-CAMPAIGN MEMORY CONTRACT" in supervisor_system_prompt()
    metadata = runtime_prompt_metadata()
    assert metadata["cross_campaign_memory"]["sha256"] == hashlib.sha256(
        memory_path.read_bytes()
    ).hexdigest()


def test_checked_in_prompt_snapshot_equals_runtime_catalog(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TREX_CROSS_CAMPAIGN_MEMORY", raising=False)
    repository_root = Path(__file__).resolve().parents[2]
    expected = (repository_root / "docs" / "all_prompts_snapshot.txt").read_text()

    assert render_prompt_catalog(("planner", "supervisor", "critic")) == expected


def test_prompt_catalog_distinguishes_available_and_campaign_roles(monkeypatch) -> None:
    monkeypatch.delenv("TREX_CROSS_CAMPAIGN_MEMORY", raising=False)
    catalog = build_prompt_catalog()

    assert catalog["schema_version"] == "trex.prompt-catalog.v2"
    assert set(catalog["roles"]) == {"planner", "supervisor", "critic"}
    assert {
        role for role, item in catalog["roles"].items()
        if item["usage_scope"] == "campaign_llm"
    } == {"planner", "supervisor"}
    assert catalog["roles"]["critic"]["usage_scope"] == "standalone_optional_llm"
    for item in catalog["roles"].values():
        assert len(item["system_prompt_sha256"]) == 64
        assert item["system_prompt"]
        assert item["dynamic_user_prompt_builder"].startswith("def ")


@pytest.mark.parametrize("guard_enabled", [True, False, None])
def test_guard_configuration_does_not_enable_standalone_llm_critic(
    monkeypatch, guard_enabled,
) -> None:
    monkeypatch.delenv("TREX_CROSS_CAMPAIGN_MEMORY", raising=False)

    metadata = runtime_prompt_metadata(critic_enabled=guard_enabled)

    assert metadata["schema_version"] == "trex.prompt-catalog.v2"
    assert metadata["deterministic_guard"] == {
        "enabled": guard_enabled,
        "record_role": "critic",
        "record_model": "deterministic_guard",
    }
    critic = metadata["roles"]["critic"]
    assert critic["usage_scope"] == "standalone_optional_llm"
    assert critic["call_configuration"]["enabled"] is False


def test_runtime_metadata_preserves_explicit_model_and_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("TREX_CROSS_CAMPAIGN_MEMORY", raising=False)
    metadata = runtime_prompt_metadata(
        model="test/model", base_url="http://example.invalid/v1",
        critic_enabled=True,
    )

    for item in metadata["roles"].values():
        assert item["call_configuration"]["model"] == "test/model"
        assert item["call_configuration"]["base_url"] == "http://example.invalid/v1"
    assert metadata["deterministic_guard"]["record_model"] == "deterministic_guard"


def test_runtime_metadata_preserves_catalog_defaults_and_prompt_identity(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TREX_CROSS_CAMPAIGN_MEMORY", raising=False)
    catalog = build_prompt_catalog()
    before = json.dumps(catalog, sort_keys=True)
    monkeypatch.setattr("trex.prompt_catalog.build_prompt_catalog", lambda: catalog)

    metadata = runtime_prompt_metadata(model="test/override", critic_enabled=True)

    assert json.dumps(catalog, sort_keys=True) == before
    for role, item in metadata["roles"].items():
        assert item["system_prompt_sha256"] == catalog["roles"][role]["system_prompt_sha256"]


def test_packaged_prompt_snapshot_matches_source_snapshot() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = repository_root / "docs" / "all_prompts_snapshot.txt"
    packaged = repository_root / "trex" / "data" / "docs" / "all_prompts_snapshot.txt"

    assert packaged.read_bytes() == source.read_bytes()
