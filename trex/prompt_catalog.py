"""Auditable catalog of available LLM prompt implementations.

The runtime prompt constants and builder functions are the single source of
truth.  This module renders them for the publication supplement and records
compact hashes in every run-provenance artifact. Catalog membership does not
mean that an implementation is invoked by a campaign: the standalone LLM
Critic is distinct from the campaign's deterministic advisory guard.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from . import critic, planner, supervisor
from .cross_campaign_memory import cross_campaign_memory_prompt_audit


PROMPT_CATALOG_SCHEMA_VERSION = "trex.prompt-catalog.v2"
PROMPT_USAGE_SCOPE = {
    "planner": "campaign_llm",
    "supervisor": "campaign_llm",
    "critic": "standalone_optional_llm",
}
PROMPT_FLOW = {
    "planner": [
        "system instructions",
        "current evidence summary",
        "active hypotheses",
        "available methods and allowed settings",
        "per-method workload limits",
        "optional bootstrap methods, critic flags, and cross-campaign memory",
    ],
    "supervisor": [
        "system instructions",
        "the same current evidence summary",
        "active HypothesisCards",
        "validated ActionCandidates",
        "current queue and recent-start context",
        "optional cross-campaign memory",
        "conditional one-shot ranking repair after a schema error",
    ],
    "critic": [
        "system instructions",
        "current evidence summary",
        "Planner recommendation",
        "flag-only audit response",
    ],
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_metadata(module: Any) -> dict[str, str]:
    path = Path(inspect.getsourcefile(module) or module.__file__).resolve()
    return {
        "path": f"trex/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_prompt_catalog() -> dict[str, Any]:
    """Return exact system prompts and source-backed dynamic builders."""

    planner_system = planner.planner_system_prompt()
    supervisor_system = supervisor.supervisor_system_prompt()
    critic_system = critic.CRITIC_SYSTEM
    return {
        "schema_version": PROMPT_CATALOG_SCHEMA_VERSION,
        "cross_campaign_memory": cross_campaign_memory_prompt_audit(),
        "roles": {
            "planner": {
                "usage_scope": PROMPT_USAGE_SCOPE["planner"],
                "runtime_source": _source_metadata(planner),
                "flow": PROMPT_FLOW["planner"],
                "system_prompt": planner_system,
                "system_prompt_sha256": _sha256_text(planner_system),
                "dynamic_user_prompt_builder": inspect.getsource(
                    planner.build_user_prompt
                ),
                "call_defaults": asdict(planner.PlannerCallConfig()),
            },
            "supervisor": {
                "usage_scope": PROMPT_USAGE_SCOPE["supervisor"],
                "runtime_source": _source_metadata(supervisor),
                "flow": PROMPT_FLOW["supervisor"],
                "system_prompt": supervisor_system,
                "system_prompt_sha256": _sha256_text(supervisor_system),
                "dynamic_user_prompt_builder": inspect.getsource(
                    supervisor.build_user_prompt
                ),
                "ranking_repair_prompt_builder": inspect.getsource(
                    supervisor.build_ranking_repair_prompt
                ),
                "call_defaults": asdict(supervisor.SupervisorCallConfig()),
            },
            "critic": {
                "usage_scope": PROMPT_USAGE_SCOPE["critic"],
                "runtime_source": _source_metadata(critic),
                "flow": PROMPT_FLOW["critic"],
                "system_prompt": critic_system,
                "system_prompt_sha256": _sha256_text(critic_system),
                "dynamic_user_prompt_builder": inspect.getsource(
                    critic.build_critic_user_prompt
                ),
                "payload_builder": inspect.getsource(critic.build_critic_payload),
                "call_defaults": asdict(critic.CriticCallConfig()),
            },
        },
    }


def runtime_prompt_metadata(
    *,
    model: str | None = None,
    base_url: str | None = None,
    critic_enabled: bool | None = None,
) -> dict[str, Any]:
    """Return available prompt identities and campaign role metadata.

    ``critic_enabled`` is retained for caller compatibility and describes the
    deterministic advisory guard, not the standalone LLM Critic. None means
    that the caller did not supply the guard's configuration. Model/endpoint
    overrides identify declared configuration, not actual model invocations.
    """

    catalog = build_prompt_catalog()
    roles: dict[str, dict[str, Any]] = {}
    for role, item in catalog["roles"].items():
        call_configuration = dict(item["call_defaults"])
        if model is not None:
            call_configuration["model"] = model
        if base_url is not None:
            call_configuration["base_url"] = base_url
        if role == "critic":
            # Campaigns use a deterministic guard, not this LLM implementation.
            call_configuration["enabled"] = False
        roles[role] = {
            "usage_scope": item["usage_scope"],
            "runtime_source": item["runtime_source"],
            "system_prompt_sha256": item["system_prompt_sha256"],
            "call_configuration": call_configuration,
        }
    return {
        "schema_version": catalog["schema_version"],
        "cross_campaign_memory": catalog["cross_campaign_memory"],
        "deterministic_guard": {
            "enabled": critic_enabled,
            "record_role": "critic",
            "record_model": "deterministic_guard",
        },
        "roles": roles,
    }


def render_prompt_catalog(roles: Iterable[str]) -> str:
    """Render a stable, human-readable publication supplement."""

    catalog = build_prompt_catalog()
    chunks = [
        "T-ReX available LLM prompt catalog",
        f"Schema: {catalog['schema_version']}",
        "Catalog membership is not a record of model invocations.",
        "Cross-campaign memory: "
        + json.dumps(catalog["cross_campaign_memory"], sort_keys=True),
        "",
    ]
    for role in roles:
        item = catalog["roles"][role]
        chunks.extend([
            "=" * 88,
            role.upper(),
            "=" * 88,
            "Usage scope: " + item["usage_scope"],
            "Runtime source: " + item["runtime_source"]["path"],
            "Runtime source SHA256: " + item["runtime_source"]["sha256"],
            "System prompt SHA256: " + item["system_prompt_sha256"],
            "Module defaults: " + json.dumps(item["call_defaults"], sort_keys=True),
            "Prompt flow: " + " -> ".join(item["flow"]),
            "",
            "[SYSTEM PROMPT]",
            item["system_prompt"].rstrip(),
            "",
            "[DYNAMIC USER-PROMPT BUILDER]",
            item["dynamic_user_prompt_builder"].rstrip(),
        ])
        if role == "supervisor":
            chunks.extend([
                "",
                "[CONDITIONAL RANKING-REPAIR BUILDER]",
                item["ranking_repair_prompt_builder"].rstrip(),
            ])
        if role == "critic":
            chunks.extend([
                "",
                "[CRITIC PAYLOAD BUILDER]",
                item["payload_builder"].rstrip(),
            ])
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render available Planner/Supervisor prompts and the optional "
            "standalone LLM Critic prompt; catalog membership does not imply use."
        )
    )
    parser.add_argument(
        "--role", choices=("all", "planner", "supervisor", "critic"), default="all"
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = build_prompt_catalog()
    roles = list(catalog["roles"]) if args.role == "all" else [args.role]
    if args.list:
        for role in roles:
            print(f"{role}: {catalog['roles'][role]['runtime_source']['path']}")
        return 0
    if args.format == "json":
        print(json.dumps({
            "schema_version": catalog["schema_version"],
            "cross_campaign_memory": catalog["cross_campaign_memory"],
            "roles": {role: catalog["roles"][role] for role in roles},
        }, indent=2, sort_keys=True))
    else:
        print(render_prompt_catalog(roles), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
