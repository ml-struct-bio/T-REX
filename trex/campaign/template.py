"""Create an explicit, editable campaign YAML template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..validation import DEFAULT_FAMILIES
from .models import CAMPAIGN_SCHEMA_VERSION, PolicyConfig, RunConfig


def campaign_template(
    *,
    name: str,
    target: str,
    archive_root: str,
    asset_root: str | None,
    target_constraint: str | None,
    target_pdb: str | None,
) -> dict[str, Any]:
    """Return a complete template with every user-facing policy visible."""

    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "name": name,
        "target": {
            "name": target,
            "asset_root": asset_root,
            "constraint": target_constraint,
            "pdb": target_pdb,
        },
        "run": {
            "archive_root": archive_root,
            "max_wall_hours": RunConfig.max_wall_hours,
            "seed": 0,
            "worker_gpus": ["1", "2", "3"],
            "enabled_families": list(DEFAULT_FAMILIES),
        },
        "llm": {
            "base_url": "http://127.0.0.1:12000/v1",
            "model": "vllm/Qwen/Qwen3.6-27B-FP8",
        },
        "policy": {
            "critic": True,
            "evidence_skip": False,
            "exemplars": True,
            "foldseek_su_tm_score": 0.60,
            "foldseek_collapse_tm_score": 0.60,
            "selector_quota_realization": "fractional_carry",
            "selector_mode_window_k": PolicyConfig.selector_mode_window_k,
            "selector_adaptive_mode_window_k": PolicyConfig.selector_adaptive_mode_window_k,
        },
        "memory": {
            "cross_campaign_path": None,
        },
        "backends": {
            "repo_root": None,
            "external_root": None,
            "complexa_repo": None,
            "legacy_complexa_repo": None,
            "complexa_python": None,
            "bindcraft_repo": None,
            "bindcraft_env": None,
            "boltzgen_repo": None,
            "boltzgen_binary": None,
            "boltzgen_cache": None,
            "foldseek_binary": None,
            "mmseqs_binary": None,
            "qwen_model_path": None,
            "qwen_model_manifest": None,
        },
    }


def write_campaign_template(
    out: Path | str,
    *,
    name: str,
    target: str,
    archive_root: str,
    asset_root: str | None = None,
    target_constraint: str | None = None,
    target_pdb: str | None = None,
    force: bool = False,
) -> Path:
    """Write one template, refusing accidental overwrite by default."""

    path = Path(out).expanduser().resolve()
    if path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = campaign_template(
        name=name,
        target=target,
        archive_root=archive_root,
        asset_root=asset_root,
        target_constraint=target_constraint,
        target_pdb=target_pdb,
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
