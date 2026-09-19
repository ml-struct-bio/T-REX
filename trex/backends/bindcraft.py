"""Deterministic launch-file construction for the built-in BindCraft route."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ADVANCED_OVERRIDE_KEYS = (
    "weights_plddt",
    "weights_pae_inter",
    "weights_iptm",
    "weights_helicity",
    "soft_iterations",
    "hard_iterations",
    "greedy_iterations",
    "num_seqs",
    "mpnn_fix_interface",
)


@dataclass(frozen=True)
class BindCraftLaunch:
    """Files and direct argv needed for one BindCraft worker."""

    argv: tuple[str, ...]
    settings_path: Path
    advanced_path: Path
    advanced_overrides: dict[str, object]


def _advanced_settings(
    default_path: Path,
    config_delta: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    overrides = {
        key: config_delta[key]
        for key in ADVANCED_OVERRIDE_KEYS
        if key in config_delta
    }
    if not overrides:
        return {}, {}
    try:
        base = json.loads(default_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        base = {}
    if not isinstance(base, dict):
        base = {}
    for key, value in overrides.items():
        if isinstance(base.get(key), bool):
            base[key] = bool(value)
        elif isinstance(base.get(key), int) and not isinstance(value, bool):
            base[key] = int(value)
        else:
            base[key] = value
    return base, overrides


def prepare_bindcraft_launch(
    *,
    output_dir: Path,
    repo: Path,
    environment_root: Path,
    target_id: str,
    target_pdb: str,
    hotspots: str,
    chains: str,
    default_lengths: tuple[int, int],
    config_delta: Mapping[str, object] | None,
) -> BindCraftLaunch:
    """Write reviewed BindCraft inputs and return its direct worker argv."""

    output_dir.mkdir(parents=True, exist_ok=True)
    designs_dir = output_dir / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    delta = config_delta or {}
    binder_min = int(delta.get("target_length_min", default_lengths[0]))
    binder_max = int(delta.get("target_length_max", default_lengths[1]))
    if binder_min > binder_max:
        binder_min, binder_max = binder_max, binder_min
    settings = {
        "design_path": str(designs_dir),
        "binder_name": f"BC_{target_id}",
        "starting_pdb": target_pdb,
        "chains": chains,
        "target_hotspot_residues": hotspots,
        "lengths": [binder_min, binder_max],
        "number_of_final_designs": int(delta.get("max_trajectories", 4)),
    }
    settings_path = output_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))

    advanced_default = (
        repo / "settings_advanced" / "default_4stage_multimer_mpnn_hardtarget.json"
    )
    advanced, overrides = _advanced_settings(advanced_default, delta)
    advanced_path = advanced_default
    if overrides:
        advanced_path = output_dir / "settings_advanced.json"
        advanced_path.write_text(json.dumps(advanced, indent=2))

    filters = repo / "settings_filters" / "default_filters.json"
    argv = (
        str(environment_root / "bin" / "python"),
        "-u",
        str(repo / "bindcraft.py"),
        "--settings",
        str(settings_path),
        "--filters",
        str(filters),
        "--advanced",
        str(advanced_path),
    )
    return BindCraftLaunch(
        argv=argv,
        settings_path=settings_path,
        advanced_path=advanced_path,
        advanced_overrides=overrides,
    )


__all__ = ["ADVANCED_OVERRIDE_KEYS", "BindCraftLaunch", "prepare_bindcraft_launch"]
