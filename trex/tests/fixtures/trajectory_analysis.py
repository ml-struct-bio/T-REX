"""Archived-result analysis fixtures for regression tests; not a campaign tool."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any
from trex.success_criteria import is_strict_success

def _path_aliases(path: str | Path) -> set[str]:
    p = Path(path)
    aliases = {str(p)}
    try:
        aliases.add(str(p.resolve()))
    except OSError:
        pass
    for alias in tuple(aliases):
        if alias.startswith("/tigress/"):
            aliases.add("/projects/" + alias[len("/tigress/"):])
        elif alias.startswith("/projects/"):
            aliases.add("/tigress/" + alias[len("/projects/"):])
    return aliases


def _summarize_descendant_scores(
    descendants: list[dict[str, str]],
    *,
    bc_by_path: dict[str, dict[str, Any]],
    refilter_by_source: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    strict_bins: set[str] = set()
    strict_count = 0
    for desc in descendants:
        parent = None
        for alias in _path_aliases(desc["pdb_path"]):
            parent = bc_by_path.get(alias)
            if parent is not None:
                break
        if not parent:
            continue
        for child in refilter_by_source.get(str(parent.get("result_id") or ""), []):
            ok = child.get("exit_status") == "ok"
            strict = ok and is_strict_success(child.get("metrics") or {})
            if strict:
                strict_count += 1
                bins = child.get("bins") or {}
                if bins.get("foldseek_su"):
                    strict_bins.add(str(bins["foldseek_su"]))
            scored.append({
                "parent_result_id": parent.get("result_id"),
                "refilter_result_id": child.get("result_id"),
                "status": child.get("exit_status"),
                "strict": bool(strict),
                "foldseek_su": (child.get("bins") or {}).get("foldseek_su"),
            })
    return {
        "descendant_score_count": len(scored),
        "descendant_strict_count": strict_count,
        "descendant_su_count": len(strict_bins),
        "descendant_scores": scored,
    }


def _select_probe_manifest(items: list[dict[str, Any]], *, max_per_target: int) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_target[item["target_key"]].append(item)
    selected: list[dict[str, Any]] = []
    bucket_order = ["no_final_descendant", "accepted_descendant", "rejected_only_descendant"]
    for target_key in sorted(by_target):
        buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in bucket_order}
        for item in sorted(by_target[target_key], key=lambda x: (x["worker_dir"], x["trajectory_stem"])):
            buckets.setdefault(item["selection_bucket"], []).append(item)
        cursors = {b: 0 for b in bucket_order}
        while sum(1 for x in selected if x["target_key"] == target_key) < max_per_target:
            progressed = False
            for bucket in bucket_order:
                idx = cursors[bucket]
                if idx >= len(buckets.get(bucket, [])):
                    continue
                selected.append(buckets[bucket][idx])
                cursors[bucket] += 1
                progressed = True
                if sum(1 for x in selected if x["target_key"] == target_key) >= max_per_target:
                    break
            if not progressed:
                break
    return selected
