"""Stable, lineage-aware route identities and display roles."""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypedDict

from ..refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from ..schemas import ActionCandidate, ResultRecord
from .attribution import _refilter_role_for_record, resolve_generating_record


class RouteIdentity(TypedDict):
    """Stable route identity resolved from one result and its lineage."""

    strategy_key: str
    root_family: str
    action_family: str
    scoring_family: str | None
    operator_id: str
    config_signature: str
    config_delta: dict[str, Any]
    parent_strategy_key: str | None
    refilter_role: str | None


def canonical_config_signature(config_delta: dict[str, Any] | None) -> str:
    """Stable compact signature for the ACTUAL validated config that ran."""
    cd = dict(config_delta or {})
    if not cd:
        return "default"
    payload = json.dumps(cd, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    # Keep a short human-readable prefix so the LLM can distinguish common knobs
    # (e.g. beam_width=4 vs beam_width=8) without expanding a long JSON blob.
    bits = []
    for k in sorted(cd)[:6]:
        v = cd[k]
        bits.append(f"{k}={v}")
    text = ",".join(bits)
    if len(cd) > 6:
        text += ",..."
    return f"{text}#{digest}"


def route_component_key(family: str, operator_id: str | None, config_delta: dict[str, Any] | None) -> str:
    op = operator_id or f"{family}_default"
    return f"{family}:{op}:{canonical_config_signature(config_delta)}"


def route_role_label(
    action_family: str,
    *,
    refilter_role: str | None = None,
    canonical_refilter_gpu_h: float = 0.0,
    scope: str = "route",
) -> str:
    """Human-facing role label for route rows.

    Backend family names are execution details; route_role is the scientific
    action/accounting role shown to the Planner and audit logs.
    """
    if scope == "family":
        if action_family == "structure_refilter":
            return "af2_refilter_family_rollup"
        if action_family == "proteinmpnn_redesign":
            return "sequence_redesign_family_rollup"
        return "generator_family_rollup"
    if action_family == "structure_refilter":
        if refilter_role == PARENT_MODEL_REFOLD:
            return "af2_parent_model_refold"
        if refilter_role == CANONICAL_SCORE_CONVERSION:
            return "af2_score_conversion"
        return "af2_refilter"
    if action_family == "proteinmpnn_redesign":
        return (
            "sequence_redesign_with_af2_score_conversion"
            if canonical_refilter_gpu_h > 0 else "sequence_redesign"
        )
    return (
        "generator_with_af2_score_conversion"
        if canonical_refilter_gpu_h > 0 else "generator"
    )


def _route_identity_for_record(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> RouteIdentity:
    """Lineage-aware route identity for cost/SU attribution.

    For a direct generator route this is simply family+operator+config. For
    canonical AF2 score-conversion it resolves to the upstream generator route.
    For proteinmpnn_redesign it preserves both the original root generator and
    the MPNN action, so Complexa->MPNN and BindCraft->MPNN are learned as
    distinct routes while SU is still counted once globally.
    """
    role = _refilter_role_for_record(r, spawning_actions=spawning_actions)
    gen = resolve_generating_record(
        r, by_result_id=by_result_id, spawning_actions=spawning_actions,
    )
    gen_ac = spawning_actions.get(gen.result_id)
    action_family = gen.backend_family
    operator_id = gen_ac.operator_id if gen_ac is not None else f"{action_family}_default"
    action_cfg = dict((gen_ac.config_delta or {}) if gen_ac is not None else {})

    root = gen
    root_ac = gen_ac
    parent_strategy_key: str | None = None
    parent_id = gen_ac.parent_result_id if gen_ac is not None else None
    # Parent-bound actions (especially proteinmpnn_redesign and intentional
    # refolds) should keep the original route context rather than becoming a
    # context-free "MPNN worked" aggregate.
    if parent_id and parent_id in by_result_id:
        parent = by_result_id[parent_id]
        root = resolve_generating_record(
            parent, by_result_id=by_result_id, spawning_actions=spawning_actions,
        )
        root_ac = spawning_actions.get(root.result_id)
        root_family_tmp = root.backend_family
        root_op_tmp = root_ac.operator_id if root_ac is not None else f"{root_family_tmp}_default"
        root_cfg_tmp = dict((root_ac.config_delta or {}) if root_ac is not None else {})
        parent_strategy_key = "route::" + route_component_key(
            root_family_tmp, root_op_tmp, root_cfg_tmp,
        )

    # A canonical score-conversion of a non-parent-bound generator has no action
    # parent other than the generator itself; keep root==action in that case.
    root_family = root.backend_family
    root_operator = root_ac.operator_id if root_ac is not None else f"{root_family}_default"
    root_cfg = dict((root_ac.config_delta or {}) if root_ac is not None else {})
    root_comp = route_component_key(root_family, root_operator, root_cfg)
    action_comp = route_component_key(action_family, operator_id, action_cfg)
    strategy_key = f"route::{action_comp}" if root_comp == action_comp else f"route::{root_comp}->{action_comp}"

    scoring_family = None
    if r.backend_family == "structure_refilter" and role == CANONICAL_SCORE_CONVERSION:
        scoring_family = "structure_refilter"
    elif r.backend_family == "structure_refilter" and role == PARENT_MODEL_REFOLD:
        scoring_family = "structure_refilter"

    return {
        "strategy_key": strategy_key,
        "root_family": root_family,
        "action_family": action_family,
        "scoring_family": scoring_family,
        "operator_id": operator_id,
        "config_signature": canonical_config_signature(action_cfg),
        "config_delta": action_cfg,
        "parent_strategy_key": parent_strategy_key,
        "refilter_role": role,
    }
