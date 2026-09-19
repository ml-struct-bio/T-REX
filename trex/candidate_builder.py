"""CandidateBuilder: deterministic frontier-based seed pool.

Maps HypothesisCard.recommended_action_families to launchable
ActionCandidates via a fixed registry. Each candidate gets a
FeasibilityCheck that the Selector must pass before launch.

MVP scope: registry is in-process dict. Real T-ReX replaces with the
capability_registry in §17 Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json

from .dedup_trust import near_miss_dedup_trusted_from_evidence as _near_miss_dedup_trusted
from .capability_registry import (
    CapabilityRegistry,
    EVAL_BUDGET_DEFAULTS_PER_FAMILY,
    compute_eval_budget,
    default_registry,
    validate_config_delta,
    validate_config_delta_partial,
)
from .refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from .schemas import (
    ActionCandidate,
    EvidenceSummary,
    FeasibilityCheck,
    HypothesisCard,
    Recipe,
    ResourceClass,
)


# Bug G (feedback): single source of truth is `capability_registry`.
# We previously kept a duplicate DEFAULT_REGISTRY dict here; that has been
# removed. Use `registry.get(family)` for operator/lane/cost/allowed_params/
# availability. This file now ONLY owns frontier construction + dedup +
# warmstart/fallback policy.


@dataclass(frozen=True)
class BuilderConfig:
    default_bucket_id: str = "rb_default"
    # Optional override; when None, builder consults registry.is_available
    # rather than this tuple. Kept for tests that need fine-grained control.
    healthy_backends_override: tuple[str, ...] | None = None
    unavailable_backends_override: tuple[str, ...] | None = None
    # Evidence backstop: keep exact productive route/config rows launchable even
    # if a Planner tick omits them. This is not a target prior; it is replay of
    # the archive-derived route-value ledger, capped so Supervisor ranking still
    # controls the batch.
    route_value_replay_enabled: bool = True
    route_value_replay_max_candidates: int = 3
    route_value_replay_min_su_per_gpu_h: float = 0.10
    route_value_replay_stale_recent_gpu_h: float = 1.0
    route_value_replay_lifetime_stale_dry_gpu_h: float = 12.0
    # Diagnostic-only routes can deserve one more continuation before SU appears,
    # but only as bounded evidence acquisition. SU/near-miss replay always outranks
    # this, pending score-conversion must be drained first, and the route stops
    # being auto-replayed once it has consumed this much route GPU-h without
    # objective progress.
    route_value_replay_diagnostic_max_gpu_h: float = 6.0
    # Delayed-feedback guard for diagnostic-only generators. After a family
    # emits a full-ish artifact batch, wait for a small canonical AF2 scoring
    # microbatch before launching the same family again with no scored signal.
    diagnostic_first_feedback_min_accepted: int = 8
    diagnostic_first_feedback_min_refilters: int = 4
    # Deep-stall/zero-yield escape guard. If the LLM keeps proposing only the
    # dry dominant root family, add a small evidence-acquisition floor from
    # distinct untried/under-tested root families. This is target-agnostic:
    # it only fires when the current evidence says the dominant route has
    # stopped buying new SU.
    deep_stall_cross_family_min_probes: int = 2
    cross_family_probe_warning_dry_gpu_h: float = 6.0
    # Early zero-SU coverage deadline. This is not a lockout period: the LLM may
    # propose BindCraft/BoltzGen/other root families before this. It turns the
    # 3-slot warmstart into 4-arm coverage: after the first meaningful completed
    # zero-SU evidence, open the missing orthogonal root probe rather than
    # replaying only the same root family.
    early_zero_su_cross_family_min_probes: int = 2
    early_zero_su_cross_family_min_worker_gpu_h: float = 0.10
    early_zero_su_cross_family_min_completed_children: int = 1
    # Auxiliary diagnostic progress can keep a no-SU route credible, but only
    # above a normalized threshold so raw advisory metrics do not become a launch
    # bypass. New SU/GPU-h and trusted near-miss remain stronger signals.
    cross_family_diagnostic_improvement_min_score: float = 0.35
    # A single/fresh SU immediately after a long deep_stall is real evidence,
    # but not enough to close the escape loop on hard targets. Keep a small
    # non-dominant-root probe floor while the run is still low-yield overall.
    deep_stall_recovery_cross_family_min_probes: int = 1
    deep_stall_recovery_history_lookback_ticks: int = 10
    deep_stall_recovery_min_deep_ticks: int = 4
    deep_stall_recovery_max_total_su_per_gpu_h: float = 0.25
    # Appendix-I.4-style hard-target safety seed. This is candidate exposure, not
    # a forced launch: it emits from the dry warning floor only for the pLDDT-low
    # / interface-plausible diagnostic signature, independent of target identity
    # or an easy/hard label. Planner can still propose complexa_mcts freely from
    # the same diagnostics; the seed only prevents omission when the route has
    # not already received a dry/low-quality probe.
    i4_mcts_seed_enabled: bool = True
    i4_mcts_seed_min_dry_gpu_h: float = 6.0
    i4_mcts_sufficient_gpu_h: float = 3.0
    i4_mcts_sufficient_completions: int = 2
    # T-ReX: Complexa reward weights are late/diagnostic levers, not first-line
    # search knobs. If the LLM proposes reward_* before enough axis evidence is
    # available, keep the family but defer the reward keys and run a material
    # search perturbation first. Once the axis has enough observations, reward
    # keys are allowed, but reward-only exact replays still get an axis-matched
    # material knob so the route changes the sampled basin rather than just the
    # scalar objective. Require more than a single 16-sample diagnostic batch so
    # early interface diagnostics do not jump straight to reward-only FK/MCTS.
    complexa_reward_evidence_min_n: int = 32


@dataclass(frozen=True)
class _RouteValueRateSignal:
    rate: float
    source: str
    has_recent_rate: bool
    lifetime_rate: float


def _is_material_parent_model_refold(config_delta: dict[str, Any] | None) -> bool:
    """Whether a manual AF2 refilter changes the folding experiment.

    Empty/default structure_refilter cards are score-conversion duplicates; the
    controller's chain lane owns those. A manual parent_model_refold is allowed
    to revisit an already scored/refiltered parent only when it actually changes
    the AF2 ensemble, recycles, or template initialization.
    """
    cd = config_delta or {}
    model_names = cd.get("model_names")
    if model_names is not None and str(model_names) != "model_1_multimer_v3":
        return True
    if "num_recycles" in cd:
        try:
            if int(float(cd["num_recycles"])) != 3:
                return True
        except (TypeError, ValueError):
            return True
    if "use_initial_guess" in cd:
        try:
            if int(float(cd["use_initial_guess"])) != 1:
                return True
        except (TypeError, ValueError):
            return True
    return False


def feasibility_for(
    family: str,
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
    registry: "CapabilityRegistry | None" = None,
    *,
    parent_pdb_available: bool = True,
) -> FeasibilityCheck:
    reasons: list[str] = []
    # fix20 #2 (2026-05-26): pre-check parent-PDB precondition for chain
    # families. Without this, the controller emits the candidate, runs it
    # through Selector (consuming a slot), and then SKIPs at exec time —
    # wasting a launch and never surfacing the failure back to the
    # Planner. With the registry metadata in place, the check is one bool
    # against `cap.requires_parent_pdb`.
    cap_for_meta = (registry or default_registry()).get(family)
    if (cap_for_meta is not None
            and cap_for_meta.requires_parent_pdb
            and not parent_pdb_available):
        reasons.append(f"no_parent_pdb:{family}")
        return FeasibilityCheck(
            backend_healthy=False,
            runtime_bucket_id=None,
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
            reasons=reasons,
        )
    # Single source of truth: capability_registry. Override tuples in cfg
    # only used for tests.
    if cfg.unavailable_backends_override is not None:
        if family in cfg.unavailable_backends_override:
            reasons.append("capability_unavailable")
            backend_healthy = False  # 2026-05-26 fix: was leaving backend_healthy=True
                                      #   even when family is in unavailable_override,
                                      #   causing Selector to launch unavailable families
        else:
            backend_healthy = (
                cfg.healthy_backends_override is None
                or family in cfg.healthy_backends_override
            )
    else:
        cap = (registry or default_registry()).get(family)
        if cap is None:
            reasons.append(f"unknown_family:{family}")
            backend_healthy = False
        elif cap.availability != "available":
            reasons.append(f"capability_{cap.availability}:{family}")
            backend_healthy = False
        else:
            backend_healthy = True
    if not backend_healthy and not any("unavailable" in r or "unknown" in r for r in reasons):
        reasons.append(f"backend_unhealthy:{family}")

    # Route cap check: if route_health backlog is saturated, route-heavy families fail
    rh = evidence.route_health
    backlog_ratio = rh.backlog_used / max(1, rh.backlog_cap)
    route_heavy = family in (
        "structure_refilter",
    )
    route_cap_ok = not (route_heavy and backlog_ratio >= 0.95)
    if not route_cap_ok:
        reasons.append("route_backlog_saturated")

    cost_ok = evidence.remaining_wall_h > 0
    if not cost_ok:
        reasons.append("no_remaining_wall_clock")
    else:
        # Remaining-wall gate (2026-05-28; rationale updated 2026-05-30 for the
        # rev3 timeout). Block a launch that cannot reach its first-SU window
        # before the wall. INTENTIONALLY uses the per-family EXPECTED runtime
        # (FAMILY_RUNTIME_H, e.g. BindCraft ~2.5H) — NOT the rev3 HARD_CEILING_S
        # (BindCraft 5H). Two reasons: (1) the expected-runtime threshold already
        # exceeds each family's SU floor (BindCraft FAMILY_RUNTIME 2.5H > its 1.5H
        # floor), so a launch that passes can run past its floor and produce SU;
        # (2) rev3 salvage-parses incrementally-written accepted designs on a
        # wall-kill, so a productive generator killed AT the wall is no longer
        # "salvage 0" — it banks the SU it made. Gating on HARD_CEILING_S instead
        # would idle the GPUs for the final ~5H of every run (over-restriction).
        # Minor: the small downstream-chain refilter cost is absorbed by the drain
        # margin. For most of the 47H run remaining_wall_h >> every family runtime
        # so this is inert; it only gates heavy generators in the final hours.
        from .critic_guard import FAMILY_RUNTIME_H
        _rt = FAMILY_RUNTIME_H.get(family)
        _DRAIN_MARGIN_H = 0.1
        if _rt is not None and _rt + _DRAIN_MARGIN_H > evidence.remaining_wall_h:
            cost_ok = False
            reasons.append(
                f"insufficient_wall_for_{family}"
                f"(~{_rt:.1f}h>{evidence.remaining_wall_h:.1f}h_remaining)"
            )

    return FeasibilityCheck(
        backend_healthy=backend_healthy,
        runtime_bucket_id=cfg.default_bucket_id if backend_healthy else None,
        compiler_ok=True,   # MVP: assume compiler always OK; real T-ReX invokes verifier
        verifier_ok=True,
        route_cap_ok=route_cap_ok,
        cost_ok=cost_ok,
        reasons=reasons,
    )


def _failed_signature_set(evidence: EvidenceSummary) -> set[tuple[str, str]]:
    """Build a set of (operator_id, config_key) tuples seen as `joint_fail`
    in the EvidenceSummary recipes. Used to dedup candidates against recently
    failed configurations.

    CRITICAL (2026-05-28): the evidence reducer splits ONE config's
    descendants by outcome into separate recipe entries that share the SAME
    (operator_id, config_delta). A stochastic config therefore appears as both
    joint_fail AND strict_success. Re-proposing a config that ALSO yields
    strict successes is correct, so it must NOT be deduped. Without this
    exclusion, 47% (77/163) of joint_fail drops across 110 archives were
    productive configs hard-rejected by the Selector — silently killing
    exploit. We exclude any signature that has a strict_success entry.
    """
    productive: set[tuple[str, str]] = set()
    for r in evidence.recipes:
        if r.recipe_class == "strict_success":
            cfg_key = json.dumps(dict(r.config_delta), sort_keys=True)
            productive.add((r.operator_id, cfg_key))

    sigs: set[tuple[str, str]] = set()
    for r in evidence.recipes:
        if r.recipe_class != "joint_fail":
            continue
        cfg_key = json.dumps(dict(r.config_delta), sort_keys=True)
        sig = (r.operator_id, cfg_key)
        if sig in productive:
            continue
        sigs.add(sig)
    return sigs


WARMSTART_FAMILIES: tuple[tuple[str, dict[str, object]], ...] = (
    # Target-agnostic cold-start warmstart. This is a cost/evidence prior, not
    # a target-family prior: one cheap/direct-scored Complexa route establishes
    # fast SU/GPU-h evidence, while BoltzGen and BindCraft provide two
    # orthogonal diagnostic-generator reads. If the first completed evidence
    # still has zero SU, the early cross-family backstop opens another
    # non-dominant root/config probe instead of replaying only the same root.
    ("complexa_beam", {"beam_width": 4, "n_branch": 4, "nsamples": 4}),
    ("boltzgen", {"num_designs": 16, "budget": 4}),
    ("bindcraft", {"max_trajectories": 2}),
)


# Families kept available to LLM cards, route-value replay, and generic cheap
# fallback, but excluded from deterministic missing-root coverage. They do not
# add a new root paradigm after Complexa beam; the coverage slot should go to
# an orthogonal generator/config probe such as FK/MCTS, BoltzGen, or BindCraft.
ROOT_COVERAGE_EXCLUDED_FAMILIES: frozenset[str] = frozenset({"complexa_best_of_n"})


# Post-cold-start fallback must remain evidence/capability driven. The cold
# start above intentionally probes three broad paradigms; after evidence exists,
# fallback scans the registry instead of hard-coding a family ladder.
GENERIC_FALLBACK_DIVERSITY_FRACTIONS: dict[str, tuple[float, float]] = {
    # key: (stalled fraction through allowed range, deep_stall fraction)
    "sc_scale_noise": (0.55, 0.90),
    "temperature": (0.55, 0.90),
    "exploration_prob": (0.55, 0.90),
    "exploration_constant": (0.55, 0.90),
    "noise_scale": (0.55, 0.90),
    "step_scale": (0.55, 0.90),
    "backbone_noise": (0.55, 0.90),
}


def _cb_mh_value(h, k, d=0):
    v = h.get(k, d) if isinstance(h, dict) else getattr(h, k, d)
    return d if v is None else v


def _is_route_light_generator_cap(cap: Any) -> bool:
    return (
        getattr(cap, "role", "generator") == "generator"
        and not bool(getattr(cap, "requires_parent_pdb", False))
    )


def _numeric_config_value(lo: float, hi: float, fraction: float) -> int | float:
    value = lo + (hi - lo) * fraction
    # Integer-valued search-count ranges are usually wider than one step.
    # Continuous [0, 1] knobs such as noise_scale must not round to 0/1.
    if float(lo).is_integer() and float(hi).is_integer() and (hi - lo) > 1.0:
        return int(round(value))
    return round(float(value), 4)


def _fallback_diversity_config(cap: Any, state_label: str) -> dict[str, object]:
    """Generic diversity bump for fallback candidates.

    Only non-budget, broadly interpretable exploration/noise knobs are filled.
    Budget expansion (beam width, trajectories, simulations, num_designs) stays
    under Planner/Supervisor control through explicit evidence-backed cards.
    """
    params: dict[str, object] = {}
    ap = getattr(cap, "allowed_params", {}) or {}
    idx = 1 if state_label == "deep_stall" else 0
    for key, fractions in GENERIC_FALLBACK_DIVERSITY_FRACTIONS.items():
        rng = ap.get(key)
        if isinstance(rng, tuple) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            params[key] = _numeric_config_value(lo, hi, fractions[idx])
    if state_label in ("stalled", "deep_stall", "strict_duplicate_collapse") and "refinement_algorithm" in ap:
        allowed = ap.get("refinement_algorithm")
        if isinstance(allowed, (list, tuple, set)) and "sequence_hallucination" in allowed:
            params.setdefault("refinement_algorithm", "sequence_hallucination")
    if not params:
        return {}
    repaired, _ = validate_config_delta_partial(cap, params)
    return repaired


def _should_pair_complexa_sequence_hallucination(
    evidence: EvidenceSummary,
    primary_axis: str | None,
    config_delta: dict[str, object],
) -> bool:
    """Target-agnostic Complexa refinement pairing.

    T-ReX's productive easy-target routes often combined backbone/interface noise
    with sequence_hallucination. If a Complexa card is already proposing an
    interface/sequence rescue or the run is in a stalled/duplicate regime, pair
    the ordinary noise/reward tweak with hallucination so the route can test the
    T-ReX-style beam-then-hallucinate operator without a target-name prior.
    """
    if config_delta.get("refinement_algorithm") == "sequence_hallucination":
        return False
    if primary_axis not in {"iPAE", "binder_scRMSD"}:
        return False
    state = str(getattr(evidence, "state_label", "") or "")
    if state in {"stalled", "deep_stall", "rescue_rich", "productive_duplicate", "strict_duplicate_collapse"}:
        return True
    return _near_miss_dedup_trusted(evidence) and bool((getattr(evidence, "near_miss_count", 0) or 0) > 0)



I4_MCTS_CONFIG: dict[str, object] = {
    # Proteina-Complexa Appendix I.4 hard-target setup, exposed as an
    # evidence-triggered candidate rather than a target-name prior.
    "nsteps": 400,
    "nsamples": 4,
    "batch_size": 16,
    "n_simulations": 20,
    "exploration_prob": 0.5,
    "exploration_constant": 1.0,
    "filter_samples_limit": 100,
    "reward_i_pae_weight": -1.0,
    "reward_plddt_weight": 1.0,
    "refinement_algorithm": "sequence_hallucination",
    "enable_greedy_optimization": True,
    "n_greedy_iters": 15,
    "greedy_percentage": 5.0,
}


def _stat_fail_rate(stat: Any | None) -> float:
    if stat is None:
        return 0.0
    n = float(_cb_mh_value(stat, "n", 0) or 0)
    if n <= 0:
        return 0.0
    return float(_cb_mh_value(stat, "fail_count", 0) or 0) / n


def _stat_pass_near_rate(stat: Any | None) -> float:
    if stat is None:
        return 0.0
    n = float(_cb_mh_value(stat, "n", 0) or 0)
    if n <= 0:
        return 0.0
    return (
        float(_cb_mh_value(stat, "pass_count", 0) or 0)
        + float(_cb_mh_value(stat, "near_pass_count", 0) or 0)
    ) / n


_COMPLEXA_NOISE_AXIS_GROUPS: dict[str, tuple[str, ...]] = {
    "structure": ("pLDDT", "binder_pLDDT_avg", "design_ptm"),
    "interface": ("iPAE", "min_ipae", "avg_ipsae", "max_ipsae", "ipTM", "design_to_target_iptm"),
    "geometry": ("binder_scRMSD", "shape_complementarity", "interface_contact_density", "hotspot_rmsd", "buried_sasa"),
}

_COMPLEXA_MATERIAL_SEARCH_KEYS: frozenset[str] = frozenset({
    "sc_scale_noise",
    "temperature",
    "beam_width",
    "n_branch",
    "nsamples",
    "replicas",
    "n_simulations",
    "exploration_prob",
    "exploration_constant",
    "refinement_algorithm",
    "n_greedy_iters",
    "enable_greedy_optimization",
    "greedy_percentage",
})


def _complexa_axis_group(primary_axis: str | None) -> tuple[str, ...]:
    axis = str(primary_axis or "")
    if axis == "pLDDT":
        return _COMPLEXA_NOISE_AXIS_GROUPS["structure"]
    if axis == "binder_scRMSD":
        return _COMPLEXA_NOISE_AXIS_GROUPS["geometry"]
    if axis == "iPAE":
        return _COMPLEXA_NOISE_AXIS_GROUPS["interface"]
    return (axis,) if axis else ()


def _axis_group_observation_n(evidence: EvidenceSummary, axes: tuple[str, ...]) -> int:
    strict = getattr(evidence, "axis_stats", {}) or {}
    diagnostic = getattr(evidence, "diagnostic_axis_stats", {}) or {}
    best = 0
    for axis in axes:
        for table in (strict, diagnostic):
            row = table.get(axis)
            if row is None:
                continue
            try:
                best = max(best, int(_cb_mh_value(row, "n", 0) or 0))
            except (TypeError, ValueError):
                continue
    return best


def _complexa_has_material_search_delta(
    config_delta: dict[str, object],
    *,
    family: str | None = None,
) -> bool:
    """Whether config_delta contains a real Complexa search-basin change.

    Default-equivalent budget keys (e.g. beam_width=4, nsamples=4, nsteps=400)
    are not material evidence. They often appear because prompts spell out the
    default lane, but they should not make scalar reward changes look like an
    evidence-backed material perturbation.
    """
    defaults = EVAL_BUDGET_DEFAULTS_PER_FAMILY.get(str(family or ""), {})
    for key, value in config_delta.items():
        if key not in _COMPLEXA_MATERIAL_SEARCH_KEYS:
            continue
        if key.startswith("reward_"):
            continue
        if key in defaults:
            try:
                if float(value) == float(defaults[key]):
                    continue
            except (TypeError, ValueError):
                if value == defaults[key]:
                    continue
        return True
    return False


def _complexa_numeric_value(cap: Any, key: str, fraction: float) -> object | None:
    rng = (getattr(cap, "allowed_params", {}) or {}).get(key)
    if not (isinstance(rng, tuple) and len(rng) == 2):
        return None
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, ValueError):
        return None
    return _numeric_config_value(lo, hi, fraction)


def _complexa_material_search_delta(
    evidence: EvidenceSummary,
    primary_axis: str | None,
    family: str,
    cap: Any,
    existing: dict[str, object],
) -> dict[str, object]:
    """Axis-matched material search tweak for T-ReX reward/config repair.

    The helper intentionally changes sampling/search first and leaves scalar
    reward reweighting as a later diagnostic lever. It stays within the same
    Complexa family the LLM selected; cross-family pivots remain Planner/
    Supervisor decisions.
    """
    ap = getattr(cap, "allowed_params", {}) or {}
    out: dict[str, object] = {}

    def add_num(key: str, fraction: float) -> bool:
        if key in existing or key in out:
            return False
        value = _complexa_numeric_value(cap, key, fraction)
        if value is None:
            return False
        out[key] = value
        return True

    def add_enum(key: str, value: object) -> bool:
        if key in existing or key in out or key not in ap:
            return False
        allowed = ap.get(key)
        if isinstance(allowed, (list, tuple, set)) and value not in allowed:
            return False
        out[key] = value
        return True

    state = str(getattr(evidence, "state_label", "") or "")
    duplicate_pressure = (
        state in {"productive_duplicate", "strict_duplicate_collapse"}
        or bool(getattr(evidence, "strict_duplicate_collapse_signal", False))
    )
    axis = str(primary_axis or "")

    if family == "complexa_mcts":
        add_num("n_simulations", 0.20)
        add_num("exploration_prob", 0.57)
        add_num("exploration_constant", 0.33)
        return out

    if family == "complexa_fk_steering":
        add_num("temperature", 0.45 if duplicate_pressure else 0.30)
        if axis in {"iPAE", "binder_scRMSD"}:
            noise = _adaptive_complexa_sc_scale_noise(evidence, primary_axis, cap)
            if noise is not None and "sc_scale_noise" not in existing:
                out.setdefault("sc_scale_noise", noise)
        return out

    if duplicate_pressure:
        if add_num("beam_width", 0.43):
            add_num("n_branch", 0.33)
            return out
        if add_num("replicas", 0.60):
            return out

    if axis == "iPAE":
        # Interface blocker: first widen/re-dock the search basin; scalar
        # interface reward is allowed only after enough interface observations.
        if add_num("beam_width", 0.43):
            add_num("n_branch", 0.33)
            return out
        noise = _adaptive_complexa_sc_scale_noise(evidence, primary_axis, cap)
        if noise is not None and "sc_scale_noise" not in existing:
            out["sc_scale_noise"] = noise
            return out
    elif axis == "binder_scRMSD":
        add_enum("refinement_algorithm", "sequence_hallucination")
        noise = _adaptive_complexa_sc_scale_noise(evidence, primary_axis, cap)
        if noise is not None and "sc_scale_noise" not in existing:
            out.setdefault("sc_scale_noise", noise)
        if out:
            return out
    elif axis == "pLDDT":
        # Low confidence is not automatically an interface-reward problem.
        # Prefer broader/alternative sampling first; pLDDT reward/MCTS can still
        # be used once this axis is repeatedly observed as the blocker.
        if add_num("beam_width", 0.35):
            add_num("n_branch", 0.33)
            return out
        add_num("sc_scale_noise", 0.35)
        if out:
            return out

    if add_num("beam_width", 0.35):
        return out
    add_num("replicas", 0.50)
    if out:
        return out
    add_num("sc_scale_noise", 0.45)
    return out


def _repair_complexa_reward_timing(
    config_delta: dict[str, object],
    *,
    evidence: EvidenceSummary,
    primary_axis: str | None,
    family: str,
    cap: Any,
    cfg: BuilderConfig,
) -> tuple[dict[str, object], list[str]]:
    """Apply T-ReX reward timing policy to a validated Complexa config.

    Sparse evidence: reward-only keys are deferred and replaced by a material
    search perturbation. If the LLM already paired the reward with material
    search (MCTS, wider beam, noise, hallucination, etc.), keep it: this is an
    evidence-guided search proposal, not a scalar-only retry. Sufficient axis
    evidence: reward keys remain, but exact reward-only replay gets a material
    search knob as well.
    """
    reward_keys = [k for k in config_delta if str(k).startswith("reward_")]
    if not reward_keys:
        return config_delta, []

    axis_n = _axis_group_observation_n(evidence, _complexa_axis_group(primary_axis))
    evidence_rich = axis_n >= int(cfg.complexa_reward_evidence_min_n)
    material_present = _complexa_has_material_search_delta(config_delta, family=family)
    material_delta = _complexa_material_search_delta(evidence, primary_axis, family, cap, config_delta)

    if not evidence_rich and material_present and family == "complexa_mcts":
        return config_delta, []

    if not evidence_rich:
        repaired = {k: v for k, v in config_delta.items() if k not in reward_keys}
        if not _complexa_has_material_search_delta(repaired, family=family):
            repaired.update(material_delta)
        return repaired, [
            f"config_delta_adjusted:reward_deferred:evidence_sparse_axis_n={axis_n}<"
            f"{int(cfg.complexa_reward_evidence_min_n)}:material_search_first"
        ]

    if not material_present and material_delta:
        repaired = dict(config_delta)
        repaired.update(material_delta)
        return repaired, [
            "config_delta_adjusted:auto_pair:axis_matched_material_search_for_reward_retry"
        ]
    return config_delta, []


def _best_axis_rates(evidence: EvidenceSummary, axes: tuple[str, ...]) -> tuple[float, float]:
    """Return (max fail rate, max pass-or-near support) for a metric group."""
    strict = getattr(evidence, "axis_stats", {}) or {}
    diagnostic = getattr(evidence, "diagnostic_axis_stats", {}) or {}
    fail = 0.0
    support = 0.0
    for axis in axes:
        row = strict.get(axis) or diagnostic.get(axis)
        if row is None:
            continue
        fail = max(fail, _stat_fail_rate(row))
        support = max(support, _stat_pass_near_rate(row))
    return fail, support


def _parent_bound_redesign_support(evidence: EvidenceSummary) -> float:
    """Whether fixed-backbone/sequence-side rescue is already improving metrics.

    ProteinMPNN progress means the parent backbone may be
    usable and the failure is more likely sequence/side-chain/fold-model choice
    than a need for large de-novo backbone perturbation. This should soften, not
    eliminate, Complexa sc_scale_noise pairing.
    """
    score = 0.0
    for row in getattr(evidence, "route_values", None) or []:
        fam = str(_cb_mh_value(row, "action_family", _cb_mh_value(row, "family", "")) or "")
        if fam not in {"proteinmpnn_redesign"}:
            continue
        try:
            score = max(score, float(_cb_mh_value(row, "diagnostic_improvement_score", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return score


def _adaptive_complexa_sc_scale_noise(
    evidence: EvidenceSummary,
    primary_axis: str | None,
    cap: Any,
) -> float | None:
    """Metric-first Complexa backbone/interface perturbation.

    `sc_scale_noise` is a cheap structure-regeneration/redesign-like knob. Its
    value should follow scientific signals, not just a coarse state label:
    pLDDT/geometry failure implies stronger backbone perturbation; isolated
    interface failure with good fold/geometry implies moderate re-docking; and
    successful parent-bound redesign/refilter progress softens the perturbation
    because sequence/side-chain/fold-model repair is already carrying signal.
    State/dry time is only a fallback when metric evidence is sparse.
    """
    rng = (getattr(cap, "allowed_params", {}) or {}).get("sc_scale_noise")
    if not (isinstance(rng, tuple) and len(rng) == 2):
        return None
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, ValueError):
        return None

    structure_fail, structure_support = _best_axis_rates(evidence, _COMPLEXA_NOISE_AXIS_GROUPS["structure"])
    interface_fail, interface_support = _best_axis_rates(evidence, _COMPLEXA_NOISE_AXIS_GROUPS["interface"])
    geometry_fail, geometry_support = _best_axis_rates(evidence, _COMPLEXA_NOISE_AXIS_GROUPS["geometry"])
    redesign_support = _parent_bound_redesign_support(evidence)
    has_metric_signal = any(
        v > 0.0 for v in (
            structure_fail, structure_support, interface_fail, interface_support,
            geometry_fail, geometry_support, redesign_support,
        )
    )

    axis = str(primary_axis or "")
    if has_metric_signal:
        if structure_fail >= 0.60 or geometry_fail >= 0.70:
            # Bad fold/global geometry: treat as stronger structure regeneration.
            fraction = 0.85
        elif interface_fail >= 0.70 and max(structure_support, geometry_support) >= 0.50:
            # Fold is plausible but interface is wrong: re-dock, not a full reset.
            fraction = 0.65
        elif interface_fail >= 0.60 or geometry_fail >= 0.50:
            fraction = 0.60
        elif axis == "pLDDT":
            fraction = 0.75
        else:
            fraction = 0.50
        if redesign_support >= 0.50:
            fraction = min(fraction, 0.50)
        elif redesign_support >= 0.35:
            fraction = min(fraction, 0.60)
        if max(structure_support, interface_support, geometry_support) >= 0.70 and max(structure_fail, interface_fail, geometry_fail) < 0.50:
            fraction = min(fraction, 0.45)
    else:
        # Fallback only: no qualified metric block yet. Keep this mild so state
        # labels alone cannot masquerade as scientific evidence.
        try:
            dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
        except (TypeError, ValueError):
            dry_gpu_h = 0.0
        state = str(getattr(evidence, "state_label", "") or "")
        fraction = 0.55
        if dry_gpu_h >= 12.0 or state == "deep_stall":
            fraction = 0.65

    return float(_numeric_config_value(lo, hi, fraction))


def _hard_target_i4_signal(evidence: EvidenceSummary, cfg: BuilderConfig) -> bool:
    """Target-agnostic Appendix-I.4 candidate trigger.

    I.4-style MCTS is just one Complexa MCTS strategy. Expose it like other
    methods when evidence shows a pLDDT/scaffold-confidence bottleneck with a
    plausible interface, rather than waiting for full deep_stall. The dry floor
    prevents easy/productive targets from paying for it before cheap routes have
    had a fair chance. This uses no target identity and is still only a
    candidate, not a forced launch.
    """
    state = str(getattr(evidence, "state_label", "") or "")
    dry_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    if state == "low_evidence":
        return False
    if dry_h < cfg.i4_mcts_seed_min_dry_gpu_h:
        return False

    hard_i4_n = 0
    for jp in getattr(evidence, "joint_patterns", None) or []:
        if (
            getattr(jp, "axes", None) == ("pLDDT", "iPAE")
            and getattr(jp, "pattern", None) == "A_fail_B_pass"
        ):
            hard_i4_n += int(getattr(jp, "count", 0) or 0)
    if hard_i4_n > 0:
        return True

    axis_stats = getattr(evidence, "axis_stats", {}) or {}
    diag_stats = getattr(evidence, "diagnostic_axis_stats", {}) or {}
    plddt_fail = (
        _stat_fail_rate(axis_stats.get("pLDDT")) >= 0.5
        or _stat_fail_rate(diag_stats.get("binder_pLDDT_avg")) >= 0.5
    )
    interface_plausible = (
        _stat_pass_near_rate(axis_stats.get("iPAE")) >= 0.25
        or _stat_pass_near_rate(diag_stats.get("min_ipae")) >= 0.25
        or _stat_pass_near_rate(diag_stats.get("ipTM")) >= 0.25
        or _stat_pass_near_rate(diag_stats.get("avg_ipsae")) >= 0.25
    )
    return bool(plddt_fail and interface_plausible)


def _i4_mcts_sufficiently_tried(evidence: EvidenceSummary, cfg: BuilderConfig) -> bool:
    """Whether the fixed I.4 MCTS safety seed should stop repeating.

    This intentionally distinguishes "tested and dry" from "started but not yet
    scored" and from "showing scientific progress". Positive SU, recent
    near-miss, or route-level diagnostic improvement means the MCTS route is not
    a dead safety seed; route-value replay or the LLM can continue it. The seed
    is suppressed only after enough completed, score-visible dry compute.
    """
    best = None
    best_gpu = -1.0
    for row in getattr(evidence, "route_values", None) or []:
        fam = str(
            _cb_mh_value(row, "action_family", None)
            or _cb_mh_value(row, "family", "")
            or ""
        )
        if fam != "complexa_mcts":
            continue
        scope = str(_cb_mh_value(row, "scope", "") or "")
        if scope == "family":
            best = row
            break
        gpu = float(_cb_mh_value(row, "route_gpu_h", 0.0) or 0.0)
        if gpu > best_gpu:
            best = row
            best_gpu = gpu
    if best is None:
        return False

    pending = int(_cb_mh_value(best, "pending_score_conversion_count", 0) or 0)
    if pending > 0:
        return False

    new_su = int(_cb_mh_value(best, "new_su", 0) or 0)
    near_recent = int(_cb_mh_value(best, "near_miss_recent", 0) or 0)
    diag_score = float(_cb_mh_value(best, "diagnostic_improvement_score", 0.0) or 0.0)
    if new_su > 0 or near_recent > 0 or diag_score >= float(cfg.cross_family_diagnostic_improvement_min_score):
        return False

    route_gpu_h = float(_cb_mh_value(best, "route_gpu_h", 0.0) or 0.0)
    completions = int(_cb_mh_value(best, "completions", 0) or 0)
    return (
        route_gpu_h >= float(cfg.i4_mcts_sufficient_gpu_h)
        and completions >= int(cfg.i4_mcts_sufficient_completions)
    )


def _diagnostic_i4_mcts_candidates(
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
    registry: CapabilityRegistry,
    existing: list[ActionCandidate],
    *,
    parent_pdb_available: bool = True,
) -> list[ActionCandidate]:
    """Appendix-I.4-like MCTS candidate from diagnostic evidence.

    This is a deterministic safety seed, not a launch command. Supervisor and
    Selector still decide whether it beats cheaper exploit/rescue candidates.
    """
    if not cfg.i4_mcts_seed_enabled:
        return []
    if "complexa_mcts" in set(cfg.unavailable_backends_override or ()):
        return []
    if not _hard_target_i4_signal(evidence, cfg):
        return []
    if _i4_mcts_sufficiently_tried(evidence, cfg):
        return []
    cap = registry.get("complexa_mcts")
    if cap is None:
        return []
    config_delta, dropped = validate_config_delta_partial(cap, I4_MCTS_CONFIG)
    if dropped or not config_delta:
        return []
    sig = (
        "complexa_mcts",
        cap.default_operator_id,
        json.dumps(dict(config_delta), sort_keys=True),
        None,
    )
    for c in existing:
        if (
            c.method_family,
            c.operator_id,
            json.dumps(dict(c.config_delta), sort_keys=True),
            c.parent_result_id,
        ) == sig:
            return []
    feas = feasibility_for(
        "complexa_mcts",
        evidence,
        cfg,
        registry=registry,
        parent_pdb_available=parent_pdb_available,
    )
    tick_tag = evidence.tick_id or "t000"
    return [ActionCandidate(
        candidate_id=f"diagnostic_i4_mcts_{tick_tag}_complexa_mcts",
        hypothesis_ids=["diagnostic_i4_mcts"],
        parent_result_id=None,
        method_family="complexa_mcts",
        operator_id=cap.default_operator_id,
        lane_id=cap.default_lane_id,
        config_delta=dict(config_delta),
        downstream_route_plan=_downstream_score_conversion_plan("complexa_mcts", cap),
        estimated_cost_class=cap.default_cost_class,
        expected_signal=(
            "diagnostic_i4_mcts pLDDT-aware MCTS: "
            "pLDDT/scaffold-confidence blocker with plausible interface evidence"
        ),
        evidence_refs=[
            "hard_target_signal",
            "diagnostic_driver_tldr",
            "joint_patterns",
            "diagnostic_axis_stats",
            "dry_since_last_SU",
        ],
        feasibility=feas,
        supervisor_mode="explore",
    )]


def _evidence_guided_fallback_seeds(
    evidence: EvidenceSummary,
    registry: CapabilityRegistry,
) -> tuple[tuple[str, str, dict[str, object]], ...]:
    """Fallback seeds ranked by evidence, not by a fixed family list."""
    cost_order = {"low": 0, "diagnostic": 1, "standard": 2, "extended": 3}
    rows: list[tuple[tuple[float, ...], str, str, dict[str, object]]] = []
    mh_map = getattr(evidence, "method_health", {}) or {}
    for fam, cap in registry.capabilities.items():
        if not _is_route_light_generator_cap(cap):
            continue
        h = mh_map.get(fam, {})
        gpu_h = float(_cb_mh_value(h, "cumulative_gpu_h", 0.0) or 0.0)
        strict = int(_cb_mh_value(h, "strict_yield_su", 0) or 0)
        chained = int(_cb_mh_value(h, "chained_strict_yield_su", 0) or 0)
        near = int(_cb_mh_value(h, "near_miss_yield", 0) or 0)
        recent_su = float(_cb_mh_value(h, "su_per_gpu_h_recent", 0.0) or 0.0)
        recent_chain = float(_cb_mh_value(h, "chained_su_per_gpu_h_recent", 0.0) or 0.0)
        recent_near = int(_cb_mh_value(h, "near_miss_yield_recent", 0) or 0)
        recent_signal = recent_su > 0.0 or recent_chain > 0.0 or recent_near > 0
        dead = gpu_h >= 3.0 and strict == 0 and chained == 0 and near == 0 and not recent_signal
        stale = gpu_h >= 6.0 and not recent_signal and (strict > 0 or chained > 0 or near > 0)
        penalty = 2 if dead else 1 if stale else 0
        params = _fallback_diversity_config(cap, evidence.state_label)
        tag = "evidence_fallback_dead_probe" if dead else "evidence_fallback"
        score = (
            float(penalty),
            -recent_su,
            -recent_chain,
            -float(recent_near),
            float(cost_order.get(cap.default_cost_class, 99)),
            float(gpu_h),
        )
        rows.append((score, tag, fam, params))
    if not rows:
        return ()
    viable = [r for r in rows if r[0][0] < 2.0] or rows
    viable.sort(key=lambda r: (r[0], r[2]))
    return tuple((tag, fam, params) for _, tag, fam, params in viable)


def _with_feasibility_reason(
    feas: FeasibilityCheck,
    reason: str,
    *,
    compiler_ok: bool | None = None,
    cost_ok: bool | None = None,
    route_cap_ok: bool | None = None,
) -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=feas.backend_healthy,
        runtime_bucket_id=feas.runtime_bucket_id,
        compiler_ok=feas.compiler_ok if compiler_ok is None else compiler_ok,
        verifier_ok=feas.verifier_ok,
        route_cap_ok=feas.route_cap_ok if route_cap_ok is None else route_cap_ok,
        cost_ok=feas.cost_ok if cost_ok is None else cost_ok,
        reasons=list(feas.reasons) + [reason],
    )


def _scaled_wall_reason(
    family: str,
    config_delta: dict[str, object] | None,
    evidence: EvidenceSummary,
) -> str | None:
    """llm-004 (2026-06-18): the remaining-wall gate in feasibility_for uses a FLAT
    per-family FAMILY_RUNTIME_H, but a heavy config_delta scales a generator's actual
    runtime (bindcraft max_trajectories, boltzgen num_designs×budget, complexa search
    expansion), so a config-heavy launch can pass the flat gate near end-of-run and
    then run to its HARD_CEILING and be wall-killed — wasting the unfinished tail in
    the SU/GPU-h denominator. Returns an infeasibility reason when the budget-scaled
    expected runtime no longer fits the remaining wall, else None. Multiplier is
    floored at 1.0 so default/empty configs are unchanged (this can only ADD
    restriction, never relax the base gate)."""
    from .critic_guard import FAMILY_RUNTIME_H
    rt = FAMILY_RUNTIME_H.get(family)
    if rt is None or evidence.remaining_wall_h is None or evidence.remaining_wall_h <= 0:
        return None
    if not config_delta:
        return None  # default config → base gate already covers it
    base = compute_eval_budget(family, {})
    scaled = compute_eval_budget(family, config_delta)
    mult = max(1.0, scaled / base) if base > 0 else 1.0
    if mult <= 1.0:
        return None
    rt_scaled = rt * mult
    _DRAIN_MARGIN_H = 0.1
    if rt_scaled + _DRAIN_MARGIN_H > evidence.remaining_wall_h:
        return (
            f"insufficient_wall_for_{family}_heavy_config"
            f"(~{rt_scaled:.1f}h[x{mult:.1f}]>{evidence.remaining_wall_h:.1f}h_remaining)"
        )
    return None


def _candidate_parent_refs(h: HypothesisCard) -> list[str]:
    """Evidence refs the LLM intended as a parent baseline, in priority order."""
    refs: list[str] = []
    for p in h.predicted_metric_changes:
        refs.extend(str(x) for x in (p.baseline_refs or []) if str(x))
    refs.extend(str(x) for x in (h.evidence_refs or []) if str(x))
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


def _shared_baseline_refs(h: HypothesisCard) -> tuple[list[str], bool]:
    """Return the card-wide comparison refs and whether axes disagree."""
    if not h.predicted_metric_changes:
        return [], False
    per_change = [
        [str(ref) for ref in (change.baseline_refs or []) if str(ref)]
        for change in h.predicted_metric_changes
    ]
    first = per_change[0]
    return first, any(refs != first for refs in per_change[1:])


def _looks_like_result_id(ref: str | None) -> bool:
    if not ref:
        return False
    s = str(ref).strip().lower()
    return len(s) == 16 and all(c in "0123456789abcdef" for c in s)


def _normalize_result_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    s = str(ref).strip()
    if _looks_like_result_id(s):
        return s
    for prefix in (
        "exemplar_", "exemplars.", "exemplars:",
        "example_", "examples.", "examples:",
        "production_panel_selected:", "production_panel_selected_ids.", "production_panel_selected_ids:",
        "production_near_miss:", "production_near_miss_ids.", "production_near_miss_ids:",
    ):
        if s.startswith(prefix):
            rid = s[len(prefix):]
            if _looks_like_result_id(rid):
                return rid
    return None


def _parent_ref_maps(evidence: EvidenceSummary) -> tuple[set[str], dict[str, str]]:
    """Return concrete result ids and aliases that can safely seed parent PDB work.

    Planner refs are broader than result ids: they may cite recipe hashes,
    exemplar labels, or aggregate evidence blocks. Only concrete archive result
    ids should become ActionCandidate.parent_result_id; aggregate refs remain
    evidence, not launch inputs.
    """
    concrete: set[str] = set()
    aliases: dict[str, str] = {}
    alias_rank: dict[str, int] = {}

    def _set_alias(alias: str | None, rid: str, priority: int) -> None:
        if not alias:
            return
        if alias not in aliases or priority < alias_rank.get(alias, 999):
            aliases[alias] = rid
            alias_rank[alias] = priority

    def _add_result_id(rid: str | None, *alias_refs: str) -> None:
        if not rid:
            return
        concrete.add(rid)
        _set_alias(rid, rid, -1)
        for alias in alias_refs:
            _set_alias(alias, rid, 0)

    for ex in getattr(evidence, "examples", None) or []:
        rid = getattr(ex, "result_id", None)
        _add_result_id(
            rid,
            f"example_{rid}" if rid else "",
            f"examples.{rid}" if rid else "",
            f"examples:{rid}" if rid else "",
        )
    for ex in getattr(evidence, "exemplars", None) or []:
        rid = getattr(ex, "result_id", None)
        _add_result_id(
            rid,
            f"exemplar_{rid}" if rid else "",
            f"exemplars.{rid}" if rid else "",
            f"exemplars:{rid}" if rid else "",
        )
    for rid in getattr(evidence, "production_panel_selected_ids", None) or []:
        _add_result_id(
            rid,
            f"production_panel_selected:{rid}",
            f"production_panel_selected_ids.{rid}",
            f"production_panel_selected_ids:{rid}",
        )
    for rid in getattr(evidence, "production_near_miss_ids", None) or []:
        _add_result_id(
            rid,
            f"production_near_miss:{rid}",
            f"production_near_miss_ids.{rid}",
            f"production_near_miss_ids:{rid}",
        )

    recipe_priority = {
        "strict_success": 0,
        "panel_ready": 1,
        "near_miss": 2,
        "joint_fail": 3,
    }
    for r in getattr(evidence, "recipes", None) or []:
        reps = list(getattr(r, "representative_result_ids", None) or [])
        if not reps:
            continue
        for rep in reps:
            _add_result_id(rep)
        recipe_hash = getattr(r, "recipe_hash", None)
        prio = recipe_priority.get(getattr(r, "recipe_class", ""), 9)
        if recipe_hash:
            _set_alias(recipe_hash, reps[0], prio)
            _set_alias(f"recipe_{recipe_hash}", reps[0], prio)
    return concrete, aliases


def _joint_fail_only_representative_ids(evidence: EvidenceSummary) -> set[str]:
    """Representatives that are only known through joint-fail evidence."""
    joint_fail: set[str] = set()
    productive_or_near: set[str] = set()
    for r in getattr(evidence, "recipes", None) or []:
        reps = set(getattr(r, "representative_result_ids", None) or [])
        if not reps:
            continue
        if getattr(r, "recipe_class", None) == "joint_fail":
            joint_fail.update(reps)
        else:
            productive_or_near.update(reps)
    return joint_fail - productive_or_near


def _joint_fail_parent_source_families(evidence: EvidenceSummary) -> dict[str, set[str]]:
    """Map joint-fail representatives to the families that generated them."""
    out: dict[str, set[str]] = {}
    for r in getattr(evidence, "recipes", None) or []:
        if getattr(r, "recipe_class", None) != "joint_fail":
            continue
        fam = getattr(r, "method_family", None)
        if not fam:
            continue
        for rep in getattr(r, "representative_result_ids", None) or []:
            if rep:
                out.setdefault(rep, set()).add(str(fam))
    return out



def _source_families_for_row(row: Any) -> set[str]:
    out: set[str] = set()
    for key in ("action_family", "root_family", "family", "method_family"):
        value = _cb_mh_value(row, key, None)
        if value:
            out.add(str(value))
    return out


def _add_parent_source(out: dict[str, set[str]], rid: str | None, families: set[str]) -> None:
    norm = _normalize_result_ref(rid)
    if norm is not None and families:
        out.setdefault(norm, set()).update(families)


def _parent_source_family_map(evidence: EvidenceSummary) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for ex in getattr(evidence, "examples", None) or []:
        _add_parent_source(out, getattr(ex, "result_id", None), _source_families_for_row(ex))
    for ex in getattr(evidence, "exemplars", None) or []:
        families = _source_families_for_row(ex)
        _add_parent_source(out, getattr(ex, "result_id", None), families)
        _add_parent_source(out, getattr(ex, "parent_result_id", None), families)
    for r in getattr(evidence, "recipes", None) or []:
        families = _source_families_for_row(r)
        for rep in getattr(r, "representative_result_ids", None) or []:
            _add_parent_source(out, rep, families)
    for row in getattr(evidence, "strategy_feedback", None) or []:
        families = _source_families_for_row(row)
        for key in ("representative_result_ids", "representative_strict_ids", "representative_near_miss_ids", "representative_failure_ids"):
            values = _cb_mh_value(row, key, []) or []
            if isinstance(values, str):
                values = [values]
            for rid in values:
                _add_parent_source(out, rid, families)
    for row in getattr(evidence, "route_values", None) or []:
        families = _source_families_for_row(row)
        refs = _cb_mh_value(row, "evidence_refs", []) or []
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            _add_parent_source(out, ref, families)
    return out


_PARENT_SOURCE_PHRASES: dict[str, tuple[str, ...]] = {
    "boltzgen": ("boltzgen backlog", "boltzgen backbones", "boltzgen backbone", "boltzgen-generated", "boltzgen generated", "boltzgen artifact", "boltzgen artifacts", "boltzgen parent"),
    "bindcraft": ("bindcraft backlog", "bindcraft backbones", "bindcraft backbone", "bindcraft-generated", "bindcraft generated", "bindcraft artifact", "bindcraft artifacts", "bindcraft parent"),
    "complexa": ("complexa backlog", "complexa backbones", "complexa backbone", "complexa-generated", "complexa generated", "complexa artifact", "complexa artifacts", "complexa parent"),
}


def _requested_parent_source_families(h: HypothesisCard, action_family: str) -> set[str]:
    if action_family not in {"proteinmpnn_redesign", "structure_refilter"}:
        return set()
    text = _hypothesis_action_text(h)
    requested: set[str] = set()
    for family, phrases in _PARENT_SOURCE_PHRASES.items():
        if any(phrase in text for phrase in phrases):
            requested.add(family)
    return requested


def _family_matches_requested_source(source_family: str, requested_family: str) -> bool:
    if requested_family == "complexa":
        return _root_family_group(source_family) == "complexa"
    return source_family == requested_family or _root_family_group(source_family) == requested_family


def _parent_source_mismatch_reason(*, hypothesis: HypothesisCard, action_family: str, parent_result_id: str | None, parent_source_families: dict[str, set[str]]) -> str | None:
    if not parent_result_id:
        return None
    requested = _requested_parent_source_families(hypothesis, action_family)
    if not requested:
        return None
    known = parent_source_families.get(parent_result_id, set())
    if not known:
        return None
    if any(_family_matches_requested_source(src, req) for src in known for req in requested):
        return None
    return "parent_source_mismatch:" + f"requested={chr(44).join(sorted(requested))}:resolved={chr(44).join(sorted(known))}:parent={parent_result_id}"


def _hypothesis_text(h: HypothesisCard) -> str:
    rt = getattr(h, "reasoning_trace", None)
    parts = [getattr(h, "claim", "")]
    if rt is not None:
        parts.extend([
            getattr(rt, "observed_signal", ""),
            getattr(rt, "inference", ""),
            getattr(rt, "action_implication", ""),
        ])
    parts.extend(str(x) for x in (getattr(h, "evidence_refs", None) or []))
    return " ".join(p for p in parts if p).lower()


def _hypothesis_action_text(h: HypothesisCard) -> str:
    """Action-intent text only, excluding observational evidence."""
    rt = getattr(h, "reasoning_trace", None)
    parts = [getattr(h, "claim", "")]
    if rt is not None:
        parts.append(getattr(rt, "action_implication", ""))
    return " ".join(p for p in parts if p).lower()


def _hypothesis_requests_existing_score_backlog(h: HypothesisCard) -> bool:
    """Whether a card is asking to score existing backlog, not generate more.

    Canonical AF2 score conversion of already accepted diagnostic artifacts is
    scheduled by the controller's chain_refilter lane. Observing an unscored
    backlog is not enough to block a generator: the action itself must ask to
    score/clear/convert existing artifacts. If the action says to generate or
    explicitly says it is not a scoring request, keep the generator feasible.
    """
    text = _hypothesis_text(h)
    action_text = _hypothesis_action_text(h)
    generation_intent = any(
        term in action_text for term in (
            "new generation", "generate new", "generate another",
            "new boltzgen", "new bindcraft", "fresh generation",
            "fresh boltzgen", "fresh bindcraft", "explore boltzgen",
            "explore bindcraft", "generate structurally", "different seed",
            "different stochastic seed",
        )
    )
    explicit_not_scoring = any(
        term in action_text for term in (
            "not a scoring request", "not a score request",
            "not score-conversion", "not scoring", "while the backlog is scored",
            "while backlog is scored",
        )
    )
    if generation_intent or explicit_not_scoring:
        return False
    has_backlog = any(
        term in text for term in (
            "diagnostic_chain_backlog", "backlog", "pending", "unscored",
            "already accepted", "accepted artifact", "accepted artifacts",
        )
    )
    has_score_action = any(
        term in action_text for term in (
            "score pending", "score existing", "score accepted",
            "score the backlog", "score backlog", "scoring backlog",
            "rescore pending", "rescore existing", "convert pending",
            "convert existing", "convert accepted", "clear pending", "clear backlog",
            "drain", "canonical", "structure_refilter", "chain_refilter",
        )
    )
    return has_backlog and has_score_action


def _health_value(row: Any, key: str, default: Any = 0) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _diagnostic_first_feedback_reason(
    family: str,
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
) -> str | None:
    """Gate repeated diagnostic generation until the first AF2 feedback lands.

    This is route-agnostic delayed-feedback control: if a diagnostic-only family
    has already emitted a batch of accepted artifacts but only a few canonical
    structure_refilter children have returned, launch score-conversion first.
    Once any strict/near/SU signal exists, or enough first-batch refilters have
    completed, the family is free for normal Supervisor/Selector allocation.
    """
    backlog = getattr(evidence, "diagnostic_chain_backlog", None) or {}
    by_family = backlog.get("by_family") if isinstance(backlog, dict) else None
    row = (by_family or {}).get(family) if isinstance(by_family, dict) else None
    if not isinstance(row, dict):
        return None
    accepted = int(row.get("accepted_artifacts", 0) or 0)
    unscored = int(row.get("unscored_artifacts", 0) or 0)
    completed = int(row.get("completed_refilters", 0) or 0)
    if accepted < max(1, cfg.diagnostic_first_feedback_min_accepted) or unscored <= 0:
        return None
    mh = (getattr(evidence, "method_health", None) or {}).get(family)
    strict = int(_health_value(mh, "strict_yield", 0) or 0)
    near = int(_health_value(mh, "near_miss_yield", 0) or 0)
    chained = int(_health_value(mh, "chained_strict_yield_su", 0) or 0)
    chained_recent = int(_health_value(mh, "chained_strict_yield_su_recent", 0) or 0)
    if strict > 0 or near > 0 or chained > 0 or chained_recent > 0:
        return None
    need = min(accepted, max(1, cfg.diagnostic_first_feedback_min_refilters))
    if completed >= need:
        return None
    return f"awaiting_first_score_conversion:{family}:completed={completed}/need={need}:unscored={unscored}"


_DIAGNOSTIC_ROUTE_DEFAULT_CONFIG: dict[str, dict[str, object]] = {
    "boltzgen": {"num_designs": 16, "budget": 4},
    "bindcraft": {"max_trajectories": 4},
}
_DIAGNOSTIC_ROUTE_FEEDBACK_FAMILIES = {"boltzgen", "bindcraft"}


def _normalized_diagnostic_route_config(
    family: str, config_delta: dict[str, object] | None
) -> dict[str, object]:
    out = dict(_DIAGNOSTIC_ROUTE_DEFAULT_CONFIG.get(family, {}))
    out.update(dict(config_delta or {}))
    return out


def _diagnostic_route_feedback_reason(
    family: str,
    config_delta: dict[str, object],
    evidence: EvidenceSummary,
) -> str | None:
    """Block same exact diagnostic route replay until pending AF2 scores return.

    The family-level gate above prevents a whole diagnostic generator from
    immediately replaying before the first score-conversion microbatch lands.
    Once that family has some feedback, this route-level gate is narrower: a
    different BoltzGen/BindCraft config can still explore, but the exact
    route/config that already has unscored artifacts must wait for its canonical
    score-conversion children before being generated again.
    Parent-bound rescue routes are intentionally excluded here because their route identity includes
    parent context; the parent/source/lineage gates handle those cases.
    """
    if family not in _DIAGNOSTIC_ROUTE_FEEDBACK_FAMILIES:
        return None
    cfg_key = json.dumps(_normalized_diagnostic_route_config(family, config_delta), sort_keys=True)
    for row in getattr(evidence, "route_values", None) or []:
        if _row_scope(row) != "route":
            continue
        if _row_family(row) != family:
            continue
        row_cfg = _cb_mh_value(row, "config_delta", {}) or {}
        if not isinstance(row_cfg, dict):
            row_cfg = {}
        if json.dumps(_normalized_diagnostic_route_config(family, row_cfg), sort_keys=True) != cfg_key:
            continue
        pending = int(_cb_mh_value(row, "pending_score_conversion_count", 0) or 0)
        if pending <= 0:
            continue
        objective_signal = (
            int(_cb_mh_value(row, "new_su", 0) or 0) > 0
            or int(_cb_mh_value(row, "new_su_recent", 0) or 0) > 0
            or int(_cb_mh_value(row, "record_recent_new_su", 0) or 0) > 0
            or int(_cb_mh_value(row, "new_su_recent_gpu", 0) or 0) > 0
            or int(_cb_mh_value(row, "medium_recent_new_su", 0) or 0) > 0
            or int(_cb_mh_value(row, "near_miss_count", 0) or 0) > 0
            or int(_cb_mh_value(row, "near_miss_recent", 0) or 0) > 0
        )
        if objective_signal:
            return None
        strategy_key = str(_cb_mh_value(row, "strategy_key", "route") or "route")
        return (
            f"awaiting_route_score_conversion:{family}:"
            f"pending={pending}:strategy={strategy_key}"
        )
    return None


def _is_aggregate_evidence_ref(ref: str) -> bool:
    if not ref:
        return True
    aggregate_exact = {
        "axis_stats",
        "diagnostic_alt_model_scores",
        "diagnostic_axis_stats",
        "diagnostic_chain_backlog",
        "duplicate_fraction",
        "examples",
        "exemplars",
        "joint_patterns",
        "method_health",
        "metric_availability",
        "near_miss_count",
        "production_panel",
        "production_panel_diversity_bins",
        "production_panel_gap_reasons",
        "production_panel_selected_ids",
        "production_near_miss_ids",
        "recipes",
        "recent_ticks_history",
        "route_health",
        "strategy_feedback",
        "top_bin_share",
    }
    aggregate_prefixes = (
        "axis_stats.",
        "diagnostic_alt_model_scores.",
        "diagnostic_axis_stats.",
        "diagnostic_chain_backlog.",
        "family_role_table",
        "method_health.",
        "recipe_",
        "route_health.",
        "strategy_feedback",
    )
    return ref in aggregate_exact or ref.startswith(aggregate_prefixes)


# Canonical AF2 score-conversion config = the FIXED SU-minting basis (matches the
# auto-chain conversion + the V5/V6.3 baseline). An LLM-proposed structure_refilter
# is an intentional `parent_model_refold` (on the E/R/X budget, credited as its own
# refold strategy) ONLY if it MATERIALLY changes this scoring config. A no-op /
# default re-score is functionally the auto-chain conversion, so it is tagged
# canonical (off-budget, SU credited to the upstream generator) — this stops the
# LLM from relabeling free score-conversion as a budgeted scientific action and
# from silently stealing SU credit from the generator that produced the backbone.


def _downstream_score_conversion_plan(family: str, cap) -> list[str]:
    """Return the explicit downstream plan for diagnostic-only generators.

    Direct-scored Complexa rows already carry official strict axes from their
    inline AF2-multimer path. If those axes are absent, the controller's
    `_record_needs_score_conversion` fallback still sends the emitted artifact
    through canonical score conversion. Do not advertise that exceptional
    fallback as a routine candidate plan; it confuses route provenance and the
    structure_refilter-vs-score-conversion distinction in prompts/logs.
    """
    if cap is not None and cap.outputs_diagnostic_only:
        return ["structure_refilter"]
    return []


_CANONICAL_REFILTER_CONFIG: dict[str, object] = {
    "model_names": "model_1_multimer_v3",
    "num_recycles": 3,
    "use_initial_guess": 1,
}


def _structure_refilter_role(method_family: str, config_delta: dict | None) -> str | None:
    if method_family != "structure_refilter":
        return None
    return (
        PARENT_MODEL_REFOLD if _is_material_parent_model_refold(config_delta)
        else CANONICAL_SCORE_CONVERSION
    )


def _resolve_parent_result_id(
    h: HypothesisCard,
    evidence: EvidenceSummary,
) -> tuple[str | None, str | None]:
    concrete, aliases = _parent_ref_maps(evidence)
    rejected: list[str] = []
    _baseline_refs, baseline_conflict = _shared_baseline_refs(h)
    if baseline_conflict:
        return None, "inconsistent_baseline_refs"
    for ref in _candidate_parent_refs(h):
        if ref in aliases:
            return aliases[ref], None
        if ref in concrete:
            return ref, None
        normalized = _normalize_result_ref(ref)
        if normalized is not None and normalized in concrete:
            return normalized, None
        if normalized is not None:
            rejected.append(f"unknown_result_id:{normalized}")
            continue
        if _is_aggregate_evidence_ref(ref):
            rejected.append(f"aggregate:{ref}")
        else:
            rejected.append(ref)
    if rejected:
        bad = ",".join(rejected[:3])
        return None, f"no_concrete_parent_result_id({bad})"
    return None, "no_concrete_parent_result_id"


def _resolve_baseline_result_id(
    h: HypothesisCard,
    evidence: EvidenceSummary,
) -> str | None:
    """Resolve the hypothesis comparison baseline without imposing launch gates."""
    concrete, aliases = _parent_ref_maps(evidence)
    baseline_refs, baseline_conflict = _shared_baseline_refs(h)
    if baseline_conflict:
        # Legacy/malformed cards with conflicting per-axis baselines cannot be
        # represented by ActionCandidate.baseline_result_id. Do not silently
        # evaluate every axis against whichever reference happened to be first.
        return None
    refs = baseline_refs + [str(ref) for ref in (h.evidence_refs or []) if str(ref)]
    for ref in refs:
        if ref in aliases:
            return aliases[ref]
        if ref in concrete:
            return ref
        normalized = _normalize_result_ref(ref)
        if normalized is not None and normalized in concrete:
            return normalized
    return None


def _warmstart_candidates(
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
    registry: CapabilityRegistry,
    has_llm_hypotheses: bool = False,
    parent_pdb_available: bool = True,
    completed_families: set[str] | None = None,
) -> list[ActionCandidate]:
    """Deterministic seed candidates.

    Two cases:
      (1) Cold archive (no recipes, state=low_evidence): emit the accepted
          three-paradigm seed lanes. Fires even if the LLM also produced cards
          — cold-start guarantees a launchable evidence floor.
      (2) Stalled/deep-stall/strict-duplicate-collapse state with no usable
          generator card: emit evidence-ranked route-light generators from the
          capability registry.
          This prevents an empty LLM tick from idling GPUs without reintroducing
          a fixed post-evidence method-family prior.

    M-1 fix (2026-05-26): Stalled-fallback no longer fires when the LLM has
    proposed hypotheses. The LLM's diagnostic rescue cards, including
    Complexa-internal tuning cards, were being out-competed for selector quota
    by deterministic fallback candidates. The fallback is a safety net for
    empty/dead-end LLM output, not a parallel exploration channel.
    """
    if evidence.state_label == "low_evidence":
        # Cold-start warmstart only when the archive carries no evidence yet.
        if evidence.recipes:
            return []
        seeds = tuple(("warmstart", fam, params) for fam, params in WARMSTART_FAMILIES)
    elif evidence.state_label in ("stalled", "deep_stall", "strict_duplicate_collapse"):
        # Fallback fires only when the LLM/builder has no usable route-light
        # generator card. Candidate families come from registry capabilities and
        # are ranked by recent SU/near-miss/chained-SU evidence plus cost.
        if has_llm_hypotheses:
            return []
        seeds = _evidence_guided_fallback_seeds(evidence, registry)
    else:
        return []

    out: list[ActionCandidate] = []
    # Skip seeds whose family was excluded at the cluster level
    # (--enabled-families). Without this, every stalled tick would
    # emit an infeasible boltzgen candidate that the selector then drops.
    cluster_unavail = set(cfg.unavailable_backends_override or ())
    completed_families = set(completed_families or ())
    for i, (seed_tag, fam, params) in enumerate(seeds):
        if fam in cluster_unavail or (seed_tag == "warmstart" and fam in completed_families):
            continue
        cap = registry.get(fam) if registry else None
        if cap is None:
            continue
        op, lane, cost = cap.default_operator_id, cap.default_lane_id, cap.default_cost_class
        feas = feasibility_for(fam, evidence, cfg, registry=registry,
                                    parent_pdb_available=parent_pdb_available)
        # Diagnostic-only capabilities emit native/advisory structures, not the
        # strict AF2-calibrated metrics, so their artifacts must chain through
        # canonical structure_refilter to enter the SU path. Use registry
        # metadata instead of a family list so provenance cannot drift.
        out.append(
            ActionCandidate(
                candidate_id=f"{seed_tag}_{evidence.tick_id}_{i:02d}_{fam}",
                hypothesis_ids=[seed_tag],
                parent_result_id=None,
                method_family=fam,
                operator_id=op,
                lane_id=lane,
                config_delta=dict(params),
                downstream_route_plan=_downstream_score_conversion_plan(fam, cap),
                estimated_cost_class=cost,
                expected_signal=f"{seed_tag} {fam} {params}",
                evidence_refs=[],
                feasibility=feas,
            )
        )
    return out



def _route_value_rate_signal(row: Any) -> _RouteValueRateSignal:
    """Decision-safe route value for replay/allocation.

    GPU-window value is the primary recent signal. Medium-window value is the
    delayed-feedback guard. The record window is only a fallback for direct-
    scored routes; for delayed score-conversion routes in older archives it may
    contain only tiny AF2 refilter records while missing the generator GPU-h.
    Lifetime value is memory, not current exploit evidence.
    """
    def _as_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    lifetime = max(0.0, _as_float(_cb_mh_value(row, "new_su_per_route_gpu_h", None)) or 0.0)

    value = _as_float(_cb_mh_value(row, "gpu_recent_new_su_per_route_gpu_h", None))
    if value is not None:
        return _RouteValueRateSignal(max(0.0, value), "gpu_recent", True, lifetime)

    value = _as_float(_cb_mh_value(row, "medium_recent_new_su_per_route_gpu_h", None))
    if value is not None:
        return _RouteValueRateSignal(max(0.0, value), "medium_recent", True, lifetime)

    route_role = str(_cb_mh_value(row, "route_role", "") or "")
    canonical_refilter_gpu_h = _as_float(_cb_mh_value(row, "canonical_refilter_gpu_h", 0.0)) or 0.0
    record_value = _as_float(_cb_mh_value(
        row,
        "record_recent_new_su_per_route_gpu_h",
        _cb_mh_value(row, "recent_new_su_per_route_gpu_h", None),
    ))
    if (
        record_value is not None
        and canonical_refilter_gpu_h <= 0.0
        and "score_conversion" not in route_role
    ):
        return _RouteValueRateSignal(max(0.0, record_value), "record_recent", True, lifetime)

    return _RouteValueRateSignal(lifetime, "lifetime_memory", False, lifetime)


def _route_value_current_rate(row: Any) -> float:
    return _route_value_rate_signal(row).rate


def _route_value_recent_new_su(row: Any) -> int:
    values: list[int] = []
    for key in (
        "new_su_recent_gpu",
        "medium_recent_new_su",
        "record_recent_new_su",
        "new_su_recent",
    ):
        try:
            values.append(int(_cb_mh_value(row, key, 0) or 0))
        except (TypeError, ValueError):
            values.append(0)
    return max(values or [0])


def _route_value_recent_gpu_h(row: Any) -> float:
    values: list[float] = []
    for key in (
        "gpu_recent_route_gpu_h",
        "medium_recent_route_gpu_h",
        "record_recent_route_gpu_h",
        "recent_route_gpu_h",
    ):
        try:
            values.append(float(_cb_mh_value(row, key, 0.0) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return max(values or [0.0])


def _route_value_replay_candidates(
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
    registry: CapabilityRegistry,
    existing: list[ActionCandidate],
    *,
    parent_pdb_available: bool = True,
) -> list[ActionCandidate]:
    # Keep exact productive route/config rows visible to Supervisor when one
    # Planner tick omits them. This is evidence replay, not a target prior.
    if not cfg.route_value_replay_enabled or cfg.route_value_replay_max_candidates <= 0:
        return []

    seen = {
        (
            c.method_family,
            c.operator_id,
            json.dumps(dict(c.config_delta), sort_keys=True),
            c.parent_result_id,
        )
        for c in existing
    }
    status_rank = {"promote": 0, "healthy": 1, "observed": 2}
    rows: list[tuple[tuple[float, ...], Any, dict[str, Any], Any]] = []
    for row in getattr(evidence, "route_values", None) or []:
        if getattr(row, "scope", None) != "route" and not (isinstance(row, dict) and row.get("scope") == "route"):
            continue
        status = str(_cb_mh_value(row, "status", "observed") or "observed")
        marginal_status = str(_cb_mh_value(row, "marginal_status", status) or status)
        # `diversify` means strict output is duplicate-heavy, not that the
        # route is dead. Keep it replayable when new SU/GPU-h is still high;
        # later rate/staleness gates decide whether it is worth launching.
        if status in {"collapse_risk", "plumbing", "untried"}:
            continue
        fam = str(
            _cb_mh_value(row, "action_family", None)
            or _cb_mh_value(row, "family", "")
            or ""
        )
        if not fam:
            continue
        cap = registry.get(fam)
        if cap is None or not _is_route_light_generator_cap(cap):
            continue
        if fam in set(cfg.unavailable_backends_override or ()):  # cluster-level family filter
            continue
        raw_cfg = _cb_mh_value(row, "config_delta", {}) or {}
        if not isinstance(raw_cfg, dict):
            raw_cfg = {}
        config_delta, dropped = validate_config_delta_partial(cap, raw_cfg) if raw_cfg else ({}, [])
        if raw_cfg and dropped and not config_delta:
            continue
        sig = (fam, cap.default_operator_id, json.dumps(dict(config_delta), sort_keys=True), None)
        if sig in seen:
            continue
        lifetime_rate = float(_cb_mh_value(row, "new_su_per_route_gpu_h", 0.0) or 0.0)
        rate_signal = _route_value_rate_signal(row)
        rate = rate_signal.rate
        new_su = int(_cb_mh_value(row, "new_su", 0) or 0)
        new_su_recent = _route_value_recent_new_su(row)
        near = int(_cb_mh_value(row, "near_miss_count", 0) or 0)
        near_recent = int(_cb_mh_value(row, "near_miss_recent", 0) or 0)
        recent_gpu = _route_value_recent_gpu_h(row)
        route_gpu_h = float(_cb_mh_value(row, "route_gpu_h", 0.0) or 0.0)
        route_role = str(_cb_mh_value(row, "route_role", "") or "")
        canonical_refilter_gpu_h = float(_cb_mh_value(row, "canonical_refilter_gpu_h", 0.0) or 0.0)
        pending_score_conversion = int(_cb_mh_value(row, "pending_score_conversion_count", 0) or 0)
        diag_score = float(_cb_mh_value(row, "diagnostic_improvement_score", 0.0) or 0.0)
        diag_axes = [str(x) for x in (_cb_mh_value(row, "diagnostic_improvement_axes", []) or []) if str(x)]
        objective_signal = new_su > 0 or new_su_recent > 0 or near > 0 or near_recent > 0
        delayed_feedback_route = (
            bool(getattr(cap, "outputs_diagnostic_only", False))
            or "score_conversion" in route_role
            or canonical_refilter_gpu_h > 0.0
        )
        diagnostic_replay = (
            not objective_signal
            and delayed_feedback_route
            and pending_score_conversion <= 0
            and diag_score >= float(cfg.cross_family_diagnostic_improvement_min_score)
            and route_gpu_h <= float(cfg.route_value_replay_diagnostic_max_gpu_h)
            and marginal_status not in {"dry_low_quality", "dry_duplicate", "collapse_risk"}
        )
        if status == "defer" and not diagnostic_replay:
            continue
        if not (objective_signal or diagnostic_replay):
            continue
        try:
            target_dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
        except (TypeError, ValueError):
            target_dry_gpu_h = 0.0
        has_gpu_or_medium_recent_signal = (
            _cb_mh_value(row, "gpu_recent_new_su_per_route_gpu_h", None) is not None
            or _cb_mh_value(row, "medium_recent_new_su_per_route_gpu_h", None) is not None
            or int(_cb_mh_value(row, "new_su_recent_gpu", 0) or 0) > 0
            or int(_cb_mh_value(row, "medium_recent_new_su", 0) or 0) > 0
        )
        stale_lifetime_memory = (
            objective_signal
            and not rate_signal.has_recent_rate
            and new_su_recent <= 0
            and near_recent <= 0
        )
        stale_record_memory = (
            objective_signal
            and rate_signal.source == "record_recent"
            and not has_gpu_or_medium_recent_signal
            and target_dry_gpu_h >= cfg.route_value_replay_lifetime_stale_dry_gpu_h
        )
        if objective_signal and (
            new_su_recent <= 0
            and near_recent <= 0
            and recent_gpu >= cfg.route_value_replay_stale_recent_gpu_h
        ):
            continue
        if (
            stale_lifetime_memory
            and target_dry_gpu_h >= cfg.route_value_replay_lifetime_stale_dry_gpu_h
        ):
            continue
        if stale_record_memory:
            continue
        if objective_signal and rate < cfg.route_value_replay_min_su_per_gpu_h and new_su_recent <= 0 and near_recent <= 0:
            continue
        signal_rank = 0 if objective_signal else 1
        stale_memory_rank = 1 if (stale_lifetime_memory or stale_record_memory) else 0
        score = (
            float(signal_rank),
            float(stale_memory_rank),
            float(status_rank.get(status, 9)),
            -rate,
            -lifetime_rate,
            -float(new_su_recent),
            -float(new_su),
            -float(near_recent),
            -float(near),
            -float(diag_score),
            route_gpu_h,
        )
        rows.append((score, row, config_delta, cap))

    rows.sort(key=lambda x: (x[0], str(_cb_mh_value(x[1], "strategy_key", ""))))
    out: list[ActionCandidate] = []
    tick_tag = evidence.tick_id or "t000"
    for idx, (_, row, config_delta, cap) in enumerate(rows[:max(0, int(cfg.route_value_replay_max_candidates))]):
        fam = str(_cb_mh_value(row, "action_family", None) or _cb_mh_value(row, "family", ""))
        feas = feasibility_for(
            fam,
            evidence,
            cfg,
            registry=registry,
            parent_pdb_available=parent_pdb_available,
        )
        refs = [str(x) for x in (_cb_mh_value(row, "evidence_refs", []) or []) if str(x)]
        if not refs:
            refs = ["route_values"]
        rate_signal = _route_value_rate_signal(row)
        rate = rate_signal.rate
        status_text = str(_cb_mh_value(row, "status", "observed") or "observed")
        diag_score = float(_cb_mh_value(row, "diagnostic_improvement_score", 0.0) or 0.0)
        diag_axes = [str(x) for x in (_cb_mh_value(row, "diagnostic_improvement_axes", []) or []) if str(x)]
        objective_signal = (
            int(_cb_mh_value(row, "new_su", 0) or 0) > 0
            or _route_value_recent_new_su(row) > 0
            or int(_cb_mh_value(row, "near_miss_count", 0) or 0) > 0
            or int(_cb_mh_value(row, "near_miss_recent", 0) or 0) > 0
        )
        recent_su_for_expected = _route_value_recent_new_su(row)
        near_recent_for_expected = int(_cb_mh_value(row, "near_miss_recent", 0) or 0)
        if not objective_signal:
            expected = f"route_value_replay {fam} status={status_text} new_su_per_gpu_h={rate:.3f}"
        elif rate_signal.source == "lifetime_memory":
            expected = (
                f"route_value_replay {fam} status={status_text} "
                f"lifetime_su_per_gpu_h={rate_signal.lifetime_rate:.3f} "
                "source=lifetime_memory"
            )
            if recent_su_for_expected > 0:
                expected += f" recent_su_count_unrated={recent_su_for_expected}"
            elif near_recent_for_expected > 0:
                expected += f" recent_near_miss_count={near_recent_for_expected}"
            else:
                expected += " no_recent_signal=1"
        else:
            expected = (
                f"route_value_replay {fam} status={status_text} "
                f"new_su_per_gpu_h={rate:.3f} source={rate_signal.source}"
            )
        if not objective_signal and diag_score > 0.0:
            expected += f" diagnostic_improvement_score={diag_score:.3f}"
            if diag_axes:
                expected += " diagnostic_axes=" + ";".join(diag_axes[:3])
        out.append(ActionCandidate(
            candidate_id=f"route_replay_{tick_tag}_{idx:02d}_{fam}",
            hypothesis_ids=["route_value_replay"],
            parent_result_id=None,
            method_family=fam,
            operator_id=cap.default_operator_id,
            lane_id=cap.default_lane_id,
            config_delta=dict(config_delta),
            downstream_route_plan=_downstream_score_conversion_plan(fam, cap),
            estimated_cost_class=cap.default_cost_class,
            expected_signal=expected,
            evidence_refs=refs[:4],
            feasibility=feas,
            supervisor_mode=(
                "exploit"
                if (
                    rate_signal.has_recent_rate
                    and rate > 0.0
                    and (
                        int(_cb_mh_value(row, "new_su", 0) or 0) > 0
                        or recent_su_for_expected > 0
                    )
                )
                else (
                    "rescue"
                    if (
                        near_recent_for_expected > 0
                        or int(_cb_mh_value(row, "near_miss_count", 0) or 0) > 0
                    )
                    else "explore"
                )
            ),
        ))
    return out


def _root_family_group(family: str | None) -> str:
    fam = str(family or "")
    if fam.startswith("complexa_"):
        return "complexa"
    return fam


def _row_scope(row: Any) -> str:
    return str(_cb_mh_value(row, "scope", "") or "")


def _row_family(row: Any) -> str:
    return str(
        _cb_mh_value(row, "action_family", None)
        or _cb_mh_value(row, "family", "")
        or ""
    )


def _route_value_record_recent_safe_for_current_signal(
    row: Any,
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
) -> bool:
    """Whether record-window route value can stand in for current progress.

    Record windows can overstate delayed score-conversion routes because the
    recent rows are often cheap AF2 children while the upstream generator GPU-h
    sits outside that record window. Keep this aligned with Selector's
    decision-safe route-rate contract: record_recent is a fallback only for
    direct/non-score-conversion routes, and not after a hard dry/stale interval.
    """
    route_role = str(_cb_mh_value(row, "route_role", "") or "")
    try:
        canonical_refilter_gpu_h = float(
            _cb_mh_value(row, "canonical_refilter_gpu_h", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        canonical_refilter_gpu_h = 0.0
    if canonical_refilter_gpu_h > 0.0 or "score_conversion" in route_role:
        return False
    try:
        dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        dry_gpu_h = 0.0
    return dry_gpu_h < float(cfg.route_value_replay_lifetime_stale_dry_gpu_h)


def _dominant_route_root_group(evidence: EvidenceSummary, registry: CapabilityRegistry) -> str | None:
    """Root group that has consumed the most generator route GPU-h.

    Exact route rows are the trusted source when present. Family rollups are a
    fallback for early/smoke EvidenceSummary payloads that do not yet carry exact
    routes; mixing both double-counts the same GPU-h and can suppress the
    cross-family escape floor.
    """
    rows = list(getattr(evidence, "route_values", None) or [])
    route_rows = [row for row in rows if _cb_mh_value(row, "scope", None) == "route"]
    family_rows = [row for row in rows if _cb_mh_value(row, "scope", None) == "family"]
    source_rows = route_rows if route_rows else family_rows

    gpu_by_root: dict[str, float] = {}
    for row in source_rows:
        fam = _row_family(row)
        cap = registry.get(fam)
        if cap is None or not _is_route_light_generator_cap(cap):
            continue
        gpu = float(_cb_mh_value(row, "route_gpu_h", 0.0) or 0.0)
        if gpu <= 0.0:
            continue
        root = _root_family_group(fam)
        gpu_by_root[root] = gpu_by_root.get(root, 0.0) + gpu
    if not gpu_by_root:
        return None
    return max(gpu_by_root, key=lambda k: (gpu_by_root[k], k))


def _pending_route_light_root_groups(
    evidence: EvidenceSummary,
    registry: CapabilityRegistry,
    *,
    exclude_root: str | None = None,
) -> set[str]:
    """Root-family groups already running/queued in the event controller."""
    pending = getattr(evidence, "pending_family_load", None) or {}
    by_family = pending.get("by_family", {}) if isinstance(pending, dict) else {}
    out: set[str] = set()
    for fam, row in by_family.items():
        cap = registry.get(str(fam))
        if cap is None or not _is_route_light_generator_cap(cap):
            continue
        if not isinstance(row, dict):
            continue
        pending_total = int(row.get("pending_total", 0) or 0)
        running = int(row.get("running", 0) or 0)
        queued = int(row.get("queued", 0) or 0)
        if max(pending_total, running + queued) <= 0:
            continue
        root = _root_family_group(str(fam))
        if root and root != exclude_root:
            out.add(root)
    return out


def _root_has_current_su(
    evidence: EvidenceSummary,
    root: str | None,
    cfg: BuilderConfig,
) -> bool:
    if not root:
        return False
    for row in getattr(evidence, "route_values", None) or []:
        fam = _row_family(row)
        if _root_family_group(fam) != root:
            continue
        if float(_cb_mh_value(row, "gpu_recent_new_su_per_route_gpu_h", 0.0) or 0.0) > 0.0:
            return True
        if float(_cb_mh_value(row, "medium_recent_new_su_per_route_gpu_h", 0.0) or 0.0) > 0.0:
            return True
        if int(_cb_mh_value(row, "new_su_recent_gpu", 0) or 0) > 0:
            return True
        if int(_cb_mh_value(row, "medium_recent_new_su", 0) or 0) > 0:
            return True
        if not _route_value_record_recent_safe_for_current_signal(row, evidence, cfg):
            continue
        if float(_cb_mh_value(row, "record_recent_new_su_per_route_gpu_h", 0.0) or 0.0) > 0.0:
            return True
        if int(_cb_mh_value(row, "record_recent_new_su", _cb_mh_value(row, "new_su_recent", 0)) or 0) > 0:
            return True
    return False


def _deep_stall_recovery_active(evidence: EvidenceSummary, cfg: BuilderConfig) -> bool:
    # Weak recovery after a long dry plateau: keep one escape probe open.
    # A new SU after deep_stall is real signal, so we do not relabel the state
    # or kill exploit. But when recent history shows sustained deep_stall and
    # the total run still has low SU/GPU-h, one fresh SU should not fully close
    # cross-family exploration. This fades as history moves past the plateau or
    # the run-level rate improves.
    state = str(getattr(evidence, "state_label", "") or "")
    if state not in {"productive", "productive_duplicate"}:
        return False
    if int(getattr(evidence, "run_su_count_delta", 0) or 0) <= 0:
        return False
    total_rate = getattr(evidence, "run_su_per_worker_gpu_h_total", None)
    if total_rate is None:
        worker_h = float(getattr(evidence, "worker_gpu_h_total", 0.0) or 0.0)
        total_su = int(getattr(evidence, "run_su_count", 0) or 0)
        total_rate = (total_su / worker_h) if worker_h > 0.0 else None
    if total_rate is not None and float(total_rate) > cfg.deep_stall_recovery_max_total_su_per_gpu_h:
        return False
    history = list(getattr(evidence, "recent_ticks_history", None) or [])
    if not history:
        return False
    lookback = max(1, int(cfg.deep_stall_recovery_history_lookback_ticks))
    recent = history[-lookback:]
    min_deep = max(1, int(cfg.deep_stall_recovery_min_deep_ticks))
    dry_deep_ticks = sum(
        1 for item in recent
        if str(item.get("state_label", "")) == "deep_stall"
        and int(item.get("run_su_count_delta", 0) or 0) == 0
    )
    return dry_deep_ticks >= min_deep


def _early_zero_su_cross_family_active(
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
    dominant_root: str | None,
) -> bool:
    """Whether zero-SU early evidence needs orthogonal root coverage.

    This is a coverage deadline, not a method prior. It lets the LLM pivot
    earlier, but if no official SU appears after the first small worker-GPU-h
    budget, the run must have the configured number of non-dominant root-family
    probes already running/queued or launchable.
    """
    state = str(getattr(evidence, "state_label", "") or "")
    if state not in {"low_evidence", "stalled"}:
        return False
    if int(getattr(evidence, "run_su_count", 0) or 0) > 0:
        return False
    if int(getattr(evidence, "run_su_count_delta", 0) or 0) > 0:
        return False
    if not dominant_root:
        return False
    if _root_has_current_su(evidence, dominant_root, cfg):
        return False
    worker_h = float(getattr(evidence, "worker_gpu_h_total", 0.0) or 0.0)
    if worker_h < float(cfg.early_zero_su_cross_family_min_worker_gpu_h):
        return False
    completed = int(getattr(evidence, "completed_children", 0) or 0)
    if completed < int(cfg.early_zero_su_cross_family_min_completed_children):
        return False
    return True


def _needs_cross_family_escape(evidence: EvidenceSummary, cfg: BuilderConfig, dominant_root: str | None) -> bool:
    state = str(getattr(evidence, "state_label", "") or "")
    dry_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    if state == "deep_stall":
        return not _root_has_current_su(evidence, dominant_root, cfg)
    if state in {"stalled", "strict_duplicate_collapse"}:
        return dry_h >= cfg.cross_family_probe_warning_dry_gpu_h and not _root_has_current_su(evidence, dominant_root, cfg)
    return False


def _cross_family_escape_target(evidence: EvidenceSummary, cfg: BuilderConfig, dominant_root: str | None) -> int:
    if _needs_cross_family_escape(evidence, cfg, dominant_root):
        return max(0, int(cfg.deep_stall_cross_family_min_probes))
    if _early_zero_su_cross_family_active(evidence, cfg, dominant_root):
        return max(0, int(cfg.early_zero_su_cross_family_min_probes))
    if _deep_stall_recovery_active(evidence, cfg):
        return max(0, int(cfg.deep_stall_recovery_cross_family_min_probes))
    return 0


def _family_route_row(evidence: EvidenceSummary, family: str) -> Any | None:
    best = None
    best_gpu = -1.0
    for row in getattr(evidence, "route_values", None) or []:
        if _row_family(row) != family:
            continue
        scope = _row_scope(row)
        gpu = float(_cb_mh_value(row, "route_gpu_h", 0.0) or 0.0)
        if scope == "family":
            return row
        if gpu > best_gpu:
            best = row
            best_gpu = gpu
    return best


def _cross_family_escape_candidates(
    evidence: EvidenceSummary,
    cfg: BuilderConfig,
    registry: CapabilityRegistry,
    existing: list[ActionCandidate],
    *,
    parent_pdb_available: bool = True,
) -> list[ActionCandidate]:
    """Ensure zero-SU/dry ticks test non-dominant root families.

    This is a launchability safeguard, not a method prior. It fires when early
    zero-SU evidence misses orthogonal root-family coverage, when the dominant
    root stopped buying SU, or when a low-yield run has just recovered from a
    sustained deep_stall. In the recovery case it keeps a smaller probe floor so
    the route that just made SU can still be exploited.
    """
    dominant_root = _dominant_route_root_group(evidence, registry)
    target = _cross_family_escape_target(evidence, cfg, dominant_root)
    if target <= 0:
        return []

    cluster_unavail = set(cfg.unavailable_backends_override or ())
    existing_roots = {
        _root_family_group(c.method_family)
        for c in existing
        if c.feasibility.all_ok()
        and (registry.get(c.method_family) is not None)
        and _is_route_light_generator_cap(registry.get(c.method_family))
        and _root_family_group(c.method_family) != dominant_root
    }
    existing_roots.update(
        _pending_route_light_root_groups(evidence, registry, exclude_root=dominant_root)
    )
    need = target - len(existing_roots)
    if need <= 0:
        return []

    cost_order = {"low": 0, "diagnostic": 1, "standard": 2, "extended": 3}
    rows: list[tuple[tuple[float, ...], str, dict[str, object], str, float]] = []
    for fam, cap in registry.capabilities.items():
        if fam in ROOT_COVERAGE_EXCLUDED_FAMILIES:
            continue
        if fam in cluster_unavail:
            continue
        if not _is_route_light_generator_cap(cap):
            continue
        root = _root_family_group(fam)
        if not root or root == dominant_root or root in existing_roots:
            continue
        row = _family_route_row(evidence, fam)
        status = str(_cb_mh_value(row, "marginal_status", _cb_mh_value(row, "status", "untried")) or "untried")
        route_gpu_h = float(_cb_mh_value(row, "route_gpu_h", 0.0) or 0.0)
        new_su = int(_cb_mh_value(row, "new_su", 0) or 0)
        near_recent = int(_cb_mh_value(row, "near_miss_recent", 0) or 0)
        rate = _route_value_current_rate(row) if row is not None else 0.0
        diag_score = float(_cb_mh_value(row, "diagnostic_improvement_score", 0.0) or 0.0)
        diag_credible = diag_score >= float(cfg.cross_family_diagnostic_improvement_min_score)
        credible = (
            status in {"untried", "under_tested", "awaiting_score_conversion"}
            or new_su > 0
            or near_recent > 0
            or rate > 0.0
            or diag_credible
        )
        if not credible:
            continue
        status_rank = {
            # SU/value-bearing routes must outrank auxiliary diagnostic support.
            "promote": 0,
            "healthy": 1,
            "productive": 1,
            "productive_but_duplicate": 2,
            "diversify": 2,
            "delayed_productive": 3,
            "delayed_productive_duplicate": 4,
            "awaiting_score_conversion": 5,
            "untried": 6,
            "under_tested": 7,
            "observed": 8,
            "dry_low_quality": 9,
            "dry": 10,
            "defer": 11,
            "collapse_risk": 12,
        }.get(status, 13)
        params = _fallback_diversity_config(cap, str(getattr(evidence, "state_label", "") or ""))
        su_value_rank = 0 if (new_su > 0 or rate > 0.0) else 1
        score = (
            float(su_value_rank),
            float(status_rank),
            -float(rate),
            -float(new_su),
            -float(near_recent),
            float(route_gpu_h),
            -float(diag_score),
            float(cost_order.get(cap.default_cost_class, 99)),
        )
        rows.append((score, fam, params, status, diag_score))
    rows.sort(key=lambda x: (x[0], x[1]))

    out: list[ActionCandidate] = []
    used_roots = set(existing_roots)
    tick_tag = evidence.tick_id or "t000"
    for _, fam, params, status, diag_score in rows:
        root = _root_family_group(fam)
        if root in used_roots:
            continue
        cap = registry.get(fam)
        if cap is None:
            continue
        feas = feasibility_for(
            fam, evidence, cfg, registry=registry,
            parent_pdb_available=parent_pdb_available,
        )
        out.append(ActionCandidate(
            candidate_id=f"evidence_fallback_cross_family_{tick_tag}_{len(out):02d}_{fam}",
            hypothesis_ids=["cross_family_escape"],
            parent_result_id=None,
            method_family=fam,
            operator_id=cap.default_operator_id,
            lane_id=cap.default_lane_id,
            config_delta=dict(params),
            downstream_route_plan=_downstream_score_conversion_plan(fam, cap),
            estimated_cost_class=cap.default_cost_class,
            expected_signal=(
                f"cross_family_escape dominant_root={dominant_root or 'unknown'} "
                f"family={fam} prior_status={status} diagnostic_score={diag_score:.2f}"
            ),
            evidence_refs=["route_values", "dry_since_last_SU", "untried_cross_family_candidates"],
            feasibility=feas,
        ))
        used_roots.add(root)
        if len(out) >= need:
            break
    return out


def build_candidates(
    hypotheses: list[HypothesisCard],
    evidence: EvidenceSummary,
    *,
    cfg: BuilderConfig | None = None,
    registry: CapabilityRegistry | None = None,
    include_warmstart: bool = True,
    warmstart_completed_families: set[str] | None = None,
    parent_pdb_available: bool = True,
    structure_refilter_result_ids: set[str] | None = None,
    structure_refilter_scored_source_ids: set[str] | None = None,
) -> list[ActionCandidate]:
    cfg = cfg or BuilderConfig()
    registry = registry or default_registry()
    structure_refilter_result_ids = structure_refilter_result_ids or set()
    structure_refilter_scored_source_ids = structure_refilter_scored_source_ids or set()
    failed_sigs = _failed_signature_set(evidence)
    joint_fail_parent_ids = _joint_fail_only_representative_ids(evidence)
    joint_fail_parent_source_families = _joint_fail_parent_source_families(evidence)
    parent_source_families = _parent_source_family_map(evidence)
    # rec 4 (2026-05-30): parents the give-up floor flagged as stuck. A candidate
    # that would REFINE one of these (parent_result_id in the set) is marked
    # infeasible below so budget flows to fresh generation instead.
    # GAP 2 (2026-06-13): per-(parent, rescue-family) exhaustion. Map
    # parent_result_id -> set of EXHAUSTED rescue families. An EMPTY set means
    # "block ALL refinement of this parent" (regenerate; pLDDT-stuck or mined-out
    # arms); a non-empty set blocks only those families, leaving an untried rescue
    # family free to attempt the same parent before full regeneration.
    stuck_exhausted: dict[str, set[str]] = {}
    for e in (getattr(evidence, "stuck_lineage_roots", None) or []):
        if isinstance(e, dict) and e.get("root_result_id"):
            stuck_exhausted[e["root_result_id"]] = set(e.get("exhausted_families") or [])
    stuck_set = set(stuck_exhausted)  # back-compat: membership = parent is flagged
    parent_artifact_ids = {
        str(x) for x in (getattr(evidence, "parent_artifact_result_ids", None) or [])
        if str(x)
    }
    out: list[ActionCandidate] = []

    # Cold-start root coverage: prepend deterministic seeds when archive is cold.
    # These do not depend on LLM output and ensure that even a Planner fallback
    # leaves us with one Complexa, one BoltzGen, and one BindCraft root probe.
    if include_warmstart:
        out.extend(_warmstart_candidates(
            evidence, cfg, registry,
            has_llm_hypotheses=bool(hypotheses),
            parent_pdb_available=parent_pdb_available,
            completed_families=warmstart_completed_families,
        ))

    if (include_warmstart
            and evidence.state_label == "low_evidence"
            and not evidence.recipes
            and out):
        # Cost-aware cold start is deterministic before any target evidence exists.
        # Planner/Supervisor control resumes after every enabled cold-start family
        # has been selected; partial-slot ticks emit only the missing families.
        return out

    seq = 0
    for h in hypotheses:
        if h.status in ("contradicted", "retired"):
            continue
        primary_axis = (
            h.predicted_metric_changes[0].axis if h.predicted_metric_changes else "iPAE"
        )
        for fam in h.recommended_action_families:
            cap = registry.get(fam)
            if cap is None:
                continue
            op = cap.default_operator_id
            lane = cap.default_lane_id
            cost = cap.default_cost_class
            feas = feasibility_for(fam, evidence, cfg, registry=registry,
                                    parent_pdb_available=parent_pdb_available)
            if cap is not None and cap.outputs_diagnostic_only:
                if _hypothesis_requests_existing_score_backlog(h):
                    feas = _with_feasibility_reason(
                        feas,
                        f"score_backlog_is_system_managed:{fam}",
                        compiler_ok=False,
                    )
                feedback_reason = _diagnostic_first_feedback_reason(fam, evidence, cfg)
                if feedback_reason:
                    feas = _with_feasibility_reason(
                        feas,
                        feedback_reason,
                        route_cap_ok=False,
                    )

            # B-010 fix (2026-05-26): partial-accept LLM config_delta. The
            # earlier strict-all policy dropped the whole dict on any single
            # invalid key (LLM commonly mixes valid `temperature=0.05` with
            # hallucinated `steering_weight=8.0`), so every Complexa launch
            # ran with default config regardless of the LLM's tuning intent.
            # Now: keep valid keys, drop only invalid ones. Eval-budget breach
            # is repaired when possible and otherwise remains an explicit
            # infeasible candidate. Never silently launch an intended config as
            # `{}`; that was a critical hidden no-op failure mode.
            suggested = h.config_delta_suggestions.get(fam, {})
            dropped_config_reasons: list[str] = []
            if cap is not None and suggested:
                config_delta, dropped_config_reasons = validate_config_delta_partial(
                    cap, suggested
                )
            else:
                config_delta = {}
            if suggested and dropped_config_reasons:
                reason = "config_delta_adjusted:" + "|".join(dropped_config_reasons)
                if len(reason) > 400:
                    reason = reason[:397] + "..."
                hard_reject = not config_delta and any(
                    "unrepairable" in r
                    or "unknown_param" in r
                    or "non_numeric" in r
                    or "not_allowed_value" in r
                    for r in dropped_config_reasons
                )
                feas = _with_feasibility_reason(
                    feas,
                    reason,
                    cost_ok=False if hard_reject else None,
                )

            # rec 3 (2026-05-30): axis-matched config SOFT-DEFAULT. When the LLM
            # named a family to fix a bottleneck but gave NO config tuning, seed a
            # remediation knob matched to the hypothesis's dominant axis so the
            # launch targets the diagnosed component instead of running plain
            # defaults. Fires ONLY when config_delta is empty (LLM tuning always
            # wins) and only for families exposing the knob. pLDDT (structural)
            # has no fixed-backbone fix → left to the give-up/regenerate floor.
            # NB: `not suggested` => the LLM gave NO config for this family. An
            # attempted-but-invalid suggestion that validated to {} is left empty,
            # not overridden by this soft-default. Also require a REAL axis
            # diagnosis (h.predicted_metric_changes) so we never seed off the
            # fallback primary_axis="iPAE" when the hypothesis predicted nothing.
            soft_defaulted = False
            if (not config_delta and not suggested and cap is not None
                    and h.predicted_metric_changes):
                ap = cap.allowed_params or {}
                if primary_axis == "binder_scRMSD" and "refinement_algorithm" in ap:
                    seed_cfg = {"refinement_algorithm": "sequence_hallucination"}
                elif primary_axis == "iPAE" and "sc_scale_noise" in ap:
                    adaptive_noise = _adaptive_complexa_sc_scale_noise(evidence, primary_axis, cap)
                    seed_cfg = {"sc_scale_noise": adaptive_noise} if adaptive_noise is not None else {"sc_scale_noise": 0.30}
                else:
                    seed_cfg = {}
                if seed_cfg:
                    config_delta, _ = validate_config_delta_partial(cap, seed_cfg)
                    soft_defaulted = bool(config_delta)

            pre_autopair_cfg_key = json.dumps(dict(config_delta), sort_keys=True)
            pre_autopair_joint_fail = (
                (op, pre_autopair_cfg_key) in failed_sigs and not soft_defaulted
            )

            if fam.startswith("complexa_") and config_delta:
                greedy_requested = (
                    int(config_delta.get("n_greedy_iters", 0) or 0) > 0
                    or bool(config_delta.get("enable_greedy_optimization", False))
                    or "greedy_percentage" in config_delta
                )
                if (greedy_requested
                        and config_delta.get("refinement_algorithm") != "sequence_hallucination"
                        and cap is not None
                        and "refinement_algorithm" in (cap.allowed_params or {})):
                    config_delta = dict(config_delta)
                    config_delta["refinement_algorithm"] = "sequence_hallucination"
                    feas = _with_feasibility_reason(
                        feas,
                        "config_delta_adjusted:auto_set:refinement_algorithm=sequence_hallucination_for_greedy_knobs",
                    )
                elif (cap is not None
                        and "refinement_algorithm" in (cap.allowed_params or {})
                        and not any(str(k).startswith("reward_") for k in config_delta)
                        and _should_pair_complexa_sequence_hallucination(evidence, primary_axis, config_delta)):
                    config_delta = dict(config_delta)
                    config_delta["refinement_algorithm"] = "sequence_hallucination"
                    feas = _with_feasibility_reason(
                        feas,
                        "config_delta_adjusted:auto_pair:refinement_algorithm=sequence_hallucination_for_complexa_rescue",
                    )

                # T-ReX reward timing: reward weights are diagnostic levers, not
                # the first search perturbation. Sparse evidence defers reward_*
                # and launches an axis-matched material search tweak first;
                # evidence-rich reward configs remain valid, but exact
                # reward-only retries get a material search knob as well.
                if cap is not None and config_delta:
                    repaired_delta, reward_timing_reasons = _repair_complexa_reward_timing(
                        config_delta,
                        evidence=evidence,
                        primary_axis=primary_axis,
                        family=fam,
                        cap=cap,
                        cfg=cfg,
                    )
                    if reward_timing_reasons:
                        config_delta, dropped_after_reward_timing = validate_config_delta_partial(
                            cap, repaired_delta
                        )
                        reasons = reward_timing_reasons + [
                            f"post_reward_timing_drop:{r}" for r in dropped_after_reward_timing
                        ]
                        feas = _with_feasibility_reason(feas, "|".join(reasons))

            # Proteina-Complexa expands best-of-n to nsamples * replicas and
            # chunks each model forward pass by dataloader.batch_size. On the
            # production 80-GB GPUs, the default batch of 16 reproducibly OOMs
            # for replicas > 4 on large multichain targets. Keep the requested
            # total search breadth, but make the already-supported chunk size
            # explicit in config_delta so execution and provenance agree.
            if fam == "complexa_best_of_n" and config_delta:
                replicas = int(config_delta.get("replicas", 4) or 4)
                batch_size = int(config_delta.get("batch_size", 16) or 16)
                if replicas > 4 and batch_size > 8:
                    config_delta = dict(config_delta)
                    config_delta["batch_size"] = 8
                    feas = _with_feasibility_reason(
                        feas,
                        "config_delta_adjusted:memory_safe_best_of_n_batch_size="
                        f"{batch_size}->8",
                    )

            if cap is not None and cap.outputs_diagnostic_only:
                route_feedback_reason = _diagnostic_route_feedback_reason(
                    fam, config_delta, evidence
                )
                if route_feedback_reason:
                    feas = _with_feasibility_reason(
                        feas,
                        route_feedback_reason,
                        route_cap_ok=False,
                    )

            # llm-004 (2026-06-18): now that config_delta is budget-final, re-check
            # the remaining-wall gate scaled by the config's eval-budget. The base
            # feasibility_for gate is config-blind (flat FAMILY_RUNTIME_H), so a
            # heavy-config generator could pass it near end-of-run and then be
            # wall-killed. Only restricts heavy configs (mult>1); default configs
            # already passed the base gate unchanged.
            if feas.cost_ok and config_delta:
                _wall_reason = _scaled_wall_reason(fam, config_delta, evidence)
                if _wall_reason:
                    feas = _with_feasibility_reason(feas, _wall_reason, cost_ok=False)

            # Q3 feedback: if this exact (operator, config) signature appears
            # as a joint_fail recipe in the current evidence, attach a caution
            # reason but do NOT make the candidate infeasible. Stochastic search
            # can need more samples, and the Planner prompt explicitly treats
            # strategy_feedback as scoped feedback rather than a ban list.
            # C-10 fix (2026-05-30): EXEMPT the rec-3 soft-default from this dedup.
            # The dedup exists to stop re-running configs the LLM PROPOSED and that
            # failed; the soft-default is the builder's fixed fallback, so once it
            # lands in joint_fail it would hard-block EVERY future un-tuned card on
            # that operator — funnelling all un-tuned remediation into one
            # permanently dead config. The LLM's own (deduped) tuning still wins,
            # and the give-up floor (rec 4) bounds a backbone that keeps failing.
            cfg_key = json.dumps(dict(config_delta), sort_keys=True)
            final_joint_fail = (op, cfg_key) in failed_sigs and not soft_defaulted
            if final_joint_fail or pre_autopair_joint_fail:
                caution_key = cfg_key if final_joint_fail else pre_autopair_cfg_key
                feas = FeasibilityCheck(
                    backend_healthy=feas.backend_healthy,
                    runtime_bucket_id=feas.runtime_bucket_id,
                    compiler_ok=feas.compiler_ok,
                    verifier_ok=feas.verifier_ok,
                    route_cap_ok=feas.route_cap_ok,
                    cost_ok=feas.cost_ok,
                    reasons=list(feas.reasons) + [
                        f"prior_joint_fail_caution_not_ban:({op},{caution_key[:50]})"
                    ],
                )

            seq += 1
            # E-feedback fix (2026-05-26): include tick_id prefix in
            # candidate_id so a re-candidated active hypothesis in a later
            # tick doesn't collide with its first-tick candidate. Previous
            # construction was `cand_{hypothesis_id}_{seq:03d}`, and `seq`
            # resets each build_candidates call — hyp_0001_00 reaching the
            # builder in both tick 1 and tick 5 would yield two distinct
            # ActionCandidate records sharing the same candidate_id, which
            # downstream lookups (phase5_launcher, audit by id) treat as a
            # single entity.
            tick_tag = evidence.tick_id or "t000"
            _refines_parent = cap is not None and getattr(cap, "requires_parent_pdb", False)
            baseline_rid = _resolve_baseline_result_id(h, evidence)
            _baseline_refs, baseline_conflict = _shared_baseline_refs(h)
            if baseline_conflict:
                feas = _with_feasibility_reason(
                    feas,
                    "inconsistent_baseline_refs",
                    compiler_ok=False,
                )
            parent_rid: str | None = None
            parent_reason: str | None = None
            if _refines_parent:
                parent_rid, parent_reason = _resolve_parent_result_id(h, evidence)
                if baseline_rid is None:
                    baseline_rid = parent_rid
                if parent_rid is None:
                    feas = _with_feasibility_reason(
                        feas,
                        f"{parent_reason}:{fam}" if parent_reason else f"no_concrete_parent_result_id:{fam}",
                        compiler_ok=False,
                    )
                elif parent_rid not in parent_artifact_ids:
                    feas = _with_feasibility_reason(
                        feas,
                        f"no_usable_parent_artifact:{parent_rid}:{fam}",
                        compiler_ok=False,
                    )
                elif (
                    mismatch_reason := _parent_source_mismatch_reason(
                        hypothesis=h,
                        action_family=fam,
                        parent_result_id=parent_rid,
                        parent_source_families=parent_source_families,
                    )
                ):
                    feas = _with_feasibility_reason(
                        feas,
                        mismatch_reason,
                        compiler_ok=False,
                    )
                elif (
                    fam == "structure_refilter"
                    and parent_rid in structure_refilter_result_ids
                    and not _is_material_parent_model_refold(config_delta)
                ):
                    # Canonical AF2 structure_refilter is a score-conversion
                    # lane, not an iterative refinement generator. A Planner may
                    # cite a prior AF2 child as evidence that a BoltzGen/BindCraft
                    # route failed, but launching AF2 on that AF2 output again
                    # with default settings is a dead refilter-of-refilter loop.
                    # Material parent_model_refold retries are handled by the
                    # condition above and allowed through.
                    feas = _with_feasibility_reason(
                        feas,
                        f"refilter_of_refilter_blocked:{parent_rid}",
                        compiler_ok=False,
                    )
                elif (
                    fam == "structure_refilter"
                    and parent_rid in structure_refilter_scored_source_ids
                    and not _is_material_parent_model_refold(config_delta)
                ):
                    # Also block a repeat default AF2 score of the same
                    # original diagnostic artifact. The first structure_refilter
                    # child is canonical score conversion; a materially changed
                    # parent_model_refold remains a valid adaptive action.
                    feas = _with_feasibility_reason(
                        feas,
                        f"structure_refilter_source_already_scored:{parent_rid}",
                        compiler_ok=False,
                    )
                elif parent_rid in joint_fail_parent_ids:
                    source_families = joint_fail_parent_source_families.get(parent_rid, set())
                    score_conversion = False
                    if fam == "structure_refilter":
                        for source_family in source_families:
                            source_cap = registry.get(source_family)
                            if source_cap is not None and source_cap.outputs_diagnostic_only:
                                score_conversion = True
                                break
                    if not score_conversion:
                        # A joint-fail-only representative is weak evidence for
                        # parent-consuming rescue, but it is not proof that a
                        # sequence redesign, material refold, or diagnostic
                        # probe cannot help. Keep it feasible as a bounded
                        # caution; repeated non-improving children are still
                        # stopped by stuck_lineage_roots/rescue_exhausted below.
                        # Canonical score-conversion of diagnostic-only outputs
                        # is exempt from even this caution because it is required
                        # to enter the official AF2/SU path.
                        feas = FeasibilityCheck(
                            backend_healthy=feas.backend_healthy,
                            runtime_bucket_id=feas.runtime_bucket_id,
                            compiler_ok=feas.compiler_ok,
                            verifier_ok=feas.verifier_ok,
                            route_cap_ok=feas.route_cap_ok,
                            cost_ok=feas.cost_ok,
                            reasons=list(feas.reasons) + [
                                f"joint_fail_parent_caution_bounded_probe:{parent_rid}"
                            ],
                        )
            # rec 4: a candidate that would REFINE a stuck backbone is marked
            # infeasible (still emitted, so the LLM's intent stays auditable) —
            # the Selector then spends budget on fresh generation, and the
            # Planner is told to regenerate from scratch. Refilter/MPNN on a
            # repeatedly non-improving parent is the polishing loop this prevents.
            # C-2 fix (2026-05-30): ONLY block families that actually consume the
            # parent backbone (requires_parent_pdb: refilter / seq_redesign). A
            # de-novo generator (complexa_*/bindcraft, requires_parent_pdb=False)
            # that merely CITES the stuck parent as lineage/evidence is a fresh
            # regeneration — the exact action the give-up floor wants — so it must
            # NOT be blocked.
            if parent_rid is not None and parent_rid in stuck_exhausted and _refines_parent:
                # GAP 2: block only if this rescue family is exhausted on the
                # parent, OR the exhausted set is empty (= block all / regenerate).
                _exh = stuck_exhausted[parent_rid]
                if not _exh or fam in _exh:
                    feas = _with_feasibility_reason(
                        feas,
                        (f"lineage_stuck_regenerate:{parent_rid}" if not _exh
                         else f"rescue_exhausted:{parent_rid}:{fam}"),
                        route_cap_ok=False,
                    )
            out.append(
                ActionCandidate(
                    candidate_id=f"cand_{tick_tag}_{h.hypothesis_id}_{seq:03d}",
                    hypothesis_ids=[h.hypothesis_id],
                    parent_result_id=parent_rid,
                    method_family=fam,
                    operator_id=op,
                    lane_id=lane,
                    config_delta=config_delta,
                    downstream_route_plan=(
                        # Diagnostic-only families are converted to strict/SU by
                        # canonical structure_refilter; Complexa carries the same
                        # plan only as a fallback when inline af2folding metrics
                        # are absent. Credit resolves back to the generating family.
                        _downstream_score_conversion_plan(fam, cap)
                    ),
                    estimated_cost_class=cost,
                    expected_signal=(
                        f"{fam} expected to improve {primary_axis}"
                    ),
                    evidence_refs=list(h.evidence_refs),
                    feasibility=feas,
                    baseline_result_id=baseline_rid,
                    refilter_role=_structure_refilter_role(fam, config_delta),
                )
            )

    i4 = _diagnostic_i4_mcts_candidates(
        evidence, cfg, registry, out, parent_pdb_available=parent_pdb_available,
    )
    if i4:
        out.extend(i4)

    has_feasible_parent_bound_rescue = any(
        c.feasibility.all_ok()
        and c.parent_result_id is not None
        and (
            c.method_family in {"proteinmpnn_redesign"}
            or (c.method_family == "structure_refilter" and _is_material_parent_model_refold(c.config_delta))
        )
        for c in out
    )
    if not has_feasible_parent_bound_rescue:
        out.extend(_route_value_replay_candidates(
            evidence,
            cfg,
            registry,
            out,
            parent_pdb_available=parent_pdb_available,
        ))

    # Deep-stall / weak-recovery escape floor: if the LLM keeps proposing only
    # the dominant root family while it is dry, or after a low-yield run just
    # recovered from a sustained deep_stall, add distinct untried/under-tested
    # root-family probes. Recovery uses a smaller floor so the fresh-SU route is
    # still exploitable.
    cross = _cross_family_escape_candidates(
        evidence, cfg, registry, out, parent_pdb_available=parent_pdb_available,
    )
    if cross:
        seen = {
            (c.method_family, json.dumps(c.config_delta, sort_keys=True))
            for c in out
        }
        out.extend(
            c for c in cross
            if (c.method_family, json.dumps(c.config_delta, sort_keys=True)) not in seen
        )

    # Generic stalled escape floor: fallback still fires when the LLM produced
    # no feasible route-light generator at all. A same-root dry generator no
    # longer suppresses the cross-family floor above.
    def _has_escape(cands: list[ActionCandidate]) -> bool:
        return any(
            c.feasibility.all_ok()
            and (registry.get(c.method_family) is not None)
            and _is_route_light_generator_cap(registry.get(c.method_family))
            for c in cands
        )

    if evidence.state_label in ("stalled", "deep_stall", "strict_duplicate_collapse") and not _has_escape(out):
        forced = _warmstart_candidates(
            evidence, cfg, registry,
            has_llm_hypotheses=False,  # force fallback regardless of card count
            parent_pdb_available=parent_pdb_available,
        )
        seen = {
            (c.method_family, json.dumps(c.config_delta, sort_keys=True))
            for c in out
        }
        out.extend(
            c for c in forced
            if (c.method_family, json.dumps(c.config_delta, sort_keys=True)) not in seen
        )

    return out
