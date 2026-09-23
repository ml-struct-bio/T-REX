"""Deterministic advisory checks for deadlines and repeated failed settings.

Emitted flags do not override selection. Target-prior judgments are reported separately,
and cross-target bias is not assessed within a single campaign. Accepts schema objects
and replayed dictionaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


# Expected family runtimes used by the deadline check.
FAMILY_RUNTIME_H: dict[str, float] = {
    "bindcraft": 9000 / 3600,
    "complexa_beam": 1800 / 3600,
    "complexa_best_of_n": 1800 / 3600,
    "complexa_fk_steering": 1800 / 3600,
    "complexa_mcts": 3600 / 3600,
    "structure_refilter": 900 / 3600,
    "proteinmpnn_redesign": 300 / 3600,
    "boltzgen": 7200 / 3600,
}


@dataclass(frozen=True)
class CriticGuardConfig:
    recent_window: int = 3          # ticks; matches LLM critic "last 3 ticks"
    min_shared_keys: int = 2        # (c): require >= N shared config keys to call a match
    min_fail_descendants: int = 2   # (c): ignore single-sample (noisy) joint_fails — a
                                    # lone bad draw must never veto a legitimate retry
    family_runtime_h: dict[str, float] = field(default_factory=lambda: dict(FAMILY_RUNTIME_H))


@dataclass(frozen=True)
class CriticGuardOutput:
    flags: list[str]                # emitted flags (categories b, c)
    judgment_flags: list[str]       # category (a), computed-but-not-emitted (reporting only)
    proposed_families: list[str]


def _g(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# Ignore shared workload settings when comparing scientific configuration changes.
# A different reward, refinement, hallucination, or exploration setting remains
# a distinct proposal even when beam width and sample counts match.
SCAFFOLDING_CONFIG_KEYS: frozenset[str] = frozenset({
    "beam_width", "n_branch", "nsamples", "nsteps", "num_recycles", "n_recycle",
    "n_simulations", "mcts_k", "max_per_method",
})


def _config_overlap(a: dict | None, b: dict | None, min_shared: int) -> bool:
    """True if two config_deltas agree on >= min_shared shared keys (all shared
    keys equal) AND the agreement is ANCHORED on >= 1 LEVERED (non-scaffolding)
    key. The levered anchor stops a false 'repeat' when two configs share only
    compute-budget scaffolding (beam_width/nsteps/...) but differ on the
    scientific lever — which lives on a NON-shared key, so the old shared-only
    test couldn't see the difference (observed live on CD45 9630421: a
    reward_max_ipsae change flagged as a repeat of a reward_avg_ipsae joint_fail).
    Conservative: a single differing shared value disqualifies."""
    if not a or not b:
        return False
    shared = set(a) & set(b)
    if len(shared) < min_shared:
        return False
    if not all(a[k] == b[k] for k in shared):
        return False
    return bool(shared - SCAFFOLDING_CONFIG_KEYS)  # >= 1 shared levered key


def deterministic_critic(
    evidence: Any,
    planner_cards: list[Any] | None,
    *,
    current_tick: int,
    cfg: CriticGuardConfig | None = None,
) -> CriticGuardOutput:
    """Deterministic flag-only audit. Returns emitted flags (b, c) plus
    computed-only judgment flags (a) for reporting."""
    cfg = cfg or CriticGuardConfig()
    cards = planner_cards or []

    proposed_families: list[str] = []
    for c in cards:
        for fam in (_g(c, "recommended_action_families") or []):
            if fam not in proposed_families:
                proposed_families.append(fam)

    recipes = _g(evidence, "recipes") or []
    remaining_wall_h = _g(evidence, "remaining_wall_h")

    def _recent(r) -> bool:
        rt = _g(r, "recency_tick")
        return rt is not None and (current_tick - rt) <= cfg.recent_window

    flags: list[str] = []

    # (b) Deadline violation — proposed family cannot finish in remaining budget.
    if isinstance(remaining_wall_h, (int, float)):
        for fam in proposed_families:
            rt = cfg.family_runtime_h.get(fam)
            if rt is not None and remaining_wall_h < rt:
                flags.append(
                    f"(b) {fam} expected ~{rt:.1f}h runtime but only "
                    f"{remaining_wall_h:.1f}h remain"
                )

    # (c) Ignored recent failure — proposed config matches a known decisive
    # joint_fail. No tight recency window: a joint_fail recipe is only present
    # in `evidence.recipes` while the evidence reducer still considers it
    # relevant, so its mere presence bounds recency. Imposing a 3-tick window
    # here misses byte-identical re-proposals that recur many ticks later
    # (observed: tick-94 joint_fail re-proposed at tick 103, 110 failed
    # samples). We require it to be a decisive multi-sample failure instead.
    #
    # CRITICAL (false-positive guard): the evidence reducer splits ONE config's
    # descendants by outcome into separate recipe entries that SHARE a
    # recipe_hash. A stochastic config can therefore appear as joint_fail AND
    # strict_success at the same tick (observed: hash 1bbe7d… = 2 strict + 18
    # fail). Re-proposing a config that ALSO yields strict successes is correct,
    # not a mistake. So we only treat a config as a "pure loser" if its
    # recipe_hash never appears as strict_success in the current evidence.
    productive_hashes = {
        _g(r, "recipe_hash") for r in recipes
        if _g(r, "recipe_class") == "strict_success"
    }
    productive_hashes.discard(None)
    recent_joint_fail = [
        r for r in recipes
        if _g(r, "recipe_class") == "joint_fail"
        and (_g(r, "descendant_count") or 0) >= cfg.min_fail_descendants
        and _g(r, "recipe_hash") not in productive_hashes
    ]
    for c in cards:
        deltas = _g(c, "config_delta_suggestions") or {}
        for fam, delta in deltas.items():
            for r in recent_joint_fail:
                if _g(r, "method_family") == fam and _config_overlap(
                    delta, _g(r, "config_delta"), cfg.min_shared_keys
                ):
                    flags.append(
                        f"(c) {fam} config matches a joint_fail recipe "
                        f"(recipe_hash={_g(r, 'recipe_hash')}, "
                        f"tick={_g(r, 'recency_tick')})"
                    )
                    break

    # (a) Target-prior contradicted — COMPUTED ONLY, not emitted. A recent
    # strict success exists from a family the planner did NOT propose, AND the
    # planner proposed nothing that has produced a recent strict success. This
    # over-fires on legitimate exploration, so we report it separately rather
    # than acting on it.
    judgment: list[str] = []
    recent_strict_fams = {
        _g(r, "method_family") for r in recipes
        if _g(r, "recipe_class") == "strict_success" and _recent(r)
    }
    recent_strict_fams.discard(None)
    if proposed_families and recent_strict_fams:
        if not (set(proposed_families) & recent_strict_fams):
            judgment.append(
                f"(a) planner proposes {sorted(proposed_families)} but recent "
                f"strict successes came only from {sorted(recent_strict_fams)}"
            )

    # de-dup emitted flags, preserve order
    seen: set[str] = set()
    deduped = [f for f in flags if not (f in seen or seen.add(f))]
    return CriticGuardOutput(
        flags=deduped,
        judgment_flags=judgment,
        proposed_families=proposed_families,
    )


_GUARD_PROMPT_HASH = hashlib.sha256(b"deterministic_critic_guard_v1").hexdigest()[:16]


def critic_guard_record(*, tick_id: str, model: str, schema_version: str, out: CriticGuardOutput):
    """Build an LLMCallRecord (role='critic') for a deterministic guard call,
    so the guard is a drop-in for the LLM critic in the audit trail. No tokens,
    ~0 latency. Imported lazily to avoid a hard schema dependency at module load."""
    from .schemas import LLMCallRecord

    return LLMCallRecord(
        call_id=f"critic_{tick_id}",
        tick_id=tick_id,
        role="critic",
        model="deterministic_guard",
        model_digest=None,
        prompt_hash=_GUARD_PROMPT_HASH,
        schema_version=schema_version,
        latency_s=0.0,
        tokens_in=0,
        tokens_out=0,
        parse_status="ok",  # type: ignore[arg-type]
        confidence=None,
        abstain=False,
        fallback_triggered=False,
        critic_flags=list(out.flags),
    )
