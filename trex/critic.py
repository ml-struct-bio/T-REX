"""T-ReX Critic LLM (§22.3) — C2_REFINED prompt, flag-only audit.

Calls vLLM with the C2_REFINED system prompt + Planner output + Evidence
summary; returns a list of critic flags. The flags are recorded in
`LLMCallRecord.critic_flags` and never override Selector decisions —
they exist only for audit and downstream analysis.

Verified by smoke 8748093 (F1=1.0 on 8 scenarios) + smoke 8764027
(F1=1.000±0.000 across 5 seeds × 8 scenarios = paper-grade robust).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from trex.llm import create_client

from . import SCHEMA_VERSION
from .provenance import model_digest_from_env
from .schemas import EvidenceSummary, LLMCallRecord, to_jsonable


# C2_REFINED prompt — verbatim from smoke 8748093 (paper-grade).
CRITIC_SYSTEM = (
    "You are a critic reviewing a Planner's recommendation. Flag ONLY if "
    "you can identify SPECIFIC, EVIDENCE-CITED issues. Do not invent target "
    "characteristics you don't have evidence for.\n\n"
    "Categories (flag ONLY with specific evidence):\n"
    "  (a) Target-prior contradicted by evidence: ONLY flag if "
    "evidence.recipes contains strict_success entries from a family "
    "DIFFERENT from the Planner's choice AND those successes are RECENT "
    "(last 3 ticks). Do NOT speculate about whether the target 'should' "
    "prefer a family you cannot verify from evidence.\n"
    "  (b) Deadline violation: cite remaining_wall_h vs expected method "
    "runtime.\n"
    "  (c) Ignored recent failure: cite the matching joint_fail recipe.\n"
    "  (d) Default-bias: ONLY flag if the SAME family was proposed in "
    "the last 3 ticks across 3+ different target_classes with no "
    "rationale shift.\n\n"
    "If you cannot identify a SPECIFIC issue cited from the evidence "
    "above, respond 'no_flags'. Novel targets, exploratory probes, and "
    "reasoned first-attempts are NOT issues. When in doubt, do not flag."
)


@dataclass(frozen=True)
class CriticCallConfig:
    model: str = "vllm/Qwen/Qwen3.6-27B-FP8"
    base_url: str = "http://127.0.0.1:8500/v1"
    max_tokens: int = 256
    temperature: float = 0.3
    enable_thinking: bool = False
    timeout_s: float = 30.0
    # E-feedback fix (2026-05-26): default disabled (was True). Stand-alone
    # tests / one-shot CLI runs previously triggered a real LLM client by
    # default, which broke offline test env (openai import). Production
    # controllers pass `enabled=bool(args.enable_critic)` explicitly so
    # this default only affects ad-hoc instantiation.
    enabled: bool = False


@dataclass(frozen=True)
class CriticOutput:
    flags: list[str]
    raw_text: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    error: str | None


def build_critic_payload(
    evidence: EvidenceSummary,
    planner_output: Any,  # PlannerOutput; avoid circular import
) -> dict:
    """Compact review payload the Supervisor/critic sees.

    v7_3: diagnostics are surfaced with the SAME pass/near/fail granularity as the
    strict axis_stats, PLUS the two-tier `below_accept`/`quality_threshold`, the
    alt-model advisory medians, and the per-design `diagnostic_blocking_axis` of
    the exemplars — so the Supervisor can see every intermediate metric the Planner
    saw (previously it got only {pass, median} for diagnostics and nothing on the
    alt-model block or per-design diagnostic blockers). Strict success/SU are
    untouched; this only widens what the critic can review.
    """
    alt = getattr(evidence, "diagnostic_alt_model_scores", None) or {}
    return {
        "target_id": evidence.target_id,
        "target_class": getattr(evidence, "target_class", None),
        "remaining_wall_h": evidence.remaining_wall_h,
        "state_label": evidence.state_label,
        "axis_stats": {k: {"pass": v.pass_count, "near_pass": v.near_pass_count,
                             "fail": v.fail_count, "median": v.median_raw}
                         for k, v in evidence.axis_stats.items()},
        "diagnostic_axis_stats": {k: {
            "pass": v.pass_count, "near_pass": v.near_pass_count,
            "fail": v.fail_count, "below_accept": v.below_accept_count,
            "median": v.median_raw, "quality_threshold": v.quality_threshold,
            "pass_threshold": v.pass_threshold,
        } for k, v in evidence.diagnostic_axis_stats.items()},
        "diagnostic_alt_model_scores": {
            fam: {sk: sv.get("median") for sk, sv in scores.items()}
            for fam, scores in alt.items()
        },
        "exemplar_diagnostic_blockers": [
            {"result_id": e.result_id, "kind": e.kind, "family": e.family,
             "diagnostic_blocking_axis": e.diagnostic_blocking_axis}
            for e in (getattr(evidence, "exemplars", None) or [])
            if e.diagnostic_blocking_axis
        ][:6],
        "recipes": [{
            "family": r.method_family, "class": r.recipe_class,
            "config_delta": r.config_delta, "recency_tick": r.recency_tick,
            "descendant_count": r.descendant_count,
        } for r in evidence.recipes[:8]],
        "planner_proposal": {
            "valid": planner_output.valid,
            "abstain": planner_output.abstain,
            "rationale": (planner_output.rationale or "")[:500],
            "cards": [{
                "claim": (c.claim or "")[:300],
                "mode_affinity": c.mode_affinity,
                "recommended_action_families": c.recommended_action_families,
            } for c in (planner_output.cards or [])[:4]],
        },
    }


def build_critic_user_prompt(
    evidence: EvidenceSummary,
    planner_output: Any,
) -> str:
    """Build the exact user message sent to the publication critic."""

    payload = build_critic_payload(evidence, planner_output)
    return (
        "Review the Planner's recommendation against the evidence. "
        "Flag specific, evidence-cited issues per the categories above, or "
        "'no_flags'.\n\n"
        + json.dumps(payload, default=str, indent=2)
    )


def call_critic(
    evidence: EvidenceSummary,
    planner_output: Any,  # PlannerOutput; avoid circular import
    cfg: CriticCallConfig,
) -> CriticOutput:
    """Call the critic. Returns parsed flags (empty if 'no_flags' or disabled)."""
    if not cfg.enabled:
        return CriticOutput(flags=[], raw_text="", latency_s=0.0,
                             tokens_in=0, tokens_out=0, error=None)

    user_text = build_critic_user_prompt(evidence, planner_output)

    # Bug A fix (2026-05-30): thread the per-call timeout (see planner.py) so a
    # hung vLLM critic request cannot block the synchronous controller reap loop.
    client = create_client(cfg.model, base_url=cfg.base_url,
                           enable_thinking=cfg.enable_thinking,
                           timeout=cfg.timeout_s, max_retries=1)
    t0 = time.time()
    try:
        resp = client.chat(
            [{"role": "user", "content": user_text}],
            system=CRITIC_SYSTEM, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
        text = (resp.text or "").strip()
        flags = _parse_flags(text)
        return CriticOutput(
            flags=flags, raw_text=text,
            latency_s=round(time.time() - t0, 2),
            tokens_in=(resp.usage or {}).get("input_tokens", 0),
            tokens_out=(resp.usage or {}).get("output_tokens", 0),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return CriticOutput(
            flags=[], raw_text="", latency_s=round(time.time() - t0, 2),
            tokens_in=0, tokens_out=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _parse_flags(text: str) -> list[str]:
    """Parse 'no_flags' or '(a) ...; (b) ...' into a list of flag strings.

    §22.8.5: strict-marker policy. Plan §22.3 promised critic flags are
    advisory; the older fallback (`flags = [text[:400]]` when no
    `(a)..(d)` markers were found) turned any Critic prose — including
    hallucinated "Looking at the evidence..." preambles — into a flag
    that the next-tick Planner prompt was instructed to address. That
    silently converted the audit-only critic into a soft veto on
    Planner exploration.

    Now: require `(a)`..`(d)` markers strictly. On parse failure, log
    the raw response and return `[]` (treat as `no_flags`).
    """
    if not text:
        return []
    if "no_flags" in text.lower()[:50]:
        return []
    flags: list[str] = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines:
        if ln.startswith(("(a)", "(b)", "(c)", "(d)")):
            flags.append(ln[:400])
    if not flags:
        # Strict mode: no recognized markers → treat as no_flags. Log
        # the raw response (truncated) so audit can spot Critic drift.
        print(
            f"  [critic._parse_flags] no (a)..(d) markers — "
            f"treating as no_flags. raw={text[:200]!r}",
            flush=True,
        )
    return flags


def critic_llm_record(
    *,
    tick_id: str,
    cfg: CriticCallConfig,
    out: CriticOutput,
) -> LLMCallRecord:
    """Build an LLMCallRecord for this critic call (audit trail)."""
    prompt_hash = hashlib.sha256(CRITIC_SYSTEM.encode()).hexdigest()[:16]
    parse_status = "timeout" if out.error else "ok"
    return LLMCallRecord(
        call_id=f"critic_{tick_id}",
        tick_id=tick_id,
        role="critic",
        model=cfg.model,
        model_digest=model_digest_from_env(),
        prompt_hash=prompt_hash,
        schema_version=SCHEMA_VERSION,
        latency_s=out.latency_s,
        tokens_in=out.tokens_in,
        tokens_out=out.tokens_out,
        parse_status=parse_status,  # type: ignore[arg-type]
        confidence=None,
        abstain=False,
        fallback_triggered=False,
        critic_flags=list(out.flags),
    )
