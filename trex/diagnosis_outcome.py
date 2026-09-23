"""Summarize measurement changes along parent-child design lineages.

For each axis blocking a parent design, record whether its descendants improve
that measurement and qualify. The evidence reducer places these advisory
summaries in EvidenceSummary.diagnosis_outcomes for subsequent LLM input.
The calculations do not change qualification or SU credit.
"""
from __future__ import annotations

from .evidence_reducer import (
    DIAGNOSTIC_AXIS_REMEDIATION,
    DIAGNOSTIC_AXIS_THRESHOLDS,
    _diagnostic_axis_value,
    resolve_generating_record,
    worst_actionable_diagnostic_axis,
    _su_key,
)
from .schemas import ActionCandidate, HypothesisCard, ResultRecord
from .success_criteria import NEAR_PASS_MARGINS, STRICT_SUCCESS, is_strict_success


DIAGNOSIS_OUTCOME_MIN_N = 3
DIAGNOSIS_IMPROVEMENT_MARGIN_FRACTION = 0.25


STRICT_AXIS_REMEDIATION: dict[str, str] = {
    "pLDDT": "bindcraft.weights_plddt↑ OR complexa_beam.reward_plddt_weight↑ OR complexa_best_of_n.reward_plddt_weight↑ OR complexa_fk_steering.reward_plddt_weight↑ OR complexa_mcts.reward_plddt_weight↑ OR complexa_beam.refinement_algorithm=sequence_hallucination OR complexa_best_of_n.refinement_algorithm=sequence_hallucination OR complexa_fk_steering.refinement_algorithm=sequence_hallucination OR complexa_mcts.refinement_algorithm=sequence_hallucination",
    "iPAE": "bindcraft.weights_pae_inter↑ OR complexa_beam.reward_i_pae_weight↓ OR complexa_beam.reward_min_ipae_weight↓ OR complexa_best_of_n.reward_i_pae_weight↓ OR complexa_best_of_n.reward_min_ipae_weight↓ OR complexa_fk_steering.reward_i_pae_weight↓ OR complexa_fk_steering.reward_min_ipae_weight↓ OR complexa_fk_steering.temperature↓ OR complexa_mcts.reward_i_pae_weight↓ OR complexa_mcts.reward_min_ipae_weight↓",
    "binder_scRMSD": "complexa_beam.sc_scale_noise↑ OR complexa_beam.refinement_algorithm=sequence_hallucination OR complexa_best_of_n.sc_scale_noise↑ OR complexa_best_of_n.refinement_algorithm=sequence_hallucination OR complexa_fk_steering.sc_scale_noise↑ OR complexa_fk_steering.refinement_algorithm=sequence_hallucination OR complexa_mcts.sc_scale_noise↑ OR complexa_mcts.refinement_algorithm=sequence_hallucination OR proteinmpnn_redesign.sampling_temp",
}


def _axis_params(axis: str) -> tuple[float, float, str, float]:
    if axis in DIAGNOSTIC_AXIS_THRESHOLDS:
        return DIAGNOSTIC_AXIS_THRESHOLDS[axis]
    if axis == "pLDDT":
        thr, direction = STRICT_SUCCESS[axis]
        return thr, thr, direction, NEAR_PASS_MARGINS[axis]
    if axis == "iPAE":
        thr, direction = STRICT_SUCCESS[axis]
        return thr, thr, direction, NEAR_PASS_MARGINS[axis]
    if axis == "binder_scRMSD":
        thr, direction = STRICT_SUCCESS[axis]
        return thr, thr, direction, NEAR_PASS_MARGINS[axis]
    raise KeyError(axis)


def _axis_value(record: ResultRecord, axis: str) -> float | None:
    if axis in STRICT_AXIS_REMEDIATION:
        v = (record.metrics or {}).get(axis)
        return float(v) if v is not None else None
    return _diagnostic_axis_value(record, axis)

def _improved(axis: str, parent_v: float, child_v: float) -> bool:
    _pass, quality, direction, margin = _axis_params(axis)
    signed = child_v - parent_v if direction == "increase" else parent_v - child_v
    if signed <= 0:
        return False
    min_delta = max(0.0, float(margin) * DIAGNOSIS_IMPROVEMENT_MARGIN_FRACTION)
    if signed >= min_delta:
        return True
    # Tiny moves are still meaningful if they cross the axis quality band.
    if direction == "increase":
        return parent_v < quality <= child_v
    return parent_v > quality >= child_v


def _strict_axis_blocked(parent: ResultRecord, axis: str) -> bool:
    pv = _axis_value(parent, axis)
    if pv is None:
        return False
    threshold, _quality, direction, _margin = _axis_params(axis)
    if axis == "binder_scRMSD":
        return pv >= threshold
    return pv < threshold if direction == "increase" else pv > threshold


def compute_diagnosis_outcomes(
    all_results: list[ResultRecord],
    *,
    spawning_actions: dict[str, ActionCandidate] | None = None,
    hypotheses: list[HypothesisCard] | None = None,
) -> dict[str, dict[str, object]]:
    """Per blocked-axis remediation outcome over the parent→child lineage.

    For each scored child with a baseline that was blocked on an ACTIONABLE
    diagnostic axis (worst_actionable_diagnostic_axis), and where both baseline
    and child/outcome carry a value on that axis: count it as a remediation
    attempt, and record whether the attempt improved the axis and whether it
    became strict.

    When ActionCandidate lineage is available, diagnostic generators are judged
    against their candidate baseline_result_id / parent_result_id, while their
    downstream auto-chain ``structure_refilter`` records contribute canonical
    strict/SU outcome credit. The auto-chain scorer is therefore not treated as a
    second LLM remediation action.

    Returns: { axis: { lever, attempts, improved, improve_rate, strict, strict_rate } }.
    """
    by_id = {r.result_id: r for r in all_results}
    spawning_actions = spawning_actions or {}
    hyp_by_id = {h.hypothesis_id: h for h in (hypotheses or [])}

    # Canonical AF2 score-conversion descendants keyed by the generator record
    # they score. These records can carry strict success even when the upstream
    # BindCraft/BoltzGen/MPNN record is diagnostic-only.
    downstream_by_generator: dict[str, list[ResultRecord]] = {}
    for r in all_results:
        ac = spawning_actions.get(r.result_id)
        is_auto_chain = (
            ac is not None
            and (
                ac.candidate_id.startswith("chain_")
                or str(ac.expected_signal or "").startswith("auto_chain:")
            )
        )
        if not is_auto_chain or r.backend_family != "structure_refilter":
            continue
        gen = resolve_generating_record(
            r, by_result_id=by_id, spawning_actions=spawning_actions)
        if gen.result_id != r.result_id:
            downstream_by_generator.setdefault(gen.result_id, []).append(r)

    agg: dict[str, dict[str, object]] = {}
    seen_attempts: set[tuple[str, str, str]] = set()

    def _baseline_records(child: ResultRecord) -> list[ResultRecord]:
        """Candidate baseline first, then legacy direct ResultRecord parents."""
        out: list[ResultRecord] = []
        ac = spawning_actions.get(child.result_id)
        if ac is not None:
            for pid in (ac.baseline_result_id, ac.parent_result_id, *(ac.evidence_refs or [])):
                if pid and pid in by_id and by_id[pid] not in out:
                    out.append(by_id[pid])
        for pid in (child.parent_ids or []):
            if pid in by_id and by_id[pid] not in out:
                out.append(by_id[pid])
        return out

    def _best_axis_value(
        axis: str, parent_v: float, outcome_records: list[ResultRecord]
    ) -> tuple[float | None, bool]:
        values = [
            v for r in outcome_records
            if (v := _axis_value(r, axis)) is not None
        ]
        if not values:
            return None, False
        improved = [v for v in values if _improved(axis, parent_v, v)]
        if improved:
            direction = _axis_params(axis)[2]
            return (
                max(improved) if direction == "increase" else min(improved),
                True,
            )
        return values[0], False

    def _attempt_axes(child_ac: ActionCandidate | None, parent: ResultRecord) -> list[str]:
        axes: list[str] = []
        blocked = worst_actionable_diagnostic_axis(parent)
        if blocked is not None:
            axes.append(blocked)
        if child_ac is not None:
            for hid in child_ac.hypothesis_ids or []:
                h = hyp_by_id.get(hid)
                if h is None:
                    continue
                for pc in h.predicted_metric_changes or []:
                    axis = str(pc.axis)
                    if axis in STRICT_AXIS_REMEDIATION and _strict_axis_blocked(parent, axis):
                        axes.append(axis)
        return list(dict.fromkeys(axes))

    for child in all_results:
        child_ac = spawning_actions.get(child.result_id)
        if (
            child_ac is not None
            and (
                child_ac.candidate_id.startswith("chain_")
                or str(child_ac.expected_signal or "").startswith("auto_chain:")
            )
        ):
            # Score-conversion records are outcomes of the upstream generator
            # attempt, not independent Planner/Supervisor remediation attempts.
            continue
        outcome_records = [child] + downstream_by_generator.get(child.result_id, [])
        for parent in _baseline_records(child):
            for axis in _attempt_axes(child_ac, parent):
                pv = _axis_value(parent, axis)
                if pv is None:
                    continue
                cv, improved = _best_axis_value(axis, pv, outcome_records)
                if cv is None:
                    continue
                key = (child.result_id, parent.result_id, axis)
                if key in seen_attempts:
                    continue
                seen_attempts.add(key)
                lever = (
                    STRICT_AXIS_REMEDIATION.get(axis)
                    if axis in STRICT_AXIS_REMEDIATION
                    else DIAGNOSTIC_AXIS_REMEDIATION.get(axis)
                )
                a = agg.setdefault(axis, {
                    "lever": lever,
                    "attempts": 0, "improved": 0, "strict": 0,
                    "strict_su_keys": set(),
                })
                a["attempts"] = int(a["attempts"]) + 1            # type: ignore[arg-type]
                if improved:
                    a["improved"] = int(a["improved"]) + 1        # type: ignore[arg-type]
                strict_records = [r for r in outcome_records if is_strict_success(r.metrics)]
                if strict_records:
                    a["strict"] = int(a["strict"]) + 1            # type: ignore[arg-type]
                    su_keys = a.setdefault("strict_su_keys", set())
                    if isinstance(su_keys, set):
                        for r in strict_records:
                            su_keys.add(_su_key(r))
    for a in agg.values():
        n = int(a["attempts"])                                # type: ignore[arg-type]
        su_keys = a.pop("strict_su_keys", set())
        unique_su = len(su_keys) if isinstance(su_keys, set) else 0
        a["unique_su"] = unique_su
        a["improve_rate"] = round(int(a["improved"]) / n, 3) if n else 0.0  # type: ignore[arg-type]
        a["strict_rate"] = round(int(a["strict"]) / n, 3) if n else 0.0     # type: ignore[arg-type]
        a["unique_su_rate"] = round(unique_su / n, 3) if n else 0.0
        a["min_n"] = DIAGNOSIS_OUTCOME_MIN_N
        a["provisional"] = n < DIAGNOSIS_OUTCOME_MIN_N
        a["prefer_key"] = "unique_su_rate"
    return agg


def format_diagnosis_outcomes_tldr(outcomes: dict[str, dict[str, object]]) -> str:
    """One-line digest for the prompt TL;DR.

    The parenthetical count is raw strict-success attempts, not Foldseek-deduped
    SU, so label it as strict to avoid inflating displayed unique credit.
    """
    if not outcomes:
        return ""
    parts = []
    for axis, a in sorted(
        outcomes.items(),
        key=lambda kv: (
            -float(kv[1].get("unique_su_rate", 0.0)),
            -int(kv[1].get("unique_su", 0)),
            -float(kv[1].get("strict_rate", 0.0)),
            -int(kv[1].get("attempts", 0)),
        ),
    ):
        lever = a.get("lever") or "?"
        parts.append(
            f"{axis}→{lever}: unique_su_rate={a.get('unique_su_rate', 0.0)}; "
            f"strict_rate={a.get('strict_rate', 0.0)}; improved {a['improved']}/{a['attempts']} "
            f"({a.get('unique_su', 0)} SU, {a['strict']} strict)"
        )
    return "remediation outcomes: " + " | ".join(parts[:4])
