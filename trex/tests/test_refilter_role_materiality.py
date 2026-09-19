"""An LLM-proposed structure_refilter is tagged parent_model_refold ONLY when it
materially changes the canonical AF2 scoring config; a no-op/default re-score is
canonical (off-budget, SU credited to the upstream generator)."""
from __future__ import annotations

from trex.candidate_builder import _structure_refilter_role
from trex.refilter_roles import (
    CANONICAL_SCORE_CONVERSION,
    PARENT_MODEL_REFOLD,
)


def test_non_refilter_family_has_no_role():
    assert _structure_refilter_role("bindcraft", {"x": 1}) is None
    assert _structure_refilter_role("complexa_beam", {}) is None


def test_empty_or_default_refilter_is_canonical():
    assert _structure_refilter_role("structure_refilter", {}) == CANONICAL_SCORE_CONVERSION
    assert _structure_refilter_role("structure_refilter", None) == CANONICAL_SCORE_CONVERSION
    # config that only restates the canonical defaults is NOT material
    assert _structure_refilter_role("structure_refilter", {
        "model_names": "model_1_multimer_v3", "num_recycles": 3,
        "use_initial_guess": 1,
    }) == CANONICAL_SCORE_CONVERSION
    # num_recycles given as a float equal to the default is still canonical
    assert _structure_refilter_role("structure_refilter", {"num_recycles": 3.0}) == CANONICAL_SCORE_CONVERSION


def test_material_change_is_parent_model_refold():
    assert _structure_refilter_role("structure_refilter", {"num_recycles": 6}) == PARENT_MODEL_REFOLD
    assert _structure_refilter_role("structure_refilter", {"use_initial_guess": 0}) == PARENT_MODEL_REFOLD
    assert _structure_refilter_role("structure_refilter", {
        "model_names": "model_1_multimer_v3,model_2_multimer_v3",
    }) == PARENT_MODEL_REFOLD
