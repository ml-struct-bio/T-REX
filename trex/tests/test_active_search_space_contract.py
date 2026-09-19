from trex.capability_registry import default_registry


ACTIVE_FIELD_COUNTS = {
    "complexa_beam": 19,
    "complexa_best_of_n": 18,
    "complexa_fk_steering": 20,
    "complexa_mcts": 20,
    "bindcraft": 12,
    "boltzgen": 6,
    "proteinmpnn_redesign": 5,
    "structure_refilter": 3,
}

def test_publication_search_space_contract() -> None:
    registry = default_registry()

    observed_active = {
        family: len(registry.capabilities[family].allowed_params)
        for family in ACTIVE_FIELD_COUNTS
    }
    assert observed_active == ACTIVE_FIELD_COUNTS
    assert sum(observed_active.values()) == 103
    assert all(
        registry.capabilities[family].availability != "legacy_archive_only"
        for family in ACTIVE_FIELD_COUNTS
    )
