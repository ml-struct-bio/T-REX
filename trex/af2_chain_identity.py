"""Resolve AF2 prediction chains without reusing input-PDB chain labels.

ColabDesign can reorder/rename chains when it saves a prediction. Input labels
are provenance, not the identity of chains in the predicted PDB. This module
only inspects existing structures; it does not run a predictor.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from sys import intern
from typing import Any


class ChainIdentityError(ValueError):
    """The predicted binder cannot be identified without guessing."""


def _chain_residues(path: Path) -> dict[str, tuple[str, ...]]:
    """Ordered CA residue names, with alternate locations/models deduplicated."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ChainIdentityError(f"Cannot read chain identity from {path}") from exc
    return _read_chain_residues(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=16384)
def _read_chain_residues(path_text: str, size: int, mtime_ns: int) -> dict[str, tuple[str, ...]]:
    # Identity is queried by structural/sequence clustering and both caches.
    # Stat-keyed reuse avoids rereading unchanged structures on each query.
    path = Path(path_text)
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ChainIdentityError(f"Cannot read chain identity from {path}") from exc
    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or len(line) < 27 or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip()
        key = (chain, line[22:27])
        if chain and key not in seen:
            chains.setdefault(chain, []).append(intern(line[17:20].strip().upper()))
            seen.add(key)
    if not chains:
        raise ChainIdentityError(f"No protein CA chains in {path}")
    return {chain: tuple(residues) for chain, residues in chains.items()}


def _sequence_hash(residues: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(residues).encode()).hexdigest()


def validate_prediction_chains(mapping: Any, predicted_pdb: Path) -> dict[str, Any]:
    """Validate a saved identity map against the predicted structure itself."""
    if not isinstance(mapping, dict):
        raise ChainIdentityError("Missing prediction chain map")
    binder = mapping.get("binder_chain")
    targets = mapping.get("target_chains")
    hashes = mapping.get("sequence_sha256")
    if (
        not isinstance(binder, str) or not binder
        or not isinstance(targets, list) or not targets
        or not all(isinstance(chain, str) and chain for chain in targets)
        or binder in targets or len(set(targets)) != len(targets)
        or not isinstance(hashes, dict)
    ):
        raise ChainIdentityError("Invalid prediction chain map")
    sequences = _chain_residues(predicted_pdb)
    if set(sequences) != {binder, *targets} or hashes != {
        chain: _sequence_hash(sequence) for chain, sequence in sequences.items()
    }:
        raise ChainIdentityError("Prediction chains differ from the verified identity map")
    return mapping


def resolve_prediction_chains(
    report: dict[str, Any],
    predicted_pdb: Path,
    *,
    report_dir: Path,
    input_binder_chain: str = "",
    input_target_chains: str = "",
) -> dict[str, Any]:
    """Use a verified map, or match legacy input/output protein sequences.

    Identical target subunits are supported as a multiset. The binder must have
    exactly one sequence match; an ambiguous or incomplete match is rejected.
    Relative input paths are interpreted relative to the report directory.
    """
    if "prediction_chains" in report:
        return validate_prediction_chains(report["prediction_chains"], predicted_pdb)
    source = str(report.get("input_pdb") or "")
    if not source:
        raise ChainIdentityError("AF2 report has no input structure")
    input_path = Path(source)
    if not input_path.is_absolute():
        input_path = report_dir / input_path
    original = _chain_residues(input_path)
    predicted = _chain_residues(predicted_pdb)
    binder = str(report.get("binder_chain") or input_binder_chain).strip()
    targets = [chain.strip() for chain in str(
        report.get("target_chain") or input_target_chains
    ).split(",") if chain.strip()]
    if binder not in original or not targets or binder in targets or len(set(targets)) != len(targets):
        raise ChainIdentityError("Missing or conflicting input chain identities")
    if any(chain not in original for chain in targets):
        raise ChainIdentityError("An input target chain is absent")
    matches = [chain for chain, sequence in predicted.items() if sequence == original[binder]]
    if len(matches) != 1:
        raise ChainIdentityError("Input binder has no unique prediction sequence match")
    predicted_binder = matches[0]
    predicted_targets = [chain for chain in predicted if chain != predicted_binder]
    if Counter(predicted[chain] for chain in predicted_targets) != Counter(original[chain] for chain in targets):
        raise ChainIdentityError("Input and predicted target chain sequences differ")
    return {
        "binder_chain": predicted_binder,
        "target_chains": predicted_targets,
        "sequence_sha256": {
            chain: _sequence_hash(sequence) for chain, sequence in predicted.items()
        },
    }
