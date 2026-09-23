"""Verified chain roles in output structures, independent of input chain labels.

Residue sequences, rather than chain names or lengths, identify the target.
Maps stored with new records remain verifiable without the original input file.
Legacy archive recovery creates in-memory records and never rewrites an archive.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from sys import intern
from typing import Any, Iterable

from .af2_chain_identity import ChainIdentityError, _chain_residues, _sequence_hash
from .schemas import ResultRecord


MAP_KEY = "output_chain_map"
STATUS_KEY = "output_chain_identity"


def chain_sequences(path: Path) -> dict[str, tuple[str, ...]]:
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        return _chain_residues(path)
    try:
        stat = path.stat()
        return _cif_sequences(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    except Exception as exc:
        raise ChainIdentityError(f"Cannot read chain identity from {path}") from exc


@lru_cache(maxsize=16384)
def _cif_sequences(path: str, size: int, modified: int) -> dict[str, tuple[str, ...]]:
    from Bio.PDB import FastMMCIFParser, MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionException

    try:
        model = next(FastMMCIFParser(QUIET=True).get_structure("output", path).get_models())
    except (KeyError, ValueError, IndexError, PDBConstructionException):
        # Valid mmCIF may contain blank lines or omit optional # separators.
        # BioPython's fast reader assumes stricter row formatting. The full
        # parser handles those files; the stat-keyed cache avoids repeat work.
        model = next(MMCIFParser(QUIET=True).get_structure("output", path).get_models())
    sequences = {
        chain.id: tuple(
            intern(residue.resname.strip().upper())
            for residue in chain
            if residue.id[0] == " " and "CA" in residue
        )
        for chain in model
    }
    return {chain: residues for chain, residues in sequences.items() if residues}


def resolve_output_chains(
    output_pdb: Path,
    target_pdb: Path,
    target_chains: Iterable[str],
) -> dict[str, Any]:
    """Match all target subunits and exactly one remaining binder, or reject.

    Identical target subunits are a multiset. If the binder is indistinguishable
    from a target subunit, more than one exclusion matches and identity is unknown.
    """
    reference = chain_sequences(target_pdb)
    selected = tuple(target_chains)
    if not selected or len(set(selected)) != len(selected):
        raise ChainIdentityError("Missing or repeated reference target chains")
    if any(chain not in reference for chain in selected):
        raise ChainIdentityError("A reference target chain is absent")
    expected = Counter(reference[chain] for chain in selected)
    output = chain_sequences(output_pdb)
    matches = [
        binder
        for binder in output
        if Counter(sequence for chain, sequence in output.items() if chain != binder)
        == expected
    ]
    if len(matches) != 1:
        raise ChainIdentityError(
            "Output has no unique binder after matching reference target sequences"
        )
    binder = matches[0]
    return {
        "binder_chain": binder,
        "target_chains": [chain for chain in output if chain != binder],
        "sequence_sha256": {
            chain: _sequence_hash(sequence) for chain, sequence in output.items()
        },
    }


def validate_output_chains(mapping: Any, output_pdb: Path) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ChainIdentityError("Missing output chain map")
    binder, targets = mapping.get("binder_chain"), mapping.get("target_chains")
    if (
        not isinstance(binder, str)
        or not binder
        or not isinstance(targets, list)
        or not targets
        or not all(isinstance(chain, str) and chain for chain in targets)
        or binder in targets
        or len(set(targets)) != len(targets)
    ):
        raise ChainIdentityError("Invalid output chain map")
    output = chain_sequences(output_pdb)
    if set(output) != {binder, *targets} or mapping.get("sequence_sha256") != {
        chain: _sequence_hash(sequence) for chain, sequence in output.items()
    }:
        raise ChainIdentityError("Output structure differs from its verified chain map")
    return mapping


def _clear_identity_dependent_bins(bins: dict[str, Any]) -> None:
    # Historical cluster labels cannot validate an unverified or changed chain.
    for key in tuple(bins):
        if key.startswith("foldseek") or key in {"sequence", "sequence_su", "seq"}:
            bins.pop(key, None)


def annotate_output_identity(
    record: ResultRecord,
    *,
    target_pdb: Path | None = None,
    target_chains: Iterable[str] = (),
) -> ResultRecord:
    """Attach verified output roles without changing scores, IDs or accounting."""
    artifacts, bins = dict(record.artifacts or {}), dict(record.bins or {})
    prior_map = artifacts.get(MAP_KEY)
    source = artifacts.get("pdb_path")
    if not source and MAP_KEY not in artifacts and STATUS_KEY not in bins:
        return record
    try:
        if not source:
            raise ChainIdentityError("Verified output identity has no structure path")
        if MAP_KEY in artifacts:
            mapping = validate_output_chains(
                json.loads(artifacts[MAP_KEY]), Path(source)
            )
        elif target_pdb is not None:
            mapping = resolve_output_chains(Path(source), target_pdb, target_chains)
        else:
            raise ChainIdentityError("Output identity requires a target reference")
        artifacts[MAP_KEY] = json.dumps(mapping, sort_keys=True)
        if artifacts[MAP_KEY] != prior_map:
            _clear_identity_dependent_bins(bins)
        artifacts["binder_chain"] = bins["binder_chain"] = mapping["binder_chain"]
        artifacts["target_chains"] = bins["target_chains"] = ",".join(
            mapping["target_chains"]
        )
        bins[STATUS_KEY] = "verified"
        bins.pop("output_chain_identity_error", None)
    except (OSError, ValueError, TypeError) as exc:
        _clear_identity_dependent_bins(bins)
        bins[STATUS_KEY] = "unresolved"
        bins["output_chain_identity_error"] = str(exc)
        for key in ("binder_chain", "binder_chain_id", "target_chains"):
            artifacts.pop(key, None)
            bins.pop(key, None)
    return replace(record, artifacts=artifacts, bins=bins)


def archive_target_reference(archive_root: Path) -> tuple[Path, tuple[str, ...]] | None:
    """Resolve a recorded reference and verify its bytes before legacy recovery."""
    provenance = archive_root / "run_provenance.json"
    if not provenance.is_file():
        return None
    try:
        target = json.loads(provenance.read_text())["target"]
        pdb, config = Path(target["pdb_path"]), Path(target["config_path"])
        ps, cs = pdb.stat(), config.stat()
        return _verified_reference(
            str(pdb.resolve()),
            ps.st_size,
            ps.st_mtime_ns,
            target.get("pdb_sha256"),
            str(config.resolve()),
            cs.st_size,
            cs.st_mtime_ns,
            target.get("config_sha256"),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ChainIdentityError(
            f"Cannot verify archive target reference: {exc}"
        ) from exc


@lru_cache(maxsize=64)
def _verified_reference(
    pdb: str,
    pdb_size: int,
    pdb_modified: int,
    expected: str,
    config: str,
    config_size: int,
    config_modified: int,
    expected_config: str,
) -> tuple[Path, tuple[str, ...]]:
    if not expected or hashlib.sha256(Path(pdb).read_bytes()).hexdigest() != expected:
        raise ChainIdentityError("Archive target PDB hash mismatch or missing hash")
    raw = Path(config).read_bytes()
    if not expected_config or hashlib.sha256(raw).hexdigest() != expected_config:
        raise ChainIdentityError(
            "Archive target constraint hash mismatch or missing hash"
        )
    return Path(pdb), tuple(json.loads(raw)["chain_ids"])


def prepare_archive_results(
    records: Iterable[ResultRecord], archive_root: Path
) -> list[ResultRecord]:
    """Recover legacy artifact identities when archived reference provenance exists."""
    rows = list(records)
    try:
        reference = archive_target_reference(archive_root)
    except ChainIdentityError:
        # Invalid provenance must not revive unchecked legacy role annotations.
        return [annotate_output_identity(row) for row in rows]
    if reference is None:
        # New self-contained maps still validate. Legacy AF2 has its own report
        # recovery; unlabelled Complexa is rejected by the clustering consumer.
        return [
            annotate_output_identity(row) if MAP_KEY in row.artifacts else row
            for row in rows
        ]
    path, chains = reference
    return [
        annotate_output_identity(row, target_pdb=path, target_chains=chains)
        for row in rows
    ]
