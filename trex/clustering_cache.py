"""Exact in-process caches for T-ReX clustering passes.

These helpers deliberately cache only *identical full input sets*. They are not
incremental clustering: if a new strict success or near-miss enters a pass, the
full pass is recomputed so a new structure can still merge with any older one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .foldseek_clusterer import (
    ClusteringResult,
    _record_binder_chain,
    _scored_pdb_path,
    cluster_archive_pdbs,
)
from .schemas import ResultRecord
from .sequence_clusterer import (
    SequenceClusteringResult,
    _sequence_for_result,
    cluster_archive_sequences,
)

_MAX_CACHE_ENTRIES = 24
_FOLDSEEK_CACHE: OrderedDict[str, ClusteringResult] = OrderedDict()
_MMSEQS_CACHE: OrderedDict[str, SequenceClusteringResult] = OrderedDict()
_CACHEABLE_STATUSES = {"ok", "no_structures", "no_sequences"}

# R4a (2026-06-01): read-through memo for binder-sequence extraction. The mmseqs
# fingerprint runs _sequence_for_result on every call (incl. cache hits); for
# PDB-only strict records that re-reads+parses the PDB each tick. Memo by
# (result_id, pdb_path, mtime_ns, binder_chain) — a stat() per call is far
# cheaper than read+parse — reusing the EXACT extraction fn (no drift). Bounded.
_SEQ_EXTRACT_CACHE: "OrderedDict[tuple, tuple[str, str]]" = OrderedDict()
_MAX_SEQ_EXTRACT_ENTRIES = 8192


def _cached_sequence_for_result(r: ResultRecord, binder_chain_id: str) -> tuple[str, str]:
    art = r.artifacts or {}
    pdb = art.get("pdb_path", "")
    mtime = -1
    if pdb:
        try:
            mtime = Path(pdb).stat().st_mtime_ns
        except OSError:
            mtime = -1
    record_chain = _record_binder_chain(r, binder_chain_id)
    key = (r.result_id, pdb, mtime, record_chain)
    hit = _SEQ_EXTRACT_CACHE.get(key)
    if hit is not None:
        _SEQ_EXTRACT_CACHE.move_to_end(key)
        return hit
    val = _sequence_for_result(r, binder_chain_id=binder_chain_id)
    _SEQ_EXTRACT_CACHE[key] = val
    _SEQ_EXTRACT_CACHE.move_to_end(key)
    while len(_SEQ_EXTRACT_CACHE) > _MAX_SEQ_EXTRACT_ENTRIES:
        _SEQ_EXTRACT_CACHE.popitem(last=False)
    return val


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {
            "path": str(path.resolve()),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    except OSError:
        return {"path": str(path), "missing": True}


def _binary_fingerprint(binary_name: str) -> dict[str, Any]:
    resolved = shutil.which(binary_name)
    if resolved is None:
        return {"requested": binary_name, "resolved": None}
    return {
        "requested": binary_name,
        "resolved": str(Path(resolved).resolve()),
        **_file_fingerprint(Path(resolved)),
    }


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _remember(
    cache: OrderedDict[str, Any],
    key: str,
    value: Any,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MAX_CACHE_ENTRIES:
        cache.popitem(last=False)


def _copy_foldseek_result(result: ClusteringResult) -> ClusteringResult:
    return replace(result, cluster_by_result_id=dict(result.cluster_by_result_id))


def _copy_sequence_result(result: SequenceClusteringResult) -> SequenceClusteringResult:
    return replace(result, cluster_by_result_id=dict(result.cluster_by_result_id))


def _foldseek_input_fingerprint(
    results: list[ResultRecord],
    *,
    target_id: str,
    foldseek_binary: str,
    min_tm_score: float,
    only_result_ids: set[str] | None,
    structure_scope: str,
    binder_chain_id: str,
    alignment_type: int | None,
) -> str:
    inputs: list[dict[str, Any]] = []
    for r in results:
        if r.target_id != target_id:
            continue
        if only_result_ids is not None and r.result_id not in only_result_ids:
            continue
        # Shared predicate with cluster_archive_pdbs — they MUST agree on which
        # records are clustered, else a cache hit could serve a clustering over a
        # different input set. _scored_pdb_path is the single source (R1/R4b).
        path = _scored_pdb_path(r)
        if path is None:
            continue
        inputs.append({
            "result_id": r.result_id,
            "artifact": _file_fingerprint(path),
            "binder_chain_id": _record_binder_chain(r, binder_chain_id),
        })
    payload = {
        "kind": "foldseek",
        "target_id": target_id,
        "binary": _binary_fingerprint(foldseek_binary),
        "min_tm_score": min_tm_score,
        "only_result_ids": sorted(only_result_ids) if only_result_ids is not None else None,
        "structure_scope": structure_scope,
        "binder_chain_id": binder_chain_id,
        "alignment_type": alignment_type,
        "min_seq_id": 0.0,
        "cov_mode": 0,
        "inputs": sorted(inputs, key=lambda x: x["result_id"]),
    }
    return _digest(payload)


def cluster_archive_pdbs_cached(
    results: list[ResultRecord],
    *,
    target_id: str,
    foldseek_binary: str = "foldseek",
    min_tm_score: float = 0.60,  # T-ReX live objective; report 0.5/0.6/0.8 as a sweep.
    timeout_seconds: int = 600,
    only_result_ids: set[str] | None = None,
    structure_scope: str = "binder_chain",
    binder_chain_id: str = "B",
    alignment_type: int | None = 1,
) -> ClusteringResult:
    """Run Foldseek with exact-input memoization.

    A cache hit is allowed only when the whole effective clustering input is
    unchanged. Failed/no-binary outcomes are intentionally not cached, so a
    transient environment problem can recover on the next tick.
    """

    key = _foldseek_input_fingerprint(
        results,
        target_id=target_id,
        foldseek_binary=foldseek_binary,
        min_tm_score=min_tm_score,
        only_result_ids=only_result_ids,
        structure_scope=structure_scope,
        binder_chain_id=binder_chain_id,
        alignment_type=alignment_type,
    )
    cached = _FOLDSEEK_CACHE.get(key)
    if cached is not None:
        _FOLDSEEK_CACHE.move_to_end(key)
        return _copy_foldseek_result(cached)

    result = cluster_archive_pdbs(
        results,
        target_id=target_id,
        foldseek_binary=foldseek_binary,
        min_tm_score=min_tm_score,
        timeout_seconds=timeout_seconds,
        only_result_ids=only_result_ids,
        structure_scope=structure_scope,
        binder_chain_id=binder_chain_id,
        alignment_type=alignment_type,
    )
    if result.status in _CACHEABLE_STATUSES:
        _remember(_FOLDSEEK_CACHE, key, _copy_foldseek_result(result))
    return result


def _sequence_input_fingerprint(
    results: list[ResultRecord],
    *,
    target_id: str,
    mmseqs_binary: str,
    min_seq_id: float,
    coverage: float,
    only_result_ids: set[str] | None,
    binder_chain_id: str,
) -> str:
    inputs: list[dict[str, Any]] = []
    for r in results:
        if r.target_id != target_id:
            continue
        if only_result_ids is not None and r.result_id not in only_result_ids:
            continue
        if r.exit_status != "ok":
            continue
        seq, source = _cached_sequence_for_result(r, binder_chain_id)
        if not seq:
            continue
        inputs.append({
            "result_id": r.result_id,
            "source": source,
            "binder_chain_id": _record_binder_chain(r, binder_chain_id),
            "length": len(seq),
            "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
        })
    payload = {
        "kind": "mmseqs",
        "target_id": target_id,
        "binary": _binary_fingerprint(mmseqs_binary),
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "only_result_ids": sorted(only_result_ids) if only_result_ids is not None else None,
        "binder_chain_id": binder_chain_id,
        "inputs": sorted(inputs, key=lambda x: x["result_id"]),
    }
    return _digest(payload)


def cluster_archive_sequences_cached(
    results: list[ResultRecord],
    *,
    target_id: str,
    mmseqs_binary: str = "mmseqs",
    min_seq_id: float = 0.90,
    coverage: float = 0.80,
    timeout_seconds: int = 600,
    only_result_ids: set[str] | None = None,
    binder_chain_id: str = "B",
) -> SequenceClusteringResult:
    key = _sequence_input_fingerprint(
        results,
        target_id=target_id,
        mmseqs_binary=mmseqs_binary,
        min_seq_id=min_seq_id,
        coverage=coverage,
        only_result_ids=only_result_ids,
        binder_chain_id=binder_chain_id,
    )
    cached = _MMSEQS_CACHE.get(key)
    if cached is not None:
        _MMSEQS_CACHE.move_to_end(key)
        return _copy_sequence_result(cached)

    result = cluster_archive_sequences(
        results,
        target_id=target_id,
        mmseqs_binary=mmseqs_binary,
        min_seq_id=min_seq_id,
        coverage=coverage,
        timeout_seconds=timeout_seconds,
        only_result_ids=only_result_ids,
        binder_chain_id=binder_chain_id,
    )
    if result.status in _CACHEABLE_STATUSES:
        _remember(_MMSEQS_CACHE, key, _copy_sequence_result(result))
    return result


def clear_clustering_caches() -> None:
    """Testing hook."""

    _FOLDSEEK_CACHE.clear()
    _MMSEQS_CACHE.clear()
    _SEQ_EXTRACT_CACHE.clear()
