"""MMseqs2-based sequence clustering for T-ReX official strict-success evidence.

This is intentionally secondary evidence. The SSOT success metric remains
AF2-Multimer strict_success deduped by binder structure (`foldseek_su`);
sequence clustering only tells the Planner whether the structurally successful
set is also sequence-diverse enough for a wet-lab panel.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .foldseek_clusterer import _pdb_for_result, _record_binder_chain
from .schemas import ResultRecord


@dataclass(frozen=True)
class SequenceClusteringResult:
    """Outcome of an MMseqs2 clustering call."""

    cluster_by_result_id: dict[str, str]
    n_sequences: int
    n_clusters: int
    status: str  # "ok" | "no_binary" | "no_sequences" | "failed" | "no_clusters"
    stderr_tail: str = ""
    min_seq_id: float = 0.90
    coverage: float = 0.80
    n_sequence_fallback: int = 0


_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def _clean_sequence(value: str | None) -> str | None:
    if not value:
        return None
    seq = "".join(ch for ch in value.upper() if ch.isalpha())
    return seq or None


def _artifact_sequence(r: ResultRecord) -> str | None:
    art = getattr(r, "artifacts", None) or {}
    for key in ("binder_sequence", "sequence", "seq"):
        seq = _clean_sequence(art.get(key))
        if seq:
            return seq
    return None


def _binder_sequence_from_pdb(
    path: Path,
    *,
    preferred_chain: str = "B",
) -> tuple[str | None, str | None]:
    """Extract a binder-chain sequence from a PDB.

    Conventionally T-ReX complex outputs use target chain A and binder chain B.
    If B is absent, use the only chain; if multiple chains are present, choose
    the first non-A chain. This mirrors binder-only Foldseek scoping.
    """
    if not preferred_chain or path.suffix.lower() != ".pdb":
        return None, None

    chains: dict[str, list[str]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith("ATOM  ") or len(line) <= 26:
                continue
            chain = line[21].strip()
            if not chain:
                continue
            resname = line[17:20].strip().upper()
            aa = _AA3_TO_1.get(resname, "X")
            res_key = (line[22:26].strip(), line[26].strip())
            if res_key in seen.setdefault(chain, set()):
                continue
            seen[chain].add(res_key)
            chains.setdefault(chain, []).append(aa)
    except OSError:
        return None, None

    seqs = {chain: "".join(seq) for chain, seq in chains.items() if seq}
    if not seqs:
        return None, None
    if preferred_chain in seqs:
        chain = preferred_chain
    elif len(seqs) == 1:
        chain = next(iter(seqs))
    else:
        non_a = sorted(ch for ch in seqs if ch != "A")
        chain = non_a[0] if non_a else sorted(seqs)[0]
    return seqs[chain], chain


def _sequence_for_result(
    r: ResultRecord,
    *,
    binder_chain_id: str = "B",
) -> tuple[str | None, str | None]:
    chain = _record_binder_chain(r, binder_chain_id)
    if not chain:
        return None, None
    seq = _artifact_sequence(r)
    if seq:
        return seq, "artifact"
    path = _pdb_for_result(r)
    if path is None:
        return None, None
    return _binder_sequence_from_pdb(path, preferred_chain=chain)


def _parse_cluster_tsv(
    tmp: Path,
    name_to_result_id: dict[str, str],
    *,
    prefix: str,
) -> dict[str, str]:
    cluster_files = sorted(tmp.glob("**/*cluster*.tsv"))
    if not cluster_files:
        return {}
    out: dict[str, str] = {}
    for path in cluster_files:
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep_name, member_name = parts[0], parts[1]
            rep_rid = name_to_result_id.get(rep_name)
            member_rid = name_to_result_id.get(member_name)
            if member_rid is None:
                continue
            out[member_rid] = f"{prefix}:{rep_rid or rep_name}"
    return out


def cluster_archive_sequences(
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
    """Cluster binder sequences for target-matched, ok ResultRecords.

    Missing MMseqs2 returns `status="no_binary"` and an empty map rather than
    silently substituting exact-match dedup. The Planner should see that the
    sequence-diversity evidence is unavailable instead of treating an
    underpowered fallback as a real sequence-cluster metric.
    """
    sequences: dict[str, str] = {}
    n_sequence_fallback = 0
    for r in results:
        if r.target_id != target_id:
            continue
        if only_result_ids is not None and r.result_id not in only_result_ids:
            continue
        if r.exit_status != "ok":
            continue
        seq, source = _sequence_for_result(r, binder_chain_id=binder_chain_id)
        if not seq:
            continue
        if source != "artifact":
            n_sequence_fallback += 1
        sequences[r.result_id] = seq

    if not sequences:
        return SequenceClusteringResult(
            cluster_by_result_id={},
            n_sequences=0,
            n_clusters=0,
            status="no_sequences",
            min_seq_id=min_seq_id,
            coverage=coverage,
            n_sequence_fallback=n_sequence_fallback,
        )
    if len(sequences) == 1:
        rid = next(iter(sequences))
        return SequenceClusteringResult(
            cluster_by_result_id={rid: f"seq{int(min_seq_id * 100)}:{rid}"},
            n_sequences=1,
            n_clusters=1,
            status="ok",
            min_seq_id=min_seq_id,
            coverage=coverage,
            n_sequence_fallback=n_sequence_fallback,
        )

    binary = shutil.which(mmseqs_binary)
    if binary is None:
        return SequenceClusteringResult(
            cluster_by_result_id={},
            n_sequences=len(sequences),
            n_clusters=0,
            status="no_binary",
            min_seq_id=min_seq_id,
            coverage=coverage,
            n_sequence_fallback=n_sequence_fallback,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="v7_mmseqs_") as raw_tmp:
            tmp = Path(raw_tmp)
            fasta = tmp / "strict_success.fasta"
            name_to_rid: dict[str, str] = {}
            lines: list[str] = []
            for idx, (rid, seq) in enumerate(sorted(sequences.items())):
                name = f"s_{idx:05d}"
                name_to_rid[name] = rid
                lines.extend([f">{name}", seq])
            fasta.write_text("\n".join(lines) + "\n")

            out_prefix = tmp / "clusters"
            work_dir = tmp / "work"
            cmd = [
                binary,
                "easy-cluster",
                str(fasta),
                str(out_prefix),
                str(work_dir),
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                "0",
                "--threads",
                "1",
            ]
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if proc.returncode != 0:
                return SequenceClusteringResult(
                    cluster_by_result_id={},
                    n_sequences=len(sequences),
                    n_clusters=0,
                    status="failed",
                    stderr_tail=(proc.stderr or proc.stdout or "")[-500:],
                    min_seq_id=min_seq_id,
                    coverage=coverage,
                    n_sequence_fallback=n_sequence_fallback,
                )

            prefix = f"seq{int(min_seq_id * 100)}"
            cluster_map = _parse_cluster_tsv(tmp, name_to_rid, prefix=prefix)
            if not cluster_map:
                return SequenceClusteringResult(
                    cluster_by_result_id={},
                    n_sequences=len(sequences),
                    n_clusters=0,
                    status="no_clusters",
                    min_seq_id=min_seq_id,
                    coverage=coverage,
                    n_sequence_fallback=n_sequence_fallback,
                )
            return SequenceClusteringResult(
                cluster_by_result_id=cluster_map,
                n_sequences=len(sequences),
                n_clusters=len(set(cluster_map.values())),
                status="ok",
                min_seq_id=min_seq_id,
                coverage=coverage,
                n_sequence_fallback=n_sequence_fallback,
            )
    except Exception as exc:  # noqa: BLE001
        return SequenceClusteringResult(
            cluster_by_result_id={},
            n_sequences=len(sequences),
            n_clusters=0,
            status="failed",
            stderr_tail=f"{type(exc).__name__}: {exc}",
            min_seq_id=min_seq_id,
            coverage=coverage,
            n_sequence_fallback=n_sequence_fallback,
        )


def apply_sequence_clusters_to_bins(
    results: list[ResultRecord],
    cluster_map: dict[str, str],
    bin_key: str = "sequence_su",
) -> int:
    n = 0
    for r in results:
        cid = cluster_map.get(r.result_id)
        if cid is None:
            continue
        r.bins[bin_key] = cid
        n += 1
    return n
