"""Parse redesigned ProteinMPNN sequences from seqs/*.fa.

Thread each sequence onto the parent backbone and store its own PDB. Qualification
measurements remain empty until standardized evaluation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..schemas import ResultRecord
from .types import ParseError, ParserContext

PROTEINMPNN_GPU_H_PER_SEQ = 0.001  # MPNN is ~ms per sequence

# Numeric ``key=value`` fields in a ProteinMPNN FASTA header. Generated-sequence
# headers (protein_mpnn_run.py:403) look like
# ``T=0.1, sample=1, score=1.234, global_score=1.456, seq_recovery=0.42``.
_MPNN_FIELD_RE = re.compile(r"(\w+)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_CHAIN_SEPARATOR_RE = re.compile(r"X{20,}")


def _parse_mpnn_header(header: str) -> dict[str, float]:
    """Extract numeric ``key=value`` fields from a ProteinMPNN FASTA header."""
    out: dict[str, float] = {}
    for k, v in _MPNN_FIELD_RE.findall(header):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out

_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
_BINDER_CHAIN_DEFAULT = "B"  # T-REX convention


def _read_fasta(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    cur_header: str | None = None
    cur_seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_header is not None:
                entries.append((cur_header, "".join(cur_seq)))
            cur_header = line[1:].strip()
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_header is not None:
        entries.append((cur_header, "".join(cur_seq)))
    return [(h, s) for h, s in entries if s]


def _chain_residue_count(pdb_path: Path, chain_id: str) -> int | None:
    try:
        seen: set[tuple[str, str, str]] = set()
        for line in pdb_path.read_text(errors="ignore").splitlines():
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            if (line[21].strip() or " ") != chain_id:
                continue
            seen.add((line[22:26], line[26:27], line[17:20]))
        return len(seen) if seen else None
    except OSError:
        return None


def _binder_sequence_from_mpnn(seq: str, parent_pdb: Path, binder_chain: str) -> str:
    """Extract the binder-chain sequence from ProteinMPNN FASTA output.

    ProteinMPNN writes multi-chain complexes as one FASTA sequence with long
    X-runs between chains. For a redesigned single binder chain, keep only the
    segment that matches the binder chain length; for AF2-collapsed chains,
    removing separators can also be the correct representation.
    """
    n_res = _chain_residue_count(parent_pdb, binder_chain)
    if n_res is None or len(seq) == n_res:
        return seq
    degapped = _CHAIN_SEPARATOR_RE.sub("", seq)
    if len(degapped) == n_res:
        return degapped
    matches = [part for part in _CHAIN_SEPARATOR_RE.split(seq) if len(part) == n_res]
    if matches:
        return matches[0]
    return seq


def _thread_sequence_onto_pdb(
    parent_pdb: Path, binder_chain: str, new_seq: str, out_pdb: Path,
) -> bool:
    """Write ``parent_pdb`` with ``binder_chain`` residues renamed to
    match ``new_seq``. Side-chain atoms stripped (AF2 rebuilds them).
    Target chain preserved verbatim. Returns True on success.

    Length mismatch (PDB binder length != len(new_seq)) → returns False;
    caller falls back to parent_pdb (no semantic improvement but no
    breakage either).
    """
    try:
        from Bio.PDB import PDBParser, PDBIO  # type: ignore
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("threaded", str(parent_pdb))
        threaded = False
        for model in structure:
            for chain in model:
                if chain.id != binder_chain:
                    continue
                binder_residues = [r for r in chain if r.id[0] == " "]
                if len(binder_residues) != len(new_seq):
                    print(f"  [thread_seq] length mismatch: "
                          f"PDB binder={len(binder_residues)} vs MPNN seq={len(new_seq)}",
                          flush=True)
                    return False
                for residue, aa1 in zip(binder_residues, new_seq):
                    aa3 = _AA1_TO_3.get(aa1)
                    if aa3 is None:
                        continue
                    residue.resname = aa3
                    for atom in list(residue):
                        if atom.id not in _BACKBONE_ATOMS:
                            residue.detach_child(atom.id)
                threaded = True
                break
            if threaded:
                break
        if not threaded:
            print(f"  [thread_seq] binder chain {binder_chain!r} not in {parent_pdb.name}",
                  flush=True)
            return False
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(out_pdb))
        return True
    except Exception as e:
        print(f"  [thread_seq] failed: {e}", flush=True)
        return False


def parse_proteinmpnn_output(
    output_dir: Path, ctx: ParserContext
) -> list[ResultRecord]:
    if not output_dir.exists():
        raise ParseError(f"ProteinMPNN output dir missing: {output_dir}")
    seqs_dir = output_dir / "seqs"
    fa_files = sorted(seqs_dir.glob("*.fa")) if seqs_dir.exists() else []
    if not fa_files:
        raise ParseError(f"ProteinMPNN: no FASTA under {seqs_dir}")

    parent_pdb_path = getattr(ctx, "parent_pdb_path", "") or ""
    binder_chain = (getattr(ctx, "binder_chain", "") or _BINDER_CHAIN_DEFAULT).strip() or _BINDER_CHAIN_DEFAULT
    threaded_dir = output_dir / "threaded_pdbs"
    if parent_pdb_path:
        threaded_dir.mkdir(parents=True, exist_ok=True)

    fam = ctx.method_family or "proteinmpnn_redesign"
    records: list[ResultRecord] = []
    n_native_skipped = 0
    n_thread_failed = 0
    for fa in fa_files:
        for idx, (header, seq) in enumerate(_read_fasta(fa.read_text())):
            # Skip the native parent FASTA entry, which lacks a sample identifier.
            if "sample=" not in header:
                n_native_skipped += 1
                continue
            blob = f"{ctx.target_id}::{ctx.candidate_id}::{fa.stem}::{idx}".encode()
            rid = hashlib.sha256(blob).hexdigest()[:16]
            thread_seq = seq
            if parent_pdb_path:
                thread_seq = _binder_sequence_from_mpnn(seq, Path(parent_pdb_path), binder_chain)
            artifacts: dict[str, str] = {
                "sequence": thread_seq,
                "fa_path": str(fa),
            }
            if parent_pdb_path:
                # Write the redesigned sequence into its own backbone structure for
                # evaluation.
                threaded_pdb = threaded_dir / f"{fa.stem}_seq{idx:03d}.pdb"
                ok = _thread_sequence_onto_pdb(
                    Path(parent_pdb_path),
                    binder_chain,
                    thread_seq,
                    threaded_pdb,
                )
                if ok:
                    artifacts["pdb_path"] = str(threaded_pdb)
                    artifacts["parent_pdb"] = parent_pdb_path
                    artifacts["threaded"] = "true"
                else:
                    # Skip threading failures; pointing to the parent would evaluate the
                    # wrong sequence. The controller retains the job compute even when
                    # no records remain.
                    n_thread_failed += 1
                    continue
            # Keep source scores as diagnostics for evaluation ordering.
            bins = {"sample_index": str(idx), "source_fa": fa.name,
                    "redesigned_sequence": seq[:64]}
            hdr_fields = _parse_mpnn_header(header)
            for k in ("score", "global_score", "seq_recovery"):
                if k in hdr_fields:
                    bins[f"mpnn_{k}"] = f"{hdr_fields[k]:.4f}"
            records.append(ResultRecord(
                result_id=rid,
                parent_ids=list(ctx.parent_ids),
                backend_family=fam,
                runtime_bucket_id=ctx.runtime_bucket_id,
                target_id=ctx.target_id,
                metrics={},  # empty by design — chain with refilter to score
                metrics_calibrated={},
                route_lineage=[fam],
                gpu_h=PROTEINMPNN_GPU_H_PER_SEQ,
                exit_status="ok",
                bins=bins,
                artifacts=artifacts,
                panel_ready=False,
                tick_id=(ctx.tick_id or None),
            ))
    if n_native_skipped or n_thread_failed:
        print(f"  [proteinmpnn] dropped {n_native_skipped} native + "
              f"{n_thread_failed} threading-failed sequence(s); "
              f"{len(records)} redesign record(s) kept", flush=True)
    return records
