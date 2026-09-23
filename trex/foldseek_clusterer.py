"""Foldseek clustering of archived binder structures.

Run strict-success clustering separately for SU counts and scored-structure clustering
for duplication evidence. Missing binaries or untrusted binder identities produce no
fallback SU credit. Cluster mappings update the in-memory evidence view; archived
records remain unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .af2_chain_identity import resolve_prediction_chains, validate_prediction_chains
from .schemas import ResultRecord
from .output_identity import MAP_KEY, STATUS_KEY, validate_output_chains


@dataclass(frozen=True)
class ClusteringResult:
    """Outcome of a Foldseek clustering call."""

    cluster_by_result_id: dict[str, str]
    n_structures: int
    n_clusters: int
    status: str  # "ok" | "no_binary" | "no_structures" | "failed" | "no_clusters"
    stderr_tail: str = ""
    structure_scope: str = "full_complex"
    n_scope_fallback: int = 0


def _safe_stem(value: str) -> str:
    """Return an ASCII filesystem-safe stem."""
    return "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value
    ).strip("_")[:80]


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _pdb_chain_ids(path: Path) -> set[str]:
    chains: set[str] = set()
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                ch = line[21].strip()
                if ch:
                    chains.add(ch)
    except OSError:
        return set()
    return chains


def _record_binder_chain(r: ResultRecord, default: str = "B") -> str:
    """Return the artifact's binder chain, or empty for unresolved AF2 identity.

    AF2 can rename input chains. For legacy records, recover output identity
    from the adjacent AF2 report and structures without rewriting the archive.
    Empty means exclude; it must never trigger a guessed-chain fallback.
    Other backends retain their explicit per-record multichain metadata.
    """
    art = getattr(r, "artifacts", None) or {}
    bins = getattr(r, "bins", None) or {}
    if bins.get(STATUS_KEY) == "unresolved":
        return ""
    if MAP_KEY in art:
        try:
            mapping = validate_output_chains(json.loads(art[MAP_KEY]), Path(art["pdb_path"]))
            return mapping["binder_chain"]
        except (OSError, ValueError, TypeError, KeyError):
            return ""
    is_af2 = (
        "af2_strict_basis" in bins or "af2_chain_identity" in bins
        or "af2_prediction_chains" in art or "af2_report_path" in art
        or Path(art.get("pdb_path", "")).name == "af2_refilter_prediction.pdb"
    )
    if is_af2:
        path = _pdb_for_result(r)
        if path is None:
            return ""
        try:
            if art.get("af2_prediction_chains"):
                mapping = validate_prediction_chains(
                    json.loads(art["af2_prediction_chains"]), path,
                )
            else:
                report_path = Path(art.get("af2_report_path") or path.with_name("af2_refilter_result.json"))
                if not report_path.is_file():
                    report_path = path.with_name("af2_refilter_result.json")
                report = json.loads(report_path.read_text())
                if not isinstance(report, dict):
                    return ""
                mapping = resolve_prediction_chains(report, path, report_dir=report_path.parent)
            return mapping["binder_chain"]
        except (OSError, ValueError, TypeError):
            return ""
    for mapping in (
        getattr(r, "artifacts", None) or {},
        getattr(r, "bins", None) or {},
    ):
        for key in ("binder_chain", "binder_chain_id"):
            value = str(mapping.get(key, "") or "").strip()
            if value:
                return value.split(",")[0].strip() or default
    # Complexa can relabel its input target. Missing output metadata cannot
    # be recovered from a campaign-level chain label. Archive-aware callers
    # prepare verified output maps before reaching this helper.
    if r.backend_family.startswith("complexa"):
        return ""
    return default


def _write_chain_only_pdb(
    src: Path,
    dest: Path,
    *,
    preferred_chain: str = "B",
) -> tuple[bool, str | None]:
    """Write a single-chain PDB for Foldseek clustering.

    T-REX evaluates binder diversity, not target diversity. Multi-chain complex PDBs
    all contain the same target chain, and Foldseek emits per-chain entries; if
    the shared target chain is allowed into clustering, the result_id-level map can
    be contaminated by target-chain clusters. Prefer the controller-recorded binder chain, with B retained only as the single-chain target convention.
    """
    if not preferred_chain or src.suffix.lower() != ".pdb":
        return False, None
    chains = _pdb_chain_ids(src)
    if not chains:
        return False, None
    if preferred_chain in chains:
        chain = preferred_chain
    else:
        return False, None

    wrote_atom = False
    out_lines: list[str] = []
    try:
        for line in src.read_text(errors="replace").splitlines():
            # Identity verification reads the first model only. Later models
            # can use the same labels for different chains.
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM  ", "HETATM")):
                if len(line) > 21 and line[21].strip() == chain:
                    out_lines.append(line)
                    wrote_atom = True
            elif line.startswith("TER"):
                if wrote_atom:
                    out_lines.append(line)
                    wrote_atom = False
        if not out_lines:
            return False, None
        dest.write_text("\n".join(out_lines) + "\nEND\n")
        return True, chain
    except OSError:
        return False, None


def _pdb_for_result(r: ResultRecord) -> Path | None:
    """Locate a usable binder structure file for a ResultRecord.

    T-REX result parsers store the structure at `artifacts["pdb_path"]`
    (BindCraft, Complexa) or under `artifacts["pdb_dir"]` (some refilter
    paths). Mirrors `controller._resolve_parent_artifact`'s
    discovery logic.
    """
    art = getattr(r, "artifacts", None) or {}
    p = art.get("pdb_path")
    if p:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
            return path
    d = art.get("pdb_dir")
    if d:
        d_path = Path(d)
        if d_path.is_dir():
            for ext in (".pdb", ".cif", ".mmcif"):
                hits = sorted(d_path.glob(f"*{ext}"))
                if hits:
                    return hits[0]
    return None


def _resolve_stem(stem: str, name_to_result_id: dict[str, str]) -> str | None:
    """Map a Foldseek output stem to a known input result ID.

    Foldseek can append chain or model suffixes, such as ``_A`` or
    ``_MODEL_3_A``. Prefer an exact input-stem match, then remove trailing
    underscore-delimited segments until a known stem is found. Return None
    when no input matches; an unknown stem cannot receive cluster credit.
    """
    if stem in name_to_result_id:
        return name_to_result_id[stem]
    base = stem
    while "_" in base:
        base = base.rpartition("_")[0]
        rid = name_to_result_id.get(base)
        if rid is not None:
            return rid
    return None


def _parse_cluster_tsv(
    tmp: Path,
    name_to_result_id: dict[str, str],
) -> dict[str, str]:
    """Parse foldseek easy-cluster's `*_cluster.tsv` output.

    Foldseek emits a `<prefix>_cluster.tsv` with two tab-separated columns:
    `<representative_member_stem>\\t<member_stem>`. Cluster_id is anchored
    to the representative's result_id so the same cluster gets the same
    label across multiple runs as long as the representative survives.
    """
    cluster_files = sorted(tmp.glob("**/*cluster*.tsv"))
    if not cluster_files:
        return {}
    out: dict[str, str] = {}
    for path in cluster_files:
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep_stem, member_stem = parts[0], parts[1]
            rep_rid = _resolve_stem(rep_stem, name_to_result_id)
            member_rid = _resolve_stem(member_stem, name_to_result_id)
            if member_rid is None:
                continue
            cluster_id = f"foldseek:{rep_rid or rep_stem}"
            out[member_rid] = cluster_id
    return out


def _scored_pdb_path(r: ResultRecord) -> Path | None:
    """Return the artifact for a successful record with at least one qualification
    measurement.

    This predicate is shared by clustering, cache fingerprints, and scored-window
    selection. Unscored intermediates are excluded.
    """
    if r.exit_status != "ok":
        return None
    m = r.metrics or {}
    if not any(k in m for k in ("pLDDT", "iPAE", "binder_scRMSD")):
        return None
    return _pdb_for_result(r)


def cluster_archive_pdbs(
    results: list[ResultRecord],
    *,
    target_id: str,
    foldseek_binary: str = "foldseek",
    min_tm_score: float = 0.60,  # T-REX live objective; report 0.5/0.6/0.8 as a sweep.
    timeout_seconds: int = 600,
    only_result_ids: set[str] | None = None,
    structure_scope: str = "binder_chain",
    binder_chain_id: str = "B",
    alignment_type: int | None = 1,
) -> ClusteringResult:
    """Cluster eligible binder structures, optionally restricted to selected result IDs.

    SU counting clusters qualified designs separately: nonqualified structures must not
    merge otherwise distinct qualified clusters. Missing Foldseek returns an empty
    mapping with no_binary status; no per-result SU fallback is allowed.
    """
    pdbs: dict[str, Path] = {}
    rec_by_rid: dict = {}  # rid -> its OWN ResultRecord (for per-record binder chain)
    for r in results:
        if r.target_id != target_id:
            continue
        if only_result_ids is not None and r.result_id not in only_result_ids:
            continue
        # scored-with-PDB predicate (shared single source — see _scored_pdb_path)
        path = _scored_pdb_path(r)
        if path is None:
            continue
        pdbs[r.result_id] = path
        rec_by_rid[r.result_id] = r

    if not pdbs:
        return ClusteringResult(
            cluster_by_result_id={},
            n_structures=0,
            n_clusters=0,
            status="no_structures",
            structure_scope=structure_scope,
        )

    binary = shutil.which(foldseek_binary)
    if binary is None:
        return ClusteringResult(
            cluster_by_result_id={},
            n_structures=len(pdbs),
            n_clusters=0,
            status="no_binary",
            structure_scope=structure_scope,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="v7_foldseek_") as raw_tmp:
            tmp = Path(raw_tmp)
            input_dir = tmp / "structures"
            input_dir.mkdir()
            name_to_rid: dict[str, str] = {}
            n_scope_fallback = 0
            for idx, (rid, path) in enumerate(sorted(pdbs.items())):
                stem = _safe_stem(rid) or f"r_{idx:05d}"
                if structure_scope == "binder_chain":
                    dest = input_dir / f"{stem}.pdb"
                    # Resolve the binder identity separately for each result.
                    ok, _chain = _write_chain_only_pdb(
                        path,
                        dest,
                        preferred_chain=_record_binder_chain(
                            rec_by_rid[rid], binder_chain_id
                        ),
                    )
                    if not ok:
                        # Skip unresolved binder chains. Mixing whole complexes with
                        # binder-only structures would create invalid clusters and
                        # misleading coverage.
                        n_scope_fallback += 1
                        continue
                else:
                    dest = input_dir / f"{stem}{path.suffix.lower() or '.pdb'}"
                    _link_or_copy(path, dest)
                name_to_rid[dest.stem] = rid

            # Validate binder extraction before the singleton shortcut; file
            # existence alone does not establish a clusterable structure.
            if not name_to_rid:
                return ClusteringResult(
                    cluster_by_result_id={},
                    n_structures=len(pdbs),
                    n_clusters=0,
                    status="no_structures",
                    structure_scope=structure_scope,
                    n_scope_fallback=n_scope_fallback,
                )
            if len(name_to_rid) == 1:
                rid = next(iter(name_to_rid.values()))
                return ClusteringResult(
                    cluster_by_result_id={rid: f"foldseek:{rid}"},
                    n_structures=len(pdbs),
                    n_clusters=1,
                    status="ok",
                    structure_scope=structure_scope,
                    n_scope_fallback=n_scope_fallback,
                )

            out_prefix = tmp / "clusters"
            work_dir = tmp / "work"
            # Use one thread and sorted inputs to make greedy representative selection stable.
            # Avoid --shuffle because it is unavailable in some supported Foldseek builds.
            cmd = [
                binary,
                "easy-cluster",
                str(input_dir),
                str(out_prefix),
                str(work_dir),
                # Cluster binder structures at the caller's TM threshold.
                # The live SU threshold is 0.6; diagnostic and post-hoc passes
                # can request different thresholds.
                "--tmscore-threshold",
                str(min_tm_score),
                "--min-seq-id",
                "0.0",
                "--cov-mode",
                "0",
                "--threads",
                "1",
            ]
            if alignment_type is not None:
                cmd.extend(["--alignment-type", str(alignment_type)])
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if proc.returncode != 0:
                return ClusteringResult(
                    cluster_by_result_id={},
                    n_structures=len(pdbs),
                    n_clusters=0,
                    status="failed",
                    structure_scope=structure_scope,
                    n_scope_fallback=n_scope_fallback,
                    stderr_tail=(proc.stderr or proc.stdout or "")[-500:],
                )
            cluster_map = _parse_cluster_tsv(tmp, name_to_rid)
            if not cluster_map:
                return ClusteringResult(
                    cluster_by_result_id={},
                    n_structures=len(pdbs),
                    n_clusters=0,
                    status="no_clusters",
                    structure_scope=structure_scope,
                    n_scope_fallback=n_scope_fallback,
                )
            n_clusters = len(set(cluster_map.values()))
            return ClusteringResult(
                cluster_by_result_id=cluster_map,
                n_structures=len(pdbs),
                n_clusters=n_clusters,
                status="ok",
                structure_scope=structure_scope,
                n_scope_fallback=n_scope_fallback,
            )
    except Exception as exc:  # noqa: BLE001
        return ClusteringResult(
            cluster_by_result_id={},
            n_structures=len(pdbs),
            n_clusters=0,
            status="failed",
            structure_scope=structure_scope,
            stderr_tail=f"{type(exc).__name__}: {exc}",
        )


def apply_clusters_to_bins(
    results: list[ResultRecord],
    cluster_map: dict[str, str],
    bin_key: str = "foldseek",
) -> int:
    """Add cluster IDs to in-memory result bins and return the number updated.

    The frozen dataclass contains a mutable bins mapping. foldseek denotes
    scored-structure clustering; foldseek_su denotes strict-only clustering.
    """
    n = 0
    for r in results:
        cid = cluster_map.get(r.result_id)
        if cid is None:
            continue
        r.bins[bin_key] = cid
        n += 1
    return n
