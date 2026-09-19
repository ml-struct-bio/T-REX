"""Foldseek-based structural clustering for T-ReX archive ResultRecords.

Plan §2.5 SSOT: run-level structurally-unique success (SU) count is
strict_success filtered THEN Foldseek-deduped. Before this module was
wired (2026-05-27), `live_tick` had no Foldseek bin and could over-count
strict records as unique. The current official path has no per-result
fallback: strict records without a trusted Foldseek SU bin do not mint SU.

This module provides the required Foldseek bin. It:
  1. Collects every ResultRecord with a usable binder PDB / CIF artifact.
  2. Runs `foldseek easy-cluster` on the structures.
  3. Returns a {result_id → cluster_id} map that `live_tick` uses to
     populate the in-memory `bins["foldseek"]` view before SU is
     computed.

Design adapted from the project archival structure-clustering implementation
(Foldseek invocation pattern, cluster TSV parsing). V5's module is bound
to its `CandidateDiagnosis` schema; this is a T-ReX-native variant working
directly on `ResultRecord`.

Foldseek-unavailable behavior: returns an empty map with status="no_binary".
Official SU has no result_id fallback: strict records remain quality evidence,
but they do not mint Foldseek-deduped SU until a trusted Foldseek bin exists.

The mapping is computed in-memory per tick; not persisted to the archive
(consistent with `live_tick`'s existing pattern of treating bins as
parser-set + live-derived). For a 48 h run with ~100-500 candidates,
re-clustering every tick is cheap (Foldseek runs in seconds on this
scale and the cluster map can be cached if needed).
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
    """ASCII-clean filesystem-safe stem (matches v5 helper)."""
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
    return default


def _write_chain_only_pdb(
    src: Path,
    dest: Path,
    *,
    preferred_chain: str = "B",
) -> tuple[bool, str | None]:
    """Write a single-chain PDB for Foldseek clustering.

    T-ReX evaluates binder diversity, not target diversity. Multi-chain complex PDBs
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
    elif len(chains) == 1:
        chain = next(iter(chains))
    else:
        non_a = sorted(ch for ch in chains if ch != "A")
        chain = non_a[0] if non_a else sorted(chains)[0]

    wrote_atom = False
    out_lines: list[str] = []
    try:
        for line in src.read_text(errors="replace").splitlines():
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

    T-ReX result parsers store the structure at `artifacts["pdb_path"]`
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
    """Map a foldseek output stem back to its result_id.

    foldseek appends suffixes to the input stem for multi-CHAIN and
    multi-MODEL inputs — e.g. a 2-chain complex `<rid>.pdb` emits `<rid>_A`
    / `<rid>_B`, and a MULTI-MODEL refilter PDB (af2_refilter saves all AF2
    models, get_best=False) emits `<rid>_MODEL_3_A` etc. The old single
    `rpartition('_')` stripped only ONE trailing token, so `<rid>_MODEL_3_A`
    never matched `<rid>` → the record got NO foldseek_su bin. Under the
    old fallback contract that over-counted SU via refilter_source/result_id;
    under the current official no-fallback contract it would silently lose SU
    credit. Fix: strip trailing `_<token>`
    segments until the remaining prefix is a known input stem. The input
    stems are `_safe_stem(result_id)` and result_id is a hex hash (no '_'),
    so the loop stops exactly at the result_id — no risk of over-stripping.
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
    """Binder structure path for a record IFF it is an eligible clustering input:
    exit_status == "ok", carries >=1 strict-axis metric (pLDDT/iPAE/binder_scRMSD),
    and has a resolvable PDB/CIF artifact. (Unscored raw intermediates — e.g.
    proteinmpnn redesigns awaiting AF2 refilter — are excluded: threaded onto the
    same backbone, they would collapse into one cluster and inflate
    duplicate_fraction; their REFILTERED outputs are the scored structures.)

    SINGLE SOURCE of the "scored structure" predicate — shared by
    cluster_archive_pdbs, the clustering-cache fingerprint, and the R1
    whole-archive windowing selection so the three cannot drift (a drift would
    let a cache hit serve a clustering over a different input set → wrong SU)."""
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
    min_tm_score: float = 0.60,  # T-ReX live objective; report 0.5/0.6/0.8 as a sweep.
    timeout_seconds: int = 600,
    only_result_ids: set[str] | None = None,
    structure_scope: str = "binder_chain",
    binder_chain_id: str = "B",
    alignment_type: int | None = 1,
) -> ClusteringResult:
    """Cluster every binder structure in the archive's ResultRecords.

    Filtering:
      - target_id match (skip foreign target results)
      - exit_status == "ok"
      - has a usable PDB / CIF artifact
      - if ``only_result_ids`` is given, restrict to that subset (used for
        the strict-only SU clustering — see SSOT note below).

    SSOT NOTE (2026-05-31, bug fix): the SU count must be the number of
    distinct clusters *among the strict-success structures clustered by
    themselves* (docstring §2.5: "strict_success filtered THEN
    Foldseek-deduped"). Clustering the WHOLE archive (incl. non-strict
    structures) and then reading strict members' labels lets a non-strict
    "hub" structure (TM>=0.80 to several strict hits that are themselves
    TM<0.80 apart) transitively merge dissimilar strict successes into one
    cluster via single-linkage — undercounting SU (verified: BetV1 s2 6 vs
    15). So `live_tick` calls this twice: once over the whole scored
    archive (for duplicate_fraction / top_bin_share, which legitimately
    need all structures) and once with ``only_result_ids`` = the strict set
    (for SU).

    Returns a `ClusteringResult` containing `cluster_by_result_id`. If
    Foldseek is unavailable, returns `status="no_binary"` and an empty
    map. Official SU has no per-result fallback; `live_tick` may still
    report strict_count, but no Foldseek-deduped SU is minted from this pass.
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
                    # BUGFIX (2026-06-13): use THIS rid's own record for the binder
                    # chain, not the stale `r` left over from the filter loop above
                    # (which extracted every structure with the LAST record's chain
                    # — corrupting SU on multichain targets, e.g. TNF trimer).
                    ok, _chain = _write_chain_only_pdb(
                        path,
                        dest,
                        preferred_chain=_record_binder_chain(
                            rec_by_rid[rid], binder_chain_id
                        ),
                    )
                    if not ok:
                        # NEW-001 (2026-06-18): do NOT fall back to copying the FULL
                        # complex into a binder-chain easy-cluster run. A full
                        # binder+target complex never reaches TM>=threshold against a
                        # binder-only chain, so it would form a bogus singleton and
                        # INFLATE the SU count (over-count by ~#fallbacks) while
                        # foldseek_su_coverage still reads 1.0 (it "got a bin"),
                        # leaving su_dedup_trusted=True on a contaminated count.
                        # Instead SKIP it: the record gets no foldseek_su bin and
                        # therefore mints no official SU/new-SU. The missing bin
                        # drops coverage<1.0 -> su_dedup_trusted=False, so state,
                        # route value, and Selector decisions cannot trust a
                        # contaminated/incomplete SU signal.
                        n_scope_fallback += 1
                        continue
                else:
                    dest = input_dir / f"{stem}{path.suffix.lower() or '.pdb'}"
                    _link_or_copy(path, dest)
                name_to_rid[dest.stem] = rid

            # Scope validation must precede the mathematical singleton shortcut.
            # Merely existing with a .pdb/.cif suffix is not proof that a binder
            # chain can be extracted: truncated files and unsupported CIF scope
            # used to mint one official SU only while they were the sole record.
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
            # `easy-cluster` is a greedy set-cover; without --threads 1
            # the same input produces wildly different cluster counts
            # round-to-round because parallel workers race on the
            # cluster-representative pick (observed 57 / 76 / 35 clusters
            # on the same 218-structure archive in job 8838208).
            #
            # Newer Foldseek builds also accept `--shuffle 0` to disable
            # input shuffling, but this build rejects it
            # ("Unrecognized parameter '--shuffle'", observed in job
            # 8855459). With single-thread the race is gone, and input
            # order is determined by sorted(pdbs.items()) above, so the
            # result is reproducible without --shuffle.
            cmd = [
                binary,
                "easy-cluster",
                str(input_dir),
                str(out_prefix),
                str(work_dir),
                # Deterministic monomer/binder-chain clustering. The caller
                # chooses the TM threshold: the OFFICIAL Proteina-Complexa SU
                # live objective is 0.6; 0.5/0.8 are post-hoc
                # sweep points only.
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
    """Inject `bins[bin_key]` into every result that has a cluster_id.

    Returns the number of results updated. ResultRecord is frozen so we
    mutate the bins dict in place (the dataclass holds a reference to the
    same dict — Python doesn't enforce dict immutability on frozen
    dataclasses). `bins` is schema-required to be a `dict[str, str]`
    (default_factory=dict), so we don't guard against None.

    bin_key="foldseek" → whole-archive cluster (duplicate_fraction /
    top_bin_share). bin_key="foldseek_su" → strict-only cluster (the SU
    count; see cluster_archive_pdbs SSOT note).
    """
    n = 0
    for r in results:
        cid = cluster_map.get(r.result_id)
        if cid is None:
            continue
        r.bins[bin_key] = cid
        n += 1
    return n
