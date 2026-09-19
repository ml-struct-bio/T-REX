"""Verify success-metric consistency across families by re-folding sampled
designs through ONE AF2 refilter (recycles=3, residue-count chain detection).

For a sample of BindCraft + Complexa designs that pass each family's NATIVE
strict criterion, re-fold the complex with af2_refilter_runner (the unified
ColabDesign AF2-multimer scorer) and compare native vs refilter metrics.

Answers:
  - Do good designs get low scRMSD under recycles=3 (validates the recycles
    0->3 + chain-detection fix)?
  - What fraction of each family's NATIVE-strict designs stay strict under the
    unified refilter (true comparable strict rate)?
  - Does Complexa native af2folding agree with the unified refilter?

Usage (under the AF2/ColabDesign venv, 1 GPU):
  python -m trex.refilter_consistency_verify --n-per-family 12 --out report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLEXA_REPO = Path(os.environ.get("TREX_COMPLEXA_REPO", REPO_ROOT / "external" / "Proteina-Complexa"))
LEGACY_COMPLEXA_REPO = Path(os.environ.get("TREX_LEGACY_COMPLEXA_REPO", COMPLEXA_REPO))
P2_PYTHON = os.environ.get("TREX_COMPLEXA_PYTHON", str(COMPLEXA_REPO / ".venv" / "bin" / "python"))
P2_AF2_DATA_DIR = os.environ.get("TREX_AF2_DATA_DIR", str(LEGACY_COMPLEXA_REPO / "community_models" / "ckpts" / "AF2"))
AR_REPO = str(REPO_ROOT)
RUNS = os.environ.get("TREX_ARCHIVE_BASE", str(REPO_ROOT / "runs"))

STRICT = dict(pLDDT=90.0, iPAE=7 / 31, binder_scRMSD=1.5)

# Target chain is copied verbatim into the complex; identify it by size, not
# "shorter==binder" (CD45 target is only 88 res, smaller than many binders).
TARGET_SIZES = {"cd45": 88, "betv1": 159, "sc2rbd": 194}


def _target_size_for_path(archive_path: str) -> int | None:
    low = archive_path.lower()
    for k, v in TARGET_SIZES.items():
        if k in low:
            return v
    return None


def _is_native_strict(m: dict) -> bool:
    p, i, r = m.get("pLDDT"), m.get("iPAE"), m.get("binder_scRMSD")
    return None not in (p, i, r) and p >= 90 and i <= 7 / 31 and r < 1.5  # official strict "<"


def _residues_per_chain(p: str) -> dict[str, int]:
    seen, cnt = set(), {}
    for ln in open(p, errors="ignore"):
        if ln.startswith("ATOM") and len(ln) > 26:
            ch, rs = ln[21].strip(), ln[22:27].strip()
            if ch and (ch, rs) not in seen:
                seen.add((ch, rs)); cnt[ch] = cnt.get(ch, 0) + 1
    return cnt


def _chains(pdb: str, target_res_count: int | None) -> tuple[str, str] | None:
    """(target_chain, binder_chain). target = chain matching the known target
    size; binder = the other. Size guess alone is unreliable (CD45 target=88)."""
    c = _residues_per_chain(pdb)
    if len(c) < 2:
        return None
    if target_res_count is not None:
        target = min(c, key=lambda k: abs(c[k] - target_res_count))
        binder = max((k for k in c if k != target), key=lambda k: c[k])
        return (target, binder)
    o = sorted(c.items(), key=lambda kv: kv[1])
    return ("A", "B") if o[0][1] >= 0.85 * o[-1][1] else (o[-1][0], o[0][0])


def _pdb_for(d: dict) -> str | None:
    a = d.get("artifacts") or {}
    if a.get("pdb_path") and os.path.exists(a["pdb_path"]):
        return a["pdb_path"]
    dd = a.get("pdb_dir")
    if dd and os.path.isdir(dd):
        pdbs = sorted(glob.glob(dd + "/*.pdb"))
        return pdbs[0] if pdbs else None
    return None


def _is_near_miss(m: dict) -> bool:
    """Good on pLDDT+iPAE but not strict — used to probe metric behaviour on
    borderline designs (does the refilter agree on the boundary?)."""
    p, i = m.get("pLDDT"), m.get("iPAE")
    return p is not None and i is not None and p >= 88 and i <= 0.30 and not _is_native_strict(m)


def collect(n_per_family: int, families: list[str], target_filter: str | None,
            include_near_miss: bool) -> dict[str, list[dict]]:
    want = {f: [] for f in families}
    for A in glob.glob(f"{RUNS}/*/*/result_records.jsonl"):
        if target_filter and target_filter.lower() not in A.lower():
            continue
        for line in open(A):
            try:
                d = json.loads(line)
            except Exception:
                continue
            fam = d.get("backend_family")
            if fam not in want or len(want[fam]) >= n_per_family:
                continue
            m = d.get("metrics") or {}
            kind = "strict" if _is_native_strict(m) else ("near_miss" if include_near_miss and _is_near_miss(m) else None)
            if kind is None:
                continue
            pdb = _pdb_for(d)
            if pdb:
                want[fam].append({"pdb": pdb, "native": m, "result_id": d.get("result_id"),
                                  "tgt_size": _target_size_for_path(A), "kind": kind})
        if all(len(v) >= n_per_family for v in want.values()):
            break
    return want


def refold(pdb: str, out_dir: Path, target_res_count: int | None,
           recycles: int, model_names: str, use_initial_guess: int) -> dict | None:
    ch = _chains(pdb, target_res_count)
    if ch is None:
        return None
    tgt, bnd = ch
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        P2_PYTHON, "-m", "trex.af2_refilter_runner",
        "--input-pdb", pdb, "--out-dir", str(out_dir),
        "--af2-data-dir", P2_AF2_DATA_DIR,
        "--target-chain", tgt, "--binder-chain", bnd,
        "--num-recycles", str(recycles), "--model-names", model_names,
        "--use-initial-guess", str(use_initial_guess),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = AR_REPO + ":" + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, env=env, timeout=900, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"  [refold-fail] {Path(pdb).name}: {e}", flush=True)
        return None
    res = out_dir / "af2_refilter_result.json"
    if not res.exists():
        return None
    mt = json.loads(res.read_text()).get("metrics") or {}
    return {"chains": (tgt, bnd),
            "pLDDT": (mt["plddt"] * 100.0) if mt.get("plddt") is not None else None,
            "iPAE": mt.get("i_pae"),
            "binder_scRMSD": mt.get("binder_scrmsd_ca")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/tmp/refilter_verify"))
    ap.add_argument("--families", default="bindcraft,complexa_beam")
    ap.add_argument("--target", default=None, help="substring filter e.g. cd45/betv1/sc2rbd")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--model-names", default="model_1_multimer_v3")
    ap.add_argument("--use-initial-guess", type=int, choices=(0, 1), default=1)
    ap.add_argument("--include-near-miss", action="store_true")
    args = ap.parse_args()

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    samples = collect(args.n_per_family, fams, args.target, args.include_near_miss)
    report = {"strict_threshold": STRICT, "recycles": args.recycles,
              "model_names": args.model_names, "use_initial_guess": args.use_initial_guess,
              "target": args.target, "families": {}}
    for fam, items in samples.items():
        rows = []
        for k, it in enumerate(items):
            rf = refold(it["pdb"], args.work / fam / str(k), it.get("tgt_size"),
                        args.recycles, args.model_names, args.use_initial_guess)
            if rf is None:
                continue
            rf_strict = _is_native_strict(rf)
            rows.append({"result_id": it["result_id"], "chains": rf["chains"],
                         "kind": it.get("kind"),
                         "native": {q: it["native"].get(q) for q in STRICT},
                         "refilter": {q: rf.get(q) for q in STRICT},
                         "native_strict": it.get("kind") == "strict",
                         "refilter_strict": rf_strict})
            print(f"[{fam}/{it.get('kind')}] {Path(it['pdb']).name[:36]} chains={rf['chains']} "
                  f"native(scRMSD={it['native'].get('binder_scRMSD'):.2f},iPAE={it['native'].get('iPAE'):.2f}) "
                  f"-> refilter(scRMSD={rf['binder_scRMSD']},iPAE={rf['iPAE']}) strict={rf_strict}",
                  flush=True)
        n = len(rows)
        kept = sum(1 for r in rows if r["refilter_strict"])
        report["families"][fam] = {
            "n_native_strict_refolded": n,
            "still_strict_under_refilter": kept,
            "retention_pct": round(100 * kept / n, 1) if n else None,
            "rows": rows,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n==== SUMMARY ====")
    for fam, r in report["families"].items():
        print(f"  {fam}: {r['still_strict_under_refilter']}/{r['n_native_strict_refolded']} "
              f"native-strict stay strict under unified refilter "
              f"({r['retention_pct']}%)")


if __name__ == "__main__":
    main()
