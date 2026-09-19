#!/usr/bin/env python3
"""Verify deterministic parity between the private source snapshot and T-ReX.

This check is read-only with respect to the source tree. Python bytecode and
pytest caches are redirected or disabled, and all reports are written beneath
the caller-supplied output directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


LEGACY_PREFIX = "auto" + "research"
LEGACY_PACKAGE = LEGACY_PREFIX + "_v7_3_3"
LEGACY_CONTROLLER = LEGACY_PACKAGE + ".phase2_v7_controller"
LEGACY_LLM = LEGACY_PREFIX + "_common.llm"


POLICY_MODULES = (
    "foldseek_clusterer.py",
    "output_parsers/af2_refilter.py",
    "candidate_builder.py",
    "capability_registry.py",
    "critic.py",
    "critic_guard.py",
    "dedup_trust.py",
    "diagnosis_outcome.py",
    "evidence_reducer.py",
    "evidence_refs.py",
    "fallback.py",
    "lifecycle.py",
    "live_tick.py",
    "panel.py",
    "planner.py",
    "prompts.py",
    "refilter_roles.py",
    "schemas.py",
    "score_conversion.py",
    "selector.py",
    "sequence_clusterer.py",
    "success_criteria.py",
    "supervisor.py",
    "unified_reasoner.py",
)


# Exact source/release AST pairs freeze every reviewed standalone difference.
# They cover extraction of typed components,
# prompt-catalog helpers, opt-in provenance-bound memory, and stricter structure
# artifact checks. Any additional edit to either side changes a hash and fails
# parity. The verifier also requires equal deterministic policy snapshots and
# passing source/release suites, so an allowlisted structural difference is not
# sufficient on its own.
APPROVED_STANDALONE_DIVERGENCES: dict[str, dict[str, str]] = {
    "foldseek_clusterer.py": {
        "source_ast_sha256": "50940f3ace8e5bd2a79e462ee39e0fea2c4c645bfc1751f4bfd1fbe9d429cd91",
        "trex_ast_sha256": "17ff690dcd9ae71bcde80b514c9ee9ecfc0ce934eabda5fa8b0d6935a806f0dd",
        "reason": "2026-09-17 AF2 prediction-chain identity correction; intentional clustering difference, covered by test_af2_chain_identity.py"
    },
    "output_parsers/af2_refilter.py": {
        "source_ast_sha256": "03c94c5a1208a036cf86cf0d3de711215f6c31a395b80c650d22387f62728606",
        "trex_ast_sha256": "61ca80c03b82d823f7bd04865d3d33eb5c7fd9d5b7a1dc4b6845fdb1818d45d7",
        "reason": "2026-09-17 AF2 prediction-chain identity correction; intentional clustering difference, covered by test_af2_chain_identity.py"
    },
    "sequence_clusterer.py": {
        "source_ast_sha256": "a458d559ccfcb1b4189484fef08ee6619181ae2681bbc7145ea32664d6bb11ac",
        "trex_ast_sha256": "2aa0df5e5478c5bbc0b53131002946bccb3f8bffd4aef91e96c76305ad7335d3",
        "reason": "2026-09-17 AF2 prediction-chain identity correction; intentional clustering difference, covered by test_af2_chain_identity.py"
    },

    "critic.py": {
        "source_ast_sha256": "1d91abf071baeda8f4c0c6807fe68ef21c64211cbdbd7beaf6ad56d9f8b897aa",
        "trex_ast_sha256": "f761826e9f9824fa02596cad045baf8602788d12585645b0aaba86ae5ffb4142",
        "reason": "extract the exact critic user prompt for the runtime prompt catalog",
    },
    "capability_registry.py": {
        "source_ast_sha256": "fc9f4ce66f17ff1d8c0911f139bac4a10fcacb8e6326cae92b2bb6579a2e779c",
        "trex_ast_sha256": "fee9b7a3e5e0e409d67eeb89ad44aa7dd9676e2d95d1cef86f3a99d55e4d5efe",
        "reason": "remove unavailable legacy archive-only refilters while preserving the active eight-family registry",
    },
    "evidence_reducer.py": {
        "source_ast_sha256": "5a994cccd551d5bdda67fe6eb1fe0ed7446a9cfb661d49316bf3bfc06d330e3c",
        "trex_ast_sha256": "6a6995ac09d391971a58e822fdfa74b5421b76db6a31fec0b71e045bf7b86d58",
        "reason": "extract attribution/route phases and remove inactive legacy-refilter presentation while preserving historical accounting",
    },
    "fallback.py": {
        "source_ast_sha256": "c328baaf3b8894df863d526f344caadc0fa0b7d6c6fe99d4c052c447ec146735",
        "trex_ast_sha256": "06fdc591e1d857a46ffe40afe678d0b50a83c640b5a8cf4b57b080b1c6bd652c",
        "reason": "clarify the documented low-confidence behavior without changing fallback policy",
    },
    "live_tick.py": {
        "source_ast_sha256": "5c4472d3af2516a28bfc68e989a491b720501a282acd43b7efc9e4e6aa4c4e17",
        "trex_ast_sha256": "fdaf18f942292246153fd3ac6217558d678608096f1e91bd6fb6a49da8a66042",
        "reason": "extract typed tick phases behind the compatible live-tick entry point",
    },
    "panel.py": {
        "source_ast_sha256": "74e916b04346d49709c2bff12884d0e7cc6a13ff3e00c40575586151e7614b87",
        "trex_ast_sha256": "d04618ed76f2fd938b5ba4f56943bfba9d1e5da3cfc4f0b376deb5159d5c26ca",
        "reason": "require real structure files instead of accepting directory-shaped paths",
    },
    "planner.py": {
        "source_ast_sha256": "c04035f479a77770748548f4dd9efafae1a091d571bbc3244804fd2a73e970c1",
        "trex_ast_sha256": "be3c435fe33ffa4f5a7868f7981cb0cbdab340ade479851d95a0cb6dd6e9f439",
        "reason": "expose prompt helpers, add opt-in audited memory and remove redundant filtering for absent legacy refilters",
    },
    "refilter_roles.py": {
        "source_ast_sha256": "7b5efe71a7fcadb94725ac1483cd9cfa91f86062fd2d31cd1f83d06d7e3017c0",
        "trex_ast_sha256": "ba2523f89d70ad51e4b7257696d6cf4360ef1a0c934386c90b0e513493568a51",
        "reason": "remove the unavailable legacy cross-verifier role from the public action contract",
    },
    "schemas.py": {
        "source_ast_sha256": "782a43883fe178ddddd1ca4e11f58798ee66da15a14775479359c0105aa92570",
        "trex_ast_sha256": "1027e58b7b602392801ce544cff78a9993f3dbdae780af22814b5b51a8efd15c",
        "reason": "remove inactive legacy-refilter schema vocabulary from the public interface",
    },
    "selector.py": {
        "source_ast_sha256": "a1862b386ce03657c38a4218f834e27c5e6f39edcc279d5f58b6f521c1ea0b04",
        "trex_ast_sha256": "a8fd9095d707d45b7e44a406ad29d833d764ef9118e17e267af595cdc7502502",
        "reason": "extract typed selection phases behind the compatible selector entry point",
    },
    "success_criteria.py": {
        "source_ast_sha256": "c9b09b8fe4f3b619753179cff679fbf85e9d66e98adbdb1d817d083b25534589",
        "trex_ast_sha256": "15bee6e4c3f44060de5f68aad0bc80a9b6b28585f12a512243c44c4032307c6d",
        "reason": "clarify canonical AF2 scoring and current BoltzGen terminology without changing thresholds",
    },
    "supervisor.py": {
        "source_ast_sha256": "15f5a389fcd367c9e1288fd3fedeaced525b38e66a5a32163d79daa778a6d311",
        "trex_ast_sha256": "4817a03b509a671acdcb68d582f5d1847300429ef720ef25f14a52d63de741bd",
        "reason": "extract prompt/repair helpers and add opt-in audited memory",
    },
}


_STANDALONE_EXTENSION_BLOCK = re.compile(
    r"(?ms)^[ \t]*# TREX_STANDALONE_EXTENSION_BEGIN[^\n]*\n"
    r".*?"
    r"^[ \t]*# TREX_STANDALONE_EXTENSION_END[^\n]*(?:\n|$)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            rows[path.relative_to(root).as_posix()] = _sha256(path)
    return rows


class _NameNormalizer(ast.NodeTransformer):
    """Normalize branding-only Python identifiers before AST comparison."""

    @staticmethod
    def _name(value: str) -> str:
        value = value.replace("trex.controller", LEGACY_CONTROLLER)
        value = value.replace("trex.llm", LEGACY_LLM)
        value = value.replace("trex", LEGACY_PACKAGE)
        value = value.replace("TREX_REPO_ROOT", "V7_AR_REPO")
        value = value.replace("TREX_EXTERNAL_ROOT", "V7_SUBGIT_ROOT")
        value = value.replace("TREX_", "V7_")
        value = value.replace("T-ReX", "__TREX_BRAND__")
        value = re.sub(
            r"\bV7(?:\.3(?:\.3)?)?\b(?!\.\d)",
            "__TREX_BRAND__",
            value,
        )
        return value

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._name(node.id)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module is not None:
            node.module = self._name(node.module)
        self.generic_visit(node)
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        node.name = self._name(node.name)
        if node.asname is not None:
            node.asname = self._name(node.asname)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = self._name(node.attr)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            node.value = self._name(node.value)
        return node


def _normalized_ast(path: Path) -> tuple[str, int]:
    text = path.read_text()
    extension_blocks = len(_STANDALONE_EXTENSION_BLOCK.findall(text))
    # Standalone-only hooks are additive and inactive in the publication
    # configuration. Exclude only explicitly delimited blocks from source-AST
    # equality; the deterministic snapshot below still proves that the default
    # registry and policy outputs are exactly equal.
    text = _STANDALONE_EXTENSION_BLOCK.sub("", text)
    tree = ast.parse(text, filename=str(path))
    tree = _NameNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return (
        ast.dump(tree, annotate_fields=True, include_attributes=False),
        extension_blocks,
    )


def _ast_comparison_status(
    module: str, source_sha256: str, trex_sha256: str
) -> tuple[str, str | None]:
    if source_sha256 == trex_sha256:
        return "exact", None
    approved = APPROVED_STANDALONE_DIVERGENCES.get(module)
    if (
        approved is not None
        and approved["source_ast_sha256"] == source_sha256
        and approved["trex_ast_sha256"] == trex_sha256
    ):
        return "approved_standalone_divergence", approved["reason"]
    return "unexpected_difference", None


# These maintenance-only modules have no corresponding frozen package file.
# Pin their implementation; an unreviewed change must fail the compatibility gate.
MAINTENANCE_MODULE_SHA256 = {
    "af2_chain_identity.py": "c7c677bfccdbccc52e405877db3a4c9a1b7eda54d39e8a444bdc194224771fc6",
    "af2_refilter_runner.py": "6a6ee2fc565c4d45f0390fb51b3272411137091c4c42e9e51f59264e0f5b81e9"
}


def _maintenance_module_checks(trex_pkg: Path) -> list[dict[str, Any]]:
    rows = []
    for relative, expected in MAINTENANCE_MODULE_SHA256.items():
        path = trex_pkg / relative
        observed = (
            _sha256(path)
            if path.is_file() else None
        )
        rows.append({"module": relative, "expected_sha256": expected,
                     "observed_sha256": observed, "accepted": observed == expected})
    return rows


def _policy_ast_parity(source_pkg: Path, trex_pkg: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in POLICY_MODULES:
        source_path = source_pkg / relative
        trex_path = trex_pkg / relative
        source_ast, source_extension_blocks = _normalized_ast(source_path)
        trex_ast, trex_extension_blocks = _normalized_ast(trex_path)
        source_sha256 = hashlib.sha256(source_ast.encode()).hexdigest()
        trex_sha256 = hashlib.sha256(trex_ast.encode()).hexdigest()
        status, reason = _ast_comparison_status(relative, source_sha256, trex_sha256)
        rows.append(
            {
                "module": relative,
                "equal": source_ast == trex_ast,
                "accepted": status != "unexpected_difference",
                "status": status,
                "reason": reason,
                "source_ast_sha256": source_sha256,
                "trex_ast_sha256": trex_sha256,
                "source_extension_blocks_excluded": source_extension_blocks,
                "trex_extension_blocks_excluded": trex_extension_blocks,
            }
        )
    maintenance = _maintenance_module_checks(trex_pkg)
    return {
        "passed": all(row["accepted"] for row in rows + maintenance),
        "modules": rows,
        "intentional_maintenance_modules": maintenance,
    }


def _controller_contract(path: Path) -> dict[str, str]:
    """Read bounded event-loop contracts without importing or running backends.

    These sections must agree exactly after branding normalization. This is
    intentionally narrower than a claim of whole-controller equivalence.
    Missing or unrecognizable sections fail closed.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def stores(node: ast.AST, name: str) -> bool:
        return any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Store)
            and child.id == name
            for child in ast.walk(node)
        )

    loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.While)
        and any(
            isinstance(statement, ast.Assign) and stores(statement, "sleep_s")
            for statement in node.body
        )
    ]
    if len(loops) != 1:
        raise ValueError("expected one campaign loop with idle backoff")
    body = loops[0].body
    refill_start = next(
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.Assign) and stores(node, "n_busy")
    )
    planning_start = next(
        index
        for index, node in enumerate(body)
        if index > refill_start
        and isinstance(node, ast.If)
        and stores(node, "round_id")
    )
    postdispatch = max(
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.If) and stores(node, "dispatched_post")
    )
    cutoffs = [
        node
        for node in body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "max_wall_h"
            for child in ast.walk(node.test)
        )
    ]
    dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dispatch_candidate_to_gpu"
    )
    run_names = [
        node
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Assign)
        and (stores(node, "cid_safe") or stores(node, "run_name"))
    ]
    if len(cutoffs) != 1 or len(run_names) != 2 or postdispatch >= len(body) - 1:
        raise ValueError("incomplete cutoff, seed-input, or idle-retry contract")
    sections = {
        "ready_queue_refill_gate": [
            *body[refill_start:planning_start],
            ast.Expr(value=body[planning_start].test),
        ],
        "idle_retry_and_backoff": body[postdispatch + 1 :],
        "launch_window_cutoff": cutoffs,
        "complexa_run_name_seed_input": run_names,
    }
    return {
        name: ast.dump(
            _NameNormalizer().visit(ast.Module(body=statements, type_ignores=[])),
            annotate_fields=True,
            include_attributes=False,
        )
        for name, statements in sections.items()
    }


def _controller_contract_parity(source_pkg: Path, trex_pkg: Path) -> dict[str, Any]:
    try:
        source = _controller_contract(source_pkg / "phase2_v7_controller.py")
        release = _controller_contract(trex_pkg / "controller.py")
    except (OSError, SyntaxError, StopIteration, ValueError) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    sections = [
        {
            "section": name,
            "equal": source[name] == release[name],
            "source_ast_sha256": hashlib.sha256(source[name].encode()).hexdigest(),
            "trex_ast_sha256": hashlib.sha256(release[name].encode()).hexdigest(),
        }
        for name in source
    ]
    return {"passed": all(row["equal"] for row in sections), "sections": sections}


def _run(
    command: list[str],
    *,
    cwd: Path,
    pythonpath: Path,
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pythonpath)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(output_dir / "pycache" / name)
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (output_dir / f"{name}.stdout.txt").write_text(proc.stdout)
    (output_dir / f"{name}.stderr.txt").write_text(proc.stderr)
    return {
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "stdout_tail": proc.stdout.splitlines()[-12:],
        "stderr_tail": proc.stderr.splitlines()[-12:],
    }


def _snapshot(
    *,
    python: str,
    root: Path,
    package: str,
    output_dir: Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    code = f"""
import json
from dataclasses import asdict
from importlib import import_module
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from {package} import SCHEMA_VERSION
from {package}.capability_registry import default_registry
from {package}.{"deterministic_smoke" if package == LEGACY_PACKAGE else "tests.policy_contracts"} import (
    s2_state_classifier_cases, s6_lifecycle_replay, s7_panel_selection,
)
from {package}.success_criteria import STRICT_SUCCESS, NEAR_PASS_MARGINS
from {package}.evidence_reducer import reduce_evidence
from {package}.planner import build_evidence_for_prompt
from {package}.live_tick import LiveTickConfig
from {package}.schemas import ResultRecord
from {LEGACY_CONTROLLER if package == LEGACY_PACKAGE else "trex.controller"} import (
    _cheap_checkpoint_evidence, _controller_sleep_seconds,
)

accounting_cases = []
for charged_total, charged_recent in [(8.0, 2.0), (0.0, 0.0), (None, None)]:
    evidence = reduce_evidence(
        tick_id="v7r001", target_id="test", target_class="test",
        elapsed_wall_h=2.0, remaining_wall_h=46.0, pending_children=0,
        worker_gpu_h_total=1.0, worker_wall_gpu_count=3.0,
        worker_wall_gpu_h_total=6.0, all_results=[], window_results=[],
        run_su_count=4, run_su_count_delta=1, duplicate_fraction=None,
        near_miss_count=0, top_bin_share=None, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="test",
        charged_gpu_count=4.0, charged_gpu_h_total=charged_total,
        charged_gpu_h_recent=charged_recent, charged_gpu_h_scope="fixture",
    )
    accounting_cases.append({{
        "evidence": asdict(evidence),
        "prompt": build_evidence_for_prompt(evidence),
    }})
checkpoint_cases = [
    _cheap_checkpoint_evidence(
        SimpleNamespace(state_label="low_evidence", run_su_count=3),
        latest_state=None, elapsed_wall_h=elapsed, remaining_wall_h=44.5,
        charged_gpu_count=charged, worker_wall_gpu_count=workers,
    )
    for elapsed, charged, workers in [(2.5, 4, 3), (0.0, 4, 3), (1.0, 0, 0)]
]
idle_backoff_cases = [
    _controller_sleep_seconds(
        dispatched=dispatched, planned=planned, pending=["job"] if pending else [],
        pool=[SimpleNamespace(busy=busy)], poll_interval_s=5.0,
    )
    for dispatched in (0, 1) for planned in (0, 1)
    for pending in (False, True) for busy in (False, True)
]
panel_module = import_module("{package}.finalize_panel")
with TemporaryDirectory() as directory:
    archive_type = import_module("{package}.archive").Archive
    empty_clusters = SimpleNamespace(
        cluster_by_result_id={{}}, n_structures=0, status="no_structures",
        structure_scope="binder", n_scope_fallback=0,
    )
    with patch.object(
        panel_module, "cluster_archive_pdbs", return_value=empty_clusters
    ) as cluster:
        archive = archive_type(directory)
        archive.append(ResultRecord(
            result_id="fixture", parent_ids=[], target_id="test",
            backend_family="test", runtime_bucket_id="test",
            metrics={{}}, metrics_calibrated={{}}, route_lineage=[],
            gpu_h=0.0, exit_status="ok",
        ))
        panel_module.finalize_panel(archive, target_id="test")
    posthoc_collapse_default = cluster.call_args.kwargs["min_tm_score"]

registry = default_registry()
families = {{}}
for family_name, family in sorted(registry.capabilities.items()):
    families[family_name] = {{
        "family": family.family,
        "role": family.role,
        "default_operator_id": family.default_operator_id,
        "default_lane_id": family.default_lane_id,
        "default_cost_class": family.default_cost_class,
        "runtime_bucket_id": family.runtime_bucket_id,
        "availability": family.availability,
        "requires_parent_pdb": family.requires_parent_pdb,
        "outputs_diagnostic_only": family.outputs_diagnostic_only,
        "allowed_params": family.allowed_params,
    }}

print(json.dumps({{
    "schema_version": SCHEMA_VERSION,
    "strict_success": STRICT_SUCCESS,
    "near_pass_margins": NEAR_PASS_MARGINS,
    "registry": families,
    "state_cases": s2_state_classifier_cases(),
    "lifecycle": s6_lifecycle_replay(),
    "panel": s7_panel_selection(),
    "accounting_evidence_and_prompt": accounting_cases,
    "checkpoint_accounting": checkpoint_cases,
    "idle_backoff": idle_backoff_cases,
    "evidence_skip_enabled_by_default": LiveTickConfig().skip.enabled,
    "posthoc_collapse_default": posthoc_collapse_default,
}}, sort_keys=True, default=lambda value: value.__dict__))
"""
    result = _run(
        [python, "-c", code],
        cwd=output_dir,
        pythonpath=root,
        output_dir=output_dir,
        name=name,
    )
    if result["returncode"] != 0:
        return {}, result
    payload = json.loads((output_dir / f"{name}.stdout.txt").read_text())
    return payload, result


_SNAPSHOT_FAMILY_ID_KEYS = frozenset(
    {"family", "action_family", "scoring_family", "backend_family"}
)


def _normalize_legacy_archive_only_families(
    source_snapshot: dict[str, Any], trex_snapshot: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Remove only registry-declared inactive legacy families before comparison.

    Historical archives retain diagnostic refilter names that were never
    available to the active campaign action space. The public release removes
    those entries. Derive exclusions from the explicit availability marker so
    missing or changed active families continue to fail parity.
    """
    source_registry = source_snapshot.get("registry", {})
    trex_registry = trex_snapshot.get("registry", {})
    if not isinstance(source_registry, dict):
        source_registry = {}
    if not isinstance(trex_registry, dict):
        trex_registry = {}
    ignored = {
        name
        for name, row in source_registry.items()
        if isinstance(row, dict)
        and row.get("availability") == "legacy_archive_only"
        and name not in trex_registry
    }

    drop = object()

    def normalize(value: Any) -> Any:
        if isinstance(value, str) and value in ignored:
            return drop
        if isinstance(value, list):
            return [item for raw in value if (item := normalize(raw)) is not drop]
        if isinstance(value, dict):
            if any(value.get(key) in ignored for key in _SNAPSHOT_FAMILY_ID_KEYS):
                return drop
            out: dict[str, Any] = {}
            for key, raw in value.items():
                if key in ignored:
                    continue
                item = normalize(raw)
                if item is not drop:
                    out[key] = item
            return out
        return value

    return normalize(source_snapshot), normalize(trex_snapshot), sorted(ignored)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trex-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    trex_root = args.trex_root.resolve()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pkg = source_root / LEGACY_PACKAGE
    trex_pkg = trex_root / "trex"

    source_manifest_before = _manifest(source_pkg)
    ast_report = _policy_ast_parity(source_pkg, trex_pkg)
    controller_report = _controller_contract_parity(source_pkg, trex_pkg)
    execution_run = _run(
        [
            args.python,
            str(trex_root / "scripts/verify_execution_parity.py"),
            "--source-root",
            str(source_root),
            "--trex-root",
            str(trex_root),
        ],
        cwd=output_dir,
        pythonpath=trex_root,
        output_dir=output_dir,
        name="execution_parity",
    )
    try:
        execution_report = json.loads(
            (output_dir / "execution_parity.stdout.txt").read_text()
        )
    except (OSError, json.JSONDecodeError) as exc:
        execution_report = {"passed": False, "error": str(exc)}
    source_snapshot, source_snapshot_run = _snapshot(
        python=args.python,
        root=source_root,
        package=LEGACY_PACKAGE,
        output_dir=output_dir,
        name="source_snapshot",
    )
    trex_snapshot, trex_snapshot_run = _snapshot(
        python=args.python,
        root=trex_root,
        package="trex",
        output_dir=output_dir,
        name="trex_snapshot",
    )

    tests: dict[str, Any] = {}
    if not args.skip_tests:
        common = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short"]
        tests["source"] = _run(
            [args.python, *common, str(source_pkg / "tests")],
            cwd=output_dir,
            pythonpath=source_root,
            output_dir=output_dir,
            name="source_pytest",
        )
        tests["trex"] = _run(
            [args.python, *common, str(trex_pkg / "tests")],
            cwd=output_dir,
            pythonpath=trex_root,
            output_dir=output_dir,
            name="trex_pytest",
        )

    source_manifest_after = _manifest(source_pkg)
    (
        source_snapshot_for_comparison,
        trex_snapshot_for_comparison,
        ignored_legacy_families,
    ) = _normalize_legacy_archive_only_families(source_snapshot, trex_snapshot)
    snapshot_equal = (
        source_snapshot_run["returncode"] == 0
        and trex_snapshot_run["returncode"] == 0
        and source_snapshot_for_comparison == trex_snapshot_for_comparison
    )
    source_unchanged = source_manifest_before == source_manifest_after
    tests_passed = all(row["returncode"] == 0 for row in tests.values())
    passed = (
        ast_report["passed"]
        and controller_report["passed"]
        and execution_run["returncode"] == 0
        and execution_report.get("passed") is True
        and source_snapshot_run["returncode"] == 0
        and trex_snapshot_run["returncode"] == 0
        and snapshot_equal
        and source_unchanged
        and tests_passed
    )
    report = {
        "schema_version": "trex_behavioral_parity.v5",
        "passed": passed,
        "source_root": str(source_root),
        "trex_root": str(trex_root),
        "policy_ast": ast_report,
        "controller_contract": controller_report,
        "execution_contract": execution_report,
        "execution_contract_run": execution_run,
        "deterministic_snapshot_equal": snapshot_equal,
        "deterministic_snapshot_ignored_legacy_archive_only_families": ignored_legacy_families,
        "source_tree_unchanged_during_check": source_unchanged,
        "source_snapshot_run": source_snapshot_run,
        "trex_snapshot_run": trex_snapshot_run,
        "tests": tests,
    }
    (output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "policy_modules_exact": sum(
                    row["equal"] for row in ast_report["modules"]
                ),
                "policy_modules_approved_divergences": sum(
                    row["status"] == "approved_standalone_divergence"
                    for row in ast_report["modules"]
                ),
                "policy_modules_total": len(ast_report["modules"]),
                "deterministic_snapshot_equal": snapshot_equal,
                "deterministic_snapshot_ignored_legacy_archive_only_families": ignored_legacy_families,
                "controller_contract_equal": controller_report["passed"],
                "execution_contract_equal": execution_report.get("execution_equal")
                is True,
                "execution_checks_passed": execution_report.get("passed") is True,
                "source_tree_unchanged_during_check": source_unchanged,
                "tests": {key: value["returncode"] for key, value in tests.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
