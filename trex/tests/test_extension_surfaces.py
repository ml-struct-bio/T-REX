from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trex import capability_registry
from trex.backend_extensions import (
    BackendAdapter,
    BackendCommand,
    BackendLaunchContext,
    clear_backend_adapter_cache,
    load_backend_adapters,
    validate_backend_command,
    validate_extension_records,
)
from trex.capability_registry import Capability, compute_eval_budget, default_registry
from trex.output_parsers import ParserContext
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    ResultRecord,
    TargetConstraint,
)
from trex.scaffold import main as backend_main
from trex.targets import main as target_main
from trex.targets import resolve_target


_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.121  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   9.121  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.620   9.500  1.00 20.00           C
TER
END
"""


def _target_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    config_dir = repo / "config" / "targets"
    asset_root = tmp_path / "assets"
    config_dir.mkdir(parents=True)
    (asset_root / "structures").mkdir(parents=True)
    (config_dir / "example.json").write_text(
        json.dumps(
            {
                "target_id": "example_v1",
                "target_class": "protein",
                "chain_ids": ["A"],
                "hotspots": ["A1"],
                "panel_size_K": 8,
            }
        )
    )
    (asset_root / "structures" / "example.pdb").write_text(_PDB)
    (config_dir / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "trex_target_registry.v1",
                "targets": {
                    "example": {
                        "config": "example.json",
                        "pdb": "structures/example.pdb",
                        "target_id": "example_v1",
                    }
                },
            }
        )
    )
    return repo, asset_root


def test_registered_target_resolver_is_single_source_of_truth(tmp_path: Path) -> None:
    repo, asset_root = _target_repo(tmp_path)
    target = resolve_target("example", repo_root=repo, asset_root=asset_root)
    assert target.target_id == "example_v1"
    assert (
        target.config_path == (repo / "config" / "targets" / "example.json").resolve()
    )
    assert target.pdb_path == (asset_root / "structures" / "example.pdb").resolve()
    assert target.registered is True
    assert target.as_json()["schema_version"] == "trex.resolved-target.v1"


def test_custom_target_requires_explicit_config_and_pdb(tmp_path: Path) -> None:
    repo, _ = _target_repo(tmp_path)
    config = tmp_path / "custom.json"
    pdb = tmp_path / "custom.pdb"
    config.write_text(
        json.dumps(
            {
                "target_id": "custom_v1",
                "target_class": "protein",
                "chain_ids": ["A"],
            }
        )
    )
    pdb.write_text(_PDB)
    target = resolve_target(
        "custom", repo_root=repo, target_config=config, target_pdb=pdb
    )
    assert target.target_id == "custom_v1"
    assert target.registered is False
    with pytest.raises(ValueError, match="both --target-config and --target-pdb"):
        resolve_target("incomplete", repo_root=repo, target_config=config)


def test_custom_target_resolves_without_repository_registry(tmp_path: Path) -> None:
    config = tmp_path / "custom.json"
    pdb = tmp_path / "custom.pdb"
    config.write_text(
        json.dumps(
            {
                "target_id": "wheel_custom_v1",
                "target_class": "protein",
                "chain_ids": ["A"],
            }
        )
    )
    pdb.write_text(_PDB)
    target = resolve_target(
        "custom",
        repo_root=tmp_path / "wheel_install_without_config",
        target_config=config,
        target_pdb=pdb,
    )
    assert target.target_id == "wheel_custom_v1"
    assert target.registered is False


def test_target_scaffolder_writes_hotspots_and_chains(tmp_path: Path) -> None:
    out = tmp_path / "target.json"
    rc = target_main(
        [
            "init",
            "--out",
            str(out),
            "--target-id",
            "new_v1",
            "--chain",
            "A",
            "--hotspot",
            "A42",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["chain_ids"] == ["A"]
    assert payload["hotspots"] == ["A42"]
    assert payload["panel_size_K"] == 8


def _adapter(family: str = "example_generator") -> BackendAdapter:
    capability = Capability(
        family=family,
        default_operator_id=f"{family}_default",
        default_lane_id=family,
        default_cost_class="diagnostic",
        runtime_bucket_id="rb_test",
        availability="available",
        allowed_params={"num_designs": (1.0, 32.0)},
        role="generator",
        outputs_diagnostic_only=True,
    )
    return BackendAdapter(
        capability=capability,
        build_command=lambda ctx: BackendCommand(
            argv=("/bin/true",), output_dir=ctx.output_root / "results"
        ),
        parse_output=lambda output_dir, ctx: [],
        budget_parameters=("num_designs",),
        budget_defaults={"num_designs": 4},
        eval_budget_cap=32,
    )


def test_backend_command_cannot_escape_worker_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must stay under"):
        validate_backend_command(
            BackendCommand(argv=("/bin/true",), output_dir=tmp_path.parent),
            tmp_path,
        )


def test_diagnostic_extension_cannot_emit_canonical_strict_fields(
    tmp_path: Path,
) -> None:
    context = ParserContext(
        target_id="target",
        runtime_bucket_id="rb",
        candidate_id="candidate",
        parent_ids=["candidate"],
        method_family="example_generator",
    )
    record = ResultRecord(
        result_id="result",
        parent_ids=["candidate"],
        target_id="target",
        backend_family="example_generator",
        runtime_bucket_id="rb",
        metrics={"pLDDT": 95.0},
        metrics_calibrated={},
        route_lineage=["example_generator"],
        gpu_h=0.0,
        exit_status="ok",
        artifacts={"pdb_path": str(tmp_path / "design.pdb")},
    )
    with pytest.raises(ValueError, match="canonical strict fields"):
        validate_extension_records(_adapter(), [record], context)


def test_env_backend_extends_registry_and_budget_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "demo_plugin.py"
    module.write_text(
        "from trex.tests.test_extension_surfaces import _adapter\n"
        "ADAPTER = _adapter('demo_generator')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TREX_DISABLE_ENTRYPOINT_BACKENDS", "1")
    monkeypatch.setenv("TREX_BACKEND_PLUGINS", "demo_plugin:ADAPTER")
    clear_backend_adapter_cache()
    try:
        assert set(load_backend_adapters(refresh=True)) == {"demo_generator"}
        registry = default_registry()
        assert registry.is_available("demo_generator")
        assert compute_eval_budget("demo_generator", {}) == 4
        assert compute_eval_budget("demo_generator", {"num_designs": 12}) == 12
    finally:
        clear_backend_adapter_cache()
        sys.modules.pop("demo_plugin", None)
        capability_registry.EVAL_BUDGET_CAP_PER_FAMILY.pop("demo_generator", None)
        capability_registry.EVAL_BUDGET_DEFAULTS_PER_FAMILY.pop("demo_generator", None)
        capability_registry.EVAL_BUDGET_FORMULA_PER_FAMILY.pop("demo_generator", None)


def test_controller_dispatches_loaded_extension_with_direct_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "dispatch_plugin.py"
    module.write_text(
        "from trex.tests.test_extension_surfaces import _adapter\n"
        "ADAPTER = _adapter('dispatch_generator')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TREX_DISABLE_ENTRYPOINT_BACKENDS", "1")
    monkeypatch.setenv("TREX_BACKEND_PLUGINS", "dispatch_plugin:ADAPTER")
    clear_backend_adapter_cache()
    try:
        from trex.archive import Archive
        from trex.controller import _dispatch_candidate_to_gpu

        target_pdb = tmp_path / "target.pdb"
        target_pdb.write_text(_PDB)
        candidate = ActionCandidate(
            candidate_id="candidate",
            hypothesis_ids=["hypothesis"],
            parent_result_id=None,
            method_family="dispatch_generator",
            operator_id="dispatch_generator_default",
            lane_id="dispatch_generator",
            config_delta={"num_designs": 4},
            downstream_route_plan=["structure_refilter"],
            estimated_cost_class="diagnostic",
            expected_signal="diagnostic structures",
            evidence_refs=[],
            feasibility=FeasibilityCheck(
                backend_healthy=True,
                runtime_bucket_id="rb_test",
                compiler_ok=True,
                verifier_ok=True,
                route_cap_ok=True,
                cost_ok=True,
            ),
        )
        outcome = _dispatch_candidate_to_gpu(
            candidate,
            gpu_id="7",
            archive=Archive(tmp_path / "archive"),
            target=TargetConstraint(
                target_id="target", target_class="protein", chain_ids=["A"]
            ),
            target_pdb=str(target_pdb),
            round_id=1,
            archive_root=tmp_path / "archive",
        )
        assert outcome is not None
        proc, out_dir, _, _, target_chains, binder_chain = outcome
        assert proc.wait(timeout=10) == 0
        assert (
            out_dir
            == (
                tmp_path
                / "archive"
                / "worker_outputs"
                / "r001_dispatch_generator_candidate"
                / "results"
            ).resolve()
        )
        assert target_chains == "A"
        assert binder_chain == "B"
    finally:
        clear_backend_adapter_cache()
        sys.modules.pop("dispatch_plugin", None)
        capability_registry.EVAL_BUDGET_CAP_PER_FAMILY.pop("dispatch_generator", None)
        capability_registry.EVAL_BUDGET_DEFAULTS_PER_FAMILY.pop(
            "dispatch_generator", None
        )
        capability_registry.EVAL_BUDGET_FORMULA_PER_FAMILY.pop(
            "dispatch_generator", None
        )


def test_backend_scaffolder_produces_importable_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "adapter"
    rc = backend_main(
        [
            "init",
            "--family",
            "new_generator",
            "--executable",
            "/bin/true",
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "trex_new_generator_backend.py").is_file()
    assert "TREX_BACKEND_PLUGINS" in (out / "README.md").read_text()
    monkeypatch.syspath_prepend(str(out))
    monkeypatch.setenv("TREX_DISABLE_ENTRYPOINT_BACKENDS", "1")
    monkeypatch.setenv("TREX_BACKEND_PLUGINS", "trex_new_generator_backend:ADAPTER")
    clear_backend_adapter_cache()
    try:
        adapters = load_backend_adapters(refresh=True)
        assert adapters["new_generator"].estimate_budget({}) == 8
    finally:
        clear_backend_adapter_cache()


def test_backend_list_json_output_is_versioned(monkeypatch, capsys) -> None:
    monkeypatch.setenv("TREX_DISABLE_ENTRYPOINT_BACKENDS", "1")
    monkeypatch.delenv("TREX_BACKEND_PLUGINS", raising=False)
    clear_backend_adapter_cache()
    try:
        assert backend_main(["list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == "trex.backend-list.v1"
        assert isinstance(payload["backends"], list)
    finally:
        clear_backend_adapter_cache()
        sys.modules.pop("trex_new_generator_backend", None)


def test_target_list_json_output_is_versioned(tmp_path: Path, capsys) -> None:
    repo, _ = _target_repo(tmp_path)
    assert target_main(["--repo-root", str(repo), "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "trex.target-list.v1"
    assert payload["targets"]["example"]["target_id"] == "example_v1"
