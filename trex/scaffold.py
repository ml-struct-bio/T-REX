"""Inspect and scaffold operator-installed T-REX backend adapters."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .backend_extensions import load_backend_adapters
from .capability_registry import default_registry


_FAMILY_RE = re.compile(r"[a-z][a-z0-9_]{1,63}")
BACKEND_LIST_SCHEMA_VERSION = "trex.backend-list.v1"


_ADAPTER_TEMPLATE = '''"""T-REX adapter for __FAMILY__.

Edit ``build_command`` to match the backend CLI and ``parse_output`` to match
its artifacts. Keep native metrics backend-prefixed: third-party adapters are
diagnostic-only and T-REX performs canonical score conversion separately.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from trex.backend_extensions import (
    BackendAdapter,
    BackendCommand,
    BackendLaunchContext,
)
from trex.capability_registry import Capability
from trex.output_parsers import ParseError, ParserContext
from trex.schemas import ResultRecord


FAMILY = "__FAMILY__"
EXECUTABLE = "__EXECUTABLE__"


def preflight() -> tuple[bool, str]:
    path = shutil.which(EXECUTABLE)
    return (path is not None, path or f"{EXECUTABLE} was not found on PATH")


def build_command(ctx: BackendLaunchContext) -> BackendCommand:
    """Return direct argv; T-REX never executes an LLM-supplied shell string."""
    out = ctx.output_root / "results"
    num_designs = int(ctx.candidate.config_delta.get("num_designs", 8))
    temperature = float(ctx.candidate.config_delta.get("temperature", 1.0))
    return BackendCommand(
        argv=(
            EXECUTABLE,
            "--target-pdb", str(ctx.target_pdb),
            "--output-dir", str(out),
            "--num-designs", str(num_designs),
            "--temperature", str(temperature),
        ),
        output_dir=out,
    )


def parse_output(output_dir: Path, ctx: ParserContext) -> list[ResultRecord]:
    """Example contract: one JSON object per design in ``results.jsonl``.

    Expected keys are ``pdb_path`` and optional ``native_metrics``/``bins``.
    Replace this parser with one that validates the real backend's outputs.
    """
    source = output_dir / "results.jsonl"
    if not source.is_file():
        raise ParseError(f"missing {source}")
    records: list[ResultRecord] = []
    for index, raw in enumerate(source.read_text().splitlines()):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            pdb = Path(row["pdb_path"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ParseError(f"malformed {source}:{index + 1}") from exc
        if not pdb.is_absolute():
            pdb = output_dir / pdb
        if not pdb.is_file():
            raise ParseError(f"missing design structure: {pdb}")
        native = {
            f"{FAMILY}_native_{key}": float(value)
            for key, value in (row.get("native_metrics") or {}).items()
        }
        bins = {str(key): str(value) for key, value in (row.get("bins") or {}).items()}
        records.append(ResultRecord(
            result_id=str(row.get("result_id") or f"{ctx.candidate_id}_{index:04d}"),
            parent_ids=list(ctx.parent_ids),
            target_id=ctx.target_id,
            backend_family=FAMILY,
            runtime_bucket_id=ctx.runtime_bucket_id,
            metrics=native,
            metrics_calibrated={},
            route_lineage=[FAMILY],
            gpu_h=0.0,
            exit_status="ok",
            bins=bins,
            artifacts={"pdb_path": str(pdb.resolve())},
            panel_ready=False,
            tick_id=ctx.tick_id,
        ))
    return records


ADAPTER = BackendAdapter(
    capability=Capability(
        family=FAMILY,
        default_operator_id=f"{FAMILY}_default",
        default_lane_id=FAMILY,
        default_cost_class="diagnostic",
        runtime_bucket_id="rb_extension_v1",
        availability="available",
        notes="operator-installed diagnostic generator",
        allowed_params={
            "num_designs": (1.0, 64.0),
            "temperature": (0.05, 2.0),
            "mode": ["default"],
        },
        role="generator",
        requires_parent_pdb=False,
        outputs_diagnostic_only=True,
        preflight=preflight,
    ),
    build_command=build_command,
    parse_output=parse_output,
    budget_parameters=("num_designs",),
    budget_defaults={"num_designs": 8},
    eval_budget_cap=64,
    hard_ceiling_seconds=7200,
)
'''


_README_TEMPLATE = """# __FAMILY__ T-REX adapter

1. Edit `__MODULE__.py` so `build_command` matches the backend CLI.
2. Replace the example `results.jsonl` parser with strict validation of the
   backend's real artifacts. Emit only backend-prefixed native diagnostics.
3. Put this directory on `PYTHONPATH` and export:

   ```bash
   export TREX_BACKEND_PLUGINS=__MODULE__:ADAPTER
   export TREX_ENABLED_FAMILIES="$TREX_ENABLED_FAMILIES,__FAMILY__"
   ```

4. Run `trex-backend validate`, `trex-validate --require-backends`, unit tests,
   and a bounded campaign smoke before a full run.

For a reusable package, expose `ADAPTER` through the Python entry-point group
`trex.backends` instead of setting `TREX_BACKEND_PLUGINS`.
"""


def _scaffold(args: argparse.Namespace) -> Path:
    family = args.family.strip()
    if not _FAMILY_RE.fullmatch(family):
        raise ValueError("--family must match [a-z][a-z0-9_]{1,63}")
    module = args.module or f"trex_{family}_backend"
    if not module.isidentifier():
        raise ValueError("--module must be a valid Python module name")
    root = Path(args.out_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        raise ValueError(f"refusing to write into non-empty {root} (use --force)")
    root.mkdir(parents=True, exist_ok=True)
    module_path = root / f"{module}.py"
    readme_path = root / "README.md"
    if not args.force and (module_path.exists() or readme_path.exists()):
        raise ValueError("adapter files already exist (use --force)")
    module_path.write_text(
        _ADAPTER_TEMPLATE.replace("__FAMILY__", family).replace(
            "__EXECUTABLE__", args.executable
        )
    )
    readme_path.write_text(
        _README_TEMPLATE.replace("__FAMILY__", family).replace("__MODULE__", module)
    )
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    list_cmd = sub.add_parser("list", help="list built-in and loaded backend families")
    list_cmd.add_argument("--json", action="store_true", dest="as_json")
    sub.add_parser("validate", help="load adapters and run their preflight hooks")
    init = sub.add_parser("init", help="create a backend adapter skeleton")
    init.add_argument("--family", required=True)
    init.add_argument("--executable", required=True)
    init.add_argument("--out-dir", required=True, type=Path)
    init.add_argument("--module")
    init.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            registry = default_registry()
            extensions = set(load_backend_adapters())
            rows = [
                {
                    "family": family,
                    "source": "extension" if family in extensions else "built-in",
                    "role": cap.role,
                    "availability": cap.availability,
                    "requires_parent_pdb": cap.requires_parent_pdb,
                    "outputs_diagnostic_only": cap.outputs_diagnostic_only,
                    "allowed_params": cap.allowed_params,
                }
                for family, cap in sorted(registry.capabilities.items())
            ]
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": BACKEND_LIST_SCHEMA_VERSION,
                            "backends": rows,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                for row in rows:
                    print(
                        f"{row['family']}\t{row['source']}\t{row['role']}\t"
                        f"{row['availability']}\t{len(row['allowed_params'])} parameters"
                    )
            return 0
        if args.command == "validate":
            adapters = load_backend_adapters(refresh=True)
            for family, adapter in sorted(adapters.items()):
                preflight = adapter.capability.preflight
                ok, detail = preflight() if preflight is not None else (True, "no hook")
                print(f"{'OK' if ok else 'FAIL'}\t{family}\t{detail}")
                if not ok:
                    return 2
            print(f"Validated {len(adapters)} backend extension(s).")
            return 0
        if args.command == "init":
            print(_scaffold(args))
            return 0
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"trex-backend: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
