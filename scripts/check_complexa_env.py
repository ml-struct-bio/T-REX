#!/usr/bin/env python3
"""Verify Complexa dependencies, imports and optional GPU/checkpoint loading."""
from __future__ import annotations
import argparse
import hashlib
from importlib import metadata, import_module, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/reproducibility/complexa_environment.json"


def dependency_errors(distributions, environment):
    """Check declared requirements, including extras required by other packages."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = {canonicalize_name(d.metadata["Name"]): d for d in distributions}
    extras = {name: set() for name in installed}
    errors = set()
    changed = True
    while changed:
        changed = False
        for name, dist in installed.items():
            for raw in dist.requires or []:
                req = Requirement(raw)
                if req.marker and not any(
                    req.marker.evaluate({**environment, "extra": extra})
                    for extra in {""} | extras[name]
                ):
                    continue
                key = canonicalize_name(req.name)
                if key not in installed:
                    errors.add(f"{name}: missing {req}")
                    continue
                if req.specifier and not req.specifier.contains(
                    installed[key].version, prereleases=True
                ):
                    errors.add(
                        f"{name}: requires {req}; installed {installed[key].version}"
                    )
                new = req.extras - extras[key]
                if new:
                    extras[key].update(new)
                    changed = True
    return sorted(errors)


def verify_install(repository, profile):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if (platform.python_version(), sys.platform, platform.machine()) != (
        profile["python"],
        profile["platform"],
        profile["machine"],
    ):
        raise ValueError("Python/platform differs from the verified profile")
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != profile["git_commit"]:
        raise ValueError("Backend revision differs from the verified profile")
    if (
        hashlib.sha256((repository / "pyproject.toml").read_bytes()).hexdigest()
        != profile["patched_pyproject_sha256"]
    ):
        raise ValueError(
            "Backend dependency metadata differs from the verified profile"
        )
    # LFS datasets/images are outside the dependency check; code and config
    # remain checked. Compute nodes need no git-lfs for this source check.
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "HEAD",
            "--name-only",
            "--",
            ".",
            ":(exclude,attr:filter=lfs)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(changed) - {"pyproject.toml"}:
        raise ValueError(
            "Backend source has tracked changes beyond dependency metadata"
        )
    lock = (ROOT / profile["lock"]).read_bytes()
    if hashlib.sha256(lock).hexdigest() != profile["lock_sha256"]:
        raise ValueError("Dependency lock SHA256 mismatch")
    distributions = list(metadata.distributions())
    installed = {canonicalize_name(d.metadata["Name"]): d for d in distributions}
    pins = [
        Requirement(line)
        for line in lock.decode().splitlines()
        if line.strip() and not line.startswith(("#", "--"))
    ]
    for pin in pins:
        dist = installed.get(canonicalize_name(pin.name))
        if dist is None:
            raise ValueError(f"Missing pinned package: {pin}")
        if pin.url:
            url, revision = pin.url.removeprefix("git+").rsplit("@", 1)
            origin = json.loads(dist.read_text("direct_url.json") or "{}")
            if (
                origin.get("url") != url
                or origin.get("vcs_info", {}).get("commit_id") != revision
            ):
                raise ValueError(f"Wrong source revision: {pin.name}")
        elif not pin.specifier.contains(dist.version, prereleases=True):
            raise ValueError(f"Wrong package version: {pin}; installed {dist.version}")
    errors = dependency_errors(distributions, default_environment())
    if errors:
        raise ValueError("Dependency conflicts:\n" + "\n".join(errors))
    for name in ("proteinfoundation", "colabdesign", "ProteinMPNN"):
        spec = util.find_spec(name)
        paths = ([spec.origin] if spec and spec.origin else []) + list(
            spec.submodule_search_locations or [] if spec else []
        )
        if not paths or not all(
            Path(p).resolve().is_relative_to(repository) for p in paths
        ):
            raise ValueError(f"{name} resolves outside the selected backend checkout")
    for module in (
        "bioservices",
        "tensorstore",
        "orbax.checkpoint",
        "numpy",
        "scipy",
        "biotite",
        "jax",
        "flax",
        "optax",
        "chex",
        "haiku",
        "torch",
        "torchaudio",
        "torchvision",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "colabdesign",
        "ProteinMPNN.protein_mpnn_utils",
        "proteinfoundation.generate",
    ):
        import_module(module)
    command = [
        sys.executable,
        "-m",
        "proteinfoundation.generate",
        "--config-path",
        str(repository / "configs"),
        "--config-name",
        "search_binder_local_pipeline",
        "--help",
    ]
    result = subprocess.run(
        command, cwd=repository, capture_output=True, text=True, timeout=180
    )
    if result.returncode:
        raise ValueError("Complexa CLI/config check failed:\n" + result.stderr[-4000:])
    return {
        "installed_packages": len(distributions),
        "pinned_dependencies": len(pins),
        "dependency_conflicts": [],
        "backend_imports_and_config": "passed",
    }


def verify_gpu(checkpoint_root):
    import gc
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    import optax
    import chex
    import flax.linen as nn
    import orbax.checkpoint as ocp
    from torch_scatter import scatter_add

    if not torch.cuda.is_available() or not any(
        d.platform == "gpu" for d in jax.devices()
    ):
        raise ValueError("Both PyTorch and JAX must detect a GPU")
    x = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
    torch.testing.assert_close(x @ x.T, (x.cpu() @ x.cpu().T).cuda())
    torch.testing.assert_close(
        scatter_add(
            torch.ones(4, device="cuda"), torch.tensor([0, 0, 1, 1], device="cuda")
        ),
        torch.tensor([2.0, 2.0], device="cuda"),
    )
    torch.cuda.synchronize()
    xj = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    y = jax.jit(lambda v: v @ v.T)(xj)
    np.testing.assert_allclose(np.asarray(y), np.asarray(xj) @ np.asarray(xj).T)
    layer = nn.Dense(3)
    variables = layer.init(jax.random.PRNGKey(0), jnp.ones((2, 4)))
    jax.jit(layer.apply)(variables, jnp.ones((2, 4))).block_until_ready()
    optimizer = optax.adam(1e-3)
    p = jnp.ones(3)
    updates, _ = optimizer.update(
        jax.grad(lambda v: jnp.sum(v * v))(p), optimizer.init(p), p
    )
    chex.assert_tree_all_finite(optax.apply_updates(p, updates))
    with tempfile.TemporaryDirectory(prefix="trex-dependency-check-") as temporary:
        path = str(Path(temporary) / "roundtrip")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, {"value": np.arange(4)})
        np.testing.assert_array_equal(checkpointer.restore(path)["value"], np.arange(4))
        checkpointer.close()
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch_jax_flax_optax_orbax": "passed",
    }
    if checkpoint_root:
        from colabdesign.af.alphafold.model.data import get_model_haiku_params
        from proteinfoundation.proteina import Proteina

        params = get_model_haiku_params(
            "model_1_multimer_v3", str(checkpoint_root / "af2")
        )
        if not params:
            raise ValueError("AF2 parameter loading failed")
        report["af2_parameter_arrays"] = len(jax.tree_util.tree_leaves(params))
        del params
        gc.collect()
        model = Proteina.load_from_checkpoint(
            str(checkpoint_root / "complexa/complexa.ckpt"),
            map_location="cpu",
            strict=False,
            autoencoder_ckpt_path=str(checkpoint_root / "complexa/complexa_ae.ckpt"),
        )
        if model.autoencoder is None:
            raise ValueError("Complexa autoencoder did not load")
        model.eval().cuda()
        torch.cuda.synchronize()
        if not all(p.device.type == "cuda" for p in model.parameters()):
            raise ValueError("Complexa parameters are not on the GPU")
        report["complexa_generator_and_autoencoder_loading"] = "passed"
        report["complexa_parameter_count"] = sum(p.numel() for p in model.parameters())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Optional T-REX-assets/checkpoints directory; requires --gpu",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.checkpoint_root and not args.gpu:
        parser.error("--checkpoint-root requires --gpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if not args.gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        profile = json.loads(PROFILE.read_text())
        report = verify_install(args.repo.resolve(), profile)
        if args.gpu:
            report["gpu"] = verify_gpu(
                args.checkpoint_root.resolve() if args.checkpoint_root else None
            )
        report["ok"] = True
        report[
            "scope"
        ] = "Dependency/import/GPU/checkpoint-loading checks; no design campaign or scientific-result equivalence test."
    except Exception as exc:
        detail = getattr(exc, "stderr", "") or ""
        report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "detail": str(detail)[-4000:],
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
