"""BoltzGen input-spec and launch construction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .process_management import WorkerLaunch, isolated_subprocess_environment


def parse_hotspot(token: str, default_chain: str) -> tuple[str, int] | None:
    """Parse ``A123``, ``123``, or ``A:123`` into a chain and residue."""

    if not token:
        return None
    value = token.strip().replace(":", "")
    if not value:
        return None
    chain = default_chain
    number = value
    if value[0].isalpha():
        chain = value[0]
        number = value[1:]
    try:
        return chain, int(number)
    except ValueError:
        return None


def _chain_positions(target_pdb: str | Path) -> dict[tuple[str, int], int]:
    """Map raw PDB residue numbers to BoltzGen's chain-local positions."""

    positions: dict[tuple[str, int], int] = {}
    seen: set[tuple[str, int, str]] = set()
    counts: dict[str, int] = {}
    try:
        lines = Path(target_pdb).read_text(errors="ignore").splitlines()
    except OSError:
        return positions
    for raw in lines:
        if not raw.startswith("ATOM") or len(raw) < 27:
            continue
        chain = raw[21].strip()
        try:
            residue_number = int(raw[22:26].strip())
        except ValueError:
            continue
        insertion_code = raw[26].strip()
        residue = (chain, residue_number, insertion_code)
        if not chain or residue in seen:
            continue
        seen.add(residue)
        counts[chain] = counts.get(chain, 0) + 1
        positions.setdefault((chain, residue_number), counts[chain])
    return positions


def write_boltzgen_yaml(
    out_dir: Path,
    target_id: str,
    target_pdb: str,
    hotspots: list[str],
    chain_ids: list[str],
    binder_chain: str,
    length_range: tuple[int, int],
) -> Path:
    """Write the reviewable BoltzGen design spec for one target.

    ``target_id`` remains in this stable interface for compatibility and for
    future metadata support, although BoltzGen's current schema does not use it.
    """

    del target_id
    target_chains = [chain for chain in chain_ids if chain] or ["A"]
    target_chain = target_chains[0]
    lower, upper = length_range
    lines = [
        "entities:",
        "  - protein:",
        f"      id: {binder_chain}",
        f"      sequence: {lower}..{upper}",
        "  - file:",
        f"      path: {target_pdb}",
        "      include:",
    ]
    for chain in target_chains:
        lines.extend(("        - chain:", f"            id: {chain}"))

    positions = _chain_positions(target_pdb)
    by_chain: dict[str, list[int]] = {}
    for hotspot in hotspots or []:
        parsed = parse_hotspot(hotspot, target_chain)
        if parsed is None:
            continue
        chain, residue_number = parsed
        chain_position = positions.get((chain, residue_number))
        if chain_position is None:
            print(
                f"  [boltzgen] WARN hotspot {hotspot} not found in target PDB; "
                "skipping",
                flush=True,
            )
            continue
        by_chain.setdefault(chain, []).append(chain_position)
    if by_chain:
        lines.append("      binding_types:")
        for chain, chain_positions in by_chain.items():
            binding = ",".join(str(position) for position in chain_positions)
            lines.extend(
                (
                    "        - chain:",
                    f"            id: {chain}",
                    f'            binding: "{binding}"',
                )
            )
        lines.append('      structure_groups: "all"')

    spec_dir = out_dir / "boltzgen_spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / "design.yaml"
    spec_path.write_text("\n".join(lines) + "\n")
    return spec_path


def prepare_boltzgen_launch(
    *,
    output_dir: Path,
    target_id: str,
    target_pdb: str,
    hotspots: list[str],
    chain_ids: list[str],
    binder_chain: str,
    length_range: tuple[int, int],
    gpu_id: str,
    binary: Path,
    repo: Path,
    cache: Path,
    config_delta: Mapping[str, object] | None,
) -> WorkerLaunch:
    """Write a BoltzGen spec and resolve its worker command and environment."""

    spec_path = write_boltzgen_yaml(
        output_dir,
        target_id,
        target_pdb,
        hotspots,
        chain_ids,
        binder_chain,
        length_range,
    )
    delta = config_delta or {}
    argv = [
        str(binary),
        "run",
        str(spec_path),
        "--output",
        str(output_dir / "boltzgen"),
        "--protocol",
        str(delta.get("protocol", "protein-anything")),
        "--num_designs",
        str(int(delta.get("num_designs", 16))),
        "--budget",
        str(int(delta.get("budget", 4))),
        "--diffusion_batch_size",
        str(int(delta.get("diffusion_batch_size", 4))),
        "--devices",
        "1",
        "--use_kernels",
        "auto",
        "--reuse",
    ]
    cache_exists = cache.exists()
    if cache_exists:
        argv.extend(("--cache", str(cache)))
    if delta.get("step_scale") is not None:
        argv.extend(("--step_scale", str(delta["step_scale"])))
    if delta.get("noise_scale") is not None:
        argv.extend(("--noise_scale", str(delta["noise_scale"])))

    environment = isolated_subprocess_environment(binary.parent.parent)
    if cache_exists:
        environment.update(
            {
                "HF_HOME": str(cache),
                "HF_HUB_CACHE": str(cache),
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "BOLTZGEN_CACHE": str(cache),
            }
        )
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return WorkerLaunch(
        argv=tuple(argv),
        output_dir=output_dir / "boltzgen",
        environment=environment,
        cwd=repo,
    )


__all__ = ["parse_hotspot", "prepare_boltzgen_launch", "write_boltzgen_yaml"]
