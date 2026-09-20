# T-REX

**Target-adaptive Rescue–Explore–eXploit** is an agentic campaign controller for
de novo protein binder design. It orchestrates generation, redesign and
evaluation jobs for one target under a finite compute budget.

Planner and Supervisor LLMs interpret campaign evidence and propose or prioritize
follow-up tests and candidate jobs. Fixed code constructs and validates candidate
jobs, enforces resource constraints and manages asynchronous execution.
Rescue addresses identified weaknesses, Explore tests alternative methods or
settings, and eXploit continues productive routes.

## Install

Use Python 3.10–3.12. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
trex --help
```

This links the controller and analysis commands to your checkout. A real campaign needs
GPU resources, external scientific backends, target structures and a configured
LLM endpoint. Follow the [installation guide](docs/installation.md) to prepare
those dependencies. Developer tests are described [separately](docs/development.md).

## Try the analysis example — CPU only

```bash
python examples/analysis_demo/run_demo.py --out ./demo-output
trex-analyze summary --archive-root ./demo-output/archive
trex-analyze trace --archive-root ./demo-output/archive --limit 10
```

The example creates a small **synthetic** archive, validates it and exports
CSV/JSON manifests and a toy PDB. It requires no GPU, LLM server, model weights
or clustering binaries. Its numbers and coordinates are test fixtures, not
scientific results. See [expected outputs](examples/analysis_demo/README.md).

## Inputs

| Input | What to provide |
| --- | --- |
| Target | Structure file and target constraints, or a registered target plus its asset root |
| Campaign | YAML specifying the target, enabled action families, budget, worker GPUs, seed and output directory |
| Environment | Paths to backend executables and model assets, plus the LLM endpoint |

See the [input reference](docs/inputs.md) and
[manuscript-to-code glossary](docs/terminology.md). YAML paths are resolved
relative to the YAML file. Environment variables are exported explicitly;
`.env` files are not loaded automatically.

## Run a campaign

After [installing the external dependencies](docs/installation.md), create a
configuration from the repository root:

```bash
trex campaign init campaign.yaml --target cd45 --archive-root ./runs/cd45
```

Edit the target asset root, backend/model paths, LLM endpoint and worker GPUs.
The generated template retains the paper-oriented defaults: a 48-hour
controller window, three worker slots and all eight action families. The
[configuration guide](docs/campaign-interface.md) explains every field;
[examples](examples/README.md) distinguish configuration inspection from an
actual campaign.

```bash
trex campaign show campaign.yaml
trex campaign preflight campaign.yaml \
  --strict --require-model --verify-backend-revisions
trex campaign run campaign.yaml --verify-backend-revisions
```

Run the last command only inside your allocated GPU environment with the LLM
server already running. It does not request GPUs or start an LLM server.
For Slurm setup, launch and resume instructions, use the [Slurm guide](docs/slurm.md).
Preflight checks configuration and assets; it does not execute a scientific backend.

Monitor from another terminal:

```bash
trex campaign status ./runs/cd45
```

## Outputs and analysis

| Output | Purpose |
| --- | --- |
| Campaign archive (`*.jsonl`) | Results, evidence, hypotheses, candidate jobs and execution decisions |
| `run_provenance*.json` | Source, model, target, environment and effective run settings |
| `worker_outputs/` | Per-job structures, scores and logs, referenced by archive records |
| Export `manifest.csv`, `manifest.json`, `pdbs/` | Ranked designs and copied structures |

```bash
trex-analyze validate --archive-root ./runs/cd45
trex-analyze summary --archive-root ./runs/cd45 --json
trex-analyze trace --archive-root ./runs/cd45 --limit 10 --json
```

The summary reads the latest recorded evidence. Final paper SU counts require
the specified post-hoc analysis. A qualified design and a structurally distinct
hit (SU) are different quantities; missing measurements are not zero successes
or measured failures. The live rate uses measured worker-wall GPU-hours.

Use `trex-export` for best-N qualified designs by score, or `trex-panel` for a
structurally deduplicated panel. Their selection rules differ. See
[outputs and analysis](docs/outputs-and-analysis.md) for complete commands,
file schemas and metric interpretation.

## Documentation and paper data

- [Installation](docs/installation.md) · [Inputs](docs/inputs.md) · [Campaign configuration](docs/campaign-interface.md)
- [Outputs and analysis](docs/outputs-and-analysis.md) · [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md) · [Terminology](docs/terminology.md) · [Development](docs/development.md)
- [Paper reproduction](docs/benchmark-reproduction.md) · [Reproducibility contract](docs/reproducibility.md)

Audit the checked-in seven-target source tables with:

```bash
python benchmarks/paper_7target_144h/verify.py
```

This checks hashes and calculations. The raw-archive DOI is pending; complete
reanalysis from raw structures requires the deposit described in the
[paper reproduction guide](docs/benchmark-reproduction.md).

## Citation and license

See [CITATION.cff](CITATION.cff). Cite the external tools and models used in your
campaign as well. T-REX code uses the [MIT License](LICENSE); external software,
weights and datasets retain their respective licenses. Computational
qualification does not establish experimental binding.
