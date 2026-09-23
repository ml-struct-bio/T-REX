# Examples

Run example commands from the repository root after installing T-REX.

| Example | Purpose | External requirements |
| --- | --- | --- |
| [Analysis demo](analysis_demo/README.md) | Create, validate, inspect and export a small synthetic archive | None; CPU only |
| [Installation profile](../.env.example) | Record installed environments and executables | Installed backends/assets; follow the [run guide](../docs/slurm.md) |
| [Campaign YAML](campaign.yaml) | Explicit campaign configuration with the standard defaults | Configured scientific backends, target assets, LLM and GPU allocation |
| [Custom target](custom_target/README.md) | Specify a target outside the built-in registry | Your target structure |

The analysis demo is an interface smoke, not a short molecular campaign. The
campaign YAML retains the default 48-hour launch window and three worker slots.
Changing a budget or enabled families defines a different campaign profile.
Create and validate a campaign with the public CLI:

```bash
trex init campaign.yaml --target cd45 --gpus 4 --hours 48 --output ./runs/cd45
trex check campaign.yaml
trex submit campaign.yaml
```

Use `trex design campaign.yaml --dry-run` for a read-only resolution check. A
real `trex design` run expects an existing GPU allocation and LLM endpoint;
`trex submit` starts the local LLM inside a new Slurm allocation. The nested
`trex campaign ...` commands remain available as the detailed compatibility API.
