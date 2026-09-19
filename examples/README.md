# Examples

Run example commands from the repository root after installing T-REX.

| Example | Purpose | External requirements |
| --- | --- | --- |
| [Analysis demo](analysis_demo/README.md) | Create, validate, inspect and export a small synthetic archive | None; CPU only |
| [Campaign YAML](campaign.yaml) | Explicit campaign configuration with the standard defaults | Configured scientific backends, target assets, LLM and GPU allocation |
| [Custom target](custom_target/README.md) | Specify a target outside the built-in registry | Your target structure |

The analysis demo is an interface smoke, not a short molecular campaign. The
campaign YAML retains the default 48-hour launch window and three worker slots.
Changing a budget or enabled families defines a different campaign profile.
Use `trex campaign show` to inspect resolved values before launching.

For a configuration-only check with your staged target:

```bash
trex campaign init campaign.yaml --target cd45 --archive-root ./runs/cd45
# Edit the target and environment paths before continuing.
trex campaign show campaign.yaml
trex campaign run campaign.yaml --dry-run
```

`--dry-run` does not launch workers or create the campaign archive. It permits
missing external backends and does not establish production readiness. Before
a real run, use the strict preflight in the [README](../README.md#run-a-campaign)
and confirm that your external environments work on the allocated GPU node.
