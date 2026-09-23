# Reproducibility

## Identify a run

Keep `run_provenance.json`, the complete campaign archive and its referenced
structures. Provenance records source, backend revisions, model and input
identities, software versions and effective run settings. The complete runtime
prompt catalog is [all_prompts_snapshot.txt](all_prompts_snapshot.txt); regenerate
it from the installed source with `python docs/all_prompts.py`.

The installation profiles pin compatible dependencies for separate environments.
They are distinct from the study software inventory in
[production_stack.json](../config/reproducibility/production_stack.json).
The serving and Complexa profile manifests record the dependency differences.
Use run-specific provenance when comparing experiments: matching checkpoints
and seeds alone does not guarantee identical stochastic or asynchronous results.

## Configuration and output semantics

The default public launcher uses a 48-hour campaign limit and a 49-hour Slurm
reservation. Effective study-run durations come from each run's provenance;
the study source launcher defaults to 47 and 48 hours, respectively. The paper's
144-worker-GPU-hour reporting denominator is separate from measured runtime.

BindCraft's controller field `max_trajectories` is forwarded as
`number_of_final_designs`: the requested count of designs passing native
BindCraft filters, not a limit on attempted trajectories. Native acceptance is
distinct from T-REX qualification and SU counting. Runtime limits can stop a
job before its requested count is reached.

Output binder/target roles are verified before clustering, export and panel
selection; unresolved identities are excluded. This behavior can change SU
counts and subsequent allocation relative to a study-source interpretation of
the same output. The public package also has distinct prompt wording and
installation profiles; it is not guaranteed to reproduce identical study outputs.

`trex-analyze summary` reports the latest online evidence. A best-N export or
limited panel is not the full campaign endpoint. See
[outputs and analysis](outputs-and-analysis.md) for record joins and interpretation.

## Validation scope

Dependency checks, checkpoint loading, model inference and a completed campaign
are separate checks. Installation and representative inference have been tested;
BindCraft's full native accepted-design quota has not been validated. These
checks do not establish every configuration's compatibility, scientific
performance, a complete 48-hour benchmark reproduction or identical outputs.
Machine-readable profile and validation scope are retained in the asset
collection's `VERSIONS.json`.

Reconstructing the paper's final tables additionally requires its frozen campaign
archives and endpoint/post-hoc analyses. The public input collection is not the
complete benchmark input set; see [asset coverage](assets.md).
Contributor tests and release checks are described in [CONTRIBUTING.md](../CONTRIBUTING.md).
