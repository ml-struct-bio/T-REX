# External backend contract

T-ReX owns campaign control, evidence reduction, action validation, dispatch,
canonical scoring policy, and archive semantics. It does not fork or vendor the
scientific generators.

## Roles

| Component | T-ReX role | Native output status |
| --- | --- | --- |
| Proteina-Complexa | de novo generation with registered search/configuration controls | Native AF2-like metrics are parsed; canonical strict fields determine direct credit where available. |
| BindCraft | hallucination-based generation and native filtering | Accepted and rejected outputs are diagnostic until the canonical score-conversion path supplies the fixed strict metrics. |
| BoltzGen | diffusion-based generation | Generated outputs are diagnostic until canonical score conversion. |
| ProteinMPNN | parent-specific sequence redesign | Redesign output requires canonical score conversion. |
| AF2 refilter | canonical score conversion or an explicitly marked advisory parent refold | Only the canonical role can mint strict/SU credit. |
| Foldseek | binder-chain structural clustering | Trusted SU requires successful clustering and coverage. |
| MMseqs2 | sequence clustering | Secondary evidence only; it does not change SU. |

## Tested revisions

Use `config/reproducibility/production_stack.json`. The Slurm preflight checks
the configured Git HEADs when `TREX_VERIFY_BACKEND_REVISIONS=1`.

AF2 parameter and ProteinMPNN weight manifests are stored beside that stack
manifest. Regular startup validates manifest integrity, file names, and byte
sizes; `trex-validate --verify-checkpoint-content` additionally rehashes every
declared byte for a release audit.

The BindCraft patch changes the executable mode bits of DSSP and DAlphaBall.
The tested campaign otherwise used the repository-provided hard-target preset
unchanged; adaptive actions could override only the registered fields. Apply
the mode patch after checking out the pinned commit:

```bash
git -C external/BindCraft apply \
  "$(pwd)/external/patches/bindcraft-production.patch"
```

`git status` will then be dirty by design. Each production
`run_provenance.json` records both the Git commit and tracked-diff digest, and
stores a patch artifact beside the archive.

## Enabled-family requirements

Preflight is conditional:

- any `complexa_*` family requires Proteina-Complexa, its Python, `env.sh`, and
  generation checkpoints;
- `bindcraft` requires the BindCraft checkout and Python;
- `boltzgen` requires the BoltzGen checkout and executable;
- `structure_refilter` requires AF2 parameters and the configured
  Complexa/AF2 Python;
- `proteinmpnn_redesign` additionally requires ProteinMPNN weights.

Foldseek is always required for the primary objective. MMseqs2 is optional.

## Environment isolation

The controller intentionally launches backend-specific executables. It strips
cluster-level CUDA library paths for backends that ship their own CUDA/JAX
runtime, preventing an inherited module from silently loading an incompatible
cuDNN/cuBLAS. Do not activate all backend environments in the controller
shell.

## Updates

New backend releases are not automatically equivalent. To update one:

1. create a new production-stack manifest;
2. run its official smoke tests;
3. run T-ReX parser/dispatch integration tests;
4. verify strict-score and artifact semantics;
5. run a bounded campaign smoke; and
6. report the new revision as a distinct experiment.
