# External backends

Third-party repositories are intentionally not vendored here; only the tested
patch is tracked in Git. Place local checkouts here or point the `TREX_*`
environment variables at installations elsewhere.

Expected default layout:

```text
external/
  Proteina-Complexa/
  BindCraft/
  BoltzGen/
  patches/
```

Backend repositories, environments, model weights, and target assets are not
redistributed by T-REX and retain their own licenses.

See [installation](../docs/installation.md) for the recorded revisions,
checkpoint locations and configuration variables.
The pinned public Complexa checkout also supplies the study's ColabDesign and
ProteinMPNN source; a second community checkout is not required.
