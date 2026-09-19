# External backends

Third-party repositories are intentionally not vendored here; only the tested
patch is tracked in Git. Place local checkouts here or point the `TREX_*`
environment variables at installations elsewhere.

Expected default layout:

```text
external/
  Proteina-Complexa/
  Proteina-Complexa-community/
  BindCraft/
  BoltzGen/
  patches/
```

Backend repositories, environments, model weights, and target assets are not
redistributed by T-ReX and retain their own licenses.
