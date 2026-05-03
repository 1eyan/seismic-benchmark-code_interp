# Research First

> Log of **open-source surveys and alignment notes**: search keywords, candidate repos, selection rationale, and deltas from the reference. Pairs with `.cursor/rules/research-first.mdc`.

## Suggested fields

Each entry should include:

- **Need** — the problem to solve.
- **Search keywords** — for later re-verification.
- **Candidates** — repo URL + commit / version.
- **Selection** — which one was adopted and why.
- **Deltas** — project-specific adaptations.

## Entry template

```markdown
## YYYY-MM-DD - Topic
- Need:
- Search keywords:
- Candidates:
- Selection:
- Deltas:
```

---

## 2026-04-22 - SEG-Y reader library selection
- Need: Read pre-stack SEG-Y files from ``/data/shared/SEGC3/`` and split them into shot gathers.
- Search keywords: "python segy reader", "segyio", "obspy segy", "seismic shot gather python".
- Candidates:
  - [segyio](https://github.com/equinor/segyio) — Equinor, the de facto open-source SEG-Y library. Fast C core, clean trace-attribute API, actively maintained.
  - [obspy](https://github.com/obspy/obspy) — broader seismology toolkit; SEG-Y support is fine but the package is heavy and optimized for earthquake seismology, not pre-stack exploration data.
  - Hand-rolled ``struct``-based parser — rejected; the IBM float / big-endian handling and header scaling already solved by segyio would need re-implementation.
- Selection: ``segyio``. Lightweight, exposes trace headers as NumPy arrays via ``f.attributes(segyio.TraceField.<name>)[:]`` which is exactly what shot splitting by FFID needs.
- Deltas: Keep coordinate scaling (``SourceGroupScalar``) out of the reader for now — returned headers are raw integers. Downstream code can apply scaling when it actually needs physical units.

---

## 2026-04-26 - DnCNN baseline reference
- Need: Register a standard DnCNN denoiser alongside the existing UNet for restoration benchmarks.
- Search keywords: "DnCNN pytorch", "Zhang DnCNN residual".
- Candidates:
  - Zhang et al., *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising* (TIP 2017) — defines depth-17 / 64-channel architecture and residual learning `x - R(x)`.
- Selection: Reimplemented from the paper description to match the project's `register_model` + YAML `params` pattern (no third-party fork vendored).
- Deltas: Hyper-parameters named to align with `UNet` (`in_channels`, `out_channels`, `base_channels`, `depth`); `in_channels` and `out_channels` must match for the residual shortcut.

---

<!-- Append new entries below this line -->
