# Techniques

> Tracks the **landed techniques** of the project (algorithms, training tricks, data-processing recipes). Record only what is already implemented and validated; put experimental ideas in `research_first.md`.

## Suggested fields

Each technique entry should include:

- **Name** — short and searchable.
- **Use case** — what problem it solves.
- **Location** — path(s) in the codebase.
- **Reference** — paper or open-source repository.
- **Known limits** — boundary conditions and caveats.

## Entry template

```markdown
## YYYY-MM-DD - Technique name
- Use case:
- Location:
- Reference:
- Known limits:
```

---

<!-- Append new entries below this line -->

## 2026-06-30 - Energy-Binned Weak Signal Evaluation (EB-WSE)
- Use case: Diagnose weak-signal loss that global MSE/SNR can hide by computing NE and SNR inside reference-energy percentile bins.
- Location: `utils/eb_wse_metrics.py`, `utils/inference_utils.py::compute_binned_metrics`, `docs/metrics/eb_wse.md`.
- Reference: Derived from the original `test/energy_binned_metrics.py` prototype; uses `scipy.ndimage.gaussian_filter` for the energy map.
- Known limits:
  - Zero-valued reference samples are excluded from binning.
  - Bin membership depends on the reference energy map and `smooth_sigma` (default `1.0`).
  - Inference reports mean-over-shots NE/SNR only; per-sample diagnostics inside the module are not exposed in `metrics_summary.json`.

## 2026-06-30 - Frequency-Binned Fidelity and Recovery Evaluation (FB-FRE)
- Use case: Diagnose frequency-specific reconstruction quality by estimating an effective band from the reference spectrum, splitting it into adaptive low/mid/high/very_high bands, and computing NE/SNR per band.
- Location: `utils/fb_fre_metrics.py`, `utils/inference_utils.py::compute_binned_metrics`, `docs/metrics/fb_fre.md`.
- Reference: Derived from the original `test/frequency_binned_metrics.py` prototype; uses `numpy.fft.rfft`/`irfft` for band-pass filtering.
- Known limits:
  - Effective band and sub-band edges are data-dependent and vary with `dt`.
  - Default `rel_threshold=0.001` (0.1% of peak power); tails below this threshold are ignored.
  - Rectangular band masks (`taper_width=0`) are used in inference; this can introduce time-domain ringing compared with tapered masks.
  - Inference reports NE, SNR, energy ratio, and Hz range only; BER and BCC from the full module are not exposed.

