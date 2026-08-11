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


## 2026-08-11 - Fourier zero-padding upsampling + complex-spectral L1 loss (Park2022 CFunet)
- Use case: 2x (both-axes) upsampling that preserves high-frequency content
  and a frequency-domain loss that penalises phase errors for seismic
  trace interpolation.
- Location: `model/interpolation/park2022_cfunet.py::FourierZeroPaddingUpsample2D`,
  `utils/losses.py::FourierL1Loss` / `CFunetMSEFourierLoss`.
- Reference: Park et al. 2022 (IEEE TGRS 60, 5917615), Eqs. 3-4 (zero-padding)
  and Eqs. 6-8 (loss). Amplitude-correction factors derived from torch.fft
  norm conventions: backward x s^2, ortho x s, forward x 1 (aligned samples
  exact on the aligned grid; interpolation points follow the sinc kernel).
- Known limits:
  - Requires even input sizes (symmetric padding is not integer for odd).
  - The paper is silent on FFT normalisation and amplitude correction; the
    "scale" default is a reproduction assumption.
  - The Fourier loss uses per-sample sum + batch mean reduction, so its scale
    depends on patch size; alpha must be tuned in that scale (paper: alpha=1
    synthetic, alpha=0.1 field).
