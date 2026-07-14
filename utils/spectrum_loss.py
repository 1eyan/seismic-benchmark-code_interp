"""F-K (frequency-wavenumber) domain losses for seismic interpolation.

Used by the BTN-SS (Spectrum Suppression) training variant from
Yuan et al. 2022 to penalise energy outside the physically plausible
signal cone in the 2D Fourier domain.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .losses import BaseLoss, register_loss


def _fk_signal_cone_mask(
    n_freq: int,
    n_wavenum: int,
    dt: float,
    dx: float,
    v_min: float,
    taper_width: int = 4,
    f_high: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build a soft (tapered) mask for the F-K domain signal cone.

    The signal cone in the F-K plane is defined by |k| < |f| / v_min.
    Energy *outside* this cone (and above ``f_high``) is considered
    non-physical and should be suppressed.

    Returns a mask of shape ``(n_freq, n_wavenum)`` where 0 = inside the
    signal cone (keep) and 1 = outside (suppress), with a smooth
    transition of width ``taper_width`` bins.

    Parameters
    ----------
    n_freq, n_wavenum : int
        F-K grid dimensions (after rfft: n_freq = n_time // 2 + 1).
    dt : float
        Time sampling interval (seconds).
    dx : float
        Trace spacing (metres).
    v_min : float
        Minimum apparent velocity (m/s) defining the cone boundary.
    taper_width : int
        Number of frequency bins for the soft transition.
    f_high : float, optional
        If given, additionally suppress energy above this frequency (Hz).
    """
    # Frequency axis (positive half only, matching rfft)
    freq = torch.arange(n_freq, device=device, dtype=dtype) / (dt * (2 * (n_freq - 1)) if n_freq > 1 else 1.0)

    # Wavenumber axis (full, centered)
    k = torch.fft.fftfreq(n_wavenum, d=dx, device=device, dtype=dtype)
    k = torch.fft.fftshift(k)

    # |k| grid: (n_freq, n_wavenum)
    k_abs = k.abs().unsqueeze(0)  # (1, n_wavenum)
    f_abs = freq.abs().unsqueeze(1)  # (n_freq, 1)

    # Cone boundary: |k| = |f| / v_min
    k_cone = f_abs / max(v_min, 1e-6)

    # Hard mask: 0 inside cone, 1 outside
    hard = (k_abs > k_cone).to(dtype=dtype)

    # Soft taper using a Hann window in the frequency direction
    if taper_width > 0 and n_freq > 2 * taper_width:
        ramp = torch.linspace(0, 1, taper_width, device=device, dtype=dtype)
        win = 0.5 * (1.0 - torch.cos(torch.pi * ramp))

        soft = hard.clone()
        for i in range(n_freq):
            if hard[i].sum() == 0 or hard[i].sum() == n_wavenum:
                continue
            # Find the transition boundary indices for this frequency
            k_boundary = k_cone[i, 0].item()
            for j in range(n_wavenum):
                kj = k_abs[0, j].item()
                dist = kj - k_boundary
                if 0 <= dist < (taper_width * k_boundary / max(n_freq - 1, 1) * 2 + 1e-8):
                    bin_idx = min(int(dist / (k_boundary / max(n_freq - 1, 1) * 2 + 1e-8)), taper_width - 1)
                    soft[i, j] = win[max(0, min(taper_width - 1, bin_idx))]
    else:
        soft = hard

    # Optional high-frequency suppression
    if f_high is not None and f_high > 0:
        f_mask = (f_abs > f_high).to(dtype=dtype)
        soft = torch.maximum(soft, f_mask)

    return soft


@register_loss("fk_spectrum_suppression")
class FKSpectrumSuppressionLoss(BaseLoss):
    """Penalise F-K energy outside the physically plausible signal cone.

    Intended as a regularisation term for BTN-SS training (Yuan et al. 2022).
    Can be used standalone or combined with a data loss via
    ``WeightedCompositeLoss``.
    """

    def __init__(
        self,
        dt: float = 0.002,
        dx: float = 1.0,
        v_min: float = 1500.0,
        taper_width: int = 4,
        f_high: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.dx = float(dx)
        self.v_min = float(v_min)
        self.taper_width = int(taper_width)
        self.f_high = f_high if f_high is not None else None
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        """Compute mean suppressed energy fraction.

        Parameters
        ----------
        pred : (B, C, H, W) or (B, H, W)
            Predicted shot gather patches.
        target : Tensor, optional (unused; accepted for API compatibility)
        """
        if pred.dim() == 4:
            pred_2d = pred[:, 0, :, :] if pred.shape[1] == 1 else pred.mean(dim=1)
        elif pred.dim() == 3:
            pred_2d = pred
        else:
            raise ValueError(
                f"Expected pred with 3 or 4 dims, got shape {pred.shape}."
            )

        B, H, W = pred_2d.shape
        cone_mask = _fk_signal_cone_mask(
            n_freq=H // 2 + 1,
            n_wavenum=W,
            dt=self.dt,
            dx=self.dx,
            v_min=self.v_min,
            taper_width=self.taper_width,
            f_high=self.f_high,
            device=pred.device,
            dtype=pred.dtype,
        )

        total_energy = pred.new_tensor(0.0)
        suppressed_energy = pred.new_tensor(0.0)

        for b in range(B):
            fk = torch.fft.rfft2(pred_2d[b])
            fk_shifted = torch.fft.fftshift(fk, dim=-1)

            power = fk_shifted.abs().pow(2)
            total_energy = total_energy + power.sum()

            suppressed = power * cone_mask
            suppressed_energy = suppressed_energy + suppressed.sum()

        return suppressed_energy / total_energy.clamp_min(self.eps)
