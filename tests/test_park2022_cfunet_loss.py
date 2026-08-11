"""Park2022 CFunet loss tests: Eq. 6 MSE, Eq. 8 complex Fourier L1, alpha scaling, gradients, attach protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation.park2022_cfunet import Park2022CFUNet  # noqa: E402
from utils.losses import CFunetMSEFourierLoss, FourierL1Loss  # noqa: E402


def _small_model():
    return Park2022CFUNet(
        in_channels=1, out_channels=1, base_channels=8, num_levels=3,
        upsample_mode="fourier_zero_padding",
    )


class TestFourierL1Standalone:
    def test_matches_manual_computation(self):
        pred = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = FourierL1Loss()
        manual = (torch.fft.fft2(pred) - torch.fft.fft2(target)).abs().sum(dim=(1, 2, 3)).mean()
        torch.testing.assert_close(loss(pred, target), manual)

    def test_complex_difference_not_magnitude_only(self):
        """A phase flip changes the complex difference but not the magnitudes."""
        target = torch.randn(1, 1, 16, 16)
        pred = -target  # |F(pred)| == |F(target)|, complex difference nonzero
        loss = FourierL1Loss()
        assert float(loss(pred, target)) > 0.0

    def test_identical_inputs_zero(self):
        x = torch.randn(2, 1, 16, 16)
        assert float(FourierL1Loss()(x, x)) == 0.0

    def test_reduction_sum(self):
        pred = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss_sum = FourierL1Loss(reduction="sum")
        per_sample = (torch.fft.fft2(pred) - torch.fft.fft2(target)).abs().sum(dim=(1, 2, 3))
        torch.testing.assert_close(loss_sum(pred, target), per_sample.sum())


class TestCFunetMSEFourierLoss:
    def _loss_and_model(self, alpha=1.0, **model_kwargs):
        loss = CFunetMSEFourierLoss(alpha=alpha)
        model = _small_model()
        loss.attach_model(model)
        return loss, model

    def test_requires_attach(self):
        loss = CFunetMSEFourierLoss()
        pred = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        with pytest.raises(RuntimeError, match="attach_model"):
            loss(pred, target)

    def test_mse_terms_match_eq6(self):
        loss, model = self._loss_and_model(alpha=0.0)
        x = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        pred = model(x)
        comp = loss.components(pred, target)
        oc = model._intermediates["coarse"]
        mse_coarse = (oc - target).square().sum(dim=(1, 2, 3)).mean()
        mse_final = (pred - target).square().sum(dim=(1, 2, 3)).mean()
        torch.testing.assert_close(comp["loss_mse"], (mse_coarse + mse_final) / 2.0)
        torch.testing.assert_close(comp["loss_total"], comp["loss_mse"])

    def test_alpha_scaling(self):
        loss, model = self._loss_and_model(alpha=0.5)
        x = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        pred = model(x)
        comp = loss.components(pred, target)
        torch.testing.assert_close(
            comp["loss_total"],
            comp["loss_mse"] + 0.5 * comp["loss_fourier"],
        )

    def test_fourier_term_uses_final_output_only(self):
        """Eq. 8: the Fourier term compares F(Of) to F(L), not the coarse output."""
        loss, model = self._loss_and_model(alpha=1.0)
        x = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        pred = model(x)
        comp = loss.components(pred, target)
        manual = (torch.fft.fft2(pred) - torch.fft.fft2(target)).abs().sum(dim=(1, 2, 3)).mean()
        torch.testing.assert_close(comp["loss_fourier"], manual)

    def test_gradients_flow_through_coarse_and_fourier(self):
        loss, model = self._loss_and_model(alpha=1.0)
        model.train()
        x = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        pred = model(x)
        loss(pred, target).backward()
        named = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert named, "model has no trainable parameters"
        for name, p in named.items():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all()

    def test_alpha_zero_disables_fourier_gradient(self):
        """alpha=0 must make the Fourier term exactly zero (Eq. 7 ablation)."""
        loss, model = self._loss_and_model(alpha=0.0)
        x = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        pred = model(x)
        comp = loss.components(pred, target)
        assert float(comp["loss_fourier"]) > 0.0  # term computed...
        torch.testing.assert_close(comp["loss_total"], comp["loss_mse"])  # ...but not added

    def test_missing_intermediates_raises(self):
        loss = CFunetMSEFourierLoss(alpha=1.0)
        model = _small_model()
        loss.attach_model(model)
        pred = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        with pytest.raises(RuntimeError, match="coarse"):
            loss(pred, target)
