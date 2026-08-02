"""Training integration tests for Liu2022WRDL: forward/backward, overfit, trainer compatibility."""

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

from model.interpolation import build_model  # noqa: E402
from model.interpolation.liu2022_wrdl import (  # noqa: E402
    Liu2022WRDL,
    FixedWaveletDWT2D,
    FixedWaveletIWT2D,
)


class TestForwardBackward:
    def test_all_conv_params_get_grad(self):
        model = Liu2022WRDL(encoder_channels=[8, 16, 32], bottleneck_blocks=1)
        x = torch.rand(2, 1, 32, 32)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_dwt_filters_have_no_grad(self):
        model = Liu2022WRDL(encoder_channels=[8, 16, 32])
        x = torch.rand(2, 1, 32, 32)
        out = model(x)
        out.mean().backward()
        assert model.dwt.filters.grad is None
        assert model.iwt.filters.grad is None


class TestOneBatchOverfit:
    def test_loss_decreases_on_single_batch(self):
        torch.manual_seed(42)
        model = Liu2022WRDL(encoder_channels=[8, 16], bottleneck_blocks=1)
        x = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_overfit_with_wrdl_loss(self):
        torch.manual_seed(42)
        from utils.losses import WRDLSSIMHuberLoss

        model = Liu2022WRDL(encoder_channels=[8, 16], bottleneck_blocks=1)
        x = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=1.0, huber_delta=1.0)
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


class TestTrainerCompatibility:
    def test_signature_has_no_mask_kwarg(self):
        """WRDL does not accept mask — it expects zero-filled missing data only."""
        import inspect
        from utils.train_utils import unwrap_ddp

        sig = inspect.signature(unwrap_ddp(Liu2022WRDL(encoder_channels=[8, 16])).forward)
        assert "mask" not in sig.parameters

    def test_accepts_kwargs(self):
        """Extra kwargs are absorbed via **kwargs."""
        model = Liu2022WRDL(encoder_channels=[8, 16])
        x = torch.rand(1, 1, 16, 16)
        out = model(x, mask=torch.ones(1, 1, 16, 16), extra=42)
        assert out.shape == (1, 1, 16, 16)

    def test_trainer_inference_via_signature(self):
        """Simulate what trainer does: signature inspection, no mask passing."""
        import inspect
        from utils.train_utils import unwrap_ddp

        model = Liu2022WRDL(encoder_channels=[8, 16])
        sig = inspect.signature(unwrap_ddp(model).forward)
        extras = {"mask": torch.ones(2, 1, 16, 16)}
        x = torch.rand(2, 1, 16, 16)

        if "mask" in sig.parameters:
            pred = model(x, **extras)
        else:
            pred = model(x)
        assert pred.shape == (2, 1, 16, 16)


class TestSerialization:
    def test_save_load_roundtrip(self):
        torch.manual_seed(42)
        model = Liu2022WRDL(encoder_channels=[8, 16], bottleneck_blocks=1)
        model.eval()
        x = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            out1 = model(x)

        state = model.state_dict()

        torch.manual_seed(99)
        model2 = Liu2022WRDL(encoder_channels=[8, 16], bottleneck_blocks=1)
        model2.load_state_dict(state)
        model2.eval()
        with torch.no_grad():
            out2 = model2(x)
        torch.testing.assert_close(out1, out2)


class TestNoNaN:
    def test_output_no_nan(self):
        model = Liu2022WRDL(encoder_channels=[8, 16, 32])
        model.eval()
        with torch.no_grad():
            x = torch.rand(2, 1, 64, 64)
            out = model(x)
        assert torch.isfinite(out).all()

    def test_gradient_no_nan(self):
        model = Liu2022WRDL(encoder_channels=[8, 16], bottleneck_blocks=1)
        x = torch.rand(2, 1, 32, 32)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name} has NaN/Inf grad"
