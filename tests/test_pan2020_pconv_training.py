"""Training integration tests for Pan2020PConvUNet: forward/backward,
one-batch overfit, trainer compatibility with mask passing."""

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


class TestForwardBackward:
    def test_all_parameters_get_grad(self) -> None:
        model = build_model({
            "type": "pan2020_pconv_unet",
            "params": {"encoder_channels": [8, 16, 32, 64, 64, 64],
                       "decoder_channels": [64, 32, 16, 8, 4, 1],
                       "output_activation": "none"},
        })
        x = torch.rand(2, 1, 32, 32)
        mask = torch.ones(2, 1, 32, 32)
        mask[:, :, :, 16:] = 0.0
        out = model(x, mask=mask)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if "mask_kernel" in name:
                assert p.grad is None, f"{name} should have no grad"
            else:
                assert p.grad is not None, f"{name} has no grad"

    def test_mask_kernel_has_no_grad(self) -> None:
        model = build_model({
            "type": "pan2020_pconv_unet",
            "params": {"encoder_channels": [4, 8, 8, 8, 8, 8],
                       "decoder_channels": [8, 8, 4, 4, 2, 1]},
        })
        x = torch.rand(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        out = model(x, mask=mask)
        out.mean().backward()
        for name, buf in model.named_buffers():
            if "mask_kernel" in name:
                assert buf.grad is None


class TestOneBatchOverfit:
    def test_loss_decreases_on_single_batch(self) -> None:
        torch.manual_seed(42)
        model = build_model({
            "type": "pan2020_pconv_unet",
            "params": {"encoder_channels": [8, 16, 32, 32, 32, 32],
                       "decoder_channels": [32, 16, 8, 4, 2, 1],
                       "output_activation": "none"},
        })
        x = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        mask = torch.ones(2, 1, 16, 16)
        mask[:, :, :, 8:] = 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            out = model(x, mask=mask)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


class TestTrainerCompatibility:
    def test_signature_inspection_passes_mask(self) -> None:
        import inspect
        from utils.train_utils import unwrap_ddp

        model = build_model({
            "type": "pan2020_pconv_unet",
            "params": {"encoder_channels": [4, 8, 8, 8, 8, 8],
                       "decoder_channels": [8, 8, 4, 4, 2, 1]},
        })
        sig = inspect.signature(unwrap_ddp(model).forward)
        assert "mask" in sig.parameters
        x = torch.rand(2, 1, 16, 16)
        mask = torch.ones(2, 1, 16, 16)
        extras = {"mask": mask}
        pred = model(x, **extras)
        assert pred.shape == (2, 1, 16, 16)

    def test_standard_model_ignores_mask(self) -> None:
        import inspect
        from utils.train_utils import unwrap_ddp
        from model.interpolation.unet import UNet

        model = UNet(in_channels=1, out_channels=1)
        sig = inspect.signature(unwrap_ddp(model).forward)
        assert "mask" not in sig.parameters
        x = torch.rand(2, 1, 16, 16)
        pred = model(x)
        assert pred.shape == (2, 1, 16, 16)


class TestInputModes:
    def test_separate_mode(self) -> None:
        model = build_model({
            "type": "pan2020_pconv_unet",
            "params": {"input_mode": "separate",
                       "encoder_channels": [4, 8, 8, 8, 8, 8],
                       "decoder_channels": [8, 8, 4, 4, 2, 1]},
        })
        x = torch.rand(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        out = model(x, mask=mask)
        assert out.shape == (1, 1, 16, 16)

    def test_packed_mode(self) -> None:
        model = build_model({
            "type": "pan2020_pconv_unet",
            "params": {"input_mode": "packed",
                       "encoder_channels": [4, 8, 8, 8, 8, 8],
                       "decoder_channels": [8, 8, 4, 4, 2, 1]},
        })
        x = torch.rand(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        packed = torch.cat([x, mask], dim=1)
        out = model(packed)
        assert out.shape == (1, 1, 16, 16)
