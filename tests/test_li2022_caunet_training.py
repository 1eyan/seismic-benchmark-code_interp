"""Training-behavior tests for Li2022CAUNet: gradients, one-batch overfit,
trainer compatibility, and mask integration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation.li2022_caunet import Li2022CAUNet  # noqa: E402
from utils.losses import ANetSSIML1Loss, build_loss  # noqa: E402
from utils.train_utils import train_one_epoch  # noqa: E402


def _small_model() -> Li2022CAUNet:
    torch.manual_seed(0)
    return Li2022CAUNet(base_channels=8, depth=2)


class TestForwardBackward:
    def test_all_parameters_receive_finite_gradients(self) -> None:
        model = _small_model()
        loss_fn = ANetSSIML1Loss()
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        loss = loss_fn(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"

    def test_ca_parameters_receive_gradients(self) -> None:
        model = _small_model()
        x = torch.rand(1, 1, 16, 16)
        model(x).sum().backward()
        # Every CA block's parameters must receive gradients.
        for m in model.modules():
            if m.__class__.__name__ == "CoordAttention2D":
                for name, p in m.named_parameters():
                    assert p.grad is not None, f"no gradient for CA.{name}"
                    assert p.grad.abs().sum() > 0, f"zero gradient for CA.{name}"

    def test_output_finite_and_input_untouched(self) -> None:
        model = _small_model().eval()
        x = torch.rand(1, 1, 32, 32)
        x_ref = x.clone()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(x, x_ref)


class TestOneBatchOverfit:
    def test_hybrid_loss_decreases(self) -> None:
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = ANetSSIML1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        target = torch.rand(4, 1, 32, 32)
        masked = target.clone()
        masked[..., :, 12:20] = 0.0  # consecutive missing block

        model.train()
        losses = []
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(masked), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        assert np.mean(losses[-5:]) < losses[0]

    def test_loss_decreases_via_alias(self) -> None:
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = build_loss({"type": "ssim_l1", "params": {}})
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        target = torch.rand(4, 1, 32, 32)
        masked = target.clone()
        masked[..., :, 12:20] = 0.0

        model.train()
        losses = []
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(masked), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        assert np.mean(losses[-5:]) < losses[0]


class TestTrainerCompatibility:
    def test_train_one_epoch_runs(self) -> None:
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = ANetSSIML1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        x = torch.rand(6, 1, 16, 16)
        y = torch.rand(6, 1, 16, 16)
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        stats = train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
        )
        assert np.isfinite(stats["train"])

    def test_train_one_epoch_with_mask_batch(self) -> None:
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = ANetSSIML1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        x = torch.rand(4, 1, 16, 16)
        y = torch.rand(4, 1, 16, 16)
        m = torch.ones(4, 1, 1, 16)
        loader = DataLoader(TensorDataset(x, y, m), batch_size=2)
        stats = train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
        )
        assert np.isfinite(stats["train"])

    def test_train_one_epoch_via_alias_loss(self) -> None:
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = build_loss({"type": "ssim_l1", "params": {}})
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        x = torch.rand(6, 1, 16, 16)
        y = torch.rand(6, 1, 16, 16)
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        stats = train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
        )
        assert np.isfinite(stats["train"])
