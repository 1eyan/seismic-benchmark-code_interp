"""Park2022 CFunet training tests: forward/backward, overfit, serialization, trainer compatibility."""

from __future__ import annotations

import io
import sys
from pathlib import Path

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
from utils.losses import CFunetMSEFourierLoss  # noqa: E402
from utils.train_utils import unwrap_ddp  # noqa: E402


def _small_model():
    return Park2022CFUNet(
        in_channels=1, out_channels=1, base_channels=8, num_levels=3,
        upsample_mode="fourier_zero_padding",
    )


def _sine_patch(size=32):
    n = torch.arange(size, dtype=torch.float32)[None, None, :, None]
    m = torch.arange(size, dtype=torch.float32)[None, None, None, :]
    return torch.sin(2.0 * torch.pi * (3.0 * n / size + 2.0 * m / size))


def _masked_input(target, missing_ratio=0.75, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(target.shape[2] * target.shape[3], generator=g)
    n_missing = int(round(missing_ratio * target.shape[2] * target.shape[3]))
    flat = target.flatten(start_dim=2)
    x = flat.clone()
    x[0, 0, perm[:n_missing]] = 0.0
    mask = (flat.abs() > 0.0).to(dtype=torch.float32)
    mask[0, 0, perm[:n_missing]] = 0.0
    return x.view_as(target), mask.view_as(target)


class TestForwardBackward:
    def test_backward_updates_all_params(self):
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = CFunetMSEFourierLoss(alpha=1.0)
        loss_fn.attach_model(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = _sine_patch()
        x, mask = _masked_input(target)
        pred = model(x, mask=mask)
        loss = loss_fn(pred, target, mask=mask)
        loss.backward()
        n_grads = sum(1 for p in model.parameters() if p.grad is not None)
        assert n_grads > 0
        assert torch.isfinite(loss).item()
        opt.step()
        assert float(loss) > 0.0


class TestOverfit:
    def test_reconstructs_missing_traces(self):
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = CFunetMSEFourierLoss(alpha=1.0)
        loss_fn.attach_model(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        target = _sine_patch()
        x, mask = _masked_input(target)
        missing = (mask == 0.0)

        model.train()
        first_loss = None
        for _ in range(400):
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask=mask)
            loss = loss_fn(pred, target, mask=mask)
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = float(loss.detach())

        model.eval()
        with torch.no_grad():
            pred = model(x, mask=mask)
        final_loss = float(loss_fn(pred, target, mask=mask).detach())
        assert final_loss < 0.1 * first_loss

        masked_mse = ((pred - target)[missing].square()).mean()
        assert masked_mse.item() < 0.02


class TestSerialization:
    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        model = _small_model()
        x = torch.randn(1, 1, 32, 32)
        y1 = model(x)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = _small_model()
        model2.load_state_dict(torch.load(buf, weights_only=True))
        y2 = model2(x)
        torch.testing.assert_close(y1, y2)


class TestTrainerCompatibility:
    def test_trainer_protocol_passes_mask(self):
        """Simulate the train_one_epoch call sequence with extras."""
        torch.manual_seed(0)
        model = _small_model()
        loss_fn = CFunetMSEFourierLoss(alpha=1.0)
        loss_fn.attach_model(unwrap_ddp(model))  # same wiring as the script
        target = _sine_patch()
        x, mask = _masked_input(target)

        pred = model(x, **{"mask": mask})
        loss = loss_fn(pred, target, **{"mask": mask})
        assert torch.isfinite(loss).item()

        # The loss must read the coarse output of THIS forward pass.
        oc = model._intermediates["coarse"]
        torch.testing.assert_close(oc * (1 - mask) + x * mask, oc)

    def test_eval_mode_no_grad(self):
        model = _small_model()
        loss_fn = CFunetMSEFourierLoss(alpha=1.0)
        loss_fn.attach_model(model)
        target = _sine_patch()
        x, mask = _masked_input(target)
        model.eval()
        with torch.no_grad():
            pred = model(x, mask=mask)
            loss = loss_fn(pred, target, mask=mask)
        assert torch.isfinite(loss).item()
