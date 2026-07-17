"""Training-behavior tests for Yu2022ANet: gradients, one-batch overfit,
trainer compatibility, and consecutive-trace mask construction."""

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

from model.interpolation.yu2022_anet import Yu2022ANet  # noqa: E402
from tools.preprocessing import mask_traces  # noqa: E402
from utils.losses import ANetSSIML1Loss  # noqa: E402
from utils.train_utils import train_one_epoch  # noqa: E402


def _small_model() -> Yu2022ANet:
    torch.manual_seed(0)
    return Yu2022ANet(base_channels=8, num_residual_blocks=2)


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

    def test_attention_parameters_receive_gradients(self) -> None:
        model = _small_model()
        x = torch.rand(1, 1, 16, 16)
        model(x).sum().backward()
        for name, p in model.attention.named_parameters():
            assert p.grad is not None, f"no gradient for attention.{name}"

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
        # 3-tuple batches pass the mask as extras; the loss must ignore it.
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


class TestConsecutiveMaskRatioRange:
    N_SHOTS = 64
    N_TRACES = 128
    N_TIME = 32

    def _make_masks(self, seed: int = 0):
        shots = np.random.default_rng(123).normal(size=(self.N_SHOTS, self.N_TRACES, self.N_TIME))
        masked, mask = mask_traces(
            shots, mode="continuous", ratio_range=(0.1, 0.3), rng=seed
        )
        return shots, masked, mask

    def test_single_contiguous_interval_per_patch(self) -> None:
        _, _, mask = self._make_masks()
        for row in mask:
            transitions = np.diff(row.astype(np.int8))
            # one 0->1 rise and at most one 1->0 fall == single contiguous run
            assert (transitions == 1).sum() <= 1
            assert (transitions == -1).sum() <= 1
            assert row.any()

    def test_whole_trace_removal(self) -> None:
        shots, masked, mask = self._make_masks()
        # Masked traces are entirely zero across the time axis.
        assert np.all(masked[mask] == 0.0)
        # Observed traces are untouched.
        assert np.array_equal(masked[~mask], shots[~mask])

    def test_ratio_within_range(self) -> None:
        _, _, mask = self._make_masks()
        counts = mask.sum(axis=1)
        low = max(1, int(np.floor(0.1 * self.N_TRACES)) - 1)
        high = int(np.ceil(0.3 * self.N_TRACES)) + 1
        assert counts.min() >= low
        assert counts.max() <= high
        # Ratios actually vary across patches (not a single fixed count).
        assert len(np.unique(counts)) > 1

    def test_seed_reproducibility(self) -> None:
        _, _, mask_a = self._make_masks(seed=7)
        _, _, mask_b = self._make_masks(seed=7)
        assert np.array_equal(mask_a, mask_b)

    def test_different_seeds_differ(self) -> None:
        _, _, mask_a = self._make_masks(seed=7)
        _, _, mask_b = self._make_masks(seed=8)
        assert not np.array_equal(mask_a, mask_b)

    def test_no_out_of_bounds(self) -> None:
        _, _, mask = self._make_masks()
        assert mask.shape == (self.N_SHOTS, self.N_TRACES)
        counts = mask.sum(axis=1)
        assert counts.max() < self.N_TRACES

    def test_random_mode_supported(self) -> None:
        shots = np.random.default_rng(0).normal(size=(8, 32, 4))
        _, mask = mask_traces(shots, mode="random", ratio_range=(0.1, 0.3), rng=0)
        counts = mask.sum(axis=1)
        assert counts.min() >= 1
        assert counts.max() <= int(np.ceil(0.3 * 32)) + 1

    def test_uniform_mode_rejected(self) -> None:
        shots = np.zeros((2, 16, 4))
        with pytest.raises(ValueError):
            mask_traces(shots, mode="uniform", ratio_range=(0.1, 0.3))

    def test_exclusive_with_missing_traces(self) -> None:
        shots = np.zeros((2, 16, 4))
        with pytest.raises(ValueError):
            mask_traces(
                shots, mode="continuous", ratio_range=(0.1, 0.3), missing_traces=4
            )

    def test_invalid_range_rejected(self) -> None:
        shots = np.zeros((2, 16, 4))
        with pytest.raises(ValueError):
            mask_traces(shots, mode="continuous", ratio_range=(0.3, 0.1))
        with pytest.raises(ValueError):
            mask_traces(shots, mode="continuous", ratio_range=(0.0, 0.3))

    def test_legacy_fixed_ratio_unchanged(self) -> None:
        # Existing single-ratio behavior must be untouched by the extension.
        shots = np.random.default_rng(0).normal(size=(4, 32, 8))
        _, mask = mask_traces(shots, mode="continuous", ratio=0.25, rng=0)
        counts = mask.sum(axis=1)
        assert np.all(counts == 8)
