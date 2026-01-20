# Copyright 2025 Xin Huang
#
# GNU General Public License v3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html


import pickle
import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import gaishi.models.unet.unet_model as unet_mod


class DummyUNetPlusPlus(nn.Module):
    """Dummy replacement for UNetPlusPlus(num_classes, input_channels=3)."""

    last_init = None

    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = int(num_classes)
        DummyUNetPlusPlus.last_init = {
            "num_classes": int(num_classes),
            "input_channels": int(input_channels),
        }
        self.head = nn.Conv2d(int(input_channels), int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        if self.num_classes == 1:
            return logits[:, 0]
        return logits


class DummyUNetPlusPlusRNNNeighborGapFusion(nn.Module):
    """Dummy replacement for UNetPlusPlusRNNNeighborGapFusion(polymorphisms=...)."""

    last_init = None

    def __init__(
        self,
        polymorphisms: int = 128,
        hidden_dim: int = 4,
        gru_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.polymorphisms = int(polymorphisms)
        DummyUNetPlusPlusRNNNeighborGapFusion.last_init = {
            "polymorphisms": int(polymorphisms),
            "hidden_dim": int(hidden_dim),
            "gru_layers": int(gru_layers),
            "bidirectional": bool(bidirectional),
        }
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 input channels, got {x.shape[1]}.")
        if x.shape[-1] != self.polymorphisms:
            raise ValueError(f"Expected width {self.polymorphisms}, got {x.shape[-1]}.")
        return self.scale * x[:, 0, :, :]  # (B, H, W)


def _make_training_h5(
    tmp_path,
    *,
    n_keys: int,
    chunk_size: int,
    n_channels: int,
    individuals: int,
    polymorphisms: int,
    force_no_positive: bool = False,
) -> tuple[str, list[str]]:
    h5_path = tmp_path / "train.h5"
    keys = [str(i) for i in range(n_keys)]
    rng = np.random.default_rng(0)

    with h5py.File(h5_path, "w") as f:
        for k in keys:
            grp = f.create_group(k)

            x = rng.integers(
                0,
                2,
                size=(chunk_size, n_channels, individuals, polymorphisms),
                dtype=np.int32,
            )
            grp.create_dataset("x_0", data=x)

            if force_no_positive:
                y = np.zeros((chunk_size, 1, individuals, polymorphisms), dtype=np.int8)
            else:
                y = rng.integers(
                    0,
                    2,
                    size=(chunk_size, 1, individuals, polymorphisms),
                    dtype=np.int8,
                )
                y[0, 0, 0, 0] = 1  # ensure at least one positive overall

            grp.create_dataset("y", data=y)

    return str(h5_path), keys


def test_train_branch_unetplusplus_two_channel(tmp_path, monkeypatch) -> None:
    # Patch the two model classes used by UNetModel
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNNNeighborGapFusion",
        DummyUNetPlusPlusRNNNeighborGapFusion,
    )

    DummyUNetPlusPlus.last_init = None
    DummyUNetPlusPlusRNNNeighborGapFusion.last_init = None

    training_data, keys = _make_training_h5(
        tmp_path,
        n_keys=20,
        chunk_size=2,
        n_channels=2,  # >=2, will slice to 2
        individuals=2,
        polymorphisms=7,
    )
    model_dir = tmp_path / "model_out"

    unet_mod.UNetModel.train(
        training_data=training_data,
        model_dir=str(model_dir),
        trained_model_file=None,
        add_channels=False,  # -> UNetPlusPlus
        n_classes=1,
        learning_rate=0.001,
        batch_size=2,  # batch_size == chunk_size => n_keys_per_batch==1
        label_noise=0.01,
        n_early=0,
        n_epochs=1,
        label_smooth=False,
    )

    assert DummyUNetPlusPlus.last_init is not None
    assert DummyUNetPlusPlus.last_init["num_classes"] == 1
    assert DummyUNetPlusPlus.last_init["input_channels"] == 2
    assert DummyUNetPlusPlusRNNNeighborGapFusion.last_init is None

    assert (model_dir / "training.log").exists()
    assert (model_dir / "validation.log").exists()
    assert (model_dir / "best.pth").exists()
    assert (model_dir / "val_keys.pkl").exists()

    with open(model_dir / "val_keys.pkl", "rb") as f:
        val_keys = pickle.load(f)

    expected_n_val = int(len(keys) * 0.05)
    assert len(val_keys) == expected_n_val


def test_train_branch_neighbor_gap_fusion_four_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNNNeighborGapFusion",
        DummyUNetPlusPlusRNNNeighborGapFusion,
    )

    DummyUNetPlusPlus.last_init = None
    DummyUNetPlusPlusRNNNeighborGapFusion.last_init = None

    training_data, _ = _make_training_h5(
        tmp_path,
        n_keys=20,
        chunk_size=2,
        n_channels=4,  # required
        individuals=3,
        polymorphisms=11,
    )
    model_dir = tmp_path / "model_out2"

    unet_mod.UNetModel.train(
        training_data=training_data,
        model_dir=str(model_dir),
        add_channels=True,  # -> UNetPlusPlusRNNNeighborGapFusion
        n_classes=1,
        batch_size=2,
        n_epochs=1,
        n_early=0,
        label_smooth=False,
    )

    assert DummyUNetPlusPlus.last_init is None
    assert DummyUNetPlusPlusRNNNeighborGapFusion.last_init is not None
    assert DummyUNetPlusPlusRNNNeighborGapFusion.last_init["polymorphisms"] == 11


def test_train_raises_when_add_channels_true_but_not_4_channels(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNNNeighborGapFusion",
        DummyUNetPlusPlusRNNNeighborGapFusion,
    )

    training_data, _ = _make_training_h5(
        tmp_path,
        n_keys=20,
        chunk_size=2,
        n_channels=3,  # invalid for add_channels=True
        individuals=2,
        polymorphisms=7,
    )
    model_dir = tmp_path / "model_out3"

    with pytest.raises(ValueError, match="expects 4 input channels"):
        unet_mod.UNetModel.train(
            training_data=training_data,
            model_dir=str(model_dir),
            add_channels=True,
            n_classes=1,
            batch_size=2,
            n_epochs=1,
            n_early=0,
            label_smooth=False,
        )


def test_train_raises_when_add_channels_true_but_n_classes_not_1(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNNNeighborGapFusion",
        DummyUNetPlusPlusRNNNeighborGapFusion,
    )

    training_data, _ = _make_training_h5(
        tmp_path,
        n_keys=20,
        chunk_size=2,
        n_channels=4,
        individuals=2,
        polymorphisms=7,
    )
    model_dir = tmp_path / "model_out4"

    with pytest.raises(ValueError, match="supports n_classes == 1"):
        unet_mod.UNetModel.train(
            training_data=training_data,
            model_dir=str(model_dir),
            add_channels=True,
            n_classes=2,  # invalid
            batch_size=2,
            n_epochs=1,
            n_early=0,
            label_smooth=False,
        )


def test_train_raises_when_no_positive_class(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNNNeighborGapFusion",
        DummyUNetPlusPlusRNNNeighborGapFusion,
    )

    training_data, _ = _make_training_h5(
        tmp_path,
        n_keys=20,
        chunk_size=2,
        n_channels=2,
        individuals=2,
        polymorphisms=7,
        force_no_positive=True,
    )
    model_dir = tmp_path / "model_out5"

    with pytest.raises(ValueError, match="no positive class"):
        unet_mod.UNetModel.train(
            training_data=training_data,
            model_dir=str(model_dir),
            add_channels=False,
            n_classes=1,
            batch_size=2,
            n_epochs=1,
            n_early=0,
            label_smooth=False,
        )
