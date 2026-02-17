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


import h5py, pickle
import numpy as np
import os, pytest, torch
import torch.nn as nn
import gaishi.models.unet.unet_model as unet_mod


class DummyUNetPlusPlus(nn.Module):
    """Dummy replacement for UNetPlusPlus."""

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


class DummyUNetPlusPlusRNN(nn.Module):
    """Dummy replacement for UNetPlusPlusRNN."""

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
        DummyUNetPlusPlusRNN.last_init = {
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
    n_reps: int,
    N: int,
    L: int,
    with_gaps: bool = True,
    force_no_positive: bool = False,
) -> str:
    """
    Create a unified-schema training HDF5 file.

    Layout written:
      /data/Ref_genotype (n,N,L) uint32
      /data/Tgt_genotype (n,N,L) uint32
      /data/Gap_to_prev  (n,N,L) int64   (optional)
      /data/Gap_to_next  (n,N,L) int64   (optional)
      /targets/Label     (n,N,L) uint8
      /meta attrs: n,N,L,Chromosome
      /meta/ref_sample_table, /meta/tgt_sample_table
      /index/ref_ids, /index/tgt_ids, /index/Seed, /index/Replicate
    """
    h5_path = tmp_path / "data.h5"
    rng = np.random.default_rng(0)

    ref = rng.integers(0, 2, size=(n_reps, N, L), dtype=np.uint32)
    tgt = rng.integers(0, 2, size=(n_reps, N, L), dtype=np.uint32)

    if with_gaps:
        gp = rng.integers(0, 10, size=(n_reps, N, L), dtype=np.int64)
        gn = rng.integers(0, 10, size=(n_reps, N, L), dtype=np.int64)

    if force_no_positive:
        y = np.zeros((n_reps, N, L), dtype=np.uint8)
    else:
        y = rng.integers(0, 2, size=(n_reps, N, L), dtype=np.uint8)
        y[0, 0, 0] = 1  # ensure at least one positive overall

    ref_ids = np.tile(np.arange(N, dtype=np.uint32)[None, :], (n_reps, 1))
    tgt_ids = np.tile(np.arange(N, dtype=np.uint32)[None, :], (n_reps, 1))
    seeds = np.arange(n_reps, dtype=np.int64)
    reps = np.arange(n_reps, dtype=np.int64)

    str_dt = h5py.string_dtype(encoding="utf-8")
    ref_table = np.asarray([f"ref_{i}" for i in range(N)], dtype=object)
    tgt_table = np.asarray([f"tgt_{i}" for i in range(N)], dtype=object)

    with h5py.File(h5_path, "w") as f:
        f.create_group("data")
        f.create_group("targets")
        f.create_group("meta")
        f.create_group("index")

        f["/meta"].attrs["n"] = int(n_reps)
        f["/meta"].attrs["N"] = int(N)
        f["/meta"].attrs["L"] = int(L)
        f["/meta"].attrs["Chromosome"] = "213"

        f.create_dataset("/meta/ref_sample_table", data=ref_table, dtype=str_dt)
        f.create_dataset("/meta/tgt_sample_table", data=tgt_table, dtype=str_dt)

        f.create_dataset("/data/Ref_genotype", data=ref, compression="lzf")
        f.create_dataset("/data/Tgt_genotype", data=tgt, compression="lzf")
        if with_gaps:
            f.create_dataset("/data/Gap_to_prev", data=gp, compression="lzf")
            f.create_dataset("/data/Gap_to_next", data=gn, compression="lzf")

        f.create_dataset("/targets/Label", data=y, compression="lzf")

        f.create_dataset("/index/ref_ids", data=ref_ids, compression="lzf")
        f.create_dataset("/index/tgt_ids", data=tgt_ids, compression="lzf")
        f.create_dataset("/index/Seed", data=seeds, compression="lzf")
        f.create_dataset("/index/Replicate", data=reps, compression="lzf")

    return str(h5_path)


def test_train_branch_unetplusplus_two_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN)

    DummyUNetPlusPlus.last_init = None
    DummyUNetPlusPlusRNN.last_init = None

    training_data = _make_training_h5(tmp_path, n_reps=40, N=2, L=7, with_gaps=True)

    model_dir = tmp_path / "model_out"
    model_path = model_dir / "best.pth"

    unet_mod.UNetModel.train(
        data=training_data,
        output=str(model_path),
        add_channels=False,  # -> UNetPlusPlus
        n_classes=1,
        learning_rate=0.001,
        batch_size=2,
        label_noise=0.01,
        n_early=0,
        n_epochs=1,
        min_delta=0.0,
        label_smooth=False,
        val_prop=0.2,
        seed=0,
    )

    assert DummyUNetPlusPlus.last_init is not None
    assert DummyUNetPlusPlus.last_init["num_classes"] == 1
    assert DummyUNetPlusPlus.last_init["input_channels"] == 2
    assert DummyUNetPlusPlusRNN.last_init is None

    assert (model_dir / "training.log").exists()
    assert (model_dir / "validation.log").exists()
    assert (model_dir / "best.pth").exists()


def test_train_branch_neighbor_gap_fusion_four_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN)

    DummyUNetPlusPlus.last_init = None
    DummyUNetPlusPlusRNN.last_init = None

    training_data = _make_training_h5(tmp_path, n_reps=40, N=3, L=11, with_gaps=True)

    model_dir = tmp_path / "model_out2"
    model_path = model_dir / "best.pth"

    unet_mod.UNetModel.train(
        data=training_data,
        output=str(model_path),
        add_channels=True,  # -> UNetPlusPlusRNN
        n_classes=1,
        batch_size=2,
        n_epochs=1,
        n_early=0,
        min_delta=0.0,
        label_smooth=False,
        val_prop=0.2,
        seed=0,
    )

    assert DummyUNetPlusPlus.last_init is None
    assert DummyUNetPlusPlusRNN.last_init is not None
    assert DummyUNetPlusPlusRNN.last_init["polymorphisms"] == 11


def test_train_raises_when_add_channels_true_but_missing_gap_datasets(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN)

    training_data = _make_training_h5(tmp_path, n_reps=40, N=2, L=7, with_gaps=False)

    model_dir = tmp_path / "model_out3"
    model_path = model_dir / "best.pth"

    # Your current pipeline will fail when trying to build 4-channel inputs without gap datasets.
    with pytest.raises(KeyError, match="Gap_to_prev|Gap_to_next"):
        unet_mod.UNetModel.train(
            data=training_data,
            output=str(model_path),
            add_channels=True,
            n_classes=1,
            batch_size=2,
            n_epochs=1,
            n_early=0,
            label_smooth=False,
            val_prop=0.2,
            seed=0,
        )


def test_train_raises_when_add_channels_true_but_n_classes_not_1(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN)

    training_data = _make_training_h5(tmp_path, n_reps=40, N=2, L=7, with_gaps=True)

    model_dir = tmp_path / "model_out4"
    model_path = model_dir / "best.pth"

    with pytest.raises(ValueError, match="supports n_classes == 1"):
        unet_mod.UNetModel.train(
            data=training_data,
            output=str(model_path),
            add_channels=True,
            n_classes=2,
            batch_size=2,
            n_epochs=1,
            n_early=0,
            label_smooth=False,
            val_prop=0.2,
            seed=0,
        )


def test_train_raises_when_no_positive_class(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN)

    training_data = _make_training_h5(
        tmp_path, n_reps=40, N=2, L=7, with_gaps=True, force_no_positive=True
    )

    model_dir = tmp_path / "model_out5"
    model_path = model_dir / "best.pth"

    with pytest.raises(ValueError, match="no positive class"):
        unet_mod.UNetModel.train(
            data=training_data,
            output=str(model_path),
            add_channels=False,
            n_classes=1,
            batch_size=2,
            n_epochs=1,
            n_early=0,
            label_smooth=False,
            val_prop=0.2,
            seed=0,
        )


def _make_h5(
    tmp_path,
    *,
    n_keys: int,
    chunk_size: int,
    n_channels: int,
    individuals: int,
    polymorphisms: int,
    force_no_positive: bool = False,
) -> tuple[str, list[str]]:
    h5_path = tmp_path / "data.h5"
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


def _save_weights(tmp_path, model, filename) -> str:
    weights_path = tmp_path / filename
    torch.save(model.state_dict(), str(weights_path))
    return str(weights_path)


def test_infer_unetplusplus_two_channel_writes_y_pred(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNN",
        DummyUNetPlusPlusRNN,
    )

    test_data, keys = _make_h5(
        tmp_path,
        n_keys=5,
        chunk_size=2,
        n_channels=2,
        individuals=3,
        polymorphisms=7,
    )

    # weights must match the model created in infer (UNetPlusPlus(num_classes=1, input_channels=2))
    dummy = DummyUNetPlusPlus(num_classes=1, input_channels=2)
    weights = _save_weights(tmp_path, dummy, filename="unet2.weights")

    out_dir = tmp_path / "infer_out_2ch"
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_mod.UNetModel.infer(
        data=test_data,
        model=weights,
        output=f"{out_dir}/data.preds.h5",
        add_channels=False,
        n_classes=1,
        x_dataset="x_0",
        y_pred_dataset="y_pred",
        device="cpu",
    )

    out_h5 = os.path.join(out_dir, "data.preds.h5")

    assert os.path.exists(out_h5)

    with h5py.File(out_h5, "r") as f:
        for k in keys:
            assert "y_pred" in f[k]
            y_pred = f[k]["y_pred"]
            assert y_pred.shape == (2, 1, 3, 7)
            assert y_pred.dtype == np.float32


def test_infer_unet_plus_plus_rnn_four_channel_writes_y_pred(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNN",
        DummyUNetPlusPlusRNN,
    )

    test_data, keys = _make_h5(
        tmp_path,
        n_keys=5,
        chunk_size=2,
        n_channels=4,
        individuals=4,
        polymorphisms=11,
    )

    # weights must match the model created in infer
    dummy = DummyUNetPlusPlusRNN(polymorphisms=11)
    weights = _save_weights(tmp_path, dummy, filename="unet4.weights")

    out_dir = tmp_path / "infer_out_4ch"
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_mod.UNetModel.infer(
        data=test_data,
        model=weights,
        output=f"{out_dir}/data.preds.h5",
        add_channels=True,
        n_classes=1,
        x_dataset="x_0",
        y_pred_dataset="y_pred",
        device="cpu",
    )

    out_h5 = os.path.join(out_dir, "data.preds.h5")

    assert os.path.exists(out_h5)

    with h5py.File(out_h5, "r") as f:
        for k in keys:
            assert "y_pred" in f[k]
            y_pred = f[k]["y_pred"]
            assert y_pred.shape == (2, 1, 4, 11)
            assert y_pred.dtype == np.float32


def test_infer_raises_when_add_channels_true_but_not_4_channels(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNN",
        DummyUNetPlusPlusRNN,
    )

    test_data, _ = _make_h5(
        tmp_path,
        n_keys=5,
        chunk_size=2,
        n_channels=3,  # invalid
        individuals=2,
        polymorphisms=7,
    )

    dummy = DummyUNetPlusPlusRNN(polymorphisms=7)
    weights = _save_weights(tmp_path, dummy, filename="bad.weights")

    out_dir = tmp_path / "infer_out_bad"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="4"):
        unet_mod.UNetModel.infer(
            data=test_data,
            model=weights,
            output=f"{out_dir}/test.pred.h5",
            add_channels=True,
            n_classes=1,
            x_dataset="x_0",
            y_pred_dataset="y_pred",
            device="cpu",
        )


def test_infer_raises_when_add_channels_true_but_n_classes_not_1(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus)
    monkeypatch.setattr(
        unet_mod,
        "UNetPlusPlusRNN",
        DummyUNetPlusPlusRNN,
    )

    test_data, _ = _make_h5(
        tmp_path,
        n_keys=5,
        chunk_size=2,
        n_channels=4,
        individuals=2,
        polymorphisms=7,
    )

    dummy = DummyUNetPlusPlusRNN(polymorphisms=7)
    weights = _save_weights(tmp_path, dummy, filename="bad_ncls.weights")

    out_dir = tmp_path / "infer_out_bad_ncls"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="n_classes|classes|supports"):
        unet_mod.UNetModel.infer(
            data=test_data,
            model=weights,
            output=f"{out_dir}/test.pred.h5",
            add_channels=True,
            n_classes=2,  # invalid for fusion
            x_dataset="x_0",
            y_pred_dataset="y_pred",
            device="cpu",
        )
