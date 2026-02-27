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


import h5py, math, pickle
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


class DummyUNetPlusPlus2(nn.Module):
    """Return logits derived from input (deterministic)."""

    def __init__(self, num_classes: int = 1, input_channels: int = 2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, N, L), we use tgt channel = 1
        tgt = x[:, 1, :, :]  # (B, N, L)
        if self.num_classes == 1:
            return tgt
        out = torch.zeros(
            (x.shape[0], self.num_classes, x.shape[2], x.shape[3]),
            device=x.device,
            dtype=x.dtype,
        )
        out[:, 0, :, :] = 0.0
        out[:, 1, :, :] = tgt
        return out


class DummyUNetPlusPlusRNN2(nn.Module):
    """4-channel dummy; return logits=tgt channel."""

    def __init__(self, polymorphisms: int = 128, **kwargs):
        super().__init__()
        self.polymorphisms = int(polymorphisms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect 4 channels when add_channels=True
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 input channels, got {x.shape[1]}.")
        if x.shape[-1] != self.polymorphisms:
            raise ValueError(f"Expected width {self.polymorphisms}, got {x.shape[-1]}.")
        return x[:, 1, :, :]  # (B, N, L)


def _save_weights(tmp_path, model, filename) -> str:
    weights_path = tmp_path / filename
    torch.save(model.state_dict(), str(weights_path))

    return str(weights_path)


def _make_inference_h5(tmp_path, *, with_gaps: bool, L: int = 5) -> str:
    """
    n=2 windows, H=2 unique tgt samples (A,B), N=3 rows per window (upsampling: [A,B,A]).
    positions overlap: window0 = 100..104, window1 = 102..106
    """
    h5_path = tmp_path / ("tiny_gaps.h5" if with_gaps else "tiny.h5")
    n = 2
    H = 2
    N = 3  # upsampled rows
    assert L == 5

    # window positions (n, L)
    pos = np.array(
        [
            [100, 101, 102, 103, 104],
            [102, 103, 104, 105, 106],
        ],
        dtype=np.int64,
    )

    # upsampling mapping per window (n, N): [A,B,A] => [0,1,0]
    tgt_ids = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.uint32,
    )

    # logits will be tgt channel -> set Tgt_genotype as known numbers (n, N, L)
    # window0: A=[1..5], B=[11..15], A=[1..5]
    # window1: A=[6..10], B=[16..20], A=[6..10]
    tgt0_A = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    tgt0_B = np.array([11, 12, 13, 14, 15], dtype=np.uint32)
    tgt1_A = np.array([6, 7, 8, 9, 10], dtype=np.uint32)
    tgt1_B = np.array([16, 17, 18, 19, 20], dtype=np.uint32)

    T = np.zeros((n, N, L), dtype=np.uint32)
    T[0, 0, :] = tgt0_A
    T[0, 1, :] = tgt0_B
    T[0, 2, :] = tgt0_A
    T[1, 0, :] = tgt1_A
    T[1, 1, :] = tgt1_B
    T[1, 2, :] = tgt1_A

    R = np.zeros_like(T, dtype=np.uint32)  # ref irrelevant for dummy
    gp = np.zeros((n, N, L), dtype=np.int64)
    gn = np.zeros((n, N, L), dtype=np.int64)

    str_dt = h5py.string_dtype("utf-8")
    with h5py.File(h5_path, "w") as f:
        f.require_group("/data")
        f.require_group("/index")
        f.require_group("/meta")
        f.require_group("/coords")

        f["/meta"].attrs["n"] = n
        f["/meta"].attrs["N"] = N
        f["/meta"].attrs["L"] = L
        f["/meta"].attrs["Chromosome"] = "1"

        f.create_dataset(
            "/meta/tgt_sample_table",
            data=np.array(["A", "B"], dtype=object),
            dtype=str_dt,
        )
        f.create_dataset(
            "/meta/ref_sample_table", data=np.array(["R0"], dtype=object), dtype=str_dt
        )

        f.create_dataset("/coords/Position", data=pos, dtype=np.int64, chunks=(1, L))
        f.create_dataset("/index/tgt_ids", data=tgt_ids, dtype=np.uint32, chunks=(1, N))

        f.create_dataset(
            "/data/Ref_genotype", data=R, dtype=np.uint32, chunks=(1, N, L)
        )
        f.create_dataset(
            "/data/Tgt_genotype", data=T, dtype=np.uint32, chunks=(1, N, L)
        )

        if with_gaps:
            f.create_dataset(
                "/data/Gap_to_prev", data=gp, dtype=np.int64, chunks=(1, N, L)
            )
            f.create_dataset(
                "/data/Gap_to_next", data=gn, dtype=np.int64, chunks=(1, N, L)
            )

    return str(h5_path)


def _sigmoid_scalar(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _gauss_weights(L: int, sigma: float) -> np.ndarray:
    mu = L // 2
    x = np.arange(L, dtype=np.float64)
    g = np.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))
    g = g / g.max()
    return g.astype(np.float64)


def _read_prob_table(path: str) -> dict[tuple[str, int], list[float]]:
    """
    Returns mapping (sample, position) -> [prob...] (len=1 for binary, len=C for multiclass)
    """
    out: dict[tuple[str, int], list[float]] = {}
    with open(path, "r", newline="") as fp:
        header = fp.readline().rstrip("\n").split("\t")
        prob_cols = [i for i, c in enumerate(header) if c.startswith("prob")]
        for line in fp:
            s, p, *rest = line.rstrip("\n").split("\t")
            pos = int(p)
            probs = [
                float(rest[i - 2]) for i in prob_cols
            ]  # header: sample position prob...
            out[(s, pos)] = probs
    return out


def test_infer_unetplusplus_two_channel_outputs_table_binary(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus2)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN2)

    h5 = _make_inference_h5(tmp_path, with_gaps=False, L=5)

    dummy = unet_mod.UNetPlusPlus(num_classes=1, input_channels=2)
    weights = _save_weights(tmp_path, dummy, filename="unet2.weights")

    out = os.path.join(tmp_path, "pred.tsv")
    unet_mod.UNetModel.infer(data=h5, model=weights, output=out, device="cpu")

    tab = _read_prob_table(out)

    # Expected (unweighted):
    # For A at pos=102:
    # window0 contributes t=2 value=3 twice; window1 contributes t=0 value=6 twice => mean = (2*3 + 2*6)/4 = 4.5
    exp_A_102 = _sigmoid_scalar(4.5)
    got_A_102 = tab[("A", 102)][0]
    assert got_A_102 == pytest.approx(exp_A_102, rel=1e-6, abs=1e-6)

    # For B at pos=102:
    # window0 t=2 value=13 once; window1 t=0 value=16 once => mean=14.5
    exp_B_102 = _sigmoid_scalar(14.5)
    got_B_102 = tab[("B", 102)][0]
    assert got_B_102 == pytest.approx(exp_B_102, rel=1e-6, abs=1e-6)


def test_infer_unet_rnn_four_channel_outputs_table_binary(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus2)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN2)

    h5 = _make_inference_h5(tmp_path, with_gaps=True, L=5)

    dummy = unet_mod.UNetPlusPlusRNN(polymorphisms=5)
    weights = _save_weights(tmp_path, dummy, filename="unet4.weights")

    out = os.path.join(tmp_path, "pred.tsv")
    unet_mod.UNetModel.infer(
        data=h5, model=weights, output=out, add_channels=True, device="cpu"
    )

    tab = _read_prob_table(out)
    # same expected as above
    assert tab[("A", 102)][0] == pytest.approx(_sigmoid_scalar(4.5), rel=1e-6, abs=1e-6)
    assert tab[("B", 102)][0] == pytest.approx(
        _sigmoid_scalar(14.5), rel=1e-6, abs=1e-6
    )


def test_infer_raises_when_add_channels_true_but_missing_gap_datasets(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus2)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN2)

    h5 = _make_inference_h5(tmp_path, with_gaps=False, L=5)

    dummy = unet_mod.UNetPlusPlusRNN(polymorphisms=5)
    weights = _save_weights(tmp_path, dummy, filename="bad.weights")

    out = os.path.join(tmp_path, "pred.tsv")

    with pytest.raises(KeyError):
        unet_mod.UNetModel.infer(
            data=h5, model=weights, output=out, add_channels=True, device="cpu"
        )


def test_infer_raises_when_add_channels_true_but_n_classes_not_1(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus2)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN2)

    h5 = _make_inference_h5(tmp_path, with_gaps=True, L=5)

    dummy = unet_mod.UNetPlusPlusRNN(polymorphisms=5)
    weights = _save_weights(tmp_path, dummy, filename="bad_ncls.weights")

    out = os.path.join(tmp_path, "pred.tsv")

    with pytest.raises(ValueError, match="n_classes|supports|1"):
        unet_mod.UNetModel.infer(
            data=h5,
            model=weights,
            output=out,
            add_channels=True,
            n_classes=2,
            device="cpu",
        )


def test_infer_site_weighting_changes_overlap_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(unet_mod, "UNetPlusPlus", DummyUNetPlusPlus2)
    monkeypatch.setattr(unet_mod, "UNetPlusPlusRNN", DummyUNetPlusPlusRNN2)

    h5 = _make_inference_h5(tmp_path, with_gaps=False, L=5)
    dummy = unet_mod.UNetPlusPlus(num_classes=1, input_channels=2)
    weights = _save_weights(tmp_path, dummy, filename="w.weights")

    out0 = os.path.join(tmp_path, "no_weight.tsv")
    out1 = os.path.join(tmp_path, "gauss.tsv")

    # no weighting
    unet_mod.UNetModel.infer(
        data=h5, model=weights, output=out0, device="cpu", site_weighting=False
    )
    # gaussian weighting
    sigma = 1.0
    unet_mod.UNetModel.infer(
        data=h5,
        model=weights,
        output=out1,
        device="cpu",
        site_weighting=True,
        sigma=sigma,
    )

    t0 = _read_prob_table(out0)
    t1 = _read_prob_table(out1)

    # For A at pos=102:
    # window0 contributes at t=2 (center, weight 1.0) value=3 twice
    # window1 contributes at t=0 (edge, weight g[0]) value=6 twice
    g = _gauss_weights(5, sigma)
    w_center = float(g[2])
    w_edge = float(g[0])
    num = 2 * 3 * w_center + 2 * 6 * w_edge
    den = 2 * w_center + 2 * w_edge
    exp = _sigmoid_scalar(num / den)

    assert t1[("A", 102)][0] == pytest.approx(exp, rel=1e-6, abs=1e-6)
    assert t1[("A", 102)][0] != pytest.approx(t0[("A", 102)][0], rel=1e-9, abs=1e-9)
