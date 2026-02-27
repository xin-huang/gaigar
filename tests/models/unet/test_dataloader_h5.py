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


import h5py
import numpy as np
import pytest
import torch

from gaishi.models.unet.dataloader_h5 import (  # noqa: F401
    H5Dataset,
    build_dataloaders_from_h5,
    make_h5_collate_fn,
)


def _write_h5(
    path: str,
    *,
    R: int = 8,
    N: int = 3,
    L: int = 5,
    with_labels: bool = True,
):
    rng = np.random.default_rng(42)

    ref = rng.integers(0, 3, size=(R, N, L), dtype=np.int16)
    tgt = rng.integers(0, 3, size=(R, N, L), dtype=np.int16)
    gp = rng.integers(0, 10, size=(R, N, L), dtype=np.int16)
    gn = rng.integers(0, 10, size=(R, N, L), dtype=np.int16)

    if with_labels:
        # Deterministic binary labels with a mix of 0/1.
        lab = (
            np.arange(R)[:, None, None]
            + np.arange(N)[None, :, None]
            + np.arange(L)[None, None, :]
        ) % 2
        lab = lab.astype(np.uint8)
    else:
        lab = None

    with h5py.File(path, "w") as h5:
        g_data = h5.create_group("data")
        g_data.create_dataset("Ref_genotype", data=ref)
        g_data.create_dataset("Tgt_genotype", data=tgt)
        g_data.create_dataset("Gap_to_prev", data=gp)
        g_data.create_dataset("Gap_to_next", data=gn)

        if with_labels:
            g_tgt = h5.create_group("targets")
            g_tgt.create_dataset("Label", data=lab)

    return ref, tgt, gp, gn, lab


@pytest.fixture()
def h5_with_labels(tmp_path):
    p = tmp_path / "toy_with_labels.h5"
    arrays = _write_h5(str(p), with_labels=True)
    return str(p), arrays


@pytest.fixture()
def h5_no_labels(tmp_path):
    p = tmp_path / "toy_no_labels.h5"
    arrays = _write_h5(str(p), with_labels=False)
    return str(p), arrays


def test_h5dataset_len_and_indexing(h5_with_labels):
    path, (ref, tgt, gp, gn, lab) = h5_with_labels
    R = ref.shape[0]

    ds_all = H5Dataset(h5_file=path, indices=None, channels=4, require_labels=True)
    assert len(ds_all) == R

    subset = [3, 0, 7]
    ds_sub = H5Dataset(h5_file=path, indices=subset, channels=2, require_labels=True)
    assert len(ds_sub) == len(subset)

    x0, y0 = ds_sub[0]
    assert x0.shape == (2,) + ref.shape[1:]
    assert y0 is not None
    assert y0.shape == (1,) + lab.shape[1:]
    assert x0.dtype == np.int32  # default x_dtype

    # Clean up open handles (dataset caches file handle).
    if ds_all._h5 is not None:
        ds_all._h5.close()
    if ds_sub._h5 is not None:
        ds_sub._h5.close()


def test_h5dataset_channels_2_and_4(h5_with_labels):
    path, (ref, tgt, gp, gn, lab) = h5_with_labels

    ds2 = H5Dataset(h5_file=path, channels=2, require_labels=True)
    x2, y2 = ds2[1]
    assert x2.shape == (2,) + ref.shape[1:]
    assert np.array_equal(x2[0], ref[1])
    assert np.array_equal(x2[1], tgt[1])
    assert y2 is not None
    assert np.array_equal(y2[0], lab[1])

    ds4 = H5Dataset(h5_file=path, channels=4, require_labels=True)
    x4, y4 = ds4[1]
    assert x4.shape == (4,) + ref.shape[1:]
    assert np.array_equal(x4[0], ref[1])
    assert np.array_equal(x4[1], tgt[1])
    assert np.array_equal(x4[2], gp[1])
    assert np.array_equal(x4[3], gn[1])
    assert y4 is not None
    assert np.array_equal(y4[0], lab[1])

    if ds2._h5 is not None:
        ds2._h5.close()
    if ds4._h5 is not None:
        ds4._h5.close()


def test_h5dataset_invalid_channels_raises(h5_with_labels):
    path, _ = h5_with_labels
    with pytest.raises(ValueError):
        _ = H5Dataset(h5_file=path, channels=3)


def test_h5dataset_missing_labels_behavior(h5_no_labels):
    path, _ = h5_no_labels

    ds_req = H5Dataset(h5_file=path, channels=2, require_labels=True)
    with pytest.raises(KeyError):
        _ = ds_req[0]

    ds_opt = H5Dataset(h5_file=path, channels=2, require_labels=False)
    x, y = ds_opt[0]
    assert x.shape[0] == 2
    assert y is None

    if ds_req._h5 is not None:
        ds_req._h5.close()
    if ds_opt._h5 is not None:
        ds_opt._h5.close()


def test_make_h5_collate_fn_no_labels():
    x = np.zeros((4, 2, 3), dtype=np.int32)
    batch = [(x, None), (x, None)]
    collate = make_h5_collate_fn(label_smooth=False)
    xb, yb = collate(batch)

    assert isinstance(xb, torch.Tensor)
    assert xb.dtype == torch.float32
    assert xb.shape == (2, 4, 2, 3)
    assert yb is None


def test_make_h5_collate_fn_label_smoothing_exact():
    # y is binary so smoothing should follow: y*(1-e) + 0.5*e
    x = np.zeros((4, 2, 3), dtype=np.int32)
    y = np.array(
        [[[1, 0, 1], [0, 1, 0]]],  # (1,N,L) where N=2, L=3
        dtype=np.float32,
    )

    batch = [(x, y), (x, y)]
    rng = np.random.default_rng(123)
    collate = make_h5_collate_fn(label_smooth=True, label_noise=0.01, rng=rng)
    xb, yb = collate(batch)

    assert yb is not None
    assert xb.shape == (2, 4, 2, 3)
    assert yb.shape == (2, 1, 2, 3)
    assert yb.dtype == torch.float32

    # Recompute expected e with the same RNG seed.
    rng2 = np.random.default_rng(123)
    y_np = np.stack([y, y], axis=0).astype(np.float32, copy=False)  # (B,1,N,L)
    e = rng2.uniform(0.0, 0.01, size=y_np.shape).astype(np.float32, copy=False)
    expected = y_np * (1.0 - e) + 0.5 * e

    assert torch.allclose(yb, torch.from_numpy(expected), atol=0.0, rtol=0.0)


def test_build_dataloaders_from_h5_shapes_indices_and_content(h5_with_labels):
    path, (ref, _tgt, _gp, _gn, _lab) = h5_with_labels
    R, N, L = ref.shape

    val_prop = 0.25
    n_val = int(R * val_prop)
    n_train = R - n_val
    batch_size = 2

    torch.manual_seed(0)

    train_loader, val_loader, train_idx, val_idx = build_dataloaders_from_h5(
        h5_file=path,
        channels=4,
        batch_size=batch_size,
        val_prop=val_prop,
        num_workers=0,
        pin_memory=False,
        seed=0,
        train_label_smooth=True,
        train_label_noise=0.01,
    )

    assert isinstance(train_idx, list)
    assert isinstance(val_idx, list)
    assert len(train_idx) == n_train
    assert len(val_idx) == n_val
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).union(set(val_idx)) == set(range(R))

    # Deterministic split given seed
    _, _, train_idx2, val_idx2 = build_dataloaders_from_h5(
        h5_file=path,
        channels=4,
        batch_size=batch_size,
        val_prop=val_prop,
        num_workers=0,
        pin_memory=False,
        seed=0,
        train_label_smooth=True,
        train_label_noise=0.01,
    )
    assert train_idx2 == train_idx
    assert val_idx2 == val_idx

    # Loader lengths respect drop_last=True
    assert len(train_loader) == n_train // batch_size
    assert len(val_loader) == n_val // batch_size

    xb_tr, yb_tr = next(iter(train_loader))
    assert xb_tr.shape == (batch_size, 4, N, L)
    assert xb_tr.dtype == torch.float32
    assert yb_tr is not None
    assert yb_tr.shape == (batch_size, 1, N, L)
    # smoothing on train: should contain non-binary values
    assert (~((yb_tr == 0.0) | (yb_tr == 1.0))).any().item() is True

    xb_val, yb_val = next(iter(val_loader))
    assert xb_val.shape == (batch_size, 4, N, L)
    assert yb_val is not None
    assert yb_val.shape == (batch_size, 1, N, L)
    # no smoothing on val: strictly binary
    assert torch.all((yb_val == 0.0) | (yb_val == 1.0)).item() is True

    # val loader is not shuffled: first element corresponds to val_idx[0]
    r0 = int(val_idx[0])
    with h5py.File(path, "r") as h5:
        exp_x = np.stack(
            [
                np.asarray(h5["/data/Ref_genotype"][r0]),
                np.asarray(h5["/data/Tgt_genotype"][r0]),
                np.asarray(h5["/data/Gap_to_prev"][r0]),
                np.asarray(h5["/data/Gap_to_next"][r0]),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        exp_y = np.asarray(h5["/targets/Label"][r0])[None, :, :].astype(
            np.float32, copy=False
        )

    assert torch.allclose(xb_val[0], torch.from_numpy(exp_x), atol=0.0, rtol=0.0)
    assert torch.allclose(yb_val[0], torch.from_numpy(exp_y), atol=0.0, rtol=0.0)

    # Close cached HDF5 handle on the underlying base dataset (shared by both subsets)
    base_ds = train_loader.dataset.dataset  # Subset.dataset -> base H5Dataset
    if getattr(base_ds, "_h5", None) is not None:
        base_ds._h5.close()
