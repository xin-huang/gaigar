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
import pytest
import torch
import numpy as np
from dataclasses import FrozenInstanceError
from gaishi.models.unet.dataloader_h5 import H5BatchSpec
from gaishi.models.unet.dataloader_h5 import H5KeyChunkDataset
from gaishi.models.unet.dataloader_h5 import make_h5_collate_fn
from gaishi.models.unet.dataloader_h5 import build_dataloaders_from_h5
from gaishi.models.unet.dataloader_h5 import split_keys


# unit tests for H5BatchSpec
def test_n_keys_per_batch_returns_expected_value() -> None:
    spec = H5BatchSpec(chunk_size=4, batch_size=32)
    assert spec.n_keys_per_batch == 8


def test_n_keys_per_batch_raises_when_not_divisible() -> None:
    spec = H5BatchSpec(chunk_size=6, batch_size=32)

    with pytest.raises(ValueError, match="must be divisible"):
        _ = spec.n_keys_per_batch


def test_h5_batch_spec_is_frozen() -> None:
    spec = H5BatchSpec(chunk_size=4, batch_size=32)

    with pytest.raises(FrozenInstanceError):
        spec.batch_size = 16  # type: ignore[misc]


def test_n_keys_per_batch_raises_when_chunk_size_is_zero() -> None:
    spec = H5BatchSpec(chunk_size=0, batch_size=32)

    with pytest.raises(ZeroDivisionError):
        _ = spec.n_keys_per_batch


# unit tests for H5KeyChunkDataset
def _make_h5_file(tmp_path, *, keys=("0", "1", "2"), with_y=True, n_channels=4):
    h5_path = tmp_path / "test.h5"
    chunk_size, individuals, polymorphisms = 4, 3, 11

    x_store = {}
    y_store = {}

    with h5py.File(h5_path, "w") as f:
        for k in keys:
            grp = f.create_group(str(k))
            x = np.random.randint(
                0,
                2,
                size=(chunk_size, n_channels, individuals, polymorphisms),
                dtype=np.int32,
            )
            grp.create_dataset("x_0", data=x)
            x_store[str(k)] = x

            if with_y:
                y = np.random.randint(
                    0,
                    2,
                    size=(chunk_size, 1, individuals, polymorphisms),
                    dtype=np.int8,
                )
                grp.create_dataset("y", data=y)
                y_store[str(k)] = y

    return str(h5_path), list(map(str, keys)), x_store, y_store


def _close_if_open(ds: H5KeyChunkDataset) -> None:
    if getattr(ds, "_h5", None) is not None:
        ds._h5.close()
        ds._h5 = None


def test_len_matches_number_of_keys(tmp_path) -> None:
    h5_path, keys, *_ = _make_h5_file(tmp_path, keys=("0", "1", "2", "3"))
    ds = H5KeyChunkDataset(h5_path=h5_path, keys=keys)

    try:
        assert len(ds) == 4
    finally:
        _close_if_open(ds)


def test_getitem_reads_correct_key_and_slices_channels(tmp_path) -> None:
    h5_path, keys, x_store, y_store = _make_h5_file(
        tmp_path, keys=("0", "1", "2"), with_y=True, n_channels=4
    )
    ds = H5KeyChunkDataset(h5_path=h5_path, keys=keys, channels=2, require_labels=True)

    try:
        x, y = ds[1]  # key "1"
        assert x.shape[1] == 2
        np.testing.assert_array_equal(x, x_store["1"][:, :2])
        assert y is not None
        np.testing.assert_array_equal(y, y_store["1"])
    finally:
        _close_if_open(ds)


def test_getitem_returns_x_as_requested_dtype(tmp_path) -> None:
    h5_path, keys, x_store, _ = _make_h5_file(
        tmp_path, keys=("0",), with_y=True, n_channels=3
    )
    ds = H5KeyChunkDataset(h5_path=h5_path, keys=keys, channels=3, x_dtype=np.float32)

    try:
        x, y = ds[0]
        assert x.dtype == np.float32
        np.testing.assert_array_equal(x, x_store["0"].astype(np.float32))
        assert y is not None
    finally:
        _close_if_open(ds)


def test_missing_labels_allowed_when_require_labels_false(tmp_path) -> None:
    h5_path, keys, x_store, _ = _make_h5_file(
        tmp_path, keys=("0", "1"), with_y=False, n_channels=2
    )
    ds = H5KeyChunkDataset(h5_path=h5_path, keys=keys, channels=2, require_labels=False)

    try:
        x, y = ds[0]
        np.testing.assert_array_equal(x, x_store["0"][:, :2])
        assert y is None
    finally:
        _close_if_open(ds)


def test_missing_labels_raises_when_require_labels_true(tmp_path) -> None:
    h5_path, keys, *_ = _make_h5_file(tmp_path, keys=("0",), with_y=False, n_channels=2)
    ds = H5KeyChunkDataset(h5_path=h5_path, keys=keys, channels=2, require_labels=True)

    try:
        with pytest.raises(KeyError, match="Missing 'y'"):
            _ = ds[0]
    finally:
        _close_if_open(ds)


def test_lazy_open_file_handle(tmp_path) -> None:
    h5_path, keys, *_ = _make_h5_file(tmp_path, keys=("0",), with_y=True, n_channels=2)
    ds = H5KeyChunkDataset(h5_path=h5_path, keys=keys, channels=2)

    try:
        assert ds._h5 is None
        _ = ds[0]
        assert ds._h5 is not None
        assert isinstance(ds._h5, h5py.File)
    finally:
        _close_if_open(ds)


def test_custom_dataset_names(tmp_path) -> None:
    h5_path = tmp_path / "test_custom.h5"
    key = "0"
    x = np.random.randint(0, 2, size=(2, 5, 3, 7), dtype=np.int32)
    y = np.random.randint(0, 2, size=(2, 1, 3, 7), dtype=np.int8)

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group(key)
        grp.create_dataset("x_alt", data=x)
        grp.create_dataset("y_alt", data=y)

    ds = H5KeyChunkDataset(
        h5_path=str(h5_path),
        keys=[key],
        channels=4,
        x_dataset="x_alt",
        y_dataset="y_alt",
        require_labels=True,
    )

    try:
        x_out, y_out = ds[0]
        np.testing.assert_array_equal(x_out, x[:, :4])
        assert y_out is not None
        np.testing.assert_array_equal(y_out, y)
    finally:
        _close_if_open(ds)


# unit tests for make_h5_collate_fn
def test_collate_concatenates_x_and_y_along_axis0() -> None:
    rng = np.random.default_rng(0)
    collate = make_h5_collate_fn(label_smooth=False, rng=rng)

    x1 = np.ones((2, 3, 4, 5), dtype=np.int32)
    y1 = np.zeros((2, 1, 4, 5), dtype=np.int8)

    x2 = np.full((2, 3, 4, 5), 2, dtype=np.int32)
    y2 = np.ones((2, 1, 4, 5), dtype=np.int8)

    x_out, y_out = collate([(x1, y1), (x2, y2)])

    assert isinstance(x_out, torch.Tensor)
    assert isinstance(y_out, torch.Tensor)

    assert x_out.dtype == torch.float32
    assert y_out.dtype == torch.float32

    assert x_out.shape == (4, 3, 4, 5)
    assert y_out.shape == (4, 1, 4, 5)

    np.testing.assert_array_equal(
        x_out.numpy(), np.concatenate([x1, x2], axis=0).astype(np.float32)
    )
    np.testing.assert_array_equal(
        y_out.numpy(), np.concatenate([y1, y2], axis=0).astype(np.float32)
    )


def test_collate_returns_none_when_no_labels_present() -> None:
    rng = np.random.default_rng(0)
    collate = make_h5_collate_fn(label_smooth=False, rng=rng)

    x1 = np.ones((2, 3, 4, 5), dtype=np.int32)
    x2 = np.full((2, 3, 4, 5), 2, dtype=np.int32)

    x_out, y_out = collate([(x1, None), (x2, None)])

    assert isinstance(x_out, torch.Tensor)
    assert x_out.shape == (4, 3, 4, 5)
    assert x_out.dtype == torch.float32
    assert y_out is None


def test_collate_ignores_none_labels_if_some_labels_exist() -> None:
    rng = np.random.default_rng(0)
    collate = make_h5_collate_fn(label_smooth=False, rng=rng)

    x1 = np.ones((2, 3, 4, 5), dtype=np.int32)
    y1 = np.zeros((2, 1, 4, 5), dtype=np.int8)

    x2 = np.full((2, 3, 4, 5), 2, dtype=np.int32)

    x_out, y_out = collate([(x1, y1), (x2, None)])

    assert y_out is not None
    assert y_out.shape == (2, 1, 4, 5)
    np.testing.assert_array_equal(y_out.numpy(), y1.astype(np.float32))


def test_label_smoothing_changes_labels_with_fixed_rng() -> None:
    # Use a fixed RNG so the smoothing is deterministic.
    rng = np.random.default_rng(123)
    collate = make_h5_collate_fn(label_smooth=True, label_noise=0.2, rng=rng)

    x = np.zeros((2, 3, 4, 5), dtype=np.int32)
    y = np.zeros((2, 1, 4, 5), dtype=np.float32)  # all zeros -> smoothed becomes 0.5*e

    x_out, y_out = collate([(x, y)])

    assert y_out is not None
    y_np = y_out.numpy()

    # For all-zero labels, Y_smooth = 0.5*e, so values should be in [0, 0.1]
    assert y_np.min() >= 0.0
    assert y_np.max() <= 0.1 + 1e-6

    # Determinism check: compute expected with the same RNG sequence
    rng2 = np.random.default_rng(123)
    e = rng2.uniform(0.0, 0.2, size=y.shape).astype(np.float32, copy=False)
    expected = y * (1.0 - e) + 0.5 * e

    np.testing.assert_allclose(y_np, expected, rtol=0, atol=1e-7)


def test_label_smoothing_preserves_ones_within_expected_range() -> None:
    rng = np.random.default_rng(999)
    collate = make_h5_collate_fn(label_smooth=True, label_noise=0.2, rng=rng)

    x = np.zeros((2, 3, 4, 5), dtype=np.int32)
    y = np.ones(
        (2, 1, 4, 5), dtype=np.float32
    )  # all ones -> smoothed becomes 1 - 0.5*e

    _, y_out = collate([(x, y)])
    assert y_out is not None
    y_np = y_out.numpy()

    # For ones, Y_smooth = 1 - 0.5*e, so values should be in [0.9, 1.0]
    assert y_np.min() >= 0.9 - 1e-6
    assert y_np.max() <= 1.0 + 1e-6


# unit tests for build_dataloaders_from_h5
def _write_h5(
    tmp_path, *, keys, chunk_size=4, n_channels=4, individuals=3, polymorphisms=11
):
    h5_path = tmp_path / "dl_test.h5"
    x_store = {}
    y_store = {}

    with h5py.File(h5_path, "w") as f:
        for k in keys:
            grp = f.create_group(str(k))
            x = np.random.randint(
                0,
                2,
                size=(chunk_size, n_channels, individuals, polymorphisms),
                dtype=np.int32,
            )
            y = np.random.randint(
                0, 2, size=(chunk_size, 1, individuals, polymorphisms), dtype=np.int8
            )
            grp.create_dataset("x_0", data=x)
            grp.create_dataset("y", data=y)
            x_store[str(k)] = x
            y_store[str(k)] = y

    return str(h5_path), x_store, y_store


def test_build_dataloaders_shapes_and_types(tmp_path) -> None:
    # 8 keys for train, 4 keys for val
    h5_path, x_store, _ = _write_h5(tmp_path, keys=[str(i) for i in range(12)])

    train_keys = [str(i) for i in range(8)]
    val_keys = [str(i) for i in range(8, 12)]

    spec = H5BatchSpec(chunk_size=4, batch_size=16)  # n_keys_per_batch=4
    channels = 2

    train_loader, val_loader = build_dataloaders_from_h5(
        h5_path=h5_path,
        train_keys=train_keys,
        val_keys=val_keys,
        channels=channels,
        spec=spec,
        num_workers=0,
        pin_memory=False,
        seed=0,
        train_label_smooth=False,
    )

    x, y = next(iter(train_loader))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    # batch_size=16 samples, channels sliced to 2
    assert x.shape == (16, 2, 3, 11)
    assert y.shape == (16, 1, 3, 11)

    xv, yv = next(iter(val_loader))
    assert xv.shape == (16, 2, 3, 11)
    assert yv.shape == (16, 1, 3, 11)


def test_build_dataloaders_channel_slicing(tmp_path) -> None:
    h5_path, x_store, _ = _write_h5(tmp_path, keys=["0", "1", "2", "3"])
    spec = H5BatchSpec(chunk_size=4, batch_size=8)  # n_keys_per_batch=2

    train_loader, _ = build_dataloaders_from_h5(
        h5_path=h5_path,
        train_keys=["0", "1", "2", "3"],
        val_keys=["0", "1"],  # not used here
        channels=3,
        spec=spec,
        num_workers=0,
        pin_memory=False,
        seed=0,
        train_label_smooth=False,
    )

    # Deterministic order is not guaranteed due to shuffle=True, so we only assert channel slicing.
    x, _ = next(iter(train_loader))
    assert x.shape[1] == 3  # sliced channels


def test_drop_last_behavior(tmp_path) -> None:
    # Train has 5 keys, with n_keys_per_batch=2 -> only 2 batches (4 keys), 1 key dropped.
    h5_path, _, _ = _write_h5(tmp_path, keys=[str(i) for i in range(10)])
    train_keys = [str(i) for i in range(5)]
    val_keys = [str(i) for i in range(5, 10)]

    spec = H5BatchSpec(chunk_size=4, batch_size=8)  # n_keys_per_batch=2

    train_loader, val_loader = build_dataloaders_from_h5(
        h5_path=h5_path,
        train_keys=train_keys,
        val_keys=val_keys,
        channels=2,
        spec=spec,
        num_workers=0,
        pin_memory=False,
        seed=0,
        train_label_smooth=False,
    )

    assert len(train_loader) == len(train_keys) // spec.n_keys_per_batch
    assert len(val_loader) == len(val_keys) // spec.n_keys_per_batch


def test_train_label_smoothing_applied_val_not_applied(tmp_path) -> None:
    # Use all-zero labels so smoothing produces >0 values; validation should remain all zeros.
    h5_path = tmp_path / "smooth_test.h5"
    chunk_size, n_channels, individuals, polymorphisms = 4, 2, 3, 7

    with h5py.File(h5_path, "w") as f:
        for k in ["0", "1", "2", "3"]:
            grp = f.create_group(k)
            x = np.zeros(
                (chunk_size, n_channels, individuals, polymorphisms), dtype=np.int32
            )
            y = np.zeros((chunk_size, 1, individuals, polymorphisms), dtype=np.float32)
            grp.create_dataset("x_0", data=x)
            grp.create_dataset("y", data=y)

    spec = H5BatchSpec(chunk_size=4, batch_size=8)  # n_keys_per_batch=2

    train_loader, val_loader = build_dataloaders_from_h5(
        h5_path=str(h5_path),
        train_keys=["0", "1"],
        val_keys=["2", "3"],
        channels=2,
        spec=spec,
        num_workers=0,
        pin_memory=False,
        seed=123,
        train_label_smooth=True,
        train_label_noise=0.2,
    )

    _, y_train = next(iter(train_loader))
    _, y_val = next(iter(val_loader))

    assert y_train is not None
    assert y_val is not None

    y_train_np = y_train.numpy()
    y_val_np = y_val.numpy()

    # Train labels should have been smoothed: for all-zero labels, values in (0, 0.1]
    assert y_train_np.max() > 0.0
    assert y_train_np.min() >= 0.0
    assert y_train_np.max() <= 0.1 + 1e-6

    # Val labels should remain exactly zero
    assert np.all(y_val_np == 0.0)


# unit tests for split_keys
def _expected_split(
    keys: list[str], val_prop: float, seed: int
) -> tuple[list[str], list[str]]:
    """Compute the expected split using the same algorithm as split_keys."""
    keys_copy = list(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys_copy)
    n_val = int(len(keys_copy) * float(val_prop))
    val_keys = keys_copy[:n_val]
    train_keys = keys_copy[n_val:]
    return train_keys, val_keys


def test_split_keys_matches_expected_shuffle_and_partition() -> None:
    keys = [f"k{i}" for i in range(100)]
    val_prop = 0.2
    seed = 123

    train_keys, val_keys = split_keys(keys, val_prop=val_prop, seed=seed)
    exp_train, exp_val = _expected_split(keys, val_prop=val_prop, seed=seed)

    assert train_keys == exp_train
    assert val_keys == exp_val
    assert len(val_keys) == int(len(keys) * val_prop)
    assert len(train_keys) + len(val_keys) == len(keys)


def test_split_keys_is_deterministic_for_same_seed() -> None:
    keys = [f"k{i}" for i in range(50)]

    out1 = split_keys(keys, val_prop=0.1, seed=7)
    out2 = split_keys(keys, val_prop=0.1, seed=7)

    assert out1 == out2


def test_split_keys_does_not_modify_input_list_in_place() -> None:
    keys = [f"k{i}" for i in range(30)]
    keys_before = list(keys)

    _ = split_keys(keys, val_prop=0.2, seed=42)

    assert keys == keys_before


def test_split_keys_val_prop_zero() -> None:
    keys = [f"k{i}" for i in range(10)]

    train_keys, val_keys = split_keys(keys, val_prop=0.0, seed=1)

    assert len(val_keys) == 0
    assert len(train_keys) == len(keys)
    assert set(train_keys) == set(keys)


def test_split_keys_val_prop_one() -> None:
    keys = [f"k{i}" for i in range(10)]

    train_keys, val_keys = split_keys(keys, val_prop=1.0, seed=1)

    assert len(val_keys) == len(keys)
    assert len(train_keys) == 0
    assert set(val_keys) == set(keys)


def test_split_keys_no_overlap_and_all_keys_present() -> None:
    keys = [f"k{i}" for i in range(77)]

    train_keys, val_keys = split_keys(keys, val_prop=0.33, seed=999)

    assert set(train_keys).isdisjoint(set(val_keys))
    assert set(train_keys).union(set(val_keys)) == set(keys)


def test_split_keys_accepts_non_list_sequence() -> None:
    keys = tuple(f"k{i}" for i in range(25))

    train_keys, val_keys = split_keys(keys, val_prop=0.2, seed=5)

    assert isinstance(train_keys, list)
    assert isinstance(val_keys, list)
    assert set(train_keys).union(set(val_keys)) == set(keys)
