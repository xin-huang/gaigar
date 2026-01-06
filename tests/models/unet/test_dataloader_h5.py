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


import numpy as np
import pytest

from gaishi.models.unet import split_keys


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
