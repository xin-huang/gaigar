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


import multiprocessing

import h5py
import numpy as np

from gaishi.utils import write_h5, write_tsv


def test_write_h5_creates_group_and_datasets(tmp_path):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    data_dict = {
        "scalar_int": 7,
        "scalar_float": 3.14,
        "str_list": ["a", "bb", "ccc"],
        "int_list": [1, 2, 3],
        "ndarray": np.array([[1, 2], [3, 4]], dtype=np.int32),
    }

    write_h5(str(h5_file), "grp1", data_dict, lock)

    with h5py.File(h5_file, "r") as f:
        assert "grp1" in f
        g = f["grp1"]

        assert set(g.keys()) == set(data_dict.keys())

        # Scalars: no compression, scalar shape
        assert g["scalar_int"].shape == ()
        assert g["scalar_int"].compression is None
        assert int(g["scalar_int"][()]) == 7

        assert g["scalar_float"].shape == ()
        assert g["scalar_float"].compression is None
        assert float(g["scalar_float"][()]) == 3.14

        # Arrays/lists: lzf compression
        assert g["int_list"].compression == "lzf"
        assert np.array_equal(g["int_list"][()], np.array([1, 2, 3]))

        assert g["ndarray"].compression == "lzf"
        assert np.array_equal(g["ndarray"][()], data_dict["ndarray"])

        # String list: stored as bytes (dtype kind 'S'), lzf compression
        assert g["str_list"].dtype.kind == "S"
        assert g["str_list"].compression == "lzf"
        assert [x for x in g["str_list"][()]] == [b"a", b"bb", b"ccc"]


def test_write_h5_raises_if_group_exists(tmp_path):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    data_dict = {"x": [1, 2, 3]}
    write_h5(str(h5_file), "grp", data_dict, lock)

    import pytest

    with pytest.raises(ValueError):
        write_h5(str(h5_file), "grp", data_dict, lock)


def test_write_tsv_appends_rows(tmp_path):
    tsv_file = tmp_path / "out.tsv"
    lock = multiprocessing.Lock()

    d1 = {"A": np.array([1, 2]), "B": 3}
    d2 = {"A": np.array([9, 8, 7]), "B": 4}

    write_tsv(str(tsv_file), d1, lock)
    write_tsv(str(tsv_file), d2, lock)

    lines = tsv_file.read_text().splitlines()
    assert lines == ["[1, 2]\t3", "[9, 8, 7]\t4"]
