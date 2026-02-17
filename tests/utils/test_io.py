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


import copy, h5py
import pytest
import multiprocessing as mp
import numpy as np
from gaishi.utils import write_h5, write_tsv


@pytest.fixture
def test_data():
    return {
        "Chromosome": "213",
        "Start": "Random",
        "End": "Random",
        "Position": np.array([0, 1, 2, 3, 4]),
        "Position_index": [0, 1, 2, 3, 4],
        "Gap_to_prev": np.array(
            [
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ],
            dtype=np.int64,
        ),
        "Gap_to_next": np.array(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
            dtype=np.int64,
        ),
        "Ref_sample": ["tsk_0_1", "tsk_0_2", "tsk_1_1", "tsk_1_2"],
        "Ref_genotype": np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1],
            ],
            dtype=np.uint32,
        ),
        "Tgt_sample": ["tsk_2_1", "tsk_2_2", "tsk_3_1", "tsk_3_2"],
        "Tgt_genotype": np.array(
            [
                [1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 1, 1],
            ],
            dtype=np.uint32,
        ),
        "Replicate": 666,
        "Seed": 4836,
        "Label": np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    }


def test_write_tsv_appends_rows(tmp_path):
    tsv_file = tmp_path / "out.tsv"
    lock = mp.Lock()

    d1 = {"A": np.array([1, 2]), "B": 3}
    d2 = {"A": np.array([9, 8, 7]), "B": 4}

    write_tsv(str(tsv_file), d1, lock)
    write_tsv(str(tsv_file), d2, lock)

    lines = tsv_file.read_text().splitlines()
    assert lines == ["[1, 2]\t3", "[9, 8, 7]\t4"]


def _read_str_1d(ds) -> list[str]:
    """Read a 1D h5py string dataset into Python strings."""
    out = []
    for x in ds[...]:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def test_write_h5_train_single_entry(tmp_path, test_data):
    h5_file = tmp_path / "train.h5"
    lock = mp.Lock()

    write_h5(str(h5_file), [test_data], ds_type="train", lock=lock)

    with h5py.File(h5_file, "r") as f:
        # Meta
        assert f["/meta"].attrs["N"] == 4
        assert f["/meta"].attrs["L"] == 5
        assert f["/meta"].attrs["Chromosome"] == "213"

        ref_table = _read_str_1d(f["/meta/ref_sample_table"])
        tgt_table = _read_str_1d(f["/meta/tgt_sample_table"])
        assert ref_table == test_data["Ref_sample"]
        assert tgt_table == test_data["Tgt_sample"]

        # Common datasets
        assert f["/data/Ref_genotype"].shape == (1, 4, 5)
        assert f["/data/Tgt_genotype"].shape == (1, 4, 5)
        assert f["/data/Gap_to_prev"].shape == (1, 4, 5)
        assert f["/data/Gap_to_next"].shape == (1, 4, 5)

        assert f["/data/Ref_genotype"].dtype == np.uint32
        assert f["/data/Tgt_genotype"].dtype == np.uint32
        assert f["/data/Gap_to_prev"].dtype == np.int64
        assert f["/data/Gap_to_next"].dtype == np.int64

        np.testing.assert_array_equal(
            f["/data/Ref_genotype"][0], test_data["Ref_genotype"]
        )
        np.testing.assert_array_equal(
            f["/data/Tgt_genotype"][0], test_data["Tgt_genotype"]
        )
        np.testing.assert_array_equal(
            f["/data/Gap_to_prev"][0], test_data["Gap_to_prev"]
        )
        np.testing.assert_array_equal(
            f["/data/Gap_to_next"][0], test_data["Gap_to_next"]
        )

        # Indices
        assert f["/index/ref_ids"].shape == (1, 4)
        assert f["/index/tgt_ids"].shape == (1, 4)
        np.testing.assert_array_equal(
            f["/index/ref_ids"][0], np.array([0, 1, 2, 3], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            f["/index/tgt_ids"][0], np.array([0, 1, 2, 3], dtype=np.uint32)
        )

        # Train-only datasets
        assert "/targets/Label" in f
        assert f["/targets/Label"].shape == (1, 4, 5)
        assert f["/targets/Label"].dtype == np.uint8
        np.testing.assert_array_equal(f["/targets/Label"][0], test_data["Label"])

        assert "/index/Seed" in f
        assert "/index/Replicate" in f
        np.testing.assert_array_equal(
            f["/index/Seed"][...], np.array([4836], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            f["/index/Replicate"][...], np.array([666], dtype=np.int64)
        )

        # Infer-only should not exist
        assert "/coords/Position" not in f


def test_write_h5_infer_single_entry(tmp_path, test_data):
    h5_file = tmp_path / "infer.h5"
    lock = mp.Lock()

    infer_entry = dict(test_data)
    infer_entry.pop("Label")
    infer_entry.pop("Seed")
    infer_entry.pop("Replicate")

    write_h5(str(h5_file), infer_entry, ds_type="infer", lock=lock)

    with h5py.File(h5_file, "r") as f:
        # Meta
        assert f["/meta"].attrs["n"] == 1
        assert f["/meta"].attrs["N"] == 4
        assert f["/meta"].attrs["L"] == 5
        assert f["/meta"].attrs["Chromosome"] == "213"

        # Common datasets exist
        assert f["/data/Ref_genotype"].shape == (1, 4, 5)
        assert f["/data/Tgt_genotype"].shape == (1, 4, 5)
        assert f["/data/Gap_to_prev"].shape == (1, 4, 5)
        assert f["/data/Gap_to_next"].shape == (1, 4, 5)

        # Infer-only dataset
        assert "/coords/Position" in f
        assert f["/coords/Position"].shape == (1, 5)
        np.testing.assert_array_equal(
            f["/coords/Position"][0], test_data["Position"].astype(np.int64)
        )

        # Train-only should not exist
        assert "/targets/Label" not in f
        assert "/index/Seed" not in f
        assert "/index/Replicate" not in f
