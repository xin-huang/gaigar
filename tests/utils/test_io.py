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
from gaishi.utils import initialize_h5, write_h5, write_tsv


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


def _assert_string_table(ds: h5py.Dataset, expected: list[str]) -> None:
    # h5py vlen utf-8 strings often show up as dtype=object
    sdt = h5py.check_string_dtype(ds.dtype)
    assert sdt is not None
    assert sdt.encoding == "utf-8"

    got = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in ds[()]
    ]
    assert got == expected


def test_initialize_h5_train_creates_expected_layout(tmp_path):
    h5_path = tmp_path / "train.h5"

    ds_type = "train"
    num = 7
    N = 3
    L = 11
    chrom = "chr7"
    ref_table = ["r0", "r1", "r2"]
    tgt_table = ["t0", "t1", "t2"]

    initialize_h5(
        h5_path,
        ds_type=ds_type,
        num_genotype_matrices=num,
        N=N,
        L=L,
        chromosome=chrom,
        ref_table=ref_table,
        tgt_table=tgt_table,
    )

    with h5py.File(h5_path, "r") as h5f:
        # groups
        assert "/data" in h5f
        assert "/index" in h5f
        assert "/meta" in h5f
        assert "/targets" in h5f
        assert "/coords" not in h5f

        # meta attrs
        meta = h5f["/meta"]
        assert int(meta.attrs["n"]) == num
        assert int(meta.attrs["N"]) == N
        assert int(meta.attrs["L"]) == L
        assert str(meta.attrs["Chromosome"]) == chrom
        assert int(meta.attrs["n_written"]) == 0

        # sample tables
        _assert_string_table(h5f["/meta/ref_sample_table"], ref_table)
        _assert_string_table(h5f["/meta/tgt_sample_table"], tgt_table)

        # datasets: data
        for name, dt in [
            ("/data/Ref_genotype", np.uint32),
            ("/data/Tgt_genotype", np.uint32),
            ("/data/Gap_to_prev", np.int64),
            ("/data/Gap_to_next", np.int64),
        ]:
            ds = h5f[name]
            assert ds.shape == (num, N, L)
            assert ds.dtype == np.dtype(dt)
            assert ds.chunks == (1, N, L)
            assert ds.compression == "lzf"

        # datasets: index ids
        ref_ids = h5f["/index/ref_ids"]
        tgt_ids = h5f["/index/tgt_ids"]
        assert ref_ids.shape == (num, N)
        assert tgt_ids.shape == (num, N)
        assert ref_ids.dtype == np.dtype(np.uint32)
        assert tgt_ids.dtype == np.dtype(np.uint32)
        assert ref_ids.chunks == (min(64, num), N)
        assert tgt_ids.chunks == (min(64, num), N)

        # datasets: train-only
        lab = h5f["/targets/Label"]
        assert lab.shape == (num, N, L)
        assert lab.dtype == np.dtype(np.uint8)
        assert lab.chunks == (1, N, L)

        seed = h5f["/index/Seed"]
        rep = h5f["/index/Replicate"]
        assert seed.shape == (num,)
        assert rep.shape == (num,)
        assert seed.dtype == np.dtype(np.int64)
        assert rep.dtype == np.dtype(np.int64)
        assert seed.chunks == (min(1024, num),)
        assert rep.chunks == (min(1024, num),)


def test_initialize_h5_infer_creates_expected_layout(tmp_path):
    h5_path = tmp_path / "infer.h5"

    ds_type = "infer"
    num = 5
    N = 4
    L = 9
    chrom = "chrX"
    ref_table = ["ra", "rb", "rc", "rd"]
    tgt_table = ["ta", "tb", "tc", "td"]

    initialize_h5(
        h5_path,
        ds_type=ds_type,
        num_genotype_matrices=num,
        N=N,
        L=L,
        chromosome=chrom,
        ref_table=ref_table,
        tgt_table=tgt_table,
    )

    with h5py.File(h5_path, "a") as h5f:
        assert "/targets" not in h5f
        assert "/coords" in h5f

        # infer-only
        pos = h5f["/coords/Position"]
        assert pos.shape == (num, L)
        assert pos.dtype == np.dtype(np.int64)
        assert pos.chunks == (1, L)

        # train-only should not exist
        assert "/index/Seed" not in h5f
        assert "/index/Replicate" not in h5f
        assert "/targets/Label" not in h5f


def test_initialize_h5_invalid_ds_type_raises(tmp_path):
    h5_path = tmp_path / "bad.h5"
    with pytest.raises(ValueError):
        initialize_h5(
            h5_path,
            ds_type="bad",
            num_genotype_matrices=3,
            N=2,
            L=2,
            chromosome="chr1",
            ref_table=["r0", "r1"],
            tgt_table=["t0", "t1"],
        )


def test_write_h5_train_single_entry(tmp_path, test_data):
    h5_file = tmp_path / "train.h5"
    lock = mp.Lock()

    cap = 10
    compression = "lzf"

    initialize_h5(
        h5_file,
        ds_type="train",
        num_genotype_matrices=cap,
        N=4,
        L=5,
        chromosome=str(test_data["Chromosome"]),
        ref_table=list(test_data["Ref_sample"]),
        tgt_table=list(test_data["Tgt_sample"]),
        compression=compression,
    )

    # append one entry (initialized already)
    write_h5(h5_file, test_data, ds_type="train", lock=lock)

    with h5py.File(h5_file, "r") as f:
        # Meta
        assert int(f["/meta"].attrs["n"]) == cap
        assert int(f["/meta"].attrs["N"]) == 4
        assert int(f["/meta"].attrs["L"]) == 5
        assert f["/meta"].attrs["Chromosome"] == "213"
        assert int(f["/meta"].attrs["n_written"]) == 1

        ref_table = _read_str_1d(f["/meta/ref_sample_table"])
        tgt_table = _read_str_1d(f["/meta/tgt_sample_table"])
        assert ref_table == list(test_data["Ref_sample"])
        assert tgt_table == list(test_data["Tgt_sample"])

        # Common datasets (preallocated capacity)
        assert f["/data/Ref_genotype"].shape == (cap, 4, 5)
        assert f["/data/Tgt_genotype"].shape == (cap, 4, 5)
        assert f["/data/Gap_to_prev"].shape == (cap, 4, 5)
        assert f["/data/Gap_to_next"].shape == (cap, 4, 5)

        assert f["/data/Ref_genotype"].dtype == np.uint32
        assert f["/data/Tgt_genotype"].dtype == np.uint32
        assert f["/data/Gap_to_prev"].dtype == np.int64
        assert f["/data/Gap_to_next"].dtype == np.int64

        assert f["/data/Ref_genotype"].compression == compression
        assert f["/data/Tgt_genotype"].compression == compression

        # Row 0 matches input
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

        # Indices (preallocated capacity)
        assert f["/index/ref_ids"].shape == (cap, 4)
        assert f["/index/tgt_ids"].shape == (cap, 4)
        np.testing.assert_array_equal(
            f["/index/ref_ids"][0], np.array([0, 1, 2, 3], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            f["/index/tgt_ids"][0], np.array([0, 1, 2, 3], dtype=np.uint32)
        )

        # Train-only datasets
        assert "/targets/Label" in f
        assert f["/targets/Label"].shape == (cap, 4, 5)
        assert f["/targets/Label"].dtype == np.uint8
        np.testing.assert_array_equal(f["/targets/Label"][0], test_data["Label"])

        assert "/index/Seed" in f
        assert "/index/Replicate" in f
        assert f["/index/Seed"].shape == (cap,)
        assert f["/index/Replicate"].shape == (cap,)
        assert int(f["/index/Seed"][0]) == int(test_data["Seed"])
        assert int(f["/index/Replicate"][0]) == int(test_data["Replicate"])

        # Infer-only should not exist
        assert "/coords/Position" not in f


def test_write_h5_infer_single_entry(tmp_path, test_data):
    h5_file = tmp_path / "infer.h5"
    lock = mp.Lock()

    infer_entry = dict(test_data)
    infer_entry.pop("Label")
    infer_entry.pop("Seed")
    infer_entry.pop("Replicate")

    cap = 10
    compression = "lzf"

    initialize_h5(
        h5_file,
        ds_type="infer",
        num_genotype_matrices=cap,
        N=4,
        L=5,
        chromosome=str(infer_entry["Chromosome"]),
        ref_table=list(infer_entry["Ref_sample"]),
        tgt_table=list(infer_entry["Tgt_sample"]),
        compression=compression,
    )

    write_h5(h5_file, infer_entry, ds_type="infer", lock=lock)

    with h5py.File(h5_file, "r") as f:
        # Meta
        assert int(f["/meta"].attrs["n"]) == cap
        assert int(f["/meta"].attrs["N"]) == 4
        assert int(f["/meta"].attrs["L"]) == 5
        assert f["/meta"].attrs["Chromosome"] == "213"
        assert int(f["/meta"].attrs["n_written"]) == 1

        # Common datasets exist (preallocated capacity)
        assert f["/data/Ref_genotype"].shape == (cap, 4, 5)
        assert f["/data/Tgt_genotype"].shape == (cap, 4, 5)
        assert f["/data/Gap_to_prev"].shape == (cap, 4, 5)
        assert f["/data/Gap_to_next"].shape == (cap, 4, 5)

        # Infer-only dataset
        assert "/coords/Position" in f
        assert f["/coords/Position"].shape == (cap, 5)
        np.testing.assert_array_equal(
            f["/coords/Position"][0], test_data["Position"].astype(np.int64)
        )

        # Train-only should not exist
        assert "/targets/Label" not in f
        assert "/index/Seed" not in f
        assert "/index/Replicate" not in f
