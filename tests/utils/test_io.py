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
import multiprocessing
import pytest
import numpy as np
from gaishi.utils import write_h5, write_tsv
from gaishi.utils.io import _normalize_hdf_entry
from gaishi.utils.io import _pack_hdf_entry
from gaishi.utils.io import _append_hdf_entries


@pytest.fixture
def test_data():
    return {
        "Chromosome": "213",
        "Start": "Random",
        "End": "Random",
        "Position": np.array([0, 1, 2, 3, 4]),
        "Position_index": [0, 1, 2, 3, 4],
        "Forward_relative_position": np.array(
            [
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ],
            dtype=np.int64,
        ),
        "Backward_relative_position": np.array(
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


def test_write_h5_single_entry_creates_group_and_updates_last_index(tmp_path, test_data):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    d = copy.deepcopy(test_data)

    nxt = write_h5(
        file_name=str(h5_file),
        entries=d,
        lock=lock,
        stepsize=192,
        is_phased=True,
        chunk_size=1,
        fwbw=True,
        start_nr=None,
        set_attributes=True,
    )
    assert nxt == 1

    with h5py.File(h5_file, "r") as f:
        assert int(f.attrs["last_index"]) == 1
        assert "0" in f

        g = f["0"]
        for name in ("x_0", "y", "indices", "pos", "ix"):
            assert name in g

        x = g["x_0"][()]
        y = g["y"][()]
        ind = g["indices"][()]
        pos = g["pos"][()]
        ix = g["ix"][()]

        # Shapes implied by current writer:
        # x: (1, 4, n_samples, n_sites)  -> 2 genotype channels + 2 fwbw channels
        assert x.shape == (1, 4, 4, 5)
        assert y.shape == (1, 1, 4, 5)
        assert ind.shape == (1, 2, 4, 2)
        assert pos.shape == (1, 1, 1, 2)
        assert ix.shape == (1, 1, 1)

        # Content sanity checks
        assert np.array_equal(x[0, 0], d["Ref_genotype"])
        assert np.array_equal(x[0, 1], d["Tgt_genotype"])
        assert np.array_equal(x[0, 2], d["Forward_relative_position"].astype(np.uint32))
        assert np.array_equal(x[0, 3], d["Backward_relative_position"].astype(np.uint32))
        assert np.array_equal(y[0, 0], d["Label"])

        # is_phased=True => hap is (hap-1)
        expected_ref = np.array([
                [0, 0],  # tsk_0_1 -> hap 1 -> 0
                [0, 1],  # tsk_0_2 -> hap 2 -> 1
                [1, 0],  # tsk_1_1 -> 0
                [1, 1],  # tsk_1_2 -> 1
            ],
            dtype=np.uint32,
        )

        expected_tgt = np.array(
            [
                [2, 0],  # tsk_2_1 -> 0
                [2, 1],  # tsk_2_2 -> 1
                [3, 0],  # tsk_3_1 -> 0
                [3, 1],  # tsk_3_2 -> 1
            ],
            dtype=np.uint32,
        )

        assert np.array_equal(ind[0, 0], expected_ref)
        assert np.array_equal(ind[0, 1], expected_tgt)

        # Start=="Random" => Start=0, End=stepsize
        assert np.array_equal(pos[0, 0, 0], np.array([0, 192], dtype=np.uint32))

        assert int(ix[0, 0, 0]) == d["Replicate"]


def test_write_h5_list_of_entries_appends_multiple_groups(tmp_path, test_data):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    d1 = copy.deepcopy(test_data)
    d1["Replicate"] = 1

    d2 = copy.deepcopy(test_data)
    d2["Replicate"] = 2

    nxt = write_h5(
        file_name=str(h5_file),
        entries=[d1, d2],
        lock=lock,
        stepsize=192,
        is_phased=True,
        chunk_size=1,
        fwbw=True,
        start_nr=None,
        set_attributes=True,
    )
    assert nxt == 2

    with h5py.File(h5_file, "r") as f:
        assert int(f.attrs["last_index"]) == 2
        assert "0" in f and "1" in f
        assert int(f["0/ix"][0, 0, 0]) == 1
        assert int(f["1/ix"][0, 0, 0]) == 2


def test_write_h5_respects_start_nr(tmp_path, test_data):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    d = copy.deepcopy(test_data)
    d["Replicate"] = 5

    nxt = write_h5(
        file_name=str(h5_file),
        entries=d,
        lock=lock,
        stepsize=192,
        is_phased=True,
        chunk_size=1,
        fwbw=True,
        start_nr=10,
        set_attributes=False,
    )
    assert nxt == 11

    with h5py.File(h5_file, "r") as f:
        assert "10" in f
        assert "0" not in f
        assert "last_index" not in f.attrs


def test_write_tsv_appends_rows(tmp_path):
    tsv_file = tmp_path / "out.tsv"
    lock = multiprocessing.Lock()

    d1 = {"A": np.array([1, 2]), "B": 3}
    d2 = {"A": np.array([9, 8, 7]), "B": 4}

    write_tsv(str(tsv_file), d1, lock)
    write_tsv(str(tsv_file), d2, lock)

    lines = tsv_file.read_text().splitlines()
    assert lines == ["[1, 2]\t3", "[9, 8, 7]\t4"]


def test_normalize_hdf_entry_random_start_phased():
    d = {
        "Start": "Random",
        "End": 999999,  # should be overwritten
        "Ref_sample": ["ref_0_1", "ref_12_2"],
        "Tgt_sample": ["tgt_3_1"],
    }

    out = _normalize_hdf_entry(d, stepsize=192, is_phased=True)

    # In-place
    assert out is d

    assert d["Start"] == 0
    assert d["End"] == 192
    assert d["StartEnd"] == [0, 192]

    # Phased: hap is 0-based (hap-1)
    assert d["Ref_sample"] == [[0, 0], [12, 1]]
    assert d["Tgt_sample"] == [[3, 0]]


def test_normalize_hdf_entry_nonrandom_start_unphased():
    d = {
        "Start": 100,
        "End": 250,
        "Ref_sample": ["ref_5_2"],
        "Tgt_sample": ["tgt_7_1", "tgt_7_2"],
    }

    _normalize_hdf_entry(d, stepsize=999, is_phased=False)

    # Start/End unchanged when Start != "Random"
    assert d["Start"] == 100
    assert d["End"] == 250
    assert d["StartEnd"] == [100, 250]

    # Unphased: hap forced to 0
    assert d["Ref_sample"] == [[5, 0]]
    assert d["Tgt_sample"] == [[7, 0], [7, 0]]


def test_normalize_hdf_entry_missing_required_key_raises_keyerror():
    d = {
        "Start": 0,
        "End": 10,
        "Ref_sample": ["ref_0_1"],
        # "Tgt_sample" missing
    }
    with pytest.raises(KeyError):
        _normalize_hdf_entry(d, stepsize=10, is_phased=True)


def test_normalize_hdf_entry_malformed_sample_id_raises():
    d = {
        "Start": 0,
        "End": 10,
        "Ref_sample": ["ref_0"],  # malformed, missing hap
        "Tgt_sample": ["tgt_1_1"],
    }

    # Current implementation may raise IndexError (missing parts) or ValueError (non-int).
    with pytest.raises((IndexError, ValueError)):
        _normalize_hdf_entry(d, stepsize=10, is_phased=True)


def test_pack_hdf_entry_orders_fields_correctly():
    ref_g = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    tgt_g = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    label = np.array([[1, 0]], dtype=np.uint8)

    ref_s = [[0, 0], [1, 1]]
    tgt_s = [[2, 0]]
    start_end = [0, 192]

    d = {
        "Ref_genotype": ref_g,
        "Tgt_genotype": tgt_g,
        "Label": label,
        "Ref_sample": ref_s,
        "Tgt_sample": tgt_s,
        "StartEnd": start_end,
        "End": 192,
        "Replicate": 7,
        "Position": [10, 20, 30],
        "Forward_relative_position": [0.1, 0.2, 0.3],
        "Backward_relative_position": [0.9, 0.8, 0.7],
    }

    packed = _pack_hdf_entry(d)

    assert isinstance(packed, list)
    assert len(packed) == 9

    # Group 0: genotypes
    assert packed[0][0] is ref_g
    assert packed[0][1] is tgt_g

    # Group 1: label
    assert packed[1][0] is label

    # Group 2: samples
    assert packed[2][0] is ref_s
    assert packed[2][1] is tgt_s

    # Group 3: StartEnd
    assert packed[3][0] is start_end

    # Group 4..8
    assert packed[4][0] == 192
    assert packed[5][0] == 7
    assert packed[6][0] == [10, 20, 30]
    assert packed[7][0] == [0.1, 0.2, 0.3]
    assert packed[8][0] == [0.9, 0.8, 0.7]


def test_pack_hdf_entry_does_not_modify_input():
    d = {
        "Ref_genotype": [1],
        "Tgt_genotype": [2],
        "Label": [3],
        "Ref_sample": [4],
        "Tgt_sample": [5],
        "StartEnd": [6],
        "End": 7,
        "Replicate": 8,
        "Position": [9],
        "Forward_relative_position": [10],
        "Backward_relative_position": [11],
    }

    before = dict(d)
    _ = _pack_hdf_entry(d)
    assert d == before


def test_pack_hdf_entry_missing_key_raises_keyerror():
    d = {
        "Ref_genotype": [1],
        "Tgt_genotype": [2],
        # "Label" missing
        "Ref_sample": [4],
        "Tgt_sample": [5],
        "StartEnd": [6],
        "End": 7,
        "Replicate": 8,
        "Position": [9],
        "Forward_relative_position": [10],
        "Backward_relative_position": [11],
    }

    with pytest.raises(KeyError):
        _pack_hdf_entry(d)


def _make_packed_entry(
    H: int = 3,
    W: int = 4,
    N: int = 2,
    rep: int = 7,
    start_end=(0, 192),
):
    """
    Create one packed entry matching the expected on-disk schema.

    entry[0] : list of 2 (H,W) arrays -> base channels
    entry[1] : list of 1 (H,W) array -> label
    entry[2] : list of 2 (N,2) arrays -> indices (ref/tgt)
    entry[3] : list of 1 (2,) array/list -> StartEnd
    entry[5] : list of 1 scalar -> replicate/index (ix)
    entry[-2], entry[-1] : (1,H,W) integer arrays -> fw/bw channels
    """
    ref = np.arange(H * W, dtype=np.uint32).reshape(H, W)
    tgt = np.arange(H * W, dtype=np.uint32).reshape(H, W) + 100

    label = np.arange(H * W, dtype=np.uint8).reshape(H, W) % 2

    ref_idx = np.array([[0, 0], [1, 1]], dtype=np.uint32)[:N]
    tgt_idx = np.array([[2, 0], [3, 1]], dtype=np.uint32)[:N]

    # Forward/backward channels must be integer dtype and shape (1, H, W)
    fw = np.zeros((1, H, W), dtype=np.int32)
    bw = np.ones((1, H, W), dtype=np.int32)

    entry = [
        [ref, tgt],  # 0: base channels (2, H, W) after np.asarray
        [label],  # 1: label (1, H, W) after np.asarray
        [ref_idx, tgt_idx],  # 2: indices (2, N, 2) after np.asarray
        [list(start_end)],  # 3: StartEnd (1, 2) after np.asarray
        [start_end[1]],  # 4: End (unused by writer)
        [rep],  # 5: Replicate -> ix
        [[10, 20, 30]],  # 6: Position (unused by writer)
        fw,  # 7: Forward_relative_position (used when fwbw=True)
        bw,  # 8: Backward_relative_position (used when fwbw=True)
    ]
    return entry


def test_append_hdf_entries_writes_one_entry_and_sets_last_index(tmp_path):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    entry = _make_packed_entry(H=3, W=4, N=2, rep=7)
    nxt = _append_hdf_entries(
        hdf_file=str(h5_file),
        input_entries=[entry],
        lock=lock,
        start_nr=None,
        chunk_size=1,
        fwbw=True,
        set_attributes=True,
    )
    assert nxt == 1

    with h5py.File(h5_file, "r") as f:
        # last_index should be updated to 1
        assert int(f.attrs["last_index"]) == 1

        # group "0" and datasets exist
        assert "0" in f
        g = f["0"]
        for name in ("x_0", "y", "indices", "pos", "ix"):
            assert name in g

        x = g["x_0"][()]
        y = g["y"][()]
        ind = g["indices"][()]
        pos = g["pos"][()]
        ix = g["ix"][()]

        # Shapes
        assert x.shape == (1, 4, 3, 4)  # 2 base + 2 fwbw channels
        assert y.shape == (1, 1, 3, 4)
        assert ind.shape == (1, 2, 2, 2)
        assert pos.shape == (1, 1, 1, 2)
        assert ix.shape == (1, 1, 1)

        # Dtypes
        assert g["x_0"].dtype == np.dtype(np.uint32)
        assert g["y"].dtype == np.dtype(np.uint8)
        assert g["indices"].dtype == np.dtype(np.uint32)
        assert g["pos"].dtype == np.dtype(np.uint32)
        assert g["ix"].dtype == np.dtype(np.uint32)

        # Content sanity
        ref = np.asarray(entry[0])[0]
        tgt = np.asarray(entry[0])[1]
        fw = np.asarray(entry[-2])[0]
        bw = np.asarray(entry[-1])[0]

        assert np.array_equal(x[0, 0], ref)
        assert np.array_equal(x[0, 1], tgt)
        assert np.array_equal(x[0, 2], fw.astype(np.uint32))
        assert np.array_equal(x[0, 3], bw.astype(np.uint32))

        assert np.array_equal(y[0, 0], np.asarray(entry[1])[0])

        assert np.array_equal(ind[0], np.asarray(entry[2], dtype=np.uint32))

        assert np.array_equal(pos[0, 0, 0], np.asarray(entry[3], dtype=np.uint32)[0])

        assert int(ix[0, 0, 0]) == entry[5][0]


def test_append_hdf_entries_appends_second_group(tmp_path):
    h5_file = tmp_path / "out.h5"
    lock = multiprocessing.Lock()

    entry0 = _make_packed_entry(rep=1)
    entry1 = _make_packed_entry(rep=2)

    nxt1 = _append_hdf_entries(
        str(h5_file), [entry0], lock=lock, chunk_size=1, fwbw=True
    )
    assert nxt1 == 1

    nxt2 = _append_hdf_entries(
        str(h5_file), [entry1], lock=lock, chunk_size=1, fwbw=True
    )
    assert nxt2 == 2

    with h5py.File(h5_file, "r") as f:
        assert "0" in f and "1" in f
        assert int(f.attrs["last_index"]) == 2
        assert int(f["0/ix"][0, 0, 0]) == 1
        assert int(f["1/ix"][0, 0, 0]) == 2
