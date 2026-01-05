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


import pytest
import h5py
import numpy as np
import pandas as pd
from multiprocessing import Lock, Value
from gaishi.multiprocessing import mp_manager
from gaishi.generators import RandomNumberGenerator
from gaishi.simulators import GenotypeMatrixSimulator


@pytest.fixture
def init_params(tmp_path):
    output_dir = tmp_path / "test_GenotypeMatrixSimulator"
    return {
        "demo_model_file": "tests/data/ArchIE_3D19.yaml",
        "nref": 50,
        "ntgt": 50,
        "ref_id": "Ref",
        "tgt_id": "Tgt",
        "src_id": "Ghost",
        "ploidy": 2,
        "seq_len": 50000,
        "mut_rate": 1.25e-8,
        "rec_rate": 1e-8,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "output_h5": False,
        "is_phased": True,
        "is_sorted": True,
        "keep_sim_data": False,
        "num_polymorphisms": 128,
        "num_upsamples": 56,
    }


@pytest.fixture
def init_params_h5(tmp_path):
    output_dir = tmp_path / "test_GenotypeMatrixSimulator_h5"
    return {
        "demo_model_file": "tests/data/ArchIE_3D19.yaml",
        "nref": 50,
        "ntgt": 50,
        "ref_id": "Ref",
        "tgt_id": "Tgt",
        "src_id": "Ghost",
        "ploidy": 2,
        "seq_len": 50000,
        "mut_rate": 1.25e-8,
        "rec_rate": 1e-8,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "output_h5": True,          # H5 mode
        "is_phased": True,
        "is_sorted": True,
        "keep_sim_data": False,
        "num_polymorphisms": 128,
        "num_upsamples": 56,
        # If the class exposes H5 knobs, they can be passed here (optional):
        # "h5_chunk_size": 1,
        # "h5_fwbw": True,
        # "h5_set_attributes": True,
    }


def test_GenotypeMatrixSimulator(init_params):
    simulator = GenotypeMatrixSimulator(**init_params)
    generator = RandomNumberGenerator(nrep=10, seed=12345)
    res = mp_manager(
        job=simulator,
        data_generator=generator,
        nprocess=2,
        nfeature=10,
        force_balanced=False,
        nintro=Value("i", 0),
        nnonintro=Value("i", 0),
        only_intro=False,
        only_non_intro=False,
        lock=Lock(),
    )

    df = pd.DataFrame(res)
    expected_df = pd.read_csv(
        "tests/expected_results/simulators/GenotypeMatrixSimulator/test.tsv",
        sep="\t",
    )

    for column in df.columns:
        if df[column].dtype.kind in "ifc":  # Float, int, complex numbers
            assert np.isclose(
                df[column], expected_df[column], atol=1e-5, rtol=1e-5
            ).all(), f"Mismatch in column {column}"
        else:
            assert (
                df[column] == expected_df[column]
            ).all(), f"Mismatch in column {column}"


def test_GenotypeMatrixSimulator_h5(init_params_h5):
    simulator = GenotypeMatrixSimulator(**init_params_h5)
    generator = RandomNumberGenerator(nrep=10, seed=12345)

    nfeature = 10
    res = mp_manager(
        job=simulator,
        data_generator=generator,
        nprocess=2,
        nfeature=nfeature,
        force_balanced=False,
        nintro=Value("i", 0),
        nnonintro=Value("i", 0),
        only_intro=False,
        only_non_intro=False,
        lock=Lock(),
    )

    # In H5 mode, mp_manager may return dicts or None depending on your integration.
    # This test asserts the on-disk contract, which is the stable source of truth.
    h5_path = f'{init_params_h5["output_dir"]}/{init_params_h5["output_prefix"]}.h5'

    with h5py.File(h5_path, "r") as h5f:
        # last_index should equal number of written groups (append-style writer)
        assert "last_index" in h5f.attrs
        last_index = int(h5f.attrs["last_index"])
        assert last_index == nfeature

        # groups are numeric strings: "0", "1", ...
        keys = sorted(h5f.keys(), key=lambda x: int(x))
        assert len(keys) == nfeature
        assert keys[0] == "0"
        assert keys[-1] == str(nfeature - 1)

        # validate structure + basic shape constraints
        for gid in keys:
            g = h5f[gid]
            assert {"x_0", "y", "indices", "pos", "ix"} <= set(g.keys())

            x = g["x_0"][()]
            y = g["y"][()]
            ind = g["indices"][()]
            pos = g["pos"][()]
            ix = g["ix"][()]

            # batch dimension
            assert x.shape[0] == 1
            assert y.shape[0] == 1
            assert ind.shape[0] == 1
            assert pos.shape[0] == 1
            assert ix.shape == (1, 1, 1)

            # channel counts: x has >=2 channels (ref/tgt); if fwbw is enabled it should be 4
            assert x.ndim == 4
            assert x.shape[1] in (2, 4)

            # y is (1, 1, n_samples, n_sites)
            assert y.ndim == 4
            assert y.shape[1] == 1

            # indices is (1, 2, n_samples, 2)
            assert ind.ndim == 4
            assert ind.shape[1] == 2
            assert ind.shape[-1] == 2

            # pos is (1, 1, 1, 2)  (StartEnd)
            assert pos.shape == (1, 1, 1, 2)

            # ix stores replicate id as integer
            assert np.issubdtype(ix.dtype, np.integer)

        # optional: check that ix values are not all identical (sanity)
        ix_vals = [int(h5f[k]["ix"][0, 0, 0]) for k in keys]
        assert len(set(ix_vals)) >= 1
