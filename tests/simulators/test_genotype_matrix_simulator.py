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
        "num_genotype_matrices": 10,
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
        "output_h5": True,  # H5 mode
        "is_phased": True,
        "is_sorted": True,
        "keep_sim_data": False,
        "num_polymorphisms": 128,
        "num_upsamples": 56,
        "num_genotype_matrices": 10,
    }


def test_GenotypeMatrixSimulator(init_params):
    simulator = GenotypeMatrixSimulator(**init_params)
    generator = RandomNumberGenerator(nrep=10, seed=12345)
    res = mp_manager(
        job=simulator,
        data_generator=generator,
        nprocess=2,
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

    _ = mp_manager(
        job=simulator,
        data_generator=generator,
        nprocess=2,
        force_balanced=False,
        nintro=Value("i", 0),
        nnonintro=Value("i", 0),
        only_intro=False,
        only_non_intro=False,
        lock=Lock(),
    )

    h5_path = f'{init_params_h5["output_dir"]}/{init_params_h5["output_prefix"]}.h5'

    with h5py.File(h5_path, "r") as h5f:
        # Required datasets for inputs
        assert "/data/Ref_genotype" in h5f
        assert "/data/Tgt_genotype" in h5f

        ref = h5f["/data/Ref_genotype"]
        tgt = h5f["/data/Tgt_genotype"]
        n = h5f["/meta"].attrs["n"]

        assert ref.ndim == 3  # (R, N, L)
        assert tgt.ndim == 3  # (R, N, L)
        assert ref.shape == tgt.shape
        assert n == 10
        assert ref.shape[0] == 10

        # dtype sanity (genotypes are typically integer-coded)
        assert np.issubdtype(ref.dtype, np.integer)
        assert np.issubdtype(tgt.dtype, np.integer)

        # Optional gap channels (present when writing 4-channel data)
        has_gp = "/data/Gap_to_prev" in h5f
        has_gn = "/data/Gap_to_next" in h5f
        assert has_gp == has_gn  # either both exist or neither

        if has_gp:
            gp = h5f["/data/Gap_to_prev"]
            gn = h5f["/data/Gap_to_next"]
            assert gp.ndim == 3 and gn.ndim == 3
            assert gp.shape == ref.shape
            assert gn.shape == ref.shape
            assert np.issubdtype(gp.dtype, np.integer) or np.issubdtype(
                gp.dtype, np.floating
            )
            assert np.issubdtype(gn.dtype, np.integer) or np.issubdtype(
                gn.dtype, np.floating
            )

        # Labels: expect (R, N, L) under /targets/Label
        assert "/targets/Label" in h5f
        lab = h5f["/targets/Label"]
        assert lab.ndim == 3
        assert lab.shape == ref.shape

        # label value sanity: in [0, 1]
        lab0 = np.asarray(lab[0])
        assert np.nanmin(lab0) >= 0.0
        assert np.nanmax(lab0) <= 1.0

        # Basic non-empty / finite sanity on the first replicate
        ref0 = np.asarray(ref[0])
        tgt0 = np.asarray(tgt[0])
        assert ref0.size > 0 and tgt0.size > 0
        assert np.isfinite(ref0).all()
        assert np.isfinite(tgt0).all()
