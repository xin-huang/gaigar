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
import shutil
import numpy as np
import pandas as pd
from multiprocessing import Lock, Value
from gaishi.multiprocessing import mp_manager
from gaishi.generators import RandomNumberGenerator
from gaishi.simulators import GenotypeMatrixSimulator


@pytest.fixture
def init_params():
    output_dir = "tests/test_GenotypeMatrixSimulator"
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
def cleanup_output_dir(request, init_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(init_params["output_dir"], ignore_errors=True)


def test_GenotypeMatrixSimulator(init_params, cleanup_output_dir):
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
