# Copyright 2024 Xin Huang
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


import os
import pytest
import shutil
import numpy as np
import pandas as pd
from gaia.simulate import lr_simulate
from gaia.simulate import unet_simulate


@pytest.fixture
def lr_simulate_params():
    output_dir = "tests/test_lr_simulate"
    return {
        "demo_model_file": "tests/data/ArchIE_3D19.yaml",
        "nrep": 100,
        "nref": 50,
        "ntgt": 50,
        "ref_id": "Ref",
        "tgt_id": "Tgt",
        "src_id": "Ghost",
        "ploidy": 2,
        "is_phased": True,
        "seq_len": 50000,
        "mut_rate": 1.25e-8,
        "rec_rate": 1e-8,
        "nprocess": 2,
        "feature_config": "tests/data/ArchIE.features.yaml",
        "intro_prop": 0.7,
        "non_intro_prop": 0.3,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "seed": 12345,
        "nfeature": 10000,
        "is_shuffled": False,
        "force_balanced": False,
        "keep_sim_data": True,
    }


@pytest.fixture
def unet_simulate_params():
    output_dir = "tests/test_unet_simulate"
    return {
        "demo_model_file": "tests/data/ArchIE_3D19.yaml",
        "nrep": 100,
        "nref": 50,
        "ntgt": 50,
        "ref_id": "Ref",
        "tgt_id": "Tgt",
        "src_id": "Ghost",
        "ploidy": 2,
        "is_phased": True,
        "seq_len": 100000,
        "mut_rate": 1.25e-8,
        "rec_rate": 1e-8,
        "nsite": 128,
        "nupsample": 56,
        "nfeature": 100,
        "nprocess": 2,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "seed": 12345,
        "output_h5": False,
        "is_sorted": True,
        "force_balanced": False,
        "only_intro": False,
        "only_non_intro": False,
        "keep_sim_data": False,
    }


@pytest.fixture
def cleanup_lr_output_dir(request, lr_simulate_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(lr_simulate_params["output_dir"], ignore_errors=True)


@pytest.fixture
def cleanup_unet_output_dir(request, unet_simulate_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(unet_simulate_params["output_dir"], ignore_errors=True)


def test_lr_simulate(lr_simulate_params, cleanup_lr_output_dir):
    lr_simulate(**lr_simulate_params)

    df = pd.read_csv(
        os.path.join(
            lr_simulate_params["output_dir"],
            f"{lr_simulate_params['output_prefix']}.features",
        ),
        sep="\t",
    )

    expected_df = pd.read_csv(
        "tests/expected_results/simulate/test.lr.simulate.features",
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


def test_unet_simulate(unet_simulate_params, cleanup_unet_output_dir):
    unet_simulate(**unet_simulate_params)

    df = pd.read_csv(
        os.path.join(
            unet_simulate_params["output_dir"],
            f"{unet_simulate_params['output_prefix']}.tsv",
        ),
        sep="\t",
    )

    assert len(df) == 100


if __name__ == "__main__":
    lr_simulate(
        demo_model_file="tests/data/ArchIE_3D19.yaml",
        nrep=100,
        nref=50,
        ntgt=50,
        ref_id="Ref",
        tgt_id="Tgt",
        src_id="Ghost",
        ploidy=2,
        is_phased=True,
        seq_len=50000,
        mut_rate=1.25e-8,
        rec_rate=1e-8,
        nprocess=2,
        feature_config="tests/data/ArchIE.features.yaml",
        intro_prop=0.7,
        non_intro_prop=0.3,
        output_prefix="test",
        output_dir="tests/test",
        seed=12345,
        nfeature=10000,
        is_shuffled=False,
        force_balanced=False,
        keep_sim_data=True,
    )
