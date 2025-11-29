# GNU General Public License v3.0
# Copyright 2024 Xin Huang
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


import os, pytest, shutil
import numpy as np
import pandas as pd
import gaishi.stats
from gaishi.simulate import simulate_feature_vectors


@pytest.fixture
def feature_vector_simulate_params(tmp_path):
    output_dir = "tests/test_feature_vector_simulate"
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
        "feature_config_file": "tests/data/ArchIE.features.yaml",
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
def cleanup_output_dir(request, feature_vector_simulate_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(feature_vector_simulate_params["output_dir"], ignore_errors=True)


def test_feature_vector_simulate(feature_vector_simulate_params, cleanup_output_dir):
    simulate_feature_vectors(**feature_vector_simulate_params)

    df = pd.read_csv(
        os.path.join(
            feature_vector_simulate_params["output_dir"],
            f"{feature_vector_simulate_params['output_prefix']}.features",
        ),
        sep="\t",
    )

    expected_df = pd.read_csv(
        "tests/expected_results/simulate/test.feature.vector.simulate.features",
        sep="\t",
    )

    pd.testing.assert_frame_equal(
        df,
        expected_df,
        check_dtype=False,
        check_like=False,
        rtol=1e-5,
        atol=1e-5,
    )
