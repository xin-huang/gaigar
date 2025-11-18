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
from gaishi.multiprocessing import mp_manager
from gaishi.generators import RandomNumberGenerator
from gaishi.simulators import FeatureVectorSimulator


@pytest.fixture
def init_params():
    output_dir = "tests/test_FeatureVectorSimulator"
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
        "is_phased": True,
        "intro_prop": 0.7,
        "non_intro_prop": 0.3,
        "feature_config_file": "tests/data/ArchIE.features.yaml",
    }


@pytest.fixture
def cleanup_output_dir(request, init_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(init_params["output_dir"], ignore_errors=True)


def test_FeatureVectorSimulator(init_params, cleanup_output_dir):
    simulator = FeatureVectorSimulator(**init_params)
    generator = RandomNumberGenerator(nrep=2, seed=12345)
    res = mp_manager(job=simulator, data_generator=generator, nprocess=2)
    res.sort(key=lambda x: (x["Replicate"]))

    df = pd.DataFrame(res)
    expected_df = pd.read_csv(
        "tests/expected_results/simulators/FeatureVectorSimulator/test.features",
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
