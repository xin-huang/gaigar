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
from gaia.utils.multiprocessing import mp_manager
from gaia.utils.generators import RandomNumberGenerator
from gaia.utils.simulators import LRTrainingDataSimulator


@pytest.fixture
def init_params():
    output_dir = "tests/test_LRTrainingDataSimulator"
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
        'feature_config': 'tests/data/ArchIE.features.yaml',
    }


@pytest.fixture
def cleanup_output_dir(request, init_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(init_params['output_dir'], ignore_errors=True)


def test_LRTrainingDataSimulator(init_params, cleanup_output_dir):
    simulator = LRTrainingDataSimulator(**init_params)
    generator = RandomNumberGenerator(nrep=2, seed=12345)
    res = mp_manager(job=simulator, data_generator=generator, nprocess=2)
    res.sort(key=lambda x: (x['Replicate']))

    df = pd.DataFrame(res)
    expected_df = pd.read_csv("tests/expected_results/simulators/LRTrainingDataSimulator/test.features", sep="\t")

    for column in df.columns:
        if df[column].dtype.kind in 'ifc':  # Float, int, complex numbers
            assert np.isclose(df[column], expected_df[column], atol=1e-5, rtol=1e-5).all(), f"Mismatch in column {column}"
        else:
            assert (df[column] == expected_df[column]).all(), f"Mismatch in column {column}"


if __name__ == '__main__':
    from cProfile import Profile
    from pstats import SortKey, Stats

    with Profile() as profile:
        simulator = LRTrainingDataSimulator(
            demo_model_file="tests/data/ArchIE_3D19.yaml",
            nref=50,
            ntgt=50,
            ref_id="Ref",
            tgt_id="Tgt",
            src_id="Ghost",
            ploidy=2,
            seq_len=50000,
            mut_rate=1.25e-8,
            rec_rate=1e-8,
            output_prefix="test",
            output_dir="tests/test_LRTrainingDataSimulator",
            is_phased=True,
            intro_prop=0.7,
            non_intro_prop=0.3,
            feature_config="tests/data/ArchIE.features.yaml",
        )

        for i in range(1000):
            simulator.run()

        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()
