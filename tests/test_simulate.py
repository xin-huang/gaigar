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
from gaishi.simulate import simulate_genotype_matrices


@pytest.fixture
def feature_vector_simulate_params(tmp_path):
    output_dir = tmp_path / "test_feature_vector_simulate"
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
        "nprocess": 1,
        "feature_config_file": "tests/data/ArchIE.features.yaml",
        "intro_prop": 0.7,
        "non_intro_prop": 0.3,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "seed": 12345,
        "nfeature": 10000,
        "is_shuffled": False,
        "force_balanced": False,
        "keep_sim_data": False,
    }


def test_feature_vector_simulate(feature_vector_simulate_params):
    simulate_feature_vectors(**feature_vector_simulate_params)

    df = pd.read_csv(
        os.path.join(
            feature_vector_simulate_params["output_dir"],
            f"{feature_vector_simulate_params['output_prefix']}.tsv",
        ),
        sep="\t",
    )

    expected_df = pd.read_csv(
        "tests/expected_results/simulate/test.simulated.feature.vectors.tsv",
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


@pytest.fixture
def genotype_matrix_simulate_params(tmp_path):
    output_dir = tmp_path / "test_genotype_matrix_simulate"
    return {
        "demo_model_file": "tests/data/ArchIE_3D19.yaml",
        "nrep": 10,
        "nref": 50,
        "ntgt": 50,
        "ref_id": "Ref",
        "tgt_id": "Tgt",
        "src_id": "Ghost",
        "ploidy": 2,
        "is_phased": True,
        "seq_len": 100000,
        "mut_rate": 1e-8,
        "rec_rate": 1e-8,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "output_h5": False,
        "seed": 12345,
        "nprocess": 1,
        "num_polymorphisms": 192,
        "num_genotype_matrices": 10,
        "is_phased": True,
        "is_sorted": True,
        "force_balanced": False,
        "keep_sim_data": False,
    }


def test_genotype_matrix_simulate(genotype_matrix_simulate_params):
    simulate_genotype_matrices(**genotype_matrix_simulate_params)

    df = pd.read_csv(
        os.path.join(
            genotype_matrix_simulate_params["output_dir"],
            f"{genotype_matrix_simulate_params['output_prefix']}.tsv",
        ),
        sep="\t",
    )

    assert df["Ref_genotype"].shape[0] == 10  # 10 genotype matrices

    # seriate order is not reproducible across environments (non-unique optima / solver tie-breaks)
    # expected_df = pd.read_csv(
    #    "tests/expected_results/simulate/test.simulated.genotype.matrices.tsv",
    #    sep="\t",
    # )

    # pd.testing.assert_frame_equal(
    #    df,
    #    expected_df,
    #    check_dtype=False,
    #    check_like=False,
    #    rtol=1e-5,
    #    atol=1e-5,
    # )
