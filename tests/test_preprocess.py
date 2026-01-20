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


import h5py, os, pytest
import numpy as np
import pandas as pd
import gaishi.stats
from gaishi.preprocess import preprocess_feature_vectors
from gaishi.preprocess import preprocess_genotype_matrices


@pytest.fixture
def feature_vector_init_params(tmp_path):
    output_dir = tmp_path / "preprocess"
    expected_dir = "tests/expected_results/simulators/MsprimeSimulator/0"
    return {
        "vcf_file": os.path.join(expected_dir, "test.0.vcf"),
        "ref_ind_file": os.path.join(expected_dir, "test.0.ref.ind.list"),
        "tgt_ind_file": os.path.join(expected_dir, "test.0.tgt.ind.list"),
        "anc_allele_file": None,
        "is_phased": True,
        "chr_name": "1",
        "win_len": 50000,
        "win_step": 50000,
        "feature_config_file": "tests/data/ArchIE.features.yaml",
        "output_dir": str(output_dir),
        "output_prefix": "test",
    }


@pytest.fixture
def genotype_matrix_init_params(tmp_path):
    output_dir = tmp_path / "preprocess"
    expected_dir = "tests/expected_results/simulators/MsprimeSimulator/0"

    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "vcf_file": os.path.join(expected_dir, "test.0.vcf"),
        "ref_ind_file": os.path.join(expected_dir, "test.0.ref.ind.list"),
        "tgt_ind_file": os.path.join(expected_dir, "test.0.tgt.ind.list"),
        "anc_allele_file": None,
        "chr_name": "1",
        "output_file": str(output_dir / "test.h5"),
        "num_polymorphisms": 5,
        "step_size": 5,
        "ploidy": 2,
        "is_phased": True,
        "num_upsamples": 56,
    }


def test_preprocess_feature_vectors(feature_vector_init_params):
    preprocess_feature_vectors(**feature_vector_init_params)

    df = pd.read_csv(
        os.path.join(
            feature_vector_init_params["output_dir"], f"{feature_vector_init_params['output_prefix']}.features"
        ),
        sep="\t",
    )

    expected_df = pd.read_csv(
        "tests/expected_results/preprocess/test.features", sep="\t"
    )

    pd.testing.assert_frame_equal(
        df,
        expected_df,
        check_dtype=False,
        check_like=False,
        rtol=1e-5,
        atol=1e-5,
    )


def test_preprocess_genotype_matrices(genotype_matrix_init_params):
    preprocess_genotype_matrices(**genotype_matrix_init_params)

    with h5py.File(genotype_matrix_init_params["output_file"], "r") as f:
        assert "last_index" in f.attrs
        last_index = int(f.attrs["last_index"])
        assert last_index >= 1

        # Check first group only (schema + dummy fields)
        g = f["0"]
        for name in ("x_0", "y", "indices", "pos", "ix"):
            assert name in g

        x = g["x_0"][()]
        y = g["y"][()]
        ind = g["indices"][()]
        pos = g["pos"][()]
        ix = g["ix"][()]

        assert x.dtype == np.uint32
        assert y.dtype == np.uint8
        assert ind.dtype == np.uint32
        assert pos.dtype == np.uint32
        assert ix.dtype == np.uint32

        # neighbor_gaps=True by default in write_h5 -> 4 channels
        assert x.shape[0] == 1
        assert x.shape[1] == 4

        # Consistency checks implied by current writer
        assert y.shape == (1, 1, x.shape[2], x.shape[3])
        assert ind.shape == (1, 2, x.shape[2], 2)
        assert pos.shape == (1, 1, 1, 2)
        assert ix.shape == (1, 1, 1)

        # For real data: dummy label should be all zeros; dummy replicate should be 0
        assert np.all(y == 0)
        assert int(f["0/ix"][0, 0, 0]) == 0
