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
            feature_vector_init_params["output_dir"],
            f"{feature_vector_init_params['output_prefix']}.features",
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
        # Meta
        assert "/meta" in f
        meta = f["/meta"]
        assert "n" in meta.attrs
        assert "N" in meta.attrs
        assert "L" in meta.attrs
        assert "Chromosome" in meta.attrs
        assert "n_written" in meta.attrs
        assert int(meta.attrs["n_written"]) >= 1

        # Required sample tables
        assert "/meta/ref_sample_table" in f
        assert "/meta/tgt_sample_table" in f

        # Required input datasets
        for name, dt in [
            ("/data/Ref_genotype", np.uint32),
            ("/data/Tgt_genotype", np.uint32),
            ("/data/Gap_to_prev", np.int64),
            ("/data/Gap_to_next", np.int64),
        ]:
            assert name in f
            ds = f[name]
            assert ds.dtype == np.dtype(dt)
            # replicate-indexed
            assert ds.ndim == 3
            assert ds.shape[0] >= 1

        # Indices
        assert "/index/ref_ids" in f
        assert "/index/tgt_ids" in f
        assert f["/index/ref_ids"].dtype == np.uint32
        assert f["/index/tgt_ids"].dtype == np.uint32
        assert f["/index/ref_ids"].ndim == 2
        assert f["/index/tgt_ids"].ndim == 2
        assert f["/index/ref_ids"].shape[0] >= 1
        assert f["/index/tgt_ids"].shape[0] >= 1

        # Infer schema
        assert "/coords/Position" in f
        pos = f["/coords/Position"]
        assert pos.dtype == np.int64
        assert pos.ndim == 2
        assert pos.shape[0] >= 1

        # Cross-dataset consistency on the first row
        ref0 = f["/data/Ref_genotype"][0]
        tgt0 = f["/data/Tgt_genotype"][0]
        gp0 = f["/data/Gap_to_prev"][0]
        gn0 = f["/data/Gap_to_next"][0]
        assert ref0.shape == tgt0.shape == gp0.shape == gn0.shape  # (N, L)

        ref_ids0 = f["/index/ref_ids"][0]
        tgt_ids0 = f["/index/tgt_ids"][0]
        assert ref_ids0.shape[0] == ref0.shape[0]  # N
        assert tgt_ids0.shape[0] == tgt0.shape[0]  # N
