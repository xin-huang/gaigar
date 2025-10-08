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
from gaigar.preprocess import lr_preprocess


@pytest.fixture
def init_params(tmp_path):
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
        "feature_config": "tests/data/ArchIE.features.yaml",
        "output_dir": str(output_dir),
        "output_prefix": "test",
    }


@pytest.fixture
def cleanup_output_dir(request, init_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(init_params["output_dir"], ignore_errors=True)


def test_lr_preprocess(init_params, cleanup_output_dir):
    lr_preprocess(**init_params)

    df = pd.read_csv(
        os.path.join(
            init_params["output_dir"], f"{init_params['output_prefix']}.features"
        ),
        sep="\t",
    )
    expected_df = pd.read_csv(
        "tests/expected_results/preprocess/test.features", sep="\t"
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
