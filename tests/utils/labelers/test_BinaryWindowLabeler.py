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


import pytest
import numpy as np
import pandas as pd
from gaia.utils.labelers import BinaryWindowLabeler


@pytest.fixture
def labeler_params():
    return {
        "win_len": 50000,
        "intro_prop": 0.7,
        "non_intro_prop": 0.3,
        "ploidy": 2,
        "is_phased": True,
    }


def test_BinaryWindowLabeler(labeler_params):
    labeler = BinaryWindowLabeler(**labeler_params)
    res = labeler.run(
        tgt_ind_file="tests/expected_results/simulators/MsprimeSimulator/0/test.0.tgt.ind.list",
        true_tract_file="tests/expected_results/simulators/MsprimeSimulator/0/test.0.true.tracts.bed",
    )

    df = pd.DataFrame(res)
    expected_df = pd.read_csv(
        "tests/expected_results/labelers/0/test.0.labels", sep="\t"
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
