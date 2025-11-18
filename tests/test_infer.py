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
import pandas as pd
import gaishi.models
import gaishi.stats
from gaishi.infer import infer


@pytest.fixture
def file_paths():
    output_dir = "tests/test_infer"
    return {
        "model": "tests/expected_results/train/test.lr.model",
        "config": "tests/data/test.config.yaml",
        "output": os.path.join(output_dir, "test.lr.predictions"),
        "output_dir": str(output_dir),
    }


@pytest.fixture
def cleanup_output_dir(request, file_paths):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(file_paths["output_dir"], ignore_errors=True)


def test_infer(file_paths, cleanup_output_dir):
    os.makedirs(file_paths["output_dir"], exist_ok=True)

    infer(
        model=file_paths["model"],
        config=file_paths["config"],
        output=file_paths["output"],
    )

    df = pd.read_csv(file_paths["output"], sep="\t")

    expected_df = pd.read_csv(
        "tests/expected_results/infer/test.lr.predictions", sep="\t"
    )

    pd.testing.assert_frame_equal(
        df,
        expected_df,
        check_dtype=False,
        check_like=False,
        rtol=1e-5,
        atol=1e-5,
    )
