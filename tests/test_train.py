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


import joblib, os, pytest, shutil
import gaishi.models
import gaishi.stats
from pathlib import Path
from gaishi.train import train


@pytest.fixture
def file_paths():
    output_dir = "tests/test_train"
    return {
        "demes": "tests/data/ArchIE_3D19.yaml",
        "config": "tests/data/test.config.yaml",
        "output": os.path.join(output_dir, "test.lr.model"),
        "output_dir": str(output_dir),
    }


@pytest.fixture
def cleanup_output_dir(request, file_paths):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(file_paths["output_dir"], ignore_errors=True)


def test_train(file_paths, cleanup_output_dir):
    os.makedirs(file_paths["output_dir"], exist_ok=True)

    train(
        demes=file_paths["demes"],
        config=file_paths["config"],
        output=file_paths["output"],
    )

    model = joblib.load(file_paths["output"])
    expected_model = joblib.load("tests/expected_results/train/test.lr.model")

    tolerance = 1e-5

    assert (
        model.coef_.shape == expected_model.coef_.shape
    ), "Model coefficient shapes do not match."
    # assert all(abs(a - b) < tolerance for a, b in zip(model.coef_.flatten(), expected_model.coef_.flatten())), "Coefficients do not match within tolerance."
    # assert abs(model.intercept_ - expected_model.intercept_) < tolerance
