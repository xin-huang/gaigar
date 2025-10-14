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


import scipy
import numpy as np
import pytest
from scipy.spatial import distance_matrix
from gaigar.stats.distance import Distance
from gaigar.stats.distance import RefDistance
from gaigar.stats.distance import TgtDistance


@pytest.fixture
def test_data():
    # gt1: (n_sites=3, n_samples1=2), gt2: (n_sites=3, n_samples2=3)
    gt1 = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    gt2 = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )

    return {
        "gt1": gt1,
        "gt2": gt2,
    }


def test_distance_compute_basic(test_data):
    gt1 = test_data["gt1"]
    gt2 = test_data["gt2"]

    params = {
        "gt1": gt1,
        "gt2": gt2,
        "key": "ref_dist",
        "all": True,
        "minimum": True,
        "maximum": True,
        "mean": True,
        "median": True,
        "variance": True,
        "skew": True,
        "kurtosis": True,
    }

    out = Distance.compute(**params)
    assert isinstance(out, dict)
    assert "All_ref_dist" in out
    assert "Minimum_ref_dist" in out
    assert "Maximum_ref_dist" in out
    assert "Mean_ref_dist" in out
    assert "Median_ref_dist" in out
    assert "Variance_ref_dist" in out
    assert "Skew_ref_dist" in out
    assert "Kurtosis_ref_dist" in out

    expected = distance_matrix(gt2.T, gt1.T)
    expected.sort(axis=-1)
    expected_skew_vals = scipy.stats.skew(
        expected,
        axis=1,  # bias=False, nan_policy="omit"
    )
    expected_skew_vals[~np.isfinite(expected_skew_vals)] = 0
    expected_kurtosis_vals = scipy.stats.kurtosis(
        expected,
        axis=1,  # bias=False, nan_policy="omit"
    )
    expected_kurtosis_vals[~np.isfinite(expected_kurtosis_vals)] = 0

    assert out["All_ref_dist"].shape == expected.shape == (gt2.shape[1], gt1.shape[1])
    assert np.allclose(out["All_ref_dist"], expected)
    assert np.allclose(out["Minimum_ref_dist"], np.min(expected, axis=1))
    assert np.allclose(out["Maximum_ref_dist"], np.max(expected, axis=1))
    assert np.allclose(out["Mean_ref_dist"], np.mean(expected, axis=1))
    assert np.allclose(out["Median_ref_dist"], np.median(expected, axis=1))
    assert np.allclose(out["Variance_ref_dist"], np.var(expected, axis=1))
    assert np.allclose(out["Skew_ref_dist"], expected_skew_vals)
    assert np.allclose(out["Kurtosis_ref_dist"], expected_kurtosis_vals)


def test_RefDistance(test_data):
    params = {
        "ref_gts": test_data["gt1"],
        "tgt_gts": test_data["gt2"],
        "all": True,
        "minimum": False,
        "maximum": False,
        "mean": False,
        "median": False,
        "variance": False,
        "skew": False,
        "kurtosis": False,
    }

    out = RefDistance.compute(**params)
    assert isinstance(out, dict)
    assert "All_ref_dist" in out
    assert "Minimum_ref_dist" not in out
    assert "Maximum_ref_dist" not in out
    assert "Mean_ref_dist" not in out
    assert "Median_ref_dist" not in out
    assert "Variance_ref_dist" not in out
    assert "Skew_ref_dist" not in out
    assert "Kurtosis_ref_dist" not in out


def test_TgtDisance(test_data):
    params = {
        "ref_gts": test_data["gt1"],
        "tgt_gts": test_data["gt2"],
        "all": False,
        "minimum": False,
        "maximum": True,
        "mean": True,
        "median": False,
        "variance": False,
        "skew": False,
        "kurtosis": True,
    }

    out = TgtDistance.compute(**params)
    assert isinstance(out, dict)
    assert "All_tgt_dist" not in out
    assert "Minimum_tgt_dist" not in out
    assert "Maximum_tgt_dist" in out
    assert "Mean_tgt_dist" in out
    assert "Median_tgt_dist" not in out
    assert "Variance_tgt_dist" not in out
    assert "Skew_tgt_dist" not in out
    assert "Kurtosis_tgt_dist" in out


def test_distance_missing_params():
    params = {}

    with pytest.raises(ValueError):
        Distance.compute(**params)
