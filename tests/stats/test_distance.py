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


import numpy as np
import pytest
from scipy.spatial import distance_matrix
from gaigar.stats.distance import Distance


def test_distance_compute_basic():
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

    out = Distance.compute(gt1=gt1, gt2=gt2, key="ref_dist")
    assert isinstance(out, dict) and "ref_dist" in out

    d = out["ref_dist"]
    expected = distance_matrix(gt2.T, gt1.T)
    expected.sort(axis=-1)

    assert d.shape == expected.shape == (gt2.shape[1], gt1.shape[1])
    assert np.allclose(d, expected)
    assert np.all(np.diff(d, axis=1) >= -1e-12)


def test_distance_custom_key():
    gt1 = np.zeros((2, 1), dtype=float)
    gt2 = np.ones((2, 2), dtype=float)

    out = Distance.compute(gt1=gt1, gt2=gt2, key="tgt_dist")
    assert "tgt_dist" in out and out["tgt_dist"].shape == (2, 1)
