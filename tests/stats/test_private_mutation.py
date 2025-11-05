# Copyright 2025 Xin Huang
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
from gaigar.stats.private_mutation import PrivateMutation


def test_private_mutation_basic_counts():
    # ref_gts: (n_sites=4, n_ref=2)
    ref = np.array(
        [
            [0, 0],  # site0 -> ref_sum=0
            [1, 0],  # site1 -> ref_sum=1
            [1, 1],  # site2 -> ref_sum=2
            [0, 0],  # site3 -> ref_sum=0
        ],
        dtype=int,
    )

    # tgt_gts: (n_sites=4, n_tgt=3)
    tgt = np.array(
        [
            [1, 0, 1],  # site0
            [0, 1, 1],  # site1
            [0, 0, 1],  # site2
            [1, 1, 0],  # site3
        ],
        dtype=int,
    )

    out = PrivateMutation.compute(ref_gts=ref, tgt_gts=tgt)
    assert "private_mutation" in out
    v = out["private_mutation"]

    # sample0: site0(OK), site3(OK) => 2
    # sample1: site3(OK)            => 1
    # sample2: site0(OK)            => 1
    expected = np.array([2, 1, 1], dtype=int)

    assert v.shape == expected.shape
    assert np.array_equal(v, expected)


def test_private_mutation_missing_params():
    params = {}

    with pytest.raises(ValueError):
        PrivateMutation.compute(**params)
