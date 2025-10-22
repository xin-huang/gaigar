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


import numpy as np
import pytest
from gaigar.stats.spectrum import Spectrum


def test_spectrum():
    # tgt_gt: (n_sites=4, n_samples=2)
    # rows = sites; cols = samples
    tgt = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=int,
    )

    out = Spectrum.compute(tgt_gts=tgt, is_phased=True, ploidy=2)["spectrum"]

    # per-site derived counts across samples: [1,1,2,0]
    # counts[:,0] = [0,1,2,0] -> bincount(minlength=3) = [2,1,1] -> [0,1,1] (set 0-bin as 0)
    # counts[:,1] = [1,0,2,0] -> bincount(minlength=3) = [2,1,1] -> [0,1,1] (set 0-bin as 0)
    expected = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=int,
    )

    assert out.shape == (2, 3)
    assert np.all(out[:, 0] == 0)
    assert np.array_equal(out, expected)


def test_spectrum_missing_params():
    params = {"ploidy": 2}

    with pytest.raises(ValueError):
        Spectrum.compute(**params)
