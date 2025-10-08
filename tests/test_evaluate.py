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


import pytest
from gaigar.evaluate import window_evaluate


@pytest.fixture
def tract_files():
    return {
        "true_tract_file": "tests/data/test.lr.true.tracts.bed",
        "inferred_tract_file": "tests/data/test.lr.inferred.tracts.bed",
    }


if __name__ == "__main__":
    window_evaluate(
        true_tract_file="tests/data/test.lr.true.tracts.bed",
        inferred_tract_file="tests/data/test.lr.inferred.tracts.bed",
        seq_len=200000000,
        sample_size=50,
        ploidy=2,
        is_phased=True,
        output="test.performance",
    )
