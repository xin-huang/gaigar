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


import os
import pytest
import numpy as np
from gaia.utils import read_data, split_genome
from gaia.utils.generators import PolymorphismDataGenerator


@pytest.fixture
def file_paths():
    expected_dir = "tests/expected_results/simulators/MsprimeSimulator/0"
    return {
        "vcf_file": os.path.join(expected_dir, "test.0.vcf"),
        "ref_ind_file": os.path.join(expected_dir, "test.0.ref.ind.list"),
        "tgt_ind_file": os.path.join(expected_dir, "test.0.tgt.ind.list"),
        "anc_allele_file": None,
        "is_phased": True,
    }


@pytest.fixture
def init_params(file_paths):
    return {
        **file_paths,
        "chr_name": "1",
        "num_polymorphisms": 10,
        "step_size": None,
        "random_polymorphisms": True,
        "seed": 12345,
    }


@pytest.fixture
def expected_params(file_paths):
    expected_data = []
    chr_name = "1"
    num_polymorphisms = 10
    ref_data, ref_samples, tgt_data, tgt_samples = read_data(**file_paths)
    pos = tgt_data[chr_name]["POS"]
    windows = split_genome(
        pos=pos,
        chr_name=chr_name,
        window_based=False,
        polymorphism_size=num_polymorphisms,
        random_polymorphisms=True,
        seed=12345,
    )

    for w in range(len(windows)):
        chr_name, idx = windows[w]
        ref_gts = ref_data[chr_name]["GT"]
        tgt_gts = tgt_data[chr_name]["GT"]
        sub_ref_gts = ref_gts[idx]
        sub_tgt_gts = tgt_gts[idx]
        sub_pos = pos[idx]

        d = {
            "chr_name": chr_name,
            "start": "Random",
            "end": "Random",
            "ploidy": 2,
            "is_phased": True,
            "ref_gts": sub_ref_gts,
            "tgt_gts": sub_tgt_gts,
            "pos": sub_pos,
            "pos_idx": idx,
        }
        expected_data.append(d)

    return expected_data


def test_PolymorphismDataGenerator(init_params, expected_params):
    generator = PolymorphismDataGenerator(**init_params)
    generated_params_list = list(generator.get())

    assert len(generated_params_list) == len(
        expected_params
    ), "The number of generated and expected parameters do not match."

    for generated, expected in zip(generated_params_list, expected_params):
        for key in generated:
            if isinstance(generated[key], np.ndarray) and isinstance(
                expected[key], np.ndarray
            ):
                assert np.array_equal(
                    generated[key], expected[key]
                ), f"Arrays do not match for key {key}."
            else:
                assert (
                    generated[key] == expected[key]
                ), f"Values do not match for key {key}."


if __name__ == "__main__":
    generator = PolymorphismDataGenerator(
        vcf_file="tests/expected_results/simulators/MsprimeSimulator/0/test.0.vcf",
        chr_name="1",
        ref_ind_file="tests/expected_results/simulators/MsprimeSimulator/0/test.0.ref.ind.list",
        tgt_ind_file="tests/expected_results/simulators/MsprimeSimulator/0/test.0.tgt.ind.list",
        num_polymorphisms=10,
        step_size=None,
        anc_allele_file=None,
        ploidy=2,
        is_phased=True,
        random_polymorphisms=True,
        seed=12345,
        num_upsamples=112,
    )

    res = list(generator.get())[0]

    print(res)
