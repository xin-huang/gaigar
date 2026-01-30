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
from gaishi.generators import PolymorphismDataGenerator
from gaishi.preprocessors import GenotypeMatrixPreprocessor


@pytest.fixture
def generator_params():
    return {
        "vcf_file": "tests/expected_results/simulators/MsprimeSimulator/0/test.0.vcf",
        "chr_name": "1",
        "ref_ind_file": "tests/expected_results/simulators/MsprimeSimulator/0/test.0.ref.ind.list",
        "tgt_ind_file": "tests/expected_results/simulators/MsprimeSimulator/0/test.0.tgt.ind.list",
        "num_polymorphisms": 10,
        "step_size": None,
        "anc_allele_file": None,
        "ploidy": 2,
        "is_phased": True,
        "random_polymorphisms": True,
        "seed": 12345,
        "num_upsamples": 112,
    }


@pytest.fixture
def preprocessor_params():
    return {
        "ref_ind_file": "tests/expected_results/simulators/MsprimeSimulator/0/test.0.ref.ind.list",
        "tgt_ind_file": "tests/expected_results/simulators/MsprimeSimulator/0/test.0.tgt.ind.list",
    }


def test_GenotypeMatricesPreprocessor(generator_params, preprocessor_params):
    generator = PolymorphismDataGenerator(**generator_params)
    preprocessor = GenotypeMatrixPreprocessor(
        **preprocessor_params,
        ref_rdm_spl_idx=generator.ref_rdm_spl_idx,
        tgt_rdm_spl_idx=generator.tgt_rdm_spl_idx,
        is_sorted=True,
    )

    sample_dicts = preprocessor.run(**list(generator.get())[0])

    assert len(sample_dicts) == 1
    assert sample_dicts[0]["Chromosome"] == "1"
    assert sample_dicts[0]["Start"] == "Random"
    assert sample_dicts[0]["End"] == "Random"
    assert len(sample_dicts[0]["Position"]) == 10
    assert len(sample_dicts[0]["Ref_sample"]) == 112 * 2
    assert len(sample_dicts[0]["Tgt_sample"]) == 112 * 2
