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


import os, pytest, yaml
from gaigar.utils import parse_ind_file
from gaigar.generators import GenomicDataGenerator
from gaigar.preprocessors import FeatureVectorPreprocessor


@pytest.fixture
def data_params():
    expected_dir = "tests/expected_results/simulators/MsprimeSimulator/0"
    return {
        "vcf_file": os.path.join(expected_dir, "test.0.vcf"),
        "ref_ind_file": os.path.join(expected_dir, "test.0.ref.ind.list"),
        "tgt_ind_file": os.path.join(expected_dir, "test.0.tgt.ind.list"),
        "anc_allele_file": None,
        "is_phased": True,
        "ploidy": 2,
        "chr_name": "1",
        "win_len": 50000,
        "win_step": 50000,
    }


@pytest.fixture
def feature_params():
    expected_dir = "tests/expected_results/simulators/MsprimeSimulator/0"
    return {
        "ref_ind_file": os.path.join(expected_dir, "test.0.ref.ind.list"),
        "tgt_ind_file": os.path.join(expected_dir, "test.0.tgt.ind.list"),
        "feature_config": "tests/data/ArchIE.features.yaml",
    }


def test_FeatureVectorPreprocessor(data_params, feature_params):
    generator = GenomicDataGenerator(**data_params)
    preprocessor = FeatureVectorPreprocessor(**feature_params)

    items = preprocessor.run(**list(generator.get())[0])

    num_features = 0
    with open(feature_params["feature_config"], "r") as f:
        features = yaml.safe_load(f)
    features = features.get("Features", {})

    tgt_samples = parse_ind_file(data_params["tgt_ind_file"])

    num_features += (
        len(features.keys()) - 3
    )  # Remove `Ref distances`, `Tgt distances`, `Spectra`
    num_features += len(features["Ref distances"].keys())
    num_features += len(features["Tgt distances"].keys()) - 1  # Remove `All`
    num_features += len(tgt_samples) * data_params["ploidy"] + 1
    num_features += len(tgt_samples) * data_params["ploidy"]

    assert (
        len(items[0].keys()) == num_features + 4
    ), "The number of estimated and expected features do not match."
