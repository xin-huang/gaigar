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


# tests/test_preprocess_config.py

from pathlib import Path

import pytest
from pydantic import ValidationError
from gaishi.configs import FeatureVectorPreprocessConfig


def _valid_kwargs():
    return {
        "vcf_file": Path("data/input.vcf.gz"),
        "chr_name": "chr1",
        "ref_ind_file": Path("config/ref.txt"),
        "tgt_ind_file": Path("config/tgt.txt"),
        "win_len": 10000,
        "win_step": 5000,
        "feature_config_file": "config/features.yaml",
        "output_dir": Path("results/preprocess"),
        "output_prefix": "lr",
        # optional fields left as defaults
    }


def test_preprocess_config_valid():
    cfg = FeatureVectorPreprocessConfig(**_valid_kwargs())

    # Paths
    assert isinstance(cfg.vcf_file, Path)
    assert isinstance(cfg.ref_ind_file, Path)
    assert isinstance(cfg.tgt_ind_file, Path)
    assert isinstance(cfg.feature_config_file, Path)
    assert isinstance(cfg.output_dir, Path)

    # output_dir should be normalized to absolute path
    assert cfg.output_dir.is_absolute()

    # Required simple fields
    assert cfg.chr_name == "chr1"
    assert cfg.win_len == 10000
    assert cfg.win_step == 5000

    # Defaults
    assert cfg.output_prefix == "lr"
    assert cfg.nprocess == 1
    assert cfg.ploidy == 2
    assert cfg.is_phased is True
    assert cfg.anc_allele_file is None


@pytest.mark.parametrize("field", ["win_len", "win_step", "nprocess", "ploidy"])
@pytest.mark.parametrize("bad_value", [0, -1])
def test_preprocess_config_positive_int_fields_must_be_gt_zero(
    field: str, bad_value: int
):
    kwargs = _valid_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValidationError):
        FeatureVectorPreprocessConfig(**kwargs)


def test_preprocess_config_output_dir_normalization():
    kwargs = _valid_kwargs()
    kwargs["output_dir"] = "relative/output"

    cfg = FeatureVectorPreprocessConfig(**kwargs)
    assert cfg.output_dir.is_absolute()
    assert cfg.output_dir.as_posix().endswith("relative/output")


def test_preprocess_config_accepts_anc_allele_file():
    kwargs = _valid_kwargs()
    kwargs["anc_allele_file"] = "config/anc_alleles.tsv"

    cfg = FeatureVectorPreprocessConfig(**kwargs)
    assert isinstance(cfg.anc_allele_file, Path)
    assert cfg.anc_allele_file.as_posix().endswith("config/anc_alleles.tsv")


def test_preprocess_config_missing_required_field_raises():
    kwargs = _valid_kwargs()
    kwargs.pop("vcf_file")

    with pytest.raises(ValidationError):
        FeatureVectorPreprocessConfig(**kwargs)
