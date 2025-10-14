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
from pydantic import ValidationError

from gaigar.configs.feature_config import FeatureConfig


def test_valid_minimal_bools():
    cfg = {
        "spectrum": True,
        "private_mutation": True,
    }
    m = FeatureConfig.model_validate(cfg)
    assert m.root["spectrum"] is True
    assert m.root["private_mutation"] is True


def test_valid_ref_tgt_dist_some_stats_true():
    cfg = {
        "ref_dist": {"mean": True, "variance": False},
        "tgt_dist": {"skew": True, "kurtosis": True},
    }
    m = FeatureConfig.model_validate(cfg)
    assert m.root["ref_dist"]["mean"] is True
    assert m.root["tgt_dist"]["kurtosis"] is True


def test_valid_dist_all_exclusive():
    cfg = {
        "ref_dist": {
            "all": True,
            "mean": False,
            "variance": False,
            "skew": False,
            "kurtosis": False,
        },
        "tgt_dist": {"all": True},
    }
    m = FeatureConfig.model_validate(cfg)
    assert m.root["ref_dist"]["all"] is True
    assert m.root["tgt_dist"]["all"] is True


def test_valid_sstar_full():
    cfg = {
        "sstar": {
            "match_bonus": 5000,
            "max_mismatch": 5,
            "mismatch_penalty": -10000,
        }
    }
    m = FeatureConfig.model_validate(cfg)
    assert m.root["sstar"]["match_bonus"] == 5000


@pytest.mark.parametrize("bad_feat", ["unknown", "SSTAR", "private"])
def test_unsupported_feature_name_raises(bad_feat):
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({bad_feat: True})
    assert "Unsupported feature" in str(ei.value)


def test_ref_dist_requires_mapping():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"ref_dist": True})
    assert "must be a mapping of stats to bools" in str(ei.value)


def test_tgt_dist_no_stat_true_raises():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"tgt_dist": {"mean": False, "variance": False}})
    assert "at least one stat must be True" in str(ei.value)


def test_dist_unknown_key_raises():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"ref_dist": {"middle": True}})
    assert "unknown dist stats" in str(ei.value)


def test_dist_value_must_be_bool():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"ref_dist": {"mean": 1}})
    assert "must be bool" in str(ei.value)


def test_spectrum_must_be_bool():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"spectrum": {"enabled": True}})
    assert "must be a boolean" in str(ei.value)


def test_sstar_requires_mapping():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"sstar": True})
    assert "sstar must be a mapping of params to integers" in str(ei.value)


def test_sstar_unknown_param_raises():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"sstar": {"window size": 100}})
    assert "unknown params" in str(ei.value)


def test_sstar_value_must_be_int():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"sstar": {"match_bonus": 5.5}})
    msg = str(ei.value)
    assert "valid integer" in msg and "fractional part" in msg

    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"sstar": {"match_bonus": True}})
    assert "must be int" in str(ei.value)


def test_sstar_max_mismatch_negative_raises():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"sstar": {"max_mismatch": -1}})
    assert "'max_mismatch' must be >= 0" in str(ei.value)


def test_sstar_mismatch_penalty_positive_raises():
    with pytest.raises(ValidationError) as ei:
        FeatureConfig.model_validate({"sstar": {"mismatch_penalty": 1}})
    assert "'mismatch_penalty' should be <= 0" in str(ei.value)


def test_combined_valid_config():
    cfg = {
        "ref_dist": {"mean": True, "variance": False},
        "tgt_dist": {"all": True},
        "spectrum": True,
        "private_mutation": True,
        "sstar": {"match_bonus": 1000, "max_mismatch": 3, "mismatch_penalty": -5000},
    }
    m = FeatureConfig.model_validate(cfg)
    assert m.root["tgt_dist"]["all"] is True
