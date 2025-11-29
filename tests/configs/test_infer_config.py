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


from pathlib import Path

import pytest
from pydantic import ValidationError
from gaishi.configs import PreprocessConfig, ModelConfig, InferConfig


def _valid_preprocess_kwargs() -> dict:
    return {
        "vcf_file": "data/input.vcf.gz",
        "chr_name": "chr1",
        "ref_ind_file": "config/ref.txt",
        "tgt_ind_file": "config/tgt.txt",
        "win_len": 10000,
        "win_step": 5000,
        "feature_config_file": "config/features.yaml",
        "output_dir": "results/infer",
        "output_prefix": "lr",
    }


def _valid_model_config() -> ModelConfig:
    return ModelConfig(
        name="logistic_regression",
        params={
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 200,
        },
    )


def test_infer_config_valid():
    preprocess_cfg = PreprocessConfig(**_valid_preprocess_kwargs())
    model_cfg = _valid_model_config()

    cfg = InferConfig(
        preprocess=preprocess_cfg,
        model=model_cfg,
    )

    # Preprocess block
    assert cfg.preprocess.chr_name == "chr1"
    assert isinstance(cfg.preprocess.vcf_file, Path)
    assert cfg.preprocess.win_len == 10000

    # Model block
    assert cfg.model.name == "logistic_regression"
    assert cfg.model.params["solver"] == "lbfgs"
    assert cfg.model.params["C"] == 1.0


def test_infer_config_missing_preprocess_raises():
    model_cfg = _valid_model_config()

    with pytest.raises(ValidationError):
        InferConfig(
            model=model_cfg,  # type: ignore[arg-type]
        )


def test_infer_config_missing_model_raises():
    preprocess_cfg = PreprocessConfig(**_valid_preprocess_kwargs())

    with pytest.raises(ValidationError):
        InferConfig(
            preprocess=preprocess_cfg,  # type: ignore[arg-type]
        )


def test_infer_config_invalid_model_name_raises():
    preprocess_cfg = PreprocessConfig(**_valid_preprocess_kwargs())

    # Pass raw dict for model so ModelConfig validation happens inside InferConfig
    bad_model = {
        "name": "random_forest",  # not allowed by Literal
        "params": {"n_estimators": 100},
    }

    with pytest.raises(ValidationError):
        InferConfig(
            preprocess=preprocess_cfg,
            model=bad_model,  # type: ignore[arg-type]
        )
