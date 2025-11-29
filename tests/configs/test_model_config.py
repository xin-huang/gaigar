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
from typing import Any
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from gaishi.configs import ModelConfig


def test_model_config_valid_logistic_regression():
    cfg = ModelConfig(
        name="logistic_regression",
        params={"C": 1.0, "penalty": "l2", "max_iter": 200},
    )

    assert cfg.name == "logistic_regression"
    assert cfg.params["C"] == 1.0
    assert cfg.params["penalty"] == "l2"
    assert cfg.params["max_iter"] == 200


def test_model_config_valid_extra_trees():
    cfg = ModelConfig(
        name="extra_trees_classifier",
        params={"n_estimators": 500, "max_depth": None, "n_jobs": -1},
    )

    assert cfg.name == "extra_trees_classifier"
    assert cfg.params["n_estimators"] == 500
    assert cfg.params["max_depth"] is None
    assert cfg.params["n_jobs"] == -1


def test_model_config_invalid_name_raises():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="random_forest",  # not in the Literal
            params={"n_estimators": 100},
        )


def test_model_config_params_default_is_empty_dict():
    cfg = ModelConfig(name="logistic_regression")
    assert cfg.params == {}
