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
from pathlib import Path
from pydantic import ValidationError
from gaishi.configs import SimulationConfig, ModelConfig, TrainConfig


def _valid_simulation_kwargs() -> dict:
    return {
        "nrep": 10,
        "nref": 20,
        "ntgt": 20,
        "ref_id": "REF",
        "tgt_id": "TGT",
        "src_id": "SRC",
        "ploidy": 2,
        "is_phased": True,
        "seq_len": 1_000_000,
        "mut_rate": 1e-8,
        "rec_rate": 1e-8,
        "nprocess": 4,
        "feature_config_file": Path("config/features.yaml"),
        "nfeature": 128,
        "is_shuffled": True,
        "force_balanced": True,
        "intro_prop": 0.5,
        "non_intro_prop": 0.5,
        "output_prefix": "train_sim",
        "output_dir": Path("results/train"),
        "keep_sim_data": False,
        "seed": 42,
    }


def _valid_model_config_logreg() -> ModelConfig:
    return ModelConfig(
        name="logistic_regression",
        params={
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 200,
        },
    )


def _valid_model_config_extra_trees() -> ModelConfig:
    return ModelConfig(
        name="extra_trees_classifier",
        params={
            "n_estimators": 500,
            "max_depth": None,
            "n_jobs": -1,
        },
    )


def test_train_config_valid_with_logistic_regression():
    sim_cfg = SimulationConfig(**_valid_simulation_kwargs())
    model_cfg = _valid_model_config_logreg()

    cfg = TrainConfig(
        simulation=sim_cfg,
        model=model_cfg,
    )

    assert cfg.simulation.nrep == 10
    assert cfg.simulation.seq_len == 1_000_000
    assert cfg.model.name == "logistic_regression"
    assert cfg.model.params["C"] == 1.0


def test_train_config_valid_with_extra_trees():
    sim_cfg = SimulationConfig(**_valid_simulation_kwargs())
    model_cfg = _valid_model_config_extra_trees()

    cfg = TrainConfig(
        simulation=sim_cfg,
        model=model_cfg,
    )

    assert cfg.model.name == "extra_trees_classifier"
    assert cfg.model.params["n_estimators"] == 500
    assert cfg.model.params["n_jobs"] == -1


def test_train_config_missing_simulation_raises():
    model_cfg = _valid_model_config_logreg()

    with pytest.raises(ValidationError):
        TrainConfig(model=model_cfg)  # type: ignore[arg-type]


def test_train_config_missing_model_type_raises():
    sim_cfg = SimulationConfig(**_valid_simulation_kwargs())

    with pytest.raises(ValidationError):
        TrainConfig(simulation=sim_cfg)  # type: ignore[arg-type]


def test_train_config_invalid_model_name_raises():
    sim_cfg = SimulationConfig(**_valid_simulation_kwargs())

    with pytest.raises(ValidationError):
        TrainConfig(
            simulation=sim_cfg,
            model=ModelConfig(
                name="random_forest",  # not allowed by Literal
                params={"n_estimators": 100},
            ),
        )
