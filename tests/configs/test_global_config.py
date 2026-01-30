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
from gaishi.configs import GlobalConfig
from gaishi.configs import ModelConfig
from gaishi.configs import FeatureVectorPreprocessConfig
from gaishi.configs import FeatureVectorSimulationConfig
from gaishi.configs import GenotypeMatrixSimulationConfig


def _valid_simulation_kwargs():
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


def _valid_preprocess_kwargs():
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


def _valid_model_config_logreg():
    return ModelConfig(
        name="logistic_regression",
        params={
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 200,
        },
    )


def _valid_model_config_extra_trees():
    return ModelConfig(
        name="extra_trees_classifier",
        params={
            "n_estimators": 500,
            "max_depth": None,
            "n_jobs": -1,
        },
    )


def test_global_config_valid_with_logistic_regression():
    sim_cfg = FeatureVectorSimulationConfig(**_valid_simulation_kwargs())
    preprocess_cfg = FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs())
    model_cfg = _valid_model_config_logreg()

    cfg = GlobalConfig(
        simulation=sim_cfg,
        preprocess=preprocess_cfg,
        model=model_cfg,
    )

    # simulation block
    assert cfg.simulation.nrep == 10
    assert cfg.simulation.seq_len == 1_000_000

    # preprocess block
    assert cfg.preprocess.chr_name == "chr1"
    assert isinstance(cfg.preprocess.vcf_file, Path)
    assert cfg.preprocess.win_len == 10000

    # model block
    assert cfg.model.name == "logistic_regression"
    assert cfg.model.params["C"] == 1.0


def test_global_config_valid_with_extra_trees():
    sim_cfg = FeatureVectorSimulationConfig(**_valid_simulation_kwargs())
    preprocess_cfg = FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs())
    model_cfg = _valid_model_config_extra_trees()

    cfg = GlobalConfig(
        simulation=sim_cfg,
        preprocess=preprocess_cfg,
        model=model_cfg,
    )

    assert cfg.model.name == "extra_trees_classifier"
    assert cfg.model.params["n_estimators"] == 500
    assert cfg.model.params["n_jobs"] == -1


def test_global_config_missing_simulation_raises():
    preprocess_cfg = FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs())
    model_cfg = _valid_model_config_logreg()

    with pytest.raises(ValidationError):
        GlobalConfig(
            preprocess=preprocess_cfg,
            model=model_cfg,
        )  # type: ignore[arg-type]


def test_infer_config_missing_preprocess_raises():
    sim_cfg = FeatureVectorSimulationConfig(**_valid_simulation_kwargs())
    model_cfg = _valid_model_config_logreg()

    with pytest.raises(ValidationError):
        GlobalConfig(
            simulation=sim_cfg,
            model=model_cfg,  # type: ignore[arg-type]
        )


def test_global_config_missing_model_type_raises():
    sim_cfg = FeatureVectorSimulationConfig(**_valid_simulation_kwargs())
    preprocess_cfg = FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs())

    with pytest.raises(ValidationError):
        GlobalConfig(
            simulation=sim_cfg,
            preprocess=preprocess_cfg,
        )  # type: ignore[arg-type]


def test_global_config_invalid_model_name_raises():
    sim_cfg = FeatureVectorSimulationConfig(**_valid_simulation_kwargs())
    preprocess_cfg = FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs())

    with pytest.raises(ValidationError):
        GlobalConfig(
            simulation=sim_cfg,
            preprocess=preprocess_cfg,
            model=ModelConfig(
                name="random_forest",  # not allowed by Literal
                params={"n_estimators": 100},
            ),
        )


def test_global_config_simulation_discriminates_feature_vector():
    cfg = GlobalConfig(
        simulation={
            "sim_type": "feature_vector",
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
            "nfeature": 128,
            "feature_config_file": Path("config/features.yaml"),
            "intro_prop": 0.5,
            "non_intro_prop": 0.5,
            "output_prefix": "train_sim",
            "output_dir": Path("results/train"),
            "seed": 42,
        },
        preprocess=FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs()),
        model=_valid_model_config_logreg(),
    )

    assert cfg.simulation.sim_type == "feature_vector"
    assert cfg.simulation.feature_config_file.name == "features.yaml"


def test_global_config_simulation_discriminates_genotype_matrix():
    cfg = GlobalConfig(
        simulation={
            "sim_type": "genotype_matrix",
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
            "nfeature": 128,
            "num_polymorphisms": 5000,
            "num_upsamples": 2,
            "output_prefix": "train_sim",
            "output_dir": Path("results/train"),
            "seed": 42,
        },
        preprocess=FeatureVectorPreprocessConfig(**_valid_preprocess_kwargs()),
        model=_valid_model_config_logreg(),
    )

    assert cfg.simulation.sim_type == "genotype_matrix"
    assert cfg.simulation.num_polymorphisms == 5000
    assert cfg.simulation.num_upsamples == 2
