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
from gaishi.configs import FeatureVectorSimulationConfig, GenotypeMatrixSimulationConfig


def _valid_feature_vector_kwargs():
    return {
        "sim_type": "feature_vector",
        "nrep": 10,
        "nref": 20,
        "ntgt": 20,
        "ref_id": "REF",
        "tgt_id": "TGT",
        "src_id": "SRC",
        "ploidy": 2,
        "is_phased": True,
        "is_shuffled": True,
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
    }


def _valid_genotype_matrix_kwargs():
    return {
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
    }


def test_feature_vector_simulation_config_valid():
    cfg = FeatureVectorSimulationConfig(**_valid_feature_vector_kwargs())

    assert cfg.sim_type == "feature_vector"
    assert cfg.nrep == 10
    assert cfg.nref == 20
    assert cfg.ntgt == 20
    assert cfg.ref_id == "REF"
    assert cfg.tgt_id == "TGT"
    assert cfg.src_id == "SRC"

    # Defaults
    assert cfg.nprocess == 1
    assert cfg.is_shuffled is True
    assert cfg.force_balanced is False
    assert cfg.keep_sim_data is False

    # Path fields
    assert isinstance(cfg.feature_config_file, Path)
    assert isinstance(cfg.output_dir, Path)
    assert cfg.output_dir.is_absolute()


def test_genotype_matrix_simulation_config_valid():
    cfg = GenotypeMatrixSimulationConfig(**_valid_genotype_matrix_kwargs())

    assert cfg.sim_type == "genotype_matrix"
    assert cfg.nrep == 10
    assert cfg.nref == 20
    assert cfg.ntgt == 20
    assert cfg.ref_id == "REF"
    assert cfg.tgt_id == "TGT"
    assert cfg.src_id == "SRC"

    # Defaults
    assert cfg.nprocess == 1
    assert cfg.force_balanced is False
    assert cfg.keep_sim_data is False

    # Genotype-matrix specific
    assert cfg.num_polymorphisms == 5000
    assert cfg.num_upsamples == 2

    # Path fields
    assert isinstance(cfg.output_dir, Path)
    assert cfg.output_dir.is_absolute()


@pytest.mark.parametrize(
    "field",
    [
        "nrep",
        "nref",
        "ntgt",
        "ploidy",
        "seq_len",
        "nfeature",
        "nprocess",
    ],
)
@pytest.mark.parametrize("bad_value", [0, -1])
def test_simulation_config_positive_int_fields_must_be_gt_zero(
    field: str, bad_value: int
):
    kwargs = _valid_feature_vector_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        FeatureVectorSimulationConfig(**kwargs)

    kwargs = _valid_genotype_matrix_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        GenotypeMatrixSimulationConfig(**kwargs)


@pytest.mark.parametrize("bad_value", [0.0, -1e-8])
def test_simulation_config_mut_rate_must_be_strictly_positive(bad_value: float):
    kwargs = _valid_feature_vector_kwargs()
    kwargs["mut_rate"] = bad_value
    with pytest.raises(ValidationError):
        FeatureVectorSimulationConfig(**kwargs)

    kwargs = _valid_genotype_matrix_kwargs()
    kwargs["mut_rate"] = bad_value
    with pytest.raises(ValidationError):
        GenotypeMatrixSimulationConfig(**kwargs)


@pytest.mark.parametrize("bad_value", [-1e-8])
def test_simulation_config_rec_rate_must_be_ge_zero(bad_value: float):
    kwargs = _valid_feature_vector_kwargs()
    kwargs["rec_rate"] = bad_value
    with pytest.raises(ValidationError):
        FeatureVectorSimulationConfig(**kwargs)

    kwargs = _valid_genotype_matrix_kwargs()
    kwargs["rec_rate"] = bad_value
    with pytest.raises(ValidationError):
        GenotypeMatrixSimulationConfig(**kwargs)


@pytest.mark.parametrize("field", ["intro_prop", "non_intro_prop"])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_feature_vector_simulation_config_props_in_closed_interval(
    field: str, value: float
):
    kwargs = _valid_feature_vector_kwargs()
    kwargs[field] = value
    cfg = FeatureVectorSimulationConfig(**kwargs)
    assert getattr(cfg, field) == value


@pytest.mark.parametrize("field", ["intro_prop", "non_intro_prop"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_feature_vector_simulation_config_props_out_of_range(field: str, value: float):
    kwargs = _valid_feature_vector_kwargs()
    kwargs[field] = value
    with pytest.raises(ValidationError):
        FeatureVectorSimulationConfig(**kwargs)


@pytest.mark.parametrize("field", ["num_polymorphisms", "num_upsamples"])
@pytest.mark.parametrize("bad_value", [0, -1])
def test_genotype_matrix_simulation_config_specific_fields_must_be_gt_zero(
    field: str, bad_value: int
):
    kwargs = _valid_genotype_matrix_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        GenotypeMatrixSimulationConfig(**kwargs)


def test_feature_vector_simulation_config_output_dir_normalization():
    kwargs = _valid_feature_vector_kwargs()
    kwargs["output_dir"] = Path("relative/path")
    cfg = FeatureVectorSimulationConfig(**kwargs)
    assert cfg.output_dir.is_absolute()
    assert cfg.output_dir.as_posix().endswith("relative/path")


def test_genotype_matrix_simulation_config_output_dir_normalization():
    kwargs = _valid_genotype_matrix_kwargs()
    kwargs["output_dir"] = Path("relative/path")
    cfg = GenotypeMatrixSimulationConfig(**kwargs)
    assert cfg.output_dir.is_absolute()
    assert cfg.output_dir.as_posix().endswith("relative/path")


def test_feature_vector_simulation_config_extra_fields_forbidden():
    kwargs = _valid_feature_vector_kwargs()
    kwargs["unknown_field"] = 123
    with pytest.raises(ValidationError):
        FeatureVectorSimulationConfig(**kwargs)


def test_genotype_matrix_simulation_config_extra_fields_forbidden():
    kwargs = _valid_genotype_matrix_kwargs()
    kwargs["unknown_field"] = 123
    with pytest.raises(ValidationError):
        GenotypeMatrixSimulationConfig(**kwargs)


def test_feature_vector_simulation_config_default_flags():
    kwargs = _valid_feature_vector_kwargs()
    kwargs.pop("is_shuffled")  # default True
    cfg = FeatureVectorSimulationConfig(**kwargs)
    assert cfg.is_shuffled is True
    assert cfg.force_balanced is False
    assert cfg.keep_sim_data is False


def test_genotype_matrix_simulation_config_default_flags():
    cfg = GenotypeMatrixSimulationConfig(**_valid_genotype_matrix_kwargs())
    assert cfg.force_balanced is False
    assert cfg.keep_sim_data is False
