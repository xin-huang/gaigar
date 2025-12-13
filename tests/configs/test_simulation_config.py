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
from gaishi.configs import SimulationConfig


def _valid_kwargs():
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
        "feature_config_file": Path("config/features.yaml"),
        "nfeature": 128,
        "intro_prop": 0.5,
        "non_intro_prop": 0.5,
        "output_prefix": "train_sim",
        "output_dir": Path("results/train"),
        "seed": 42,
    }


def test_simulation_config_valid():
    cfg = SimulationConfig(**_valid_kwargs())

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
    # output_dir should be normalized to absolute path
    assert cfg.output_dir.is_absolute()


@pytest.mark.parametrize(
    "field", ["nrep", "nref", "ntgt", "ploidy", "seq_len", "nfeature"]
)
@pytest.mark.parametrize("bad_value", [0, -1])
def test_simulation_config_positive_int_fields_must_be_gt_zero(
    field: str, bad_value: int
):
    kwargs = _valid_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValidationError):
        SimulationConfig(**kwargs)


@pytest.mark.parametrize("field", ["mut_rate"])
@pytest.mark.parametrize("bad_value", [0.0, -1e-8])
def test_simulation_config_mut_rate_must_be_strictly_positive(
    field: str, bad_value: float
):
    kwargs = _valid_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValidationError):
        SimulationConfig(**kwargs)


@pytest.mark.parametrize("field", ["rec_rate"])
@pytest.mark.parametrize("bad_value", [-1e-8])
def test_simulation_config_rec_rate_must_be_ge_zero(field: str, bad_value: float):
    kwargs = _valid_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValidationError):
        SimulationConfig(**kwargs)


@pytest.mark.parametrize("field", ["intro_prop", "non_intro_prop"])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_simulation_config_props_in_closed_interval(field: str, value: float):
    kwargs = _valid_kwargs()
    kwargs[field] = value

    cfg = SimulationConfig(**kwargs)
    assert getattr(cfg, field) == value


@pytest.mark.parametrize("field", ["intro_prop", "non_intro_prop"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_simulation_config_props_out_of_range(field: str, value: float):
    kwargs = _valid_kwargs()
    kwargs[field] = value

    with pytest.raises(ValidationError) as excinfo:
        SimulationConfig(**kwargs)


def test_simulation_config_output_dir_normalization():
    kwargs = _valid_kwargs()
    kwargs["output_dir"] = Path("relative/path")

    cfg = SimulationConfig(**kwargs)
    assert cfg.output_dir.is_absolute()
    # The last parts of the path should be preserved
    assert cfg.output_dir.as_posix().endswith("relative/path")


def test_simulation_config_default_flags():
    cfg = SimulationConfig(**_valid_kwargs())

    assert cfg.is_shuffled is True
    assert cfg.force_balanced is False
    assert cfg.keep_sim_data is False
