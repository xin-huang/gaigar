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


import yaml
from gaishi.configs import GlobalConfig
from gaishi.registries.model_registry import MODEL_REGISTRY
from gaishi.simulate import simulate_feature_vectors
from gaishi.simulate import simulate_genotype_matrices
from gaishi.utils import UniqueKeyLoader


def train(
    demes: str,
    config: str,
    output: str,
) -> None:
    """
    Run simulation and model training from YAML configuration.

    Parameters
    ----------
    demes : str
        Path to the demography (demes) YAML file used for simulation.
    config : str
        Path to the gaishi configuration YAML file.
    output : str
        Output path or directory passed to the model's `train` method, used
        to store the trained model.
    """
    try:
        with open(config, "r") as f:
            config_dict = yaml.load(f, Loader=UniqueKeyLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file '{config}': {e}")

    global_config = GlobalConfig(**config_dict)
    if global_config.simulation.sim_type == "feature_vector":
        simulate_feature_vectors(
            demo_model_file=demes,
            **global_config.simulation.model_dump(),
        )

        data = f"{global_config.simulation.output_dir}/{global_config.simulation.output_prefix}.features"
    elif global_config.simulation.sim_type == "genotype_matrix":
        simulate_genotype_matrices(
            demo_model_file=demes,
            **global_config.simulation.model_dump(),
        )

        data = f"{global_config.simulation.output_dir}/{global_config.simulation.output_prefix}.h5"
    else:
        raise ValueError("")

    model_name = global_config.model.name
    model_params = global_config.model.params
    model_cls = MODEL_REGISTRY.get(model_name)
    model_cls.train(
        data=data,
        output=output,
        **model_params,
    )
