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
from gaishi.configs import InferConfig
from gaishi.registries.model_registry import MODEL_REGISTRY
from gaishi.preprocess import preprocess_feature_vectors
from gaishi.utils import UniqueKeyLoader


def infer(
    model_file: str, 
    config: str, 
    output: str,
) -> None:
    """ """
    try:
        with open(config, "r") as f:
            config_dict = yaml.load(f, Loader=UniqueKeyLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file '{config}': {e}")

    infer_config = InferConfig(**config_dict)
    preprocess_feature_vectors(
        **infer_config.preprocess.model_dump(),
    )

    data = f"{infer_config.preprocess.output_dir}/{infer_config.preprocess.output_prefix}.features"
    model_name = infer_config.model.name
    model_params = infer_config.model.params
    model_cls = MODEL_REGISTRY.get(model_name)
    model_cls.infer(
        data=data,
        model=model,
        output=output,
        **model_params,
    )
