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


from pydantic import BaseModel
from gaishi.configs import ModelConfig
from gaishi.configs import SimulationConfig
from gaishi.configs import PreprocessConfig


class GlobalConfig(BaseModel):
    """
    Top-level config for runing gaishi

    - training: simulation + model details.
    - infer: preprocess + model details.
    """

    # Simulation block
    simulation: SimulationConfig

    # Preprocess block
    preprocess: PreprocessConfig

    # Model choice
    model: ModelConfig

    # Generic training options
    # test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Hold-out test fraction")
    # val_size: float = Field(
    #    0.0,
    #    ge=0.0,
    #    lt=1.0,
    #    description="Optional validation fraction (0 means no explicit val split)",
    # )
    # shuffle_before_split: bool = True
