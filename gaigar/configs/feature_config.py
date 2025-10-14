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


SUPPORTED_STATISTICS = [
    "ref_dist",
    "tgt_dist",
    "spectrum",
    "num_private",
    "sstar",
]


from pydantic import RootModel, field_validator, ValidationError
from typing import Dict, Literal, List, Optional, Union


class FeatureConfig(
    RootModel[
        Dict[
            str,
            Union[bool, Dict[str, Union[bool, int]]],
        ]
    ]
):
    pass
