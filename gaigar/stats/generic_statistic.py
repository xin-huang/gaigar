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


from abc import ABC, abstractmethod
from typing import Dict, Any


class GenericStatistic(ABC):
    """
    Generic class for all statistics.

    This class provides a generic interface for implementing specific statistical measures
    from genotype matrices, typically representing different populations or samples.
    """

    @staticmethod
    @abstractmethod
    def compute(self, **kwargs) -> Dict[str, Any]:
        """
        Computes the statistic based on the input genotype data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments specific to the statistic being implemented.

        Returns
        -------
        dict
            A dictionary containing the results of the statistic computation.
        """
        pass
