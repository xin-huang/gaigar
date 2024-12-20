# Copyright 2024 Xin Huang
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


class DataLabeler(ABC):
    """
    Abstract base class for labeling data based on true tracts and target individual files.

    Subclasses are expected to implement the `run` method to perform specific labeling strategies,
    given a set of initial parameters defined during instantiation.

    """

    def __init__(self, ploidy: int, is_phased: bool):
        """
        Initializes a new DataLabeler instance with the given configuration.

        Parameters
        ----------
        ploidy : int
            Ploidy of the genome.
        is_phased : bool
            Indicates whether the tracts are phased.

        Notes
        -----
        This class provides a common interface for data labeling strategies.
        Implementing subclasses must provide a specific mechanism for applying
        labels to data based on the parameters provided at instantiation and
        any additional runtime arguments.

        """
        self.ploidy = ploidy
        self.is_phased = is_phased

    @abstractmethod
    def run(self, **kwargs):
        """
        Execute the data labeling process.

        This method should be implemented by subclasses to define specific labeling
        strategies. It utilizes the class's initialization parameters along with any
        additional keyword arguments provided at runtime.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments providing extra context or control
            over the labeling process. The accepted keys and values depend on
            the specific subclass implementation.

        Returns
        -------
        The method can return results or status information if applicable. This depends on the implementation
        details of each subclass.

        Raises
        ------
        Subclasses may raise exceptions if errors occur during the simulation process. Implementers should
        document any exceptions that their method implementation may raise.

        """
        pass
