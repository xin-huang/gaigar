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


import numpy as np
from typing import Dict, Any
from scipy.spatial import distance_matrix

from gaigar.registries.stat_registry import STAT_REGISTRY
from gaigar.stats import GenericStatistic


@STAT_REGISTRY.register("distance")
class Distance(GenericStatistic):
    """
    Pairwise Euclidean distance statistic.

    Computes pairwise Euclidean distances between columns (samples) of two
    genotype matrices and returns the result under a caller-provided key.
    """

    @staticmethod
    def compute(*, gt1: np.ndarray, gt2: np.ndarray, key: str) -> Dict[str, Any]:
        """
        Computes pairwise Euclidean distances for two genotype matrices.

        Parameters
        ----------
        gt1 : np.ndarray
            Genotype matrix 1 with shape (n_sites, n_samples1). Samples are columns.
        gt2 : np.ndarray
            Genotype matrix 2 with shape (n_sites, n_samples2). Samples are columns.
        key : str
            The dictionary key to use for the returned distance matrix (e.g., 'ref_dist' or 'tgt_dist').

        Returns
        -------
        dict
            A dictionary {key: np.ndarray} where the array has shape
            (n_samples2, n_samples1) and is sorted along the last axis.
        """
        dists = Distance._cal_dist(gt1, gt2)

        return {key: dists}

    @staticmethod
    def _cal_dist(gt1: np.ndarray, gt2: np.ndarray) -> np.ndarray:
        """
        Core distance computation (Euclidean).

        Parameters
        ----------
        gt1 : np.ndarray
            Genotype matrix 1 with shape (n_sites, n_samples1).
        gt2 : np.ndarray
            Genotype matrix 2 with shape (n_sites, n_samples2).

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_samples2, n_samples1), computed as
            distance_matrix(gt2.T, gt1.T) and sorted along the last axis.
        """
        dists = distance_matrix(np.transpose(gt2), np.transpose(gt1))
        dists.sort()

        return dists
