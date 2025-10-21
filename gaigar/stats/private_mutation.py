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
from gaigar.registries.stat_registry import STAT_REGISTRY
from gaigar.stats import GenericStatistic


@STAT_REGISTRY.register("num_private")
class PrivateMutation(GenericStatistic):
    """
    Private-variant count statistic.

    Counts, for each target sample (column), the number of loci where the
    target carries derived alleles (>0) while the reference population carries
    none at that locus (sum over reference samples equals 0).
    """

    @staticmethod
    def compute(*, ref_gts: np.ndarray, tgt_gts: np.ndarray) -> Dict[str, Any]:
        """
        Computes per-sample private-variant counts.

        Parameters
        ----------
        ref_gts : np.ndarray
            Reference genotype matrix of shape `(n_sites, n_ref_samples)`.
        tgt_gts : np.ndarray
            Target genotype matrix of shape `(n_sites, n_tgt_samples)`.

        Returns
        -------
        dict
            `{private_mutation: np.ndarray}` where the array has length
            `n_tgt_samples`, giving counts of private variants per target sample.
        """
        counts = PrivateMutation._cal_mut_num(ref_gts, tgt_gts)

        return {"private_mutation": counts}

    @staticmethod
    def _cal_mut_num(ref_gt: np.ndarray, tgt_gt: np.ndarray) -> np.ndarray:
        """
        Core computation: count private mutations per target sample.

        Parameters
        ----------
        ref_gt : np.ndarray
            Genotype matrix from the reference population `(n_sites, n_ref_samples)`.
        tgt_gt : np.ndarray
            Genotype matrix from the target population `(n_sites, n_tgt_samples)`.

        Returns
        -------
        np.ndarray
            Vector of length `n_tgt_samples` with the number of private mutations
            per target sample.
        """
        ref_sum = np.sum(ref_gt, axis=1, keepdims=True)  # (n_sites, 1)
        mut_num = np.sum((tgt_gt > 0) * (ref_sum == 0), axis=0)

        return mut_num
