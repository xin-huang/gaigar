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


import scipy
import numpy as np
from typing import Dict, Any
from scipy.spatial import distance_matrix
from gaishi.registries.stat_registry import STAT_REGISTRY
from gaishi.stats import GenericStatistic


class Distance(GenericStatistic):
    """
    Pairwise Euclidean distance statistic.

    Computes pairwise Euclidean distances between columns (samples) of two
    genotype matrices and returns the result under a caller-provided key.
    """

    @staticmethod
    def compute(**kwargs) -> Dict[str, Any]:
        """
        Computes pairwise Euclidean distances for two genotype matrices.

        Parameters
        ----------
        **kwargs
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

        Raises
        ------
        ValueError
            If any required key is missing.
        """
        gt1, gt2, key = Distance.require(kwargs, "gt1", "gt2", "key")
        dists = Distance._cal_dist(gt1, gt2)
        items = {}
        if kwargs.get("all", False):
            items[f"All_{key}"] = dists
        if kwargs.get("minimum", False):
            items[f"Minimum_{key}"] = np.min(dists, axis=1)
        if kwargs.get("maximum", False):
            items[f"Maximum_{key}"] = np.max(dists, axis=1)
        if kwargs.get("mean", False):
            items[f"Mean_{key}"] = np.mean(dists, axis=1)
        if kwargs.get("median", False):
            items[f"Median_{key}"] = np.median(dists, axis=1)
        if kwargs.get("variance", False):
            items[f"Variance_{key}"] = np.var(dists, axis=1)
        if kwargs.get("skew", False):
            skew_vals = scipy.stats.skew(
                dists,
                axis=1,  # bias=False, nan_policy="omit"
            )
            skew_vals[~np.isfinite(skew_vals)] = 0
            items[f"Skew_{key}"] = skew_vals
        if kwargs.get("kurtosis", False):
            kurtosis_vals = scipy.stats.kurtosis(
                dists,
                axis=1,  # bias=False, nan_policy="omit"
            )
            kurtosis_vals[~np.isfinite(kurtosis_vals)] = 0
            items[f"Kurtosis_{key}"] = kurtosis_vals

        return items

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


@STAT_REGISTRY.register("ref_dist")
class RefDistance(GenericStatistic):
    """
    Computes pairwise Euclidean distances between columns (samples) of the
    reference genotype matrix (`ref_gts`) and the target genotype matrix
    (`tgt_gts`).
    """

    @staticmethod
    def compute(**kwargs) -> Dict[str, Any]:
        """
        Computes distances from reference to target samples.

        Parameters
        ----------
        **kwargs
            ref_gts : np.ndarray
                Reference genotype matrix of shape `(n_sites, n_ref_samples)`.
            tgt_gts : np.ndarray
                Target genotype matrix of shape `(n_sites, n_tgt_samples)`.

        Returns
        -------
        dict
            `{'ref_dist': np.ndarray}` where the array has shape
            `(n_tgt_samples, n_ref_samples)`; rows correspond to target samples,
            columns to reference samples. Each row is sorted in non-decreasing order.
        """
        ref_gts, tgt_gts = RefDistance.require(kwargs, "ref_gts", "tgt_gts")
        kwargs |= {
            "gt1": ref_gts,
            "gt2": tgt_gts,
            "key": "ref_dist",
        }

        return Distance.compute(**kwargs)


@STAT_REGISTRY.register("tgt_dist")
class TgtDistance(GenericStatistic):
    """
    Computes pairwise Euclidean distances between columns (samples) of the
    target genotype matrix (`tgt_gts`) itself.
    """

    @staticmethod
    def compute(**kwargs):
        """
        Computes self-distances among target samples.

        Parameters
        ----------
        **kwargs
            tgt_gts : np.ndarray
                Target genotype matrix of shape `(n_sites, n_tgt_samples)`.

        Returns
        -------
        dict
            `{'tgt_dist': np.ndarray}` where the array has shape
            `(n_tgt_samples, n_tgt_samples)`; rows and columns both correspond
            to target samples. Each row is sorted in non-decreasing order.
        """

        (tgt_gts,) = TgtDistance.require(kwargs, "tgt_gts")
        kwargs |= {
            "gt1": tgt_gts,
            "gt2": tgt_gts,
            "key": "tgt_dist",
        }

        return Distance.compute(**kwargs)
