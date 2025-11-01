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
from typing import Any, Dict
from gaigar.registries.stat_registry import STAT_REGISTRY
from gaigar.stats import GenericStatistic


@STAT_REGISTRY.register("spectrum")
class Spectrum(GenericStatistic):
    """
    Spectrum statistic.

    Computes per-individual allele-frequency spectra (ArchIE-style i-ton)
    from a target population genotype matrix.

    Notes
    -----
    This implementation is stateless. Use `compute` as a static
    function; no instance initialization is required.
    """

    @staticmethod
    def compute(**kwargs) -> Dict[str, Any]:
        """
        Computes per-individual frequency spectra for the target population.

        Parameters
        ----------
        **kwargs
            tgt_gts : np.ndarray
                Genotype matrix of shape `(n_sites, n_samples)` for the target population.
            is_phased : bool
                If `True`, treat genotypes as phased haplotypes (effective ploidy = 1).
            ploidy : int
                Sample ploidy when `is_phased` is `False` (ignored otherwise).

        Returns
        -------
        dict
            A dictionary with a single key:
            `{"spectrum": np.ndarray}`, where the array has shape
            `(n_samples, n_bins)` and `n_bins = n_samples * ploidy + 1`.
            Bin 0 is zeroed to exclude non-segregating sites (ArchIE behavior).

        Raises
        ------
        ValueError
            If any required key is missing.
        """
        tgt_gts, is_phased, ploidy = Spectrum.require(
            kwargs, "tgt_gts", "is_phased", "ploidy"
        )
        spec = Spectrum._calc_n_ton(tgt_gts, is_phased=is_phased, ploidy=ploidy)

        return {"spectrum": spec}

    @staticmethod
    def _calc_n_ton(tgt_gt: np.ndarray, is_phased: bool, ploidy: int) -> np.ndarray:
        """
        Computes individual frequency spectra for samples (ArchIE-style i-ton).

        Parameters
        ----------
        tgt_gt : np.ndarray
            Genotype matrix of shape (n_sites, n_samples) for the target population.
        is_phased : bool
            If True, treat genotypes as phased haplotypes (ploidy is coerced to 1).
        ploidy : int
            Ploidy of the genomes (ignored when `is_phased` is True).

        Returns
        -------
        np.ndarray
            Spectra array of shape (n_samples, n_bins), where
            `n_bins = n_samples * ploidy + 1`. Bin 0 is set to 0 to mimic ArchIE
            behavior (non-segregating sites are not counted).
        """
        if is_phased:
            ploidy = 1

        mut_num, sample_num = tgt_gt.shape
        iv = np.ones((sample_num, 1))
        counts = (tgt_gt > 0) * np.matmul(tgt_gt, iv)
        spectra = np.array(
            [
                np.bincount(
                    counts[:, idx].astype("int64"), minlength=sample_num * ploidy + 1
                )
                for idx in range(sample_num)
            ]
        )
        # ArchIE does not count non-segregating sites
        spectra[:, 0] = 0

        return spectra
