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


import numpy as np
from typing import Any
from seriate import seriate
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, cdist
from gaia.utils import parse_ind_file
from gaia.utils.preprocessors import DataPreprocessor


class GenotypeMatricesPreprocessor(DataPreprocessor):
    """
    A preprocessor subclass for generating genotype matrices from genomic data.

    This class extends DataPreprocessor to include additional functionality for sorting
    genotype matrices with seriation and linear sum assignment.

    """

    def __init__(
        self,
        ref_ind_file: str,
        tgt_ind_file: str,
        ref_rdm_spl_idx: list,
        tgt_rdm_spl_idx: list,
        is_sorted: bool,
    ):
        """
        Initializes a new instance of GenotypeMatricesPreprocessor with specific parameters.

        Parameters:
        -----------
        ref_ind_file : str
            Path to the file listing reference individual identifiers.
        tgt_ind_file : str
            Path to the file listing target individual identifiers.
        ref_rdm_spl_idx : list
            List of random sample indices for reference individuals.
        tgt_rdm_spl_idx : list
            List of random sample indices for target individuals.
        is_sorted: bool
            Indicates whether the genotype matrices should be sorted.

        """
        ref_samples = parse_ind_file(ref_ind_file)
        tgt_samples = parse_ind_file(tgt_ind_file)

        self.samples = {
            "Ref": ref_samples,
            "Tgt": tgt_samples,
        }

        self.is_sorted = is_sorted
        self.ref_rdm_spl_idx = ref_rdm_spl_idx
        self.tgt_rdm_spl_idx = tgt_rdm_spl_idx

    def run(
        self,
        chr_name: str,
        start: int,
        end: int,
        ploidy: int,
        is_phased: bool,
        ref_gts: np.ndarray,
        tgt_gts: np.ndarray,
        pos: np.ndarray,
        pos_idx: list,
    ) -> list[dict[str, Any]]:
        """
        Executes the feature vector generation process for a specified genomic window.

        Parameters
        ----------
        chr_name : str
            Name of the chromosome.
        start : int
            Start position of the genomic window.
        end : int
            End position of the genomic window.
        ploidy : int
            Ploidy of the samples, typically 2 for diploid organisms.
        is_phased : bool
            Indicates whether the genomic data is phased.
        ref_gts : np.ndarray
            Genotype array for the reference individuals.
        tgt_gts : np.ndarray
            Genotype array for the target individuals.
        pos : np.ndarray
            Array of variant positions within the genomic window.
        pos_idx : list
            List of position indices corresponding to the genomic window.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries with the formatted data for the genomic window.

        """

        ref_samples = self._create_sample_name_list(
            samples=self.samples["Ref"],
            ploidy=ploidy,
            is_phased=is_phased,
        )
        tgt_samples = self._create_sample_name_list(
            samples=self.samples["Tgt"],
            ploidy=ploidy,
            is_phased=is_phased,
        )

        ref_gts = np.transpose(ref_gts)
        tgt_gts = np.transpose(tgt_gts)

        if self.is_sorted:
            tgt_gts, tgt_samples = self._sort_tgt_genotypes(
                tgt_gts, tgt_samples, self.tgt_rdm_spl_idx
            )
            ref_gts, ref_samples = self._sort_ref_genotypes(
                ref_gts, tgt_gts, ref_samples, self.ref_rdm_spl_idx
            )
        else:
            for i in self.ref_rdm_spl_idx:
                ref_samples.append(ref_samples[i])
            for i in self.tgt_rdm_spl_idx:
                tgt_samples.append(tgt_samples[i])

        fwd_rlt_pos = np.diff(pos, prepend=[pos[0]])
        bwd_rlt_pos = np.diff(pos, append=[pos[-1]])

        fwd_rlt_pos = np.broadcast_to(fwd_rlt_pos, ref_gts.shape)
        bwd_rlt_pos = np.broadcast_to(bwd_rlt_pos, ref_gts.shape)

        data_dict = {
            "Chromosome": chr_name,
            "Start": start,
            "End": end,
            "Position": pos,
            "Position_index": pos_idx,
            "Forward_relative_position": fwd_rlt_pos,
            "Backward_relative_position": bwd_rlt_pos,
            "Ref_sample": ref_samples,
            "Ref_genotype": ref_gts,
            "Tgt_sample": tgt_samples,
            "Tgt_genotype": tgt_gts,
        }

        return [data_dict]

    def _create_sample_name_list(
        self,
        samples: list,
        ploidy: int,
        is_phased: bool,
    ) -> list[str]:
        """
        Creates a list of sample names, including phased information if applicable.

        Parameters
        ----------
        samples : list of str
            List of original sample identifiers.
        ploidy : int
            The ploidy level of the samples (e.g., 2 for diploid).
        is_phased : bool
            Indicates if the sample names should include phased information.

        Returns
        -------
        list of str
            A list of sample names, with phased information if applicable.

        """

        if is_phased:
            num_samples = len(samples) * ploidy
        else:
            num_samples = len(samples)

        samples = [
            f"{samples[int(i/ploidy)]}_{i%ploidy+1}" if is_phased else samples[i]
            for i in range(num_samples)
        ]

        return samples

    def _sort_ref_genotypes(
        self,
        ref_gts: np.ndarray,
        tgt_gts: np.ndarray,
        samples: list,
        rdm_spl_idx: list,
        metric: str = "euclidean",
    ) -> tuple[np.ndarray, list]:
        """
        Sorts reference genotypes to best match target genotypes using linear sum assignment.

        Parameters
        ----------
        ref_gts : np.ndarray
            The reference genotypes.
        tgt_gts : np.ndarray
            The target genotypes.
        samples : list
            List of sample identifiers corresponding to ref_gts.
        rdm_spl_idx: list
            List of random sample indices for reference individuals.
        metric : str, optional
            Distance metric to use. Default: 'euclidean'.

        Returns
        -------
        tuple[np.ndarray, list]
            A tuple containing the sorted reference genotypes array and the reordered list of sample identifiers.

        """

        for i in rdm_spl_idx:
            samples.append(samples[i])

        D = cdist(tgt_gts, ref_gts, metric=metric)
        D[np.where(np.isnan(D))] = 0
        _, idx = linear_sum_assignment(D)

        sorted_ref_gts = ref_gts[idx]
        sorted_samples = [samples[i] for i in idx]

        return sorted_ref_gts, sorted_samples

    def _sort_tgt_genotypes(
        self,
        gts: np.ndarray,
        samples: list,
        rdm_spl_idx: list,
        metric: str = "euclidean",
    ) -> tuple[np.ndarray, list]:
        """
        Sorts target genotypes using seriation based on a specified distance metric.

        Parameters
        ----------
        gts : np.ndarray
            The target genotypes.
        samples : list
            List of sample identifiers corresponding to gts.
        rdm_spl_idx: list
            List of random sample indices for target individuals.
        metric : str, optional
            Distance metric to use. Default: 'euclidean'.

        Returns
        -------
        tuple[np.ndarray, list]
            A tuple containing the sorted target genotypes array and the reordered list of sample identifiers.

        """

        for i in rdm_spl_idx:
            samples.append(samples[i])

        D = pdist(gts, metric=metric)
        D[np.where(np.isnan(D))] = 0
        idx = seriate(D, timeout=0)

        sorted_gts = gts[idx]
        sorted_samples = [samples[i] for i in idx]

        return sorted_gts, sorted_samples
