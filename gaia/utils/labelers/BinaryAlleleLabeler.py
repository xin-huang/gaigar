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


import allel
import numpy as np
import pandas as pd
from gaia.utils.labelers import DataLabeler


class BinaryAlleleLabeler(DataLabeler):
    """
    A labeler for binary classification of alleles based on introgression information.

    """

    def __init__(self, ploidy: int, is_phased: bool, num_polymorphisms: int):
        """
        Initializes a new instance of BinaryAlleleLabeler with given ploidy, phasing status, and number of polymorphisms.

        Parameters
        ----------
        ploidy : int
            Ploidy of the organism (e.g., 2 for diploid).
        is_phased : bool
            Whether the data is phased.
        num_polymorphisms : int
            Number of polymorphisms to consider.

        """
        super().__init__(
            ploidy=ploidy,
            is_phased=is_phased,
        )

        self.num_polymorphisms = num_polymorphisms

    def run(
        self, tgt_ind_file: str, vcf_file: str, true_tract_file: str, rep: int = None
    ) -> pd.DataFrame:
        """
        Runs the allele labeling process.

        Parameters
        ----------
        tgt_ind_file : str
            Path to the target individual file.
        vcf_file : str
            Path to the VCF file.
        true_tract_file : str
            Path to the true tract file.
        rep : int, optional
            A repetition identifier (default is None).

        Returns
        -------
        pd.DataFrame
            A DataFrame with sample names and their corresponding allele labels.

        Raises
        ------
        ValueError
            If the number of polymorphisms in the VCF file is less than the expected number.
        FileNotFoundError
            If any of the input files are not found.

        """
        vcf = allel.read_vcf(vcf_file)
        positions = vcf["variants/POS"]

        if len(positions) < self.num_polymorphisms:
            raise ValueError(
                f"Number of polymorphisms in {vcf_file} is less than {self.num_polymorphisms}."
            )

        labels = {}

        try:
            with open(tgt_ind_file, "r") as f:
                if self.is_phased is True:
                    for line in f:
                        sample = line.rstrip()
                        for p in range(self.ploidy):
                            sample_name = f"{sample}_{p+1}"
                            labels[sample_name] = {
                                "Sample": sample_name,
                                "Label": np.zeros(len(positions), dtype=int),
                            }
                else:
                    for line in f:
                        sample = line.rstrip()
                        labels[sample] = {
                            "Sample": sample,
                            "Label": np.zeros(len(positions), dtype=int),
                        }
        except FileNotFoundError:
            raise FileNotFoundError(f"tgt_ind_file {tgt_ind_file} not found.")

        try:
            with open(true_tract_file, "r") as f:
                for line in f:
                    chr_name, start, end, sample = line.rstrip().split("\t")
                    start, end = int(start), int(end)
                    if sample in labels:
                        labels[sample]["Label"][
                            (positions >= start) & (positions < end)
                        ] = 1
        except FileNotFoundError:
            raise FileNotFoundError(f"true_tract_file {true_tract_file} not found.")

        if rep is not None:
            for s in labels:
                labels[s]["Replicate"] = rep

        return labels
