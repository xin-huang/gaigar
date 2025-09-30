# GNU General Public License v3.0
# Copyright 2024 Xin Huang
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
from gaia.utils import read_data, create_windows
from gaia.utils.generators import DataGenerator


class GenomicDataGenerator(DataGenerator):
    """
    Generates genomic data for each specified window from VCF and other related files.

    """
    def __init__(self, vcf_file: str, chr_name: str, ref_ind_file: str, tgt_ind_file: str,
                 win_len: int, win_step: int, anc_allele_file: str = None,
                 ploidy: int = 2, is_phased: bool = True):
        """
        Initializes a new instance of GenomicDataGenerator.

        Parameters
        ----------
        vcf_file : str
            The path to the VCF file containing variant data.
        chr_name : str
            The name of the chromosome to process data for.
        ref_ind_file : str
            The path to the file containing identifiers for reference individuals.
        tgt_ind_file : str
            The path to the file containing identifiers for target individuals.
        win_len : int
            The length of each window in base pairs.
        win_step : int
            The step size between windows in base pairs.
        anc_allele_file : str, optional
            The path to the file containing ancestral allele information.
            Default: None.
        ploidy : int, optional
            The ploidy of the genome. Default: 2.
        is_phased : bool, optional
            Specifies whether the genotype data is phased. Default: True.
    
        Raises
        ------
        ValueError
            If `win_len` is less than or equal to 0, if `win_step` is negative, 
            if `ploidy` is less than or equal to 0, or if `chr_name` is not in the VCF file.

        """
        if win_len <= 0:
            raise ValueError("win_len must be greater than 0.")

        if win_step < 0:
            raise ValueError("win_step must be non-negative.")

        if ploidy <= 0:
            raise ValueError("ploidy must be greater than 0.")

        self.ploidy = ploidy
        self.is_phased = is_phased

        ref_data, ref_samples, tgt_data, tgt_samples = read_data(vcf_file, ref_ind_file, tgt_ind_file, anc_allele_file, is_phased)

        if chr_name not in tgt_data:
            raise ValueError(f"{chr_name} is not present in the VCF file.")

        windows = create_windows(tgt_data[chr_name]['POS'], chr_name, win_step, win_len)

        self.data = []
        for w in range(len(windows)):
            chr_name, start, end = windows[w]
            ref_gts = ref_data[chr_name]['GT']
            tgt_gts = tgt_data[chr_name]['GT']
            pos = tgt_data[chr_name]['POS']
            idx = (pos>start)*(pos<=end)
            sub_ref_gts = ref_gts[idx]
            sub_tgt_gts = tgt_gts[idx]
            sub_pos = pos[idx]

            d = {
                'chr_name': chr_name,
                'start': start,
                'end': end,
                'ploidy': self.ploidy,
                'is_phased': self.is_phased,
                'ref_gts': sub_ref_gts,
                'tgt_gts': sub_tgt_gts,
                'pos': sub_pos,
            }

            self.data.append(d)


    def get(self):
        """
        Yields genomic data for each window.

        Yields
        ------
        dict
            A dictionary containing chromosome name, start and end positions, 
            ploidy and phase information, reference and target genotypes, 
            and positions for each window.

        """
        for d in self.data:
            yield d


    def __len__(self):
        return len(self.data)
