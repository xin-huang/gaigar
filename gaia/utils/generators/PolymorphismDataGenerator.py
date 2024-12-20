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
from gaia.utils import read_data, split_genome
from gaia.utils.generators import DataGenerator


class PolymorphismDataGenerator(DataGenerator):
    """
    A class to generate polymorphism data from VCF files.

    This class processes genetic data from VCF files and generates polymorphisms,
    handling various parameters such as chromosome name, number of polymorphisms,
    ploidy, and random sampling.

    """

    def __init__(
        self,
        vcf_file: str,
        chr_name: str,
        ref_ind_file: str,
        tgt_ind_file: str,
        num_polymorphisms: int,
        step_size: int = None,
        anc_allele_file: str = None,
        ploidy: int = 2,
        is_phased: bool = True,
        random_polymorphisms: bool = False,
        seed: int = None,
        num_upsamples: int = None,
    ):
        """
        Initializes the PolymorphismDataGenerator with the given parameters.

        Parameters
        ----------
        vcf_file : str
            Path to the VCF file containing genetic data.
        chr_name : str
            Name of the chromosome to be processed.
        ref_ind_file : str
            Path to the file listing reference individual identifiers.
        tgt_ind_file : str
            Path to the file listing target individual identifiers.
        num_polymorphisms : int
            Number of polymorphisms to generate.
        step_size : int, optional
            Step size for polymorphism generation. Default: None.
        anc_allele_file : str, optional
            Path to the file containing ancestral allele information. Default: None.
        ploidy : int, optional
            Ploidy level of the organisms. Default: 2.
        is_phased : bool, optional
            Indicates whether the genotype data is phased. Default: True.
        random_polymorphisms : bool, optional
            Indicates whether polymorphisms should be randomly selected. Default: False.
        seed : int, optional
            Seed for random number generation. Default: None.
        num_upsamples : int, optional
            Number of samples after upsampling. Default: None.

        Raises
        ------
        ValueError
            If `num_polymorphisms` is less than or equal to 0.
            If `step_size` is not None and is less than 0.
            If `ploidy` is less than or equal to 0.
            If `chr_name` is not present in the VCF file.

        """

        if num_polymorphisms <= 0:
            raise ValueError("`num_polymorphisms` must be greater than 0.")

        if step_size is not None and step_size < 0:
            raise ValueError("`step_size` must be non-negative.")

        if ploidy <= 0:
            raise ValueError("`ploidy` must be greater than 0.")

        self.ploidy = ploidy
        self.is_phased = is_phased

        ref_data, ref_samples, tgt_data, tgt_samples = read_data(
            vcf_file, ref_ind_file, tgt_ind_file, anc_allele_file, is_phased
        )

        if chr_name not in tgt_data:
            raise ValueError(f"{chr_name} is not present in the VCF file.")

        pos = tgt_data[chr_name]["POS"]
        ref_gts = ref_data[chr_name]["GT"]
        tgt_gts = tgt_data[chr_name]["GT"]
        num_refs = len(ref_samples)
        num_tgts = len(tgt_samples)

        if is_phased:
            num_refs *= ploidy
            num_tgts *= ploidy
            if num_upsamples is not None:
                num_upsamples *= ploidy

        polymorphisms = split_genome(
            pos=pos,
            chr_name=chr_name,
            polymorphism_size=num_polymorphisms,
            step_size=step_size,
            window_based=False,
            random_polymorphisms=random_polymorphisms,
            seed=seed,
        )

        ref_gts, self.ref_rdm_spl_idx = self._upsample(ref_gts, num_refs, num_upsamples)
        tgt_gts, self.tgt_rdm_spl_idx = self._upsample(tgt_gts, num_tgts, num_upsamples)

        self.data = []
        for p in range(len(polymorphisms)):
            chr_name, idx = polymorphisms[p]

            if random_polymorphisms:
                start = "Random"
                end = "Random"
            else:
                start = idx[0]
                end = idx[-1] + 1

            sub_pos = pos[idx]
            sub_ref_gts = ref_gts[idx]
            sub_tgt_gts = tgt_gts[idx]

            d = {
                "chr_name": chr_name,
                "start": start,
                "end": end,
                "ploidy": self.ploidy,
                "is_phased": self.is_phased,
                "ref_gts": sub_ref_gts,
                "tgt_gts": sub_tgt_gts,
                "pos": sub_pos,
                "pos_idx": idx,
            }

            self.data.append(d)

    def get(self):
        """
        Generator to yield polymorphism data.

        Yields
        ------
        dict
            A dictionary containing polymorphism data for each window.

        """

        for d in self.data:
            yield d

    def _upsample(
        self, gts: np.ndarray, num_samples: int, num_upsamples: int = None
    ) -> tuple[np.ndarray, list]:
        """
        Upsamples the genotype data.

        Parameters
        ----------
        gts : np.ndarray
            Genotype data array.
        num_samples : int
            Number of samples in the genotype data.
        num_upsamples : int, optional
            Number of samples after upsampling. Default is None.

        Returns
        -------
        tuple[np.ndarray, list]
            A tuple containing the upsampled genotype data array
            and the list of indices of randomly sampled data.

        """

        if num_upsamples is None:
            return gts, []
        else:
            num_samples_added = max(num_upsamples - num_samples, 0)
            random_samples = np.random.randint(num_samples, size=int(num_samples_added))
            gts = np.transpose(gts).tolist()
            gts_upsampled = gts
            for idx in random_samples:
                gts_upsampled.append(gts[idx])

            gts_upsampled = np.transpose(gts_upsampled)

            return gts_upsampled, random_samples

    def __len__(self):
        """
        Returns the number of polymorphism windows.

        Returns
        -------
        int
            Number of polymorphism windows.

        """

        return len(self.data)
