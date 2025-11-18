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


import allel
import math
import numpy as np
from multiprocessing import Process, Queue


def parse_ind_file(filename):
    """
    Description:
        Helper function to read sample information from files.

    Arguments:
        filename str: Name of the file containing sample information.

    Returns:
        samples list: Sample information.
    """

    f = open(filename, "r")
    samples = [l.rstrip() for l in f.readlines()]
    f.close()

    if len(samples) == 0:
        raise Exception(f"No sample is found in {filename}! Please check your data.")

    return samples


def read_geno_data(vcf, ind, anc_allele_file, filter_missing):
    """
    Description:
        Helper function to read genotype data from VCF files.

    Arguments:
        vcf str: Name of the VCF file containing genotype data.
        ind list: List containing names of samples.
        anc_allele_file str: Name of the BED file containing ancestral allele information.
        filter_missing bool: Indicating whether filtering missing data or not.

    Returns:
        data dict: Genotype data.
    """

    vcf = allel.read_vcf(str(vcf), alt_number=1, samples=ind)
    gt = vcf["calldata/GT"]
    chr_names = np.unique(vcf["variants/CHROM"])
    samples = vcf["samples"]
    pos = vcf["variants/POS"]
    ref = vcf["variants/REF"]
    alt = vcf["variants/ALT"]

    if anc_allele_file != None:
        anc_allele = read_anc_allele(anc_allele_file)
    data = dict()
    for c in chr_names:
        if c not in data.keys():
            data[c] = dict()
            data[c]["POS"] = pos
            data[c]["REF"] = ref
            data[c]["ALT"] = alt
            data[c]["GT"] = gt
        index = np.where(vcf["variants/CHROM"] == c)
        data = filter_data(data, c, index)
        # Remove missing data
        if filter_missing:
            index = data[c]["GT"].count_missing(axis=1) == len(samples)
            data = filter_data(data, c, ~index)
        if anc_allele_file != None:
            data = check_anc_allele(data, anc_allele, c)

    return data


def filter_data(data, c, index):
    """
    Description:
        Helper function to filter genotype data.

    Arguments:
        data dict: Genotype data for filtering.
            c str: Names of chromosomes.
        index numpy.ndarray: A boolean array determines variants to be removed.

    Returns:
        data dict: Genotype data after filtering.
    """

    data[c]["POS"] = data[c]["POS"][index]
    data[c]["REF"] = data[c]["REF"][index]
    data[c]["ALT"] = data[c]["ALT"][index]
    data[c]["GT"] = allel.GenotypeArray(data[c]["GT"][index])

    return data


def read_data(vcf_file, ref_ind_file, tgt_ind_file, anc_allele_file, is_phased):
    """
    Description:
        Helper function for reading data from reference and target populations.

    Arguments:
        vcf str: Name of the VCF file containing genotype data from reference, target, and source populations.
        ref_ind_file str: Name of the file containing sample information from reference populations.
        tgt_ind_file str: Name of the file containing sample information from target populations.
        anc_allele_file str: Name of the file containing ancestral allele information.
        phased bool: If True, use phased genotypes; otherwise, use unphased genotypes.

    Returns:
        ref_data dict: Genotype data from reference populations.
        ref_samples list: Sample information from reference populations.
        tgt_data dict: Genotype data from target populations.
        tgt_samples list: Sample information from target populations.
    """

    ref_data = ref_samples = tgt_data = tgt_samples = None
    if ref_ind_file is not None:
        ref_samples = parse_ind_file(ref_ind_file)
        ref_data = read_geno_data(vcf_file, ref_samples, anc_allele_file, True)

    if tgt_ind_file is not None:
        tgt_samples = parse_ind_file(tgt_ind_file)
        tgt_data = read_geno_data(vcf_file, tgt_samples, anc_allele_file, True)

    if (ref_ind_file is not None) and (tgt_ind_file is not None):
        chr_names = tgt_data.keys()
        for c in chr_names:
            # Remove variants fixed in both the reference and target individuals
            ref_fixed_variants = np.sum(ref_data[c]["GT"].is_hom_alt(), axis=1) == len(
                ref_samples
            )
            tgt_fixed_variants = np.sum(tgt_data[c]["GT"].is_hom_alt(), axis=1) == len(
                tgt_samples
            )
            fixed_index = np.logical_and(ref_fixed_variants, tgt_fixed_variants)
            index = np.logical_not(fixed_index)
            fixed_pos = ref_data[c]["POS"][fixed_index]
            ref_data = filter_data(ref_data, c, index)
            tgt_data = filter_data(tgt_data, c, index)

    if is_phased:
        for c in chr_names:
            mut_num, ind_num, ploidy = ref_data[c]["GT"].shape
            ref_data[c]["GT"] = np.reshape(
                ref_data[c]["GT"].values, (mut_num, ind_num * ploidy)
            )
            mut_num, ind_num, ploidy = tgt_data[c]["GT"].shape
            tgt_data[c]["GT"] = np.reshape(
                tgt_data[c]["GT"].values, (mut_num, ind_num * ploidy)
            )
    else:
        for c in chr_names:
            ref_data[c]["GT"] = np.sum(ref_data[c]["GT"], axis=2)
            tgt_data[c]["GT"] = np.sum(tgt_data[c]["GT"], axis=2)

    return ref_data, ref_samples, tgt_data, tgt_samples


def get_ref_alt_allele(ref, alt, pos):
    """
    Description:
        Helper function to index REF and ALT alleles with genomic positions.

    Arguments:
        ref list: REF alleles.
        alt list: ALT alleles.
        pos list: Genomic positions.

    Returns:
        ref_allele dict: REF alleles.
        alt_allele dict: ALT alleles.
    """

    ref_allele = dict()
    alt_allele = dict()

    for i in range(len(pos)):
        r = ref[i]
        a = alt[i]
        p = pos[i]
        ref_allele[p] = r
        alt_allele[p] = a

    return ref_allele, alt_allele


def read_anc_allele(anc_allele_file):
    """
    Description:
        Helper function to read ancestral allele information from files.

    Arguments:
        anc_allele_file str: Name of the BED file containing ancestral allele information.

    Returns:
        anc_allele dict: Ancestral allele information.
    """

    anc_allele = dict()
    with open(anc_allele_file, "r") as f:
        for line in f.readlines():
            e = line.rstrip().split()
            if e[0] not in anc_allele:
                anc_allele[e[0]] = dict()
            anc_allele[e[0]][int(e[2])] = e[3]

    if not anc_allele:
        raise Exception(f"No ancestral allele is found! Please check your data.")

    return anc_allele


def check_anc_allele(data, anc_allele, c):
    """
    Description:
        Helper function to check whether the REF or ALT allele is the ancestral allele.
        If the ALT allele is the ancestral allele, then the genotypes in this position will be flipped.
        If neither the REF nor ALT allele is the ancestral allele, then this position will be removed.
        If a position has no the ancestral allele information, the this position will be removed.

    Arguments:
        data dict: Genotype data for checking ancestral allele information.
        anc_allele dict: Ancestral allele information for checking.
        c str: Name of the chromosome.

    Returns:
        data dict: Genotype data after checking.
    """

    ref_allele, alt_allele = get_ref_alt_allele(
        data[c]["REF"], data[c]["ALT"], data[c]["POS"]
    )
    # Remove variants not in the ancestral allele file
    intersect_snps = np.intersect1d(list(ref_allele.keys()), list(anc_allele[c].keys()))
    # Remove variants that neither the ref allele nor the alt allele is the ancestral allele
    removed_snps = []
    # Flip variants that the alt allele is the ancestral allele
    flipped_snps = []

    for v in intersect_snps:
        if (anc_allele[c][v] != ref_allele[v]) and (anc_allele[c][v] != alt_allele[v]):
            removed_snps.append(v)
        elif anc_allele[c][v] == alt_allele[v]:
            flipped_snps.append(v)

    intersect_snps = np.in1d(data[c]["POS"], intersect_snps)
    data = filter_data(data, c, intersect_snps)

    if len(removed_snps) != 0:
        remained_snps = np.logical_not(np.in1d(data[c]["POS"], removed_snps))
        data = filter_data(data, c, remained_snps)

    is_flipped_snps = np.in1d(data[c]["POS"], flipped_snps)
    # Assume no missing data
    for i in range(len(data[c]["POS"])):
        if is_flipped_snps[i]:
            data[c]["GT"][i] = allel.GenotypeVector(abs(data[c]["GT"][i] - 1))

    return data


def create_windows(pos, chr_name, win_step, win_len):
    """
    Description:
        Creates sliding windows along the genome.

    Arguments:
        pos numpy.ndarray: Positions for the variants.
        chr_name str: Name of the chromosome.
        win_step int: Step size of sliding windows.
        win_len int: Length of sliding windows.

    Returns:
        windows list: List of sliding windows along the genome.
    """
    win_start = (pos[0] + win_step) // win_step * win_step - win_len
    if win_start < 0:
        win_start = 0
    last_pos = pos[-1]

    windows = []
    while last_pos > win_start:
        win_end = win_start + win_len
        windows.append((chr_name, win_start, win_end))
        win_start += win_step

    return windows
