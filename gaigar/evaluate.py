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


import pandas as pd
import pyranges as pr
import numpy as np
from gaigar.stats.metrics import cal_pr


def window_evaluate(
    true_tract_file: str, 
    inferred_tract_file: str,
    seq_len: int, 
    sample_size: int,
    ploidy: int,
    is_phased: bool,
    output: str
) -> None:
    """
    Evaluates model performance with precision and recall based on genomic windows by comparing
    true and inferred introgressed tracts.

    Parameters
    ----------
    true_tract_file : str
        File path for the BED file containing true introgressed fragments.
    inferred_tract_file : str
        File path for the BED file containing inferred introgressed fragments.
    seq_len : int
        The total length of the sequence in base pairs.
    sample_size : int
        The number of samples analyzed in the study.
    ploidy : int
        The ploidy of the genomes being analyzed.
    is_phased : bool
        Indicates whether the genetic data is phased.
    output : str
        File path for the output file storing the model performance metrics in a tab-separated format.

    """
    try:
        true_tracts = pd.read_csv(
            true_tract_file, sep="\t", header=None, names=['Chromosome', 'Start', 'End', 'Sample']
        )
    except pd.errors.EmptyDataError:
        true_tracts_samples = []
    else:
        true_tracts_samples = true_tracts['Sample'].unique()
        true_tracts = pr.PyRanges(true_tracts).merge(by='Sample')

    try:
        inferred_tracts = pd.read_csv(
            inferred_tract_file, sep="\t", header=None, names=['Chromosome', 'Start', 'End', 'Sample']
        )
        inferred_tracts['End'] = inferred_tracts['End'].clip(upper=seq_len)
    except pd.errors.EmptyDataError:
        inferred_tracts_samples = []
    else:
        inferred_tracts_samples = inferred_tracts['Sample'].unique()
        inferred_tracts = pr.PyRanges(inferred_tracts).merge(by='Sample')

    if is_phased:
        sample_size = sample_size * ploidy

    res = pd.DataFrame(
        columns=[
            'Sample', 
            'Sequence_length',
            'True_tracts_length', 
            'Inferred_tracts_length', 
            'True_positives_length', 
            'False_positives_length',
            'True_negatives_length',
            'False_negatives_length',
        ]
    )

    #sum_ntrue_tracts = 0
    #sum_ninferred_tracts = 0
    #sum_ntrue_positives = 0

    for s in np.intersect1d(true_tracts_samples, inferred_tracts_samples):
        ind_true_tracts = true_tracts[true_tracts.Sample == s]
        ind_inferred_tracts = inferred_tracts[inferred_tracts.Sample == s]
        ind_overlaps = ind_true_tracts.intersect(ind_inferred_tracts)

        ntrue_tracts = np.sum([x[1].End.astype('int') - x[1].Start.astype('int') for x in ind_true_tracts])
        ninferred_tracts = np.sum([x[1].End.astype('int') - x[1].Start.astype('int') for x in ind_inferred_tracts])
        ntrue_positives = np.sum([x[1].End.astype('int') - x[1].Start.astype('int') for x in ind_overlaps])
        nfalse_positives = ninferred_tracts - ntrue_positives
        nfalse_negatives = ntrue_tracts - ntrue_positives
        ntrue_negatives = seq_len - ntrue_tracts - nfalse_positives
        #precision, recall = cal_pr(ntrue_tracts, ninferred_tracts, ntrue_positives)
        res.loc[len(res.index)] = [
            s, seq_len, 
            ntrue_tracts, ninferred_tracts, 
            ntrue_positives, nfalse_positives,
            ntrue_negatives, nfalse_negatives,
        ]

        #sum_ntrue_tracts += ntrue_tracts
        #sum_ninferred_tracts += ninferred_tracts
        #sum_ntrue_positives += ntrue_positives

    for s in np.setdiff1d(true_tracts_samples, inferred_tracts_samples):
        # ninferred_tracts = 0
        ind_true_tracts = true_tracts[true_tracts.Sample == s]
        ntrue_tracts = np.sum([x[1].End.astype('int') - x[1].Start.astype('int') for x in ind_true_tracts])
        res.loc[len(res.index)] = [
            s, seq_len,
            ntrue_tracts, 0, 
            0, 0,
            seq_len, ntrue_tracts,
        ]

        #sum_ntrue_tracts += ntrue_tracts

    for s in np.setdiff1d(inferred_tracts_samples, true_tracts_samples):
        # ntrue_tracts = 0
        ind_inferred_tracts = inferred_tracts[inferred_tracts.Sample == s]
        ninferred_tracts = np.sum([x[1].End.astype('int') - x[1].Start.astype('int') for x in ind_inferred_tracts])
        res.loc[len(res.index)] = [
            s, seq_len,
            0, ninferred_tracts, 
            0, ninferred_tracts,
            seq_len-ninferred_tracts, 0,
        ]

        #sum_ninferred_tracts += ninferred_tracts

    #sum_nfalse_positives = sum_ninferred_tracts - sum_ntrue_positives
    #sum_nfalse_negatives = sum_ntrue_tracts - sum_ntrue_positives
    #sum_ntrue_negatives = seq_len * sample_size - sum_ntrue_tracts - sum_nfalse_positives
    
    res = res.sort_values(by=['Sample'])

    numeric_columns = res.select_dtypes(include=[float, int]).columns
    column_means = res[numeric_columns].mean()
    mean_df = pd.DataFrame([column_means], index=['Mean'])
    mean_df.insert(0, 'Sample', 'Average')
    res = pd.concat([res, mean_df])

    #total_precision, total_recall = cal_pr(sum_ntrue_tracts, sum_ninferred_tracts, sum_ntrue_positives)
    #total_len = seq_len*sample_size
    #res.loc[len(res.index)] = [
    #    'Total', 
    #    total_len,
    #    sum_ntrue_tracts, 
    #    sum_ninferred_tracts, 
    #    sum_ntrue_positives/sum_ntrue_tracts*100 if sum_ntrue_tracts != 0 else 0,
    #    sum_nfalse_positives/(total_len-sum_ntrue_tracts)*100,
    #    sum_ntrue_negatives/(total_len-sum_ntrue_tracts)*100,
    #    sum_nfalse_negatives/sum_ntrue_tracts*100 if sum_ntrue_tracts != 0 else 0,
    #]

    res.fillna('NaN').to_csv(output, sep="\t", index=False)
