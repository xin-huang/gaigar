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


import os
import numpy as np
import pandas as pd
from typing import Any
from gaishi.labelers import GenericLabeler


class BinaryWindowLabeler(GenericLabeler):
    """
    A labeler for binary classification of genomic windows based on introgression proportions.

    This labeler analyzes genomic tracts to label windows as introgressed or not, based on a specified
    proportion threshold. It operates on genomic data for a specified target population, using phased
    or unphased tracts and considering specified ploidy.

    """

    def __init__(
        self,
        ploidy: int,
        is_phased: bool,
        win_len: int,
        intro_prop: float,
        non_intro_prop: float,
    ):
        """
        Initializes a new instance of BinaryWindowLabeler with specific parameters.

        Parameters
        ----------
        ploidy : int
            Ploidy of the samples, typically 2 for diploid organisms.
        is_phased : bool
            Indicates whether the data should be considered as phased.
        win_len : int
            Length of the window for analysis. Must be greater than 0.
        intro_prop : float
            Proportion threshold for labeling a window as introgressed. Must be between 0 and 1.
        non_intro_prop : float
            Proportion threshold for labeling a window as non-introgressed. Must be between 0 and 1.

        Raises
        ------
        ValueError
            If `win_len` is not greater than 0, or `intro_prop` is not within the inclusive range [0, 1],
            or `not_intro_prop` is not within the inclusive range [0, 1],
            or the sum of `intro_prop` and `not_intro_prop` is not within the inclusive range [0, 1].

        """
        if win_len <= 0:
            raise ValueError(f"win_len {win_len} must be greater than 0.")

        if not 0 <= intro_prop <= 1:
            raise ValueError(
                f"intro_prop {intro_prop} must be within the inclusive range [0, 1]."
            )

        if not 0 <= non_intro_prop <= 1:
            raise ValueError(
                f"not_intro_prop {not_intro_prop} must be within the inclusive range [0, 1]."
            )

        if not 0 <= (intro_prop + non_intro_prop) <= 1:
            raise ValueError(
                f"the sum of intro_prop {intro_prop} and non_intro_prop {non_intro_prop} must be within the inclusive range [0, 1]."
            )

        super().__init__(
            ploidy=ploidy,
            is_phased=is_phased,
        )

        self.win_len = win_len
        self.intro_prop = intro_prop
        self.non_intro_prop = non_intro_prop

    def run(
        self, tgt_ind_file: str, true_tract_file: str, rep: int = None
    ) -> list[dict[str, Any]]:
        """
        Executes the labeling process by analyzing genomic tracts and labeling
        windows as introgressed or not, based on a specified proportion threshold.

        This method operates on genomic data for a specified target population,
        using either phased or unphased tracts and considering the specified ploidy.
        It labels each window with 1 if the proportion of introgressed tracts within
        the window exceeds the introgression threshold (`intro_prop`), with 0 if
        below the non-introgression threshold (`not_intro_prop`), and with -1 for
        windows that do not meet either condition.

        Parameters
        ----------
        tgt_ind_file : str
            The path to the file containing target individual identifiers. If not
            specified, `tgt_ind_file` from the initialization will be used.
        true_tract_file : str
            The path to the file containing true tracts. This file provides the
            information on introgressed genomic regions. If not specified, `true_tract_file`
            from the initialization will be used.
        rep : int or None, optional
            Used to specify the replicate number for the simulated data.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries containing the label for each samples.

        Raises
        ------
        FileNotFoundError
            If `tgt_ind_file` or `true_tract_file` is not found.

        """
        label_df = pd.DataFrame(columns=["Chromosome", "Start", "End", "Sample"])

        try:
            with open(tgt_ind_file, "r") as f:
                if self.is_phased is True:
                    for line in f:
                        sample = line.rstrip()
                        for p in range(self.ploidy):
                            label_df.loc[len(label_df.index)] = [
                                1,
                                0,
                                self.win_len,
                                f"{sample}_{p+1}",
                            ]
                else:
                    for line in f:
                        sample = line.rstrip()
                        label_df.loc[len(label_df.index)] = [1, 0, self.win_len, sample]
        except FileNotFoundError:
            raise FileNotFoundError(f"tgt_ind_file {tgt_ind_file} not found.")

        try:
            true_tract_df = pd.read_csv(
                true_tract_file,
                sep="\t",
                header=None,
                names=["Chromosome", "Start", "End", "Sample"],
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"true_tract_file {true_tract_file} not found.")
        except pd.errors.EmptyDataError:
            label_df["Label"] = 0.0
        else:
            true_tract_df["Len"] = true_tract_df["End"] - true_tract_df["Start"]
            true_tract_df = (
                true_tract_df.groupby(by=["Sample"])["Len"].sum().reset_index()
            )
            true_tract_df["Prop"] = true_tract_df["Len"] / self.win_len
            conditions = [
                true_tract_df["Prop"] >= self.intro_prop,
                true_tract_df["Prop"] <= self.non_intro_prop,
            ]
            choices = [1, 0]
            true_tract_df["Label"] = np.select(conditions, choices, default=-1)
            label_df = label_df.merge(
                true_tract_df[["Sample", "Label"]], on="Sample", how="left"
            ).fillna(0)

            if rep is not None:
                label_df["Replicate"] = rep

        label_df["Label"] = label_df["Label"].astype("int8")

        labels = label_df.to_dict(orient="records")

        return labels
