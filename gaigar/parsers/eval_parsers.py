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


import argparse
from gaigar.parsers.argument_validation import positive_int, existed_file


def _run_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluates the model performance with inferred tracts against the truth.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object obtained from argparse, containing the following evaluation parameters:
        - true_tracts : Path to the file containing the true genetic tracts.
        - inferred_tracts : Path to the file containing the inferred genetic tracts.
        - seq_len : The length of the sequence to evaluate.
        - sample_size : The number of samples used in the evaluation.
        - ploidy: Ploidy of the input data.
        - phased: Indicates if the input data is phased (boolean).
        - output : Path where the evaluation results will be saved.

    """
    from gaigar.evaluate import window_evaluate

    window_evaluate(
        true_tract_file=args.true_tracts,
        inferred_tract_file=args.inferred_tracts,
        seq_len=args.seq_len,
        sample_size=args.sample_size,
        ploidy=args.ploidy,
        is_phased=args.phased,
        output=args.output,
    )


def add_eval_parsers(subparsers: argparse.ArgumentParser) -> None:
    """
    Initializes and configures the command-line interface parser
    for evaluating model performance.

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A command-line interface parser to be configured.

    """
    parser = subparsers.add_parser("eval", help="evaluate model performance")

    parser.add_argument(
        "--true-tracts",
        type=existed_file,
        required=True,
        help="name of the BED file containing the true introgressed fragments",
        dest="true_tracts",
    )

    parser.add_argument(
        "--inferred-tracts",
        type=existed_file,
        required=True,
        help="name of the BED file containing the inferred introgressed fragments",
        dest="inferred_tracts",
    )

    parser.add_argument(
        "--seq-len",
        type=positive_int,
        required=True,
        help="length of the sequence to evaluate",
        dest="seq_len",
    )

    parser.add_argument(
        "--sample-size",
        type=positive_int,
        required=True,
        help="number of samples used in the evaluation",
        dest="sample_size",
    )

    parser.add_argument(
        "--ploidy",
        type=positive_int,
        required=True,
        help="ploidy of genomes used in the evaluation",
    )

    parser.add_argument(
        "--phased",
        action="store_true",
        help="enable to use phased genotypes; default: False",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="name of the file storing the performance measures",
    )

    parser.set_defaults(runner=_run_evaluate)
