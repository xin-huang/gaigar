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


import argparse, os, sys
from gaigar.parsers.argument_validation import (
    positive_int,
    positive_number,
    existed_file,
)


def _run_simulation(args: argparse.Namespace) -> None:
    """
    Executes a simulation process with specified parameters.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object obtained from argparse, containing simulation parameters:
        - demes: File path to the demographic model specification in YAML format.
        - replicate: Number of simulation replicates.
        - nref: Size of the reference population.
        - ntgt: Size of the target population.
        - ref_id: Identifier for the reference population.
        - tgt_id: Identifier for the target population.
        - src_id: Identifier for the source population.
        - ploidy: Ploidy of the organisms being simulated.
        - is_phased: Indicates if the simulated data should be phased.
        - seq_len: Length of the sequence to simulate.
        - mut_rate: Mutation rate to use in the simulation.
        - rec_rate: Recombination rate to use in the simulation.
        - nprocess: Number of processes to use for parallel simulations.
        - feature_config: Configuration file specifying features to be estimated.
        - intro_prop: Proportion that determines a fragment as introgressed.
        - non_intro_prop: Proportion that determines a fragment as non-introgressed.
        - output_prefix: Prefix for output files.
        - output_dir: Directory where output files will be saved.
        - seed: Random seed for reproducibility.
        - nfeature: Number of features to simulate.
        - is_shuffled: Boolean flag to shuffle the feature vectors for training or not.
        - forced_balanced: Boolean flag to ensure a balanced distribution of introgressed and
                           non-introgressed classes in the training data.
        - keep_sim_data: Boolean flag to keep or discard simulation data.

    Returns
    -------
    None.

    """
    import demes
    from gaigar.simulate import lr_simulate

    demog = demes.load(args.demes)
    pops = [d.name for d in demog.demes]
    if args.ref_id not in pops:
        print(
            f"gaigar lr simulate: error: argument --ref_id: Population {args.ref_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)
    if args.tgt_id not in pops:
        print(
            f"gaigar lr simulate: error: argument --tgt_id: Population {args.tgt_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)
    if args.src_id not in pops:
        print(
            f"gaigar lr simulate: error: argument --src_id: Population {args.src_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)

    lr_simulate(
        demo_model_file=args.demes,
        nrep=args.replicate,
        nref=args.nref,
        ntgt=args.ntgt,
        ref_id=args.ref_id,
        tgt_id=args.tgt_id,
        src_id=args.src_id,
        ploidy=args.ploidy,
        is_phased=args.phased,
        seq_len=args.seq_len,
        mut_rate=args.mut_rate,
        rec_rate=args.rec_rate,
        nprocess=args.nprocess,
        feature_config=args.feature_config,
        intro_prop=args.intro_prop,
        non_intro_prop=args.non_intro_prop,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        seed=args.seed,
        nfeature=args.nfeature,
        is_shuffled=args.is_shuffled,
        force_balanced=args.force_balanced,
        keep_sim_data=args.keep_sim_data,
    )


def _run_preprocess(args: argparse.Namespace) -> None:
    """
    Runs the preprocessing step using parameters provided in `args`.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object obtained from argparse. Expected attributes include:
        - vcf: Path to the VCF file to preprocess.
        - ref: Path to the reference individual file.
        - tgt: Path to the target individual file.
        - anc_allele: Path to the ancestral allele file.
        - nprocess: Number of processes to use.
        - features: Configuration file for features.
        - ploidy: Ploidy of the input data.
        - phased: Indicates if the input data is phased (boolean).
        - win_len: Window length for processing.
        - win_step: Step size for the window.
        - output_dir: Directory where output files will be saved.
        - output_prefix: Prefix for output file names.

    Returns
    -------
    None.

    """
    from gaigar.preprocess import lr_preprocess

    lr_preprocess(
        vcf_file=args.vcf,
        chr_name=args.chr_name,
        ref_ind_file=args.ref,
        tgt_ind_file=args.tgt,
        anc_allele_file=None,
        nprocess=args.nprocess,
        feature_config=args.feature_config,
        ploidy=args.ploidy,
        is_phased=args.phased,
        win_len=args.win_len,
        win_step=args.win_step,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )


def _run_training(args: argparse.Namespace) -> None:
    """
    Trains a model using provided training data and specified algorithm.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object obtained from argparse, containing training parameters:
        - training_data: Path to the training data file.
        - model_file: Path to save the trained model.
        - solver:
        - penalty:
        - max_iter:
        - seed: Random seed for reproducibility.
        - is_scaled:

    Returns
    -------
    None.

    """
    from gaigar.train import lr_train

    lr_train(
        training_data=args.training_data,
        model_file=args.model_file,
        solver=args.solver,
        penalty=args.penalty,
        max_iter=args.max_iter,
        seed=args.seed,
        is_scaled=args.is_scaled,
    )


def _run_inference(args: argparse.Namespace) -> None:
    """
    Performs inference using a trained model and input features.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object obtained from argparse, containing inference parameters:
        - inference_data: Path to the file containing input features for inference.
        - model_file: Path to the trained model file.
        - output_dir: Directory to save prediction results.
        - output_prefix: Prefix for prediction output files.
        - is_scaled:

    Returns
    -------
    None.

    """
    from gaigar.infer import lr_infer

    lr_infer(
        inference_data=args.inference_data,
        model_file=args.model_file,
        output_file=args.output_file,
        is_scaled=args.is_scaled,
    )


def add_lr_parsers(subparsers: argparse.ArgumentParser) -> None:
    """
    Initializes and configures the command-line interface parser
    for using logistic regression models.

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A command-line interface parser to be configured.

    Returns
    -------
    None.

    """
    lr_parsers = subparsers.add_parser("lr", help="use logistic regression models")
    lr_subparsers = lr_parsers.add_subparsers(dest="lr_subparsers")

    # Arguments for the simulate subcommand
    parser = lr_subparsers.add_parser("simulate", help="simulate data for training")
    parser.add_argument(
        "--demes",
        type=existed_file,
        required=True,
        help="demographic model in the DEMES format",
    )
    parser.add_argument(
        "--nref",
        type=positive_int,
        required=True,
        help="number of samples in the reference population",
    )
    parser.add_argument(
        "--ntgt",
        type=positive_int,
        required=True,
        help="number of samples in the target population",
    )
    parser.add_argument(
        "--ref-id",
        type=str,
        required=True,
        help="name of the reference population in the demographic model",
        dest="ref_id",
    )
    parser.add_argument(
        "--tgt-id",
        type=str,
        required=True,
        help="name of the target population in the demographic model",
        dest="tgt_id",
    )
    parser.add_argument(
        "--src-id",
        type=str,
        required=True,
        help="name of the source population in the demographic model",
        dest="src_id",
    )
    parser.add_argument(
        "--seq-len",
        type=positive_int,
        required=True,
        help="length of the simulated genomes",
        dest="seq_len",
    )
    parser.add_argument(
        "--ploidy",
        type=positive_int,
        default=2,
        help="ploidy of the simulated genomes; default: 2",
    )
    parser.add_argument(
        "--phased",
        action="store_true",
        help="enable to use phased genotypes; default: False",
    )
    parser.add_argument(
        "--mut-rate",
        type=positive_number,
        default=1e-8,
        help="mutation rate per base pair per generation for the simulation; default: 1e-8",
        dest="mut_rate",
    )
    parser.add_argument(
        "--rec-rate",
        type=positive_number,
        default=1e-8,
        help="recombination rate per base pair per generation for the simulation; default: 1e-8",
        dest="rec_rate",
    )
    parser.add_argument(
        "--replicate",
        type=positive_int,
        default=1,
        help="number of replications per batch for the simulation, which will continue until the number of feature vectors specified by the --nfeature argument is obtained; default: 1",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="prefix of the output file name",
        dest="output_prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory of the output files",
        dest="output_dir",
    )
    parser.add_argument(
        "--feature-config",
        type=existed_file,
        required=True,
        help="name of the YAML file specifying what features should be used",
        dest="feature_config",
    )
    parser.add_argument(
        "--nfeature",
        type=positive_int,
        default=1e6,
        help="number of feature vectors should be generated; default: 1e6",
    )
    parser.add_argument(
        "--introgressed-prop",
        type=positive_number,
        default=0.7,
        help="proportion that determines a fragment as introgressed; default: 0.7",
        dest="intro_prop",
    )
    parser.add_argument(
        "--non-introgressed-prop",
        type=positive_number,
        default=0.3,
        help="proportion that determinse a fragment as non-introgressed; default: 0.3",
        dest="non_intro_prop",
    )
    parser.add_argument(
        "--keep-simulated-data",
        action="store_true",
        help="enable to keep simulated data; default: False",
        dest="keep_sim_data",
    )
    parser.add_argument(
        "--shuffle-data",
        action="store_true",
        help="enable to shuffle the feature vectors for training; default: False",
        dest="is_shuffled",
    )
    parser.add_argument(
        "--force-balanced",
        action="store_true",
        help="enable to ensure a balanced distribution of introgressed and non-introgressed classes in the feature vectors for training; default: False",
        dest="force_balanced",
    )
    parser.add_argument(
        "--nprocess",
        type=positive_int,
        default=1,
        help="number of processes for the simulation; default: 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for the simulation; default: None",
    )
    parser.set_defaults(runner=_run_simulation)

    # Arguments for the preprocess subcommand
    parser = lr_subparsers.add_parser(
        "preprocess", help="preprocess data for inference"
    )
    parser.add_argument(
        "--vcf",
        type=existed_file,
        required=True,
        help="name of the VCF file containing genotypes from samples",
    )
    parser.add_argument(
        "--chr-name",
        type=str,
        required=True,
        help="name of the chromosome in the VCF file for being processed; default: chr_name",
    )
    parser.add_argument(
        "--ref",
        type=existed_file,
        required=True,
        help="name of the file containing population information for samples without introgression",
    )
    parser.add_argument(
        "--tgt",
        type=existed_file,
        required=True,
        help="name of the file containing population information for samples for detecting ghost introgressed fragments",
    )
    parser.add_argument(
        "--feature-config",
        type=existed_file,
        required=True,
        help="name of the YAML file specifying what features should be used",
        dest="feature_config",
    )
    parser.add_argument(
        "--phased",
        action="store_true",
        help="enable to use phased genotypes; default: False",
    )
    parser.add_argument(
        "--ploidy", type=positive_int, default=2, help="ploidy of genomes; default: 2"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="prefix of the output files",
        dest="output_prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory storing the output files",
        dest="output_dir",
    )
    parser.add_argument(
        "--win-len",
        type=positive_int,
        default=50000,
        help="length of the window to calculate statistics as input features; default: 50000",
        dest="win_len",
    )
    parser.add_argument(
        "--win-step",
        type=positive_int,
        default=10000,
        help="step size for moving windows along genomes when calculating statistics; default: 10000",
        dest="win_step",
    )
    parser.add_argument(
        "--nprocess",
        type=positive_int,
        default=1,
        help="number of processes for the training; default: 1",
    )
    parser.set_defaults(runner=_run_preprocess)

    # Arguments for the train subcommand
    parser = lr_subparsers.add_parser("train", help="train a logistic regression model")
    parser.add_argument(
        "--training-data",
        type=existed_file,
        required=True,
        help="name of the file containing features to training",
        dest="training_data",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="file storing the trained model",
        dest="model_file",
    )
    parser.add_argument(
        "--solver", type=str, default="newton-cg", help="default: newton-cg"
    )
    parser.add_argument("--penalty", type=str, default=None, help="default: None")
    parser.add_argument(
        "--max-iteration",
        type=positive_int,
        default=10000,
        help="default: 10000",
        dest="max_iter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for the training algorithm; default: None",
    )
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="enable to use scaled training data; default: False",
        dest="is_scaled",
    )
    parser.set_defaults(runner=_run_training)

    # Arguments for the infer subcommand
    parser = lr_subparsers.add_parser(
        "infer",
        help="infer ghost introgressed fragments with a given logistic regression model",
    )
    parser.add_argument(
        "--inference-data",
        type=existed_file,
        required=True,
        help="name of the file storing features for inference",
        dest="inference_data",
    )
    parser.add_argument(
        "--model-file",
        type=existed_file,
        required=True,
        help="name of the file storing the trained model",
        dest="model_file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="name of the output file storing the predictions",
        dest="output_file",
    )
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="enable to use scaled inference data; default: False",
        dest="is_scaled",
    )
    parser.set_defaults(runner=_run_inference)
