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


import argparse
import sys
from gaia.parsers.argument_validation import positive_int
from gaia.parsers.argument_validation import positive_number
from gaia.parsers.argument_validation import existed_file


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
        - output_prefix: Prefix for output files.
        - output_dir: Directory where output files will be saved.
        - seed: Random seed for reproducibility.
        - nfeature: Number of features to simulate.
        - num_polymorphisms: Number of polymorphisms in each genotype matrix to simulate.
        - num_upsamples: Number of samples after upsampling.
        - output_h5: Boolean flag to save output in HDF5 format.
        - is_sorted: Boolean flag to indicate whether to sort the genotype matrices.
        - only_intro: Boolean flag to simulate only introgressed fragments.
        - only_non_intro: Boolean flag to simulate only non-introgressed fragments.
        - force_balanced: Boolean flag to ensure a balanced distribution of introgressed and
                          non-introgressed classes in the training data.
        - keep_sim_data: Boolean flag to keep or discard simulation data.


    """

    import demes
    from gaia.simulate import unet_simulate

    demog = demes.load(args.demes)
    pops = [d.name for d in demog.demes]
    if args.ref_id not in pops:
        print(
            f"gaia unet simulate: error: argument --ref_id: Population {args.ref_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)
    if args.tgt_id not in pops:
        print(
            f"gaia unet simulate: error: argument --tgt_id: Population {args.tgt_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)
    if args.src_id not in pops:
        print(
            f"gaia unet simulate: error: argument --src_id: Population {args.src_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)

    unet_simulate(
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
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        seed=args.seed,
        nfeature=args.nfeature,
        nsite=args.nsite,
        nupsample=args.nupsample,
        output_h5=args.output_h5,
        is_sorted=args.is_sorted,
        only_intro=args.only_intro,
        only_non_intro=args.only_non_intro,
        force_balanced=args.force_balanced,
        keep_sim_data=args.keep_sim_data,
    )


def _run_preprocess():
    pass


def _run_training(args: argparse.Namespace) -> None:
    """ """

    from gaia.train import unet_train

    unet_train(
        training_data=args.training_data,
        model_dir=args.model_dir,
        trained_model_file=args.trained_model_file,
    )


def _run_inference():
    pass


def add_unet_parsers(subparsers: argparse.ArgumentParser) -> None:
    """
    Initializes and configures the command-line interface parser
    for using U-Net models.

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A command-line interface parser to be configured.

    """

    unet_parsers = subparsers.add_parser("unet", help="use U-Net models")
    unet_subparsers = unet_parsers.add_subparsers(dest="unet_subparsers")

    # Arguments for the simulate subcommand
    parser = unet_subparsers.add_parser("simulate", help="simulate data for training")
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
        "--output-h5",
        action="store_true",
        help="enable to store the output in HDF5 format; default: False",
        dest="output_h5",
    )
    parser.add_argument(
        "--nfeature",
        type=positive_int,
        default=1e6,
        help="number of feature vectors should be generated; default: 1e6",
    )
    parser.add_argument(
        "--nsite",
        type=positive_int,
        default=192,
        help="number of polymorphisms in each genotype matrix; default: 192",
    )
    parser.add_argument(
        "--nupsample",
        type=positive_int,
        default=None,
        help="number of samples after sampling; default: None",
    )
    parser.add_argument(
        "--sort-genotype-matrices",
        action="store_true",
        help="enable to sort genotype matrices; default: False",
        dest="is_sorted",
    )
    parser.add_argument(
        "--only-introgression-data",
        action="store_true",
        help="enable to simulate introgression data only; default: False",
        dest="only_intro",
    )
    parser.add_argument(
        "--only-non-introgression-data",
        action="store_true",
        help="enable to simulate non-introgression data only; defaul: False",
        dest="only_non_intro",
    )
    parser.add_argument(
        "--keep-simulated-data",
        action="store_true",
        help="enable to keep simulated data; default: False",
        dest="keep_sim_data",
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
    parser = unet_subparsers.add_parser(
        "preprocess", help="preprocess data for inference"
    )
    parser.set_defaults(runner=_run_preprocess)

    # Arguments for the train subcommand
    parser = unet_subparsers.add_parser("train", help="train a UNet model")
    parser.add_argument(
        "--training-data",
        type=existed_file,
        required=True,
        help="path to the HDF5 file containing genotype matrices for training",
        dest="training_data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="path to the directory where the trained model and logs will be stored",
        dest="model_dir",
    )
    parser.add_argument(
        "--trained-model-file",
        type=existed_file,
        default=None,
        help="path to the file storing the trained model for continuing training; default: None",
        dest="trained_model_file",
    )
    parser.set_defaults(runner=_run_training)

    # Arguments for the infer subcommand
    parser = unet_subparsers.add_parser(
        "infer", help="infer ghost introgressed fragments with a given UNet model"
    )
    parser.set_defaults(runner=_run_inference)
