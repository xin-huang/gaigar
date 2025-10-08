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


def _run_simulation():
    pass


def _run_preprocess():
    pass


def _run_training(args: argparse.Namespace) -> None:
    pass


#    """
#    """
#    from gaigar.train import unet_train
#    unet_train(
#        training_data=args.training_data,
#        model_dir=args.model_dir,
#        trained_model_file=args.trained_model_file,
#    )


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

    Returns
    -------
    None.

    """
    unet_parsers = subparsers.add_parser("unet", help="use U-Net models")
    unet_subparsers = unet_parsers.add_subparsers(dest="unet_subparsers")

    # Arguments for the simulate subcommand
    parser = unet_subparsers.add_parser("simulate", help="simulate data for training")
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
