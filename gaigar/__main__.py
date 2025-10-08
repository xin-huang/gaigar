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
from gaigar.parsers.lr_parsers import add_lr_parsers
from gaigar.parsers.unet_parsers import add_unet_parsers
from gaigar.parsers.eval_parsers import add_eval_parsers


def _set_sigpipe_handler() -> None:
    """
    Sets the signal handler for SIGPIPE signals on POSIX systems.

    """
    import os, signal

    if os.name == "posix":
        # Set signal handler for SIGPIPE to quietly kill the program.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def _gaigar_cli_parser() -> argparse.ArgumentParser:
    """
    Initializes and configures the command-line interface parser
    for gaigar.

    Returns
    -------
    top_parser : argparse.ArgumentParser
        A configured command-line interface parser.

    """
    top_parser = argparse.ArgumentParser()
    subparsers = top_parser.add_subparsers(dest="subparsers")
    add_lr_parsers(subparsers)
    add_unet_parsers(subparsers)
    add_eval_parsers(subparsers)

    return top_parser


def main(arg_list: list = None) -> None:
    """
    Main entry for gaigar.

    Parameters
    ----------
    arg_list : list, optional
        A list containing arguments for gaigar. Default: None.

    """
    _set_sigpipe_handler()
    parser = _gaigar_cli_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
