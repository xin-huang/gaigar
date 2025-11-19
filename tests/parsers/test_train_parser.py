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


import pytest
import argparse
from gaishi.parsers.train_parser import add_train_parser


@pytest.fixture
def parser():
    # Initialize the argument parser with a subparser for the 'train' command
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="command")
    add_train_parser(subparsers)
    return main_parser


def test_add_score_parser(parser):
    # Simulate command-line arguments to parse
    args = parser.parse_args(
        [
            "train",
            "--demes",
            "tests/data/ArchIE_3D19.yaml",
            "--config",
            "tests/data/ArchIE.features.yaml",
            "--output",
            "output/results.tsv",
        ]
    )

    # Validate parsed arguments
    assert args.command == "train"
    assert args.demes == "tests/data/ArchIE_3D19.yaml"
    assert args.config == "tests/data/ArchIE.features.yaml"
    assert args.output == "output/results.tsv"
