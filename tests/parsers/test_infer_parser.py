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
from gaishi.parsers.infer_parser import add_infer_parser


@pytest.fixture
def parser():
    # Initialize the argument parser with a subparser for the 'infer' command
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="command")
    add_infer_parser(subparsers)
    return main_parser


def test_add_score_parser(parser):
    # Simulate command-line arguments to parse
    args = parser.parse_args(
        [
            "infer",
            "--model",
            "tests/data/test.lr.model",
            "--config",
            "tests/data/ArchIE.features.yaml",
            "--output",
            "output/results.tsv",
        ]
    )

    # Validate parsed arguments
    assert args.command == "infer"
    assert args.model == "tests/data/test.lr.model"
    assert args.config == "tests/data/ArchIE.features.yaml"
    assert args.output == "output/results.tsv"
