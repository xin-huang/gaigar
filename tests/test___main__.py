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


import pytest, os, signal, argparse
from unittest.mock import patch
from gaigar.__main__ import _set_sigpipe_handler, _gaigar_cli_parser


@pytest.mark.skipif(os.name != 'posix', reason="Test only applicable on POSIX systems")
@patch('signal.signal')
def test_set_sigpipe_handler(mock_signal):
    _set_sigpipe_handler()
    mock_signal.assert_called_once_with(signal.SIGPIPE, signal.SIG_DFL)


def test_gaigar_cli_parser():
    parser = _gaigar_cli_parser()

    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(['lr']) 
    assert hasattr(args, 'subparsers'), "Parsed args do not have the 'subparsers' attribute"
    assert args.subparsers == 'lr', "The 'subparsers' attribute does not correctly capture the sub-command name"
