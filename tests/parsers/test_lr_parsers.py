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


import pytest, argparse
from gaigar.parsers.lr_parsers import add_lr_parsers


@pytest.fixture
def lr_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparsers")
    add_lr_parsers(subparsers)
    return parser


@pytest.mark.skip(reason="Test later")
def test_lr_simulate_subparser(lr_parser):
    simulate_args = [
        "lr",
        "simulate",
        "--demes",
        "demo_model",
        "--nref",
        "100",
        "--ntgt",
        "100",
        "--ref-id",
        "REF-ID",
        "--tgt-id",
        "TGT-ID",
        "--src-id",
        "SRC-ID",
        "--seq-len",
        "100000",
        "--ploidy",
        "2",
        "--mut-rate",
        "1e-8",
        "--rec-rate",
        "1e-8",
        "--feature-config",
        "feature-config",
        "--replicate",
        "1",
        "--output-prefix",
        "out",
        "--output-dir",
        "./output",
    ]
    args = lr_parser.parse_args(simulate_args)

    assert args.subparsers == "lr"
    assert args.lr_subparsers == "simulate"
    assert args.demes == "demo_model"
    assert args.nref == 100
    assert args.ntgt == 100
    assert args.ref_id == "REF-ID"
    assert args.tgt_id == "TGT-ID"
    assert args.src_id == "SRC-ID"
    assert args.seq_len == 100000
    assert args.ploidy == 2
    assert args.mut_rate == 1e-8
    assert args.rec_rate == 1e-8
    assert args.replicate == 1
    assert args.feature_config == "feature-config"
    assert args.output_prefix == "out"
    assert args.output_dir == "./output"


@pytest.mark.skip(reason="Test later")
def test_lr_preprocess_subparser(lr_parser):
    preprocess_args = [
        "lr",
        "preprocess",
        "--vcf",
        "VCF",
        "--ref",
        "REF",
        "--tgt",
        "TGT",
        "--features",
        "features",
        "--phased",
        "--ploidy",
        "2",
        "--output-prefix",
        "out",
        "--output-dir",
        "./output",
        "--win-len",
        "50000",
        "--win-step",
        "10000",
        "--nprocess",
        "2",
    ]
    args = lr_parser.parse_args(preprocess_args)

    assert args.subparsers == "lr"
    assert args.lr_subparsers == "preprocess"
    assert args.vcf == "VCF"
    assert args.ref == "REF"
    assert args.tgt == "TGT"
    assert args.anc_allele == "anc-allele"
    assert args.features == "features"
    assert args.phased
    assert args.ploidy == 2
    assert args.output_prefix == "out"
    assert args.output_dir == "./output"
    assert args.win_len == 50000
    assert args.win_step == 10000
    assert args.nprocess == 2


@pytest.mark.skip(reason="Test later")
def test_lr_train_subparser(lr_parser):
    train_args = [
        "lr",
        "train",
        "--training-data",
        "training-data",
        "--model-file",
        "model-file",
        "--seed",
        "12345",
    ]
    args = lr_parser.parse_args(train_args)

    assert args.subparsers == "lr"
    assert args.lr_subparsers == "train"
    assert args.training_data == "training-data"
    assert args.model_file == "model-file"
    assert args.seed == 12345


@pytest.mark.skip(reason="Test later")
def test_lr_infer_subparser(lr_parser):
    test_args = [
        "lr",
        "infer",
        "--features",
        "features",
        "--model-file",
        "model-file",
        "--output-prefix",
        "out",
        "--output-dir",
        "./output",
    ]
    args = lr_parser.parse_args(test_args)

    assert args.subparsers == "lr"
    assert args.lr_subparsers == "infer"
    assert args.model_file == "model-file"
    assert args.output_prefix == "out"
    assert args.output_dir == "./output"
