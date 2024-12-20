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


import os
import pytest
import shutil
from gaia.utils.multiprocessing import mp_manager
from gaia.utils.simulators import MsprimeSimulator
from gaia.utils.generators import RandomNumberGenerator


@pytest.fixture
def sim_params():
    output_dir = "tests/test_MsprimeSimulator"
    return {
        "demo_model_file": "tests/data/ArchIE_3D19.yaml",
        "nref": 50,
        "ntgt": 50,
        "ref_id": "Ref",
        "tgt_id": "Tgt",
        "src_id": "Ghost",
        "ploidy": 2,
        "seq_len": 50000,
        "mut_rate": 1.25e-8,
        "rec_rate": 1e-8,
        "output_prefix": "test",
        "output_dir": str(output_dir),
        "is_phased": True,
    }


@pytest.fixture
def cleanup_output_dir(request, sim_params):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(sim_params["output_dir"], ignore_errors=True)


def compare_files(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        file1_content = f1.read()
        file2_content = f2.read()
        assert file1_content == file2_content, "Files do not match."


def test_MsprimeSimulator(sim_params, cleanup_output_dir):
    nprocess = 2
    nrep = 2

    simulator = MsprimeSimulator(**sim_params)
    generator = RandomNumberGenerator(nrep=nrep, seed=12345)

    mp_manager(job=simulator, data_generator=generator, nprocess=nprocess)
    for i in range(nrep):
        ref_ind_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.ref.ind.list",
        )
        tgt_ind_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.tgt.ind.list",
        )
        seed_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.seedmsprime",
        )
        bed_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.true.tracts.bed",
        )
        vcf_file = os.path.join(
            sim_params["output_dir"], f"{i}", f"{sim_params['output_prefix']}.{i}.vcf"
        )

        expected_dir = "tests/expected_results/simulators/MsprimeSimulator"

        expected_ref_ind_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.ref.ind.list"
        )
        expected_tgt_ind_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.tgt.ind.list"
        )
        expected_seed_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.seedmsprime"
        )
        expected_bed_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.true.tracts.bed"
        )
        expected_vcf_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.vcf"
        )

        compare_files(ref_ind_file, expected_ref_ind_file)
        compare_files(tgt_ind_file, expected_tgt_ind_file)
        compare_files(seed_file, expected_seed_file)
        compare_files(bed_file, expected_bed_file)
        compare_files(vcf_file, expected_vcf_file)
