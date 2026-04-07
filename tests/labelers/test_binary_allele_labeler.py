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


import allel
import pytest
import numpy as np
from gaishi.labelers import BinaryAlleleLabeler


@pytest.fixture
def set_vcf_positions(monkeypatch):
    def _set(pos):
        pos = np.asarray(pos, dtype=int)
        monkeypatch.setattr(allel, "read_vcf", lambda _vcf_path: {"variants/POS": pos})

    return _set


def test_run_phased_labels_and_rep(set_vcf_positions, tmp_path):
    set_vcf_positions([10, 20, 30, 40, 50])

    tgt_ind_file = tmp_path / "tgt.txt"
    tgt_ind_file.write_text("ind1\nind2\n")

    true_tract_file = tmp_path / "tracts.tsv"
    true_tract_file.write_text(
        "\n".join(
            [
                "chr1\t15\t45\tind1_1",  # => 20,30,40
                "chr1\t0\t25\tind2_2",  # => 10,20
                "chr1\t30\t40\tind1_2",  # => 30 (end exclusive)
                "chr1\t1\t100\tghost_1",  # ignored
            ]
        )
        + "\n"
    )

    labeler = BinaryAlleleLabeler(ploidy=2, is_phased=True, num_polymorphisms=5)
    out = labeler.run(
        tgt_ind_file=str(tgt_ind_file),
        vcf_file=str(tmp_path / "dummy.vcf"),
        true_tract_file=str(true_tract_file),
        rep=3,
    )

    assert isinstance(out, dict)
    assert set(out.keys()) == {"ind1_1", "ind1_2", "ind2_1", "ind2_2"}

    assert np.array_equal(out["ind1_1"]["Label"], np.array([0, 1, 1, 1, 0], dtype=int))
    assert np.array_equal(out["ind1_2"]["Label"], np.array([0, 0, 1, 0, 0], dtype=int))
    assert np.array_equal(out["ind2_1"]["Label"], np.array([0, 0, 0, 0, 0], dtype=int))
    assert np.array_equal(out["ind2_2"]["Label"], np.array([1, 1, 0, 0, 0], dtype=int))

    for s in out:
        assert out[s]["Sample"] == s
        assert out[s]["Replicate"] == 3


def test_run_unphased_labels(set_vcf_positions, tmp_path):
    set_vcf_positions([100, 200, 300])

    tgt_ind_file = tmp_path / "tgt.txt"
    tgt_ind_file.write_text("indA\nindB\n")

    true_tract_file = tmp_path / "tracts.tsv"
    true_tract_file.write_text(
        "\n".join(
            [
                "chr1\t150\t301\tindA",  # => 200,300
                "chr1\t0\t150\tindB",  # => 100
            ]
        )
        + "\n"
    )

    labeler = BinaryAlleleLabeler(ploidy=2, is_phased=False, num_polymorphisms=3)
    out = labeler.run(
        tgt_ind_file=str(tgt_ind_file),
        vcf_file=str(tmp_path / "dummy.vcf"),
        true_tract_file=str(true_tract_file),
    )

    assert set(out.keys()) == {"indA", "indB"}
    assert np.array_equal(out["indA"]["Label"], np.array([0, 1, 1], dtype=int))
    assert np.array_equal(out["indB"]["Label"], np.array([1, 0, 0], dtype=int))


def test_run_raises_if_too_few_polymorphisms(set_vcf_positions, tmp_path):
    set_vcf_positions([1, 2, 3])

    tgt_ind_file = tmp_path / "tgt.txt"
    tgt_ind_file.write_text("ind1\n")

    true_tract_file = tmp_path / "tracts.tsv"
    true_tract_file.write_text("chr1\t0\t10\tind1\n")

    labeler = BinaryAlleleLabeler(ploidy=2, is_phased=False, num_polymorphisms=4)

    with pytest.raises(ValueError, match=r"less than 4"):
        labeler.run(
            tgt_ind_file=str(tgt_ind_file),
            vcf_file=str(tmp_path / "dummy.vcf"),
            true_tract_file=str(true_tract_file),
        )


def test_run_missing_tgt_ind_file(set_vcf_positions, tmp_path):
    set_vcf_positions([1, 2, 3, 4])

    labeler = BinaryAlleleLabeler(ploidy=2, is_phased=False, num_polymorphisms=4)

    with pytest.raises(FileNotFoundError, match=r"tgt_ind_file .* not found"):
        labeler.run(
            tgt_ind_file=str(tmp_path / "nope.txt"),
            vcf_file=str(tmp_path / "dummy.vcf"),
            true_tract_file=str(tmp_path / "tracts.tsv"),
        )


def test_run_missing_true_tract_file(set_vcf_positions, tmp_path):
    set_vcf_positions([1, 2, 3, 4])

    tgt_ind_file = tmp_path / "tgt.txt"
    tgt_ind_file.write_text("ind1\n")

    labeler = BinaryAlleleLabeler(ploidy=2, is_phased=False, num_polymorphisms=4)

    with pytest.raises(FileNotFoundError, match=r"true_tract_file .* not found"):
        labeler.run(
            tgt_ind_file=str(tgt_ind_file),
            vcf_file=str(tmp_path / "dummy.vcf"),
            true_tract_file=str(tmp_path / "nope.tsv"),
        )
