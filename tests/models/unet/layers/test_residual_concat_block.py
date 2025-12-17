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
import torch

from gaishi.models.unet.layers import ResidualConcatBlock


@pytest.mark.parametrize(
    "batch,in_ch,out_ch,k,h,w",
    [
        (2, 3, 8, 3, 32, 32),
        (1, 2, 16, 5, 17, 19),
        (4, 1, 4, 3, 8, 13),
    ],
)
def test_residual_concat_block_output_shape(batch, in_ch, out_ch, k, h, w):
    block = ResidualConcatBlock(
        in_channels=in_ch,
        out_channels=out_ch,
        k=k,
    )
    x = torch.randn(batch, in_ch, h, w)

    y = block(x)

    assert y.shape == (batch, out_ch * 2, h, w)


def test_residual_concat_block_backward_pass():
    torch.manual_seed(0)

    block = ResidualConcatBlock(in_channels=3, out_channels=8, k=3)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)

    y = block(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    grads = [p.grad for p in block.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_residual_concat_block_is_deterministic_in_eval_mode():
    torch.manual_seed(0)

    block = ResidualConcatBlock(in_channels=3, out_channels=8, k=3).eval()
    x = torch.randn(2, 3, 16, 16)

    y1 = block(x)
    y2 = block(x)

    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("bad_k", [0, -3])
def test_residual_concat_block_rejects_invalid_kernel_size(bad_k):
    with pytest.raises(ValueError):
        _ = ResidualConcatBlock(in_channels=3, out_channels=8, k=bad_k)


def test_residual_concat_block_rejects_even_kernel_size():
    with pytest.raises(ValueError):
        _ = ResidualConcatBlock(in_channels=3, out_channels=8, k=4)
