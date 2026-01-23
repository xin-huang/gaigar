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
from gaishi.models.unet.layers import UNetPlusPlus


@pytest.mark.parametrize(
    "batch,input_channels,h,w",
    [
        (2, 3, 64, 64),
        (1, 3, 128, 96),
        (4, 1, 32, 160),
    ],
)
def test_unetplusplus_output_shape_binary(batch, input_channels, h, w):
    model = UNetPlusPlus(num_classes=1, input_channels=input_channels)
    x = torch.randn(batch, input_channels, h, w)

    y = model(x)

    assert y.shape == (batch, h, w)


@pytest.mark.parametrize(
    "batch,input_channels,num_classes,h,w",
    [
        (2, 3, 2, 64, 64),
        (1, 3, 4, 128, 96),
        (3, 1, 3, 32, 160),
    ],
)
def test_unetplusplus_output_shape_multiclass(batch, input_channels, num_classes, h, w):
    model = UNetPlusPlus(num_classes=num_classes, input_channels=input_channels)
    x = torch.randn(batch, input_channels, h, w)

    y = model(x)

    assert y.shape == (batch, num_classes, h, w)


def test_unetplusplus_backward_pass_binary():
    torch.manual_seed(0)

    model = UNetPlusPlus(num_classes=1, input_channels=3)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)

    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_unetplusplus_backward_pass_multiclass():
    torch.manual_seed(0)

    model = UNetPlusPlus(num_classes=3, input_channels=3)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)

    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("h,w", [(65, 64), (64, 65), (63, 63), (30, 64)])
def test_unetplusplus_raises_on_invalid_spatial_size(h, w):
    """
    UNet++ with 4 pooling operations requires H and W to be divisible by 2^4=16
    to keep tensor shapes compatible for concatenation.
    """
    model = UNetPlusPlus(num_classes=1, input_channels=3)
    x = torch.randn(1, 3, h, w)

    with pytest.raises(RuntimeError):
        _ = model(x)
