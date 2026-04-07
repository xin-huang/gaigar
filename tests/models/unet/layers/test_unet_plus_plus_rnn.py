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
from gaishi.models.unet.layers import UNetPlusPlusRNN


@pytest.mark.parametrize(
    "batch,h,w,polymorphisms,num_classes",
    [
        (2, 32, 128, 128, 3),
        (1, 64, 128, 128, 2),
        (3, 16, 256, 256, 5),
    ],
)
def test_unetplusplus_rnn_output_shape(batch, h, w, polymorphisms, num_classes):
    model = UNetPlusPlusRNN(
        num_classes=num_classes,
        polymorphisms=polymorphisms,
        hidden_dim=4,
        gru_layers=1,
        bidirectional=True,
    )
    x = torch.randn(batch, 4, h, w)

    y = model(x)

    assert y.shape == (batch, num_classes, h, w)


def test_unetplusplus_rnn_backward_pass():
    torch.manual_seed(0)

    model = UNetPlusPlusRNN(
        num_classes=4,
        polymorphisms=128,
        hidden_dim=4,
        gru_layers=1,
        bidirectional=True,
    )
    x = torch.randn(2, 4, 32, 128, requires_grad=True)

    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_unetplusplus_rnn_runs_with_bidirectional_toggle(bidirectional):
    model = UNetPlusPlusRNN(
        num_classes=3,
        polymorphisms=128,
        hidden_dim=4,
        gru_layers=2,
        bidirectional=bidirectional,
    )
    x = torch.randn(1, 4, 32, 128)

    y = model(x)

    assert y.shape == (1, 3, 32, 128)


@pytest.mark.parametrize("bad_channels", [1, 2, 3, 5])
def test_unetplusplus_rnn_rejects_invalid_channel_count(bad_channels):
    model = UNetPlusPlusRNN(num_classes=3, polymorphisms=128)
    x = torch.randn(1, bad_channels, 32, 128)

    with pytest.raises(ValueError):
        _ = model(x)


def test_unetplusplus_rnn_rejects_mismatched_width():
    model = UNetPlusPlusRNN(num_classes=3, polymorphisms=128)
    x = torch.randn(1, 4, 32, 127)

    with pytest.raises(ValueError):
        _ = model(x)


def test_unetplusplus_rnn_is_deterministic_in_eval_mode():
    torch.manual_seed(0)

    model = UNetPlusPlusRNN(num_classes=3, polymorphisms=128).eval()
    x = torch.randn(1, 4, 32, 128)

    y1 = model(x)
    y2 = model(x)

    assert torch.allclose(y1, y2)
