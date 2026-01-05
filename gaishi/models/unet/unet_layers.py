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


import torch
import torch.nn as nn


class ResidualConcatBlock(nn.Module):
    """
    Convolutional block with residual accumulation and channel-wise concatenation.

    This block applies a stack of ``n_layers`` 2D convolutions. Each convolution is followed
    by instance normalization and spatial dropout. Starting from the second layer, a residual
    connection is applied within the block by adding the current layer output to the previous
    layer output. Finally, outputs from all layers are concatenated along the channel dimension
    and passed through an ELU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels of the first convolution.
    out_channels : int
        Number of output channels of each convolutional layer inside the block.
    k : int, default=3
        Convolution kernel size (square kernel ``k x k``).
    n_layers : int, default=2
        Number of convolutional layers inside the block.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(B, out_channels * n_layers, H, W)``.

    Raises
    ------
    ValueError
        If ``n_layers < 1``.
    ValueError
        If ``k < 1``.
    ValueError
        If ``k`` is even.

    Notes
    -----
    - The output channel dimension equals ``out_channels * n_layers`` due to concatenation.
    - Padding is set to keep spatial resolution unchanged for odd ``k``.
    - This block is often used in encoder-decoder architectures where the channel expansion
      from concatenation is expected downstream.

    Attributes
    ----------
    conv_layers : torch.nn.ModuleList
        List of convolutional layers.
    post_layers : torch.nn.ModuleList
        List of post-processing modules (InstanceNorm2d + Dropout2d) applied after each conv.
    act : torch.nn.Module
        Activation function applied after concatenation (ELU).
    """

    def __init__(
        self, in_channels: int, out_channels: int, k: int = 3, n_layers: int = 2
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if k % 2 == 0:
            raise ValueError(
                f"k must be odd to preserve spatial shape for residual addition, got {k}."
            )

        pad = (k + 1) // 2 - 1

        self.conv_layers = nn.ModuleList()
        self.post_layers = nn.ModuleList()

        current_in = in_channels
        for _ in range(n_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    current_in,
                    out_channels,
                    kernel_size=(k, k),
                    stride=(1, 1),
                    padding=(pad, pad),
                )
            )
            self.post_layers.append(
                nn.Sequential(
                    nn.InstanceNorm2d(out_channels),
                    nn.Dropout2d(0.1),
                )
            )
            current_in = out_channels

        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, out_channels * n_layers, H, W)``.
        """
        layer_outputs = [self.post_layers[0](self.conv_layers[0](x))]

        for i in range(1, len(self.post_layers)):
            prev = layer_outputs[-1]
            curr = self.post_layers[i](self.conv_layers[i](prev))
            layer_outputs.append(curr + prev)

        return self.act(torch.cat(layer_outputs, dim=1))
