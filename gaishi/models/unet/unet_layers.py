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


class UNetPlusPlus(nn.Module):
    """
    UNet++ segmentation network.

    This implementation follows the UNet++ design where decoder features at a given resolution
    are constructed through nested, dense skip connections. At each decoder node, feature maps
    from earlier nodes at the same resolution are concatenated with an upsampled feature map
    from the next deeper resolution, then processed by a convolutional block.

    The network uses ``ResidualConcatBlock`` at each grid node. With the default block setting
    (``n_layers=2``), each node produces the expected channel dimension for the predefined
    channel schedule.

    Parameters
    ----------
    num_classes : int
        Number of output classes. If ``num_classes == 1``, the forward pass returns a tensor
        of shape ``(B, H, W)``. Otherwise it returns a tensor of shape ``(B, C, H, W)``.
    input_channels : int, default=3
        Number of input channels.

    Returns
    -------
    torch.Tensor
        Model output logits. Shape is ``(B, H, W)`` if ``num_classes == 1``, otherwise
        ``(B, num_classes, H, W)``.

    Notes
    -----
    - Downsampling is performed by max pooling with stride 2.
    - Upsampling is performed by bilinear interpolation with scale factor 2.
    - The nested structure is expressed by nodes of the form ``x_{i,j}``, where ``i`` is
      the depth (downsampling level) and ``j`` is the decoder stage at the same resolution.
      The top row nodes ``x_{0,1}`` ... ``x_{0,4}`` progressively concatenate earlier top-row
      outputs, forming dense skip connections.

    Attributes
    ----------
    downsample : torch.nn.Module
        Max pooling layer used for downsampling.
    upsample : torch.nn.Module
        Bilinear upsampling layer used for upsampling.
    output_head : torch.nn.Conv2d
        Final ``1x1`` convolution mapping features to logits.
    """

    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()

        channel_dims = [32, 64, 128, 256, 512]

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder column (j=0)
        self.node00 = ResidualConcatBlock(input_channels, channel_dims[0] // 2)
        self.node10 = ResidualConcatBlock(channel_dims[0], channel_dims[1] // 2)
        self.node20 = ResidualConcatBlock(channel_dims[1], channel_dims[2] // 2)
        self.node30 = ResidualConcatBlock(channel_dims[2], channel_dims[3] // 2)
        self.node40 = ResidualConcatBlock(channel_dims[3], channel_dims[4] // 2)

        # Nested decoder nodes
        self.node01 = ResidualConcatBlock(
            channel_dims[0] + channel_dims[1], channel_dims[0] // 2
        )
        self.node11 = ResidualConcatBlock(
            channel_dims[1] + channel_dims[2], channel_dims[1] // 2
        )
        self.node21 = ResidualConcatBlock(
            channel_dims[2] + channel_dims[3], channel_dims[2] // 2
        )
        self.node31 = ResidualConcatBlock(
            channel_dims[3] + channel_dims[4], channel_dims[3] // 2
        )

        self.node02 = ResidualConcatBlock(
            channel_dims[0] * 2 + channel_dims[1], channel_dims[0] // 2
        )
        self.node12 = ResidualConcatBlock(
            channel_dims[1] * 2 + channel_dims[2], channel_dims[1] // 2
        )
        self.node22 = ResidualConcatBlock(
            channel_dims[2] * 2 + channel_dims[3], channel_dims[2] // 2
        )

        self.node03 = ResidualConcatBlock(
            channel_dims[0] * 3 + channel_dims[1], channel_dims[0] // 2
        )
        self.node13 = ResidualConcatBlock(
            channel_dims[1] * 3 + channel_dims[2], channel_dims[1] // 2
        )

        self.node04 = ResidualConcatBlock(
            channel_dims[0] * 4 + channel_dims[1], channel_dims[0] // 2
        )

        self.output_head = nn.Conv2d(channel_dims[0], num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Output logits. Shape is ``(B, H, W)`` if ``num_classes == 1``, otherwise
            ``(B, num_classes, H, W)``.
        """
        feat00 = self.node00(x)
        feat10 = self.node10(self.downsample(feat00))
        feat01 = self.node01(torch.cat([feat00, self.upsample(feat10)], dim=1))

        feat20 = self.node20(self.downsample(feat10))
        feat11 = self.node11(torch.cat([feat10, self.upsample(feat20)], dim=1))
        feat02 = self.node02(torch.cat([feat00, feat01, self.upsample(feat11)], dim=1))

        feat30 = self.node30(self.downsample(feat20))
        feat21 = self.node21(torch.cat([feat20, self.upsample(feat30)], dim=1))
        feat12 = self.node12(torch.cat([feat10, feat11, self.upsample(feat21)], dim=1))
        feat03 = self.node03(
            torch.cat([feat00, feat01, feat02, self.upsample(feat12)], dim=1)
        )

        feat40 = self.node40(self.downsample(feat30))
        feat31 = self.node31(torch.cat([feat30, self.upsample(feat40)], dim=1))
        feat22 = self.node22(torch.cat([feat20, feat21, self.upsample(feat31)], dim=1))
        feat13 = self.node13(
            torch.cat([feat10, feat11, feat12, self.upsample(feat22)], dim=1)
        )
        feat04 = self.node04(
            torch.cat([feat00, feat01, feat02, feat03, self.upsample(feat13)], dim=1)
        )

        logits = self.output_head(feat04)

        if self.num_classes == 1:
            return logits[:, 0]
        return logits
