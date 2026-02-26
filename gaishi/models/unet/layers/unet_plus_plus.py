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
from gaishi.models.unet.layers import ResidualConcatBlock


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
        Number of output classes.
    input_channels : int,
        Number of input channels.

    Returns
    -------
    torch.Tensor
        Model output logits ``(B, num_classes, H, W)``.

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

    def __init__(self, num_classes: int, input_channels: int):
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
            Output logits ``(B, num_classes, H, W)``.
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

        return logits
