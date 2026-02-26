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
from gaishi.models.unet.layers import UNetPlusPlus


class UNetPlusPlusRNN(nn.Module):
    """
    UNet++ backbone with neighbor-gap feature fusion via a bidirectional GRU.

    The input is split into two parts:
    - Convolutional channels: the first two channels are passed to a UNet++ backbone.
    - Neighbor-gap channels: the last two channels encode distances to neighboring variants
      (gap_to_prev, gap_to_next), and are fused with the UNet++ logits using a GRU along
      the width dimension.

    The fusion treats each row as a sequence:
    - Sequence length is ``W`` (width).
    - GRU batch dimension is ``B * H`` (batch times height).

    Parameters
    ----------
    num_classes : int
        Number of classes.
    polymorphisms : int
        Expected width ``W`` of the input. The final MLP maps a length-``W`` vector to length-``W``.
        Default: 128.
    hidden_dim : int
        Hidden size of the GRU. Default: 4.
    gru_layers : int
        Number of stacked GRU layers. Default: 1.
    bidirectional : bool
        If True, uses a bidirectional GRU. Default: True.

    Raises
    ------
    ValueError
        If the input does not have 4 channels.
    ValueError
        If the input width does not match ``polymorphisms``.
    """

    def __init__(
        self,
        num_classes: int,
        polymorphisms: int = 128,
        hidden_dim: int = 4,
        gru_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.polymorphisms = polymorphisms
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional

        self.backbone = UNetPlusPlus(num_classes, input_channels=2)

        self.gru = nn.GRU(
            input_size=num_classes + 2,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        directions = 2 if bidirectional else 1

        self.out_proj = nn.Linear(hidden_dim * directions, num_classes)

        self.mlp = nn.Sequential(
            nn.Linear(self.polymorphisms, 256),
            nn.LayerNorm((256,)),
            nn.Linear(256, self.polymorphisms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 4, H, W)``.
            - ``x[:, 0:2]``: convolutional channels for UNet++.
            - ``x[:, 2:4]``: neighbor-gap channels (gap_to_prev, gap_to_next).

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, H, W)``.
        """
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 input channels, got {x.shape[1]}.")
        if x.shape[-1] != self.polymorphisms:
            raise ValueError(
                f"Expected width W == polymorphisms == {self.polymorphisms}, got {x.shape[-1]}."
            )

        conv_input = x[:, 0:2]  # (B, 2, H, W)
        neighbor_gaps = x[:, 2:4]  # (B, 2, H, W)

        unet_logits = self.backbone(conv_input)  # (B, C, H, W)
        b, c, h, w = unet_logits.shape

        # (B, H, W, C)
        unet_feat = unet_logits.permute(0, 2, 3, 1)
        # (B, H, W, 2)
        gap_feat = neighbor_gaps.permute(0, 2, 3, 1)

        # sequence view: (B*H, W, C+2)
        gru_input = torch.cat([unet_feat, gap_feat], dim=-1).reshape(b * h, w, c + 2)

        gru_output, _ = self.gru(gru_input)  # (B*H, W, hidden*dir)

        # per-position class logits: (B*H, W, C)
        fused = self.out_proj(gru_output)

        # (B*H, C, W) -> (B*H*C, W) -> MLP -> (B*H*C, W) -> (B*H, C, W) -> (B*H, W, C)
        tmp = fused.permute(0, 2, 1).reshape(b * h * c, w)
        tmp = self.mlp(tmp)
        fused = tmp.view(b * h, c, w).permute(0, 2, 1)

        # back to (B, C, H, W)
        return fused.view(b, h, w, c).permute(0, 3, 1, 2)
