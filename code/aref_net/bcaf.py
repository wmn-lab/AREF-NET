# Adapted from the local CGAFusion implementation; AGPL-3.0.
# Original implementation identifies TIP2024 CGAFusion, arXiv:2301.04805.
"""Boundary-Context Adaptive Fusion with the original computation preserved."""

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.max(x, dim=1, keepdim=True)[0]
        return self.sa(torch.cat([average, maximum], dim=1))


class ChannelPrior(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(
            2 * dim, dim, 7, padding=3, padding_mode="reflect",
            groups=dim, bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prior):
        b, c, h, w = x.shape
        paired = torch.cat([x.unsqueeze(2), prior.unsqueeze(2)], dim=2)
        return self.sigmoid(self.pa2(paired.reshape(b, 2 * c, h, w)))


class BCAF(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=8):
        super().__init__()
        self.sa = SpatialPrior()
        self.ca = ChannelPrior(out_dim, reduction)
        self.pa = PixelGate(out_dim)
        self.conv = nn.Conv2d(out_dim, out_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv_adjust = nn.ModuleList(
            Conv(channels, out_dim, 1) if channels != out_dim else nn.Identity()
            for channels in in_dim
        )

    def forward(self, data):
        rgb = self.conv_adjust[0](data[0])
        edge = self.conv_adjust[1](data[1])
        initial = rgb + edge
        prior = self.sa(initial) + self.ca(initial)
        # Preserve both sigmoid operations in the existing trained implementation.
        gate = self.sigmoid(self.pa(initial, prior))
        return self.conv(initial + gate * rgb + (1 - gate) * edge)
