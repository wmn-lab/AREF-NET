# Derived from the local Ultralytics/YOLOMM implementation; AGPL-3.0.
"""Edge-Structure Recalibration Block; parameter names remain compatible."""

import torch
from torch import nn
from timm.models.layers import SqueezeExcite

from ultralytics.nn.modules.block import C3k
from ultralytics.nn.modules.conv import Conv


class StructureMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv(channels, channels, 7, 1, 3, g=channels, act=False)
        self.conv1 = Conv(channels, channels, 3, 1, 1, g=channels, act=False)
        self.dim = channels
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))


class _ConvBN(nn.Sequential):
    def __init__(self, c1, c2, bn_weight_init=1):
        super().__init__()
        self.add_module("c", nn.Conv2d(c1, c2, 1, bias=False))
        self.add_module("bn", nn.BatchNorm2d(c2))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class ChannelMixer(nn.Sequential):
    def __init__(self, c1, c2):
        super().__init__(
            _ConvBN(c1, 2 * c1),
            nn.GELU(),
            _ConvBN(2 * c1, c2, bn_weight_init=0),
        )


class ESRUnit(nn.Module):
    def __init__(self, inp, oup, use_se=True):
        super().__init__()
        self.identity = inp == oup
        self.token_mixer = nn.Sequential(
            StructureMixer(inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = ChannelMixer(inp, oup)

    def forward(self, x):
        structure = self.token_mixer(x)
        out = self.channel_mixer(structure)
        return out + structure if self.identity else out


class ESRInnerBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(ESRUnit(hidden, hidden) for _ in range(n)))


class ESRB(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            ESRInnerBlock(self.c, self.c, 2, shortcut, g)
            if c3k else ESRUnit(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        features = list(self.cv1(x).chunk(2, dim=1))
        features.extend(module(features[-1]) for module in self.m)
        return self.cv2(torch.cat(features, dim=1))
