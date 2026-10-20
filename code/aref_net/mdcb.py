# Derived from the local Ultralytics/YOLOMM implementation; AGPL-3.0.
"""Morphology-aware Dilated Context Block; parameter names remain compatible."""

import torch
from torch import nn

from ultralytics.nn.modules.block import C3k
from ultralytics.nn.modules.conv import Conv


class MDCUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_3x3 = Conv(dim, dim // 2, 3)
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)
        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        local = self.conv_3x3(x)
        features = [
            self.conv_3x3_d1(local),
            self.conv_3x3_d3(local),
            self.conv_3x3_d5(local),
        ]
        return self.conv_1x1(torch.cat(features, dim=1)) + x


class MDCInnerBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(MDCUnit(hidden) for _ in range(n)))


class MDCB(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            MDCInnerBlock(self.c, self.c, 2, shortcut, g)
            if c3k else MDCUnit(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        features = list(self.cv1(x).chunk(2, dim=1))
        features.extend(module(features[-1]) for module in self.m)
        return self.cv2(torch.cat(features, dim=1))
