"""
SolarTriageNet: a compact residual CNN for 40x24 single-channel thermal crops.

Design choices (see report, Section "Architecture"):
* Tiny input (960 pixels) -> 3 stages with 2x2 pooling (40x24 -> 20x12 -> 10x6 -> 5x3);
  deeper nets would pool the image away.
* Residual 3x3 blocks with BatchNorm: stable training from scratch on ~14k images.
* Global average pooling + dropout: few parameters, less overfitting than a large FC layer.
* One 12-way softmax head. The anomaly score used for triage is 1 - P(No-Anomaly),
  so the same network both flags modules and suggests the fault type.
"""
from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = (nn.Identity() if c_in == c_out else
                     nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class SolarTriageNet(nn.Module):
    def __init__(self, n_classes: int = 12, channels=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        layers, c_prev = [], 1
        for c in channels:
            layers += [ResBlock(c_prev, c), nn.MaxPool2d(2)]
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_prev, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
