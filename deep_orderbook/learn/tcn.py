# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with dilated convolutions for temporal data."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (
            (kernel_size[0] - 1) * dilation[0],  # Time dimension pad for causality
            (kernel_size[1] - 1) * dilation[1] // 2,  # Symmetric pad for Price
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, dilation=dilation
        )
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()
        self.padding = padding

    def forward(self, x):
        residual = x
        # Padding: Left, Right, Top, Bottom
        pad = (self.padding[1], self.padding[1], self.padding[0], 0)
        x = F.pad(x, pad)
        out = self.relu(self.conv1(x))
        out = F.pad(out, pad)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class TCNModel(nn.Module):
    """Temporal Convolutional Network with Residual Blocks."""

    def __init__(self, input_channels, output_channels, num_levels = 4):
        super().__init__()
        levels = [input_channels] + [64] * num_levels
        dilations = [(2**i, 1) for i in range(num_levels)]  # Exponential dilation
        kernel_size = (3, 3)
        self.layers = nn.ModuleList()
        for i in range(num_levels):
            self.layers.append(
                ResidualBlock(
                    in_channels=levels[i],
                    out_channels=levels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                )
            )
        self.final_conv = nn.Conv2d(
            in_channels=levels[-1], out_channels=output_channels, kernel_size=1
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.final_conv(out)
        # Adjust PriceDimension to match target (e.g., 32)
        out = F.adaptive_avg_pool2d(out, (out.shape[2], 32))
        return out
