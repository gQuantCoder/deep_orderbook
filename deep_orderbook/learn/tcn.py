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
    """Temporal Convolutional Network with Residual Blocks.
    
    A neural network that processes 2D temporal data using dilated convolutions
    and residual connections. The network progressively increases the receptive field
    through exponentially increasing dilations.
    
    Args:
        input_channels (int): Number of input channels in the data
        output_channels (int): Number of output channels to produce
        num_levels (int, optional): Number of residual blocks, each with increasing dilation. Defaults to 4.
        num_side_lvl (int, optional): Number of price levels per side. Total output levels will be 2*num_side_lvl. Defaults to 4.
    
    Architecture:
        1. Multiple ResidualBlocks with increasing dilation rates
        2. Each block maintains 8 channels internally
        3. Final 1x1 convolution to map to desired output channels
        4. Adaptive pooling to ensure consistent price dimension output
    """

    def __init__(self, input_channels, output_channels, num_levels = 4, num_side_lvl = 4):
        super().__init__()
        # Each level processes 8 channels internally
        levels = [input_channels] + [8] * num_levels
        # Dilation increases exponentially in time dimension only (2^i, 1)
        dilations = [(2**i, 1) for i in range(num_levels)]
        kernel_size = (3, 3)
        
        # Store the number of side levels for output sizing
        self.num_side_lvl = num_side_lvl
        
        # Create stack of residual blocks
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
        # Final 1x1 convolution to map to desired output channels
        self.final_conv = nn.Conv2d(
            in_channels=levels[-1], out_channels=output_channels, kernel_size=1
        )

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time, price]
            
        Returns:
            torch.Tensor: Output tensor with shape [batch, output_channels, time, 2*num_side_lvl]
                         where the price dimension matches the shaper config
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.final_conv(out)
        # Ensure price dimension matches the shaper config using adaptive pooling
        out = F.adaptive_avg_pool2d(out, (out.shape[2], 2 * self.num_side_lvl))
        return out
