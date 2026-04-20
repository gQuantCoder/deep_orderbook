import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_orderbook.utils import logger


class ResidualBlock(nn.Module):
    """Residual Block with dilated convolutions for temporal data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        num_side_lvl (int, optional): Number of price levels per side in input. Defaults to 4.
        target_side_width (int, optional): Target number of price levels per side in output. Defaults to 4.

    Architecture:
        1. Multiple ResidualBlocks with increasing dilation rates
        2. Each block maintains 8 channels internally
        3. Progressive reduction convolutions to match target output width
        4. Final 1x1 convolution to map to desired output channels
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_levels: int = 4,
        num_side_lvl: int = 4,
        target_side_width: int = 4,
    ) -> None:
        super().__init__()

        # Calculate receptive field length
        kernel_t = 3  # temporal kernel size
        receptive_length = 1
        for i in range(num_levels):
            dilation = 2**i
            receptive_length += (kernel_t - 1) * dilation

        logger.warning(f"TCN receptive length (timesteps): {receptive_length}")

        # Store for potential future use
        self.receptive_length = receptive_length

        # Each level processes 8 channels internally
        levels = [input_channels] + [8] * num_levels
        # Dilation increases exponentially in time dimension only (2^i, 1)
        dilations = [(2**i, 1) for i in range(num_levels)]
        kernel_size = (3, 3)

        # Store the target width for output sizing
        self.target_side_width = target_side_width

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

        # Add progressive reduction layers to match the target width
        # Calculate how many reduction steps we need
        input_width = 2 * num_side_lvl
        target_width = 2 * target_side_width
        self.reduction_layers = nn.ModuleList()

        # Add reduction layers as needed
        while input_width > target_width:
            # Use small 2x1 kernels for progressive reduction
            self.reduction_layers.append(
                nn.Conv2d(
                    in_channels=levels[-1],
                    out_channels=levels[-1],
                    kernel_size=(1, 2),  # Only reduce in the price dimension
                    stride=(1, 2),  # Reduce by factor of 2
                    padding=(0, 0),  # No padding needed for 2x1 kernel
                )
            )
            input_width = input_width // 2

        # Final 1x1 convolution to map to desired output channels
        self.final_conv = nn.Conv2d(
            in_channels=levels[-1], out_channels=output_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time, price]

        Returns:
            torch.Tensor: Output tensor with shape [batch, output_channels, time, 2*target_side_width]
                         where the price dimension matches the target width
        """
        out = x
        for layer in self.layers:
            out = layer(out)

        # Apply progressive reduction
        for reduction in self.reduction_layers:
            out = reduction(out)
            out = F.relu(out)  # Add non-linearity between reductions

        # Final 1x1 convolution
        out = self.final_conv(out)

        # Fine-tune to exact size if needed
        out = F.adaptive_avg_pool2d(out, (out.shape[2], 2 * self.target_side_width))
        return out


# ---------------------------------------------------------------------------
# Minimum RF guard — works on any model with a .receptive_field attribute
# ---------------------------------------------------------------------------

def _min_n_levels_for_rf(target_rf: int, kernel: int = 3) -> int:
    return max(math.ceil(math.log2((target_rf - 1) / (kernel - 1) + 1)), 1)


def assert_min_receptive_field_2d(model: "OrderBook2DTCN", min_rf: int = 1024) -> None:
    """Hard gate: raise if the model's temporal receptive field is below min_rf ticks.

    Must be called immediately after constructing any 2D model.  A model that
    sees fewer than 1024 ticks back cannot capture order-book microstructure
    patterns that operate on timescales of tens of seconds to minutes.
    """
    if model.receptive_field < min_rf:
        raise ValueError(
            f"OrderBook2DTCN receptive field {model.receptive_field} ticks is below "
            f"the mandatory minimum {min_rf}.  Use num_levels>={_min_n_levels_for_rf(min_rf)} "
            f"(currently {model.num_levels}).  Refusing to train a model that is blind "
            f"to 99%+ of its input window."
        )


# ---------------------------------------------------------------------------
# THE canonical 2-D production model for order-book experiments
# ---------------------------------------------------------------------------

class OrderBook2DTCN(nn.Module):
    """2D causal dilated TCN — the ONLY permitted model for experiment scripts.

    Contract
    --------
    * Input shape:  (B, T, bins, 3)  — batch × time × price-bins × channels
      The 3 channels are: [net_signed_depth, buy_trade_vol, sell_trade_vol].
      **NEVER flatten (bins × channels) into a 1-D feature vector before this model.**
    * The model permutes internally to (B, 3, T, bins) for Conv2d.
    * Output shapes (both shape (B, T, levels)):
        reg  — softplus-activated t2l regression values
        cls  — raw logits for BCE classification head (same target as reg)

    Why 2D matters
    --------------
    The order book is a 2D structure: each tick has a *spatial* layout of
    depth across price bins.  A 1D TCN (Conv1d over flat 48 features) treats
    bin 7 and bin 8 as unrelated features — it cannot learn "bid depth rising
    at the best bid while ask depth thins" which is the canonical imbalance
    signal.  Conv2d kernels slide across BOTH the time axis AND the price-bin
    axis, allowing the model to detect cross-bin patterns.

    Minimum receptive field (temporal, kernel=3, doubling dilations)
    ---------------------------------------------------------------
    RF = 1 + 2 * (2^num_levels − 1)

    | num_levels | RF ticks | @ 100ms | @ 1000ms |
    |------------|----------|---------|----------|
    |  8         |  511     |  51 s   |  8.5 min |
    |  10        | 2047     |  204 s  |  34 min  |  ← default
    |  12        | 8191     |  819 s  |  136 min |

    Minimum requirement: RF ≥ 1024 ticks (enforced by assert_min_receptive_field_2d).
    """

    MIN_REQUIRED_RF: int = 1024

    def __init__(
        self,
        bins: int = 16,
        in_ch: int = 3,
        hid: int = 8,
        out_levels: int = 8,
        num_levels: int = 10,
        trunk_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        kernel_t = 3

        # Temporal receptive field (dilation doubles each level, price dilation=1)
        self.receptive_field = 1 + (kernel_t - 1) * sum(2**i for i in range(num_levels))

        dilations = [(2**i, 1) for i in range(num_levels)]
        channels = [in_ch] + [hid] * num_levels
        self.blocks = nn.ModuleList([
            ResidualBlock(channels[i], channels[i + 1], (kernel_t, 3), dilations[i])
            for i in range(num_levels)
        ])
        self.dropout = nn.Dropout2d(trunk_dropout)

        # Price-axis reduction: (bins) → (out_levels) via adaptive pool after heads
        self.reg_head = nn.Conv2d(hid, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(hid, 1, kernel_size=1)
        self.out_levels = out_levels

        logger.warning(
            f"OrderBook2DTCN: num_levels={num_levels}, hid={hid}, bins={bins}, "
            f"RF={self.receptive_field} ticks "
            f"({self.receptive_field * 0.1:.0f}s @100ms  "
            f"{self.receptive_field * 1.0:.0f}s @1000ms)"
        )

    def forward(
        self, x_bt_bins_ch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_bt_bins_ch: (B, T, bins, 3) — the raw books tensor, NOT flattened.

        Returns:
            reg:  (B, T, out_levels)  softplus-activated regression output
            cls:  (B, T, out_levels)  raw logits for classification
        """
        # (B, T, bins, 3) → (B, 3, T, bins)  for Conv2d
        x = x_bt_bins_ch.permute(0, 3, 1, 2)
        h = x
        for block in self.blocks:
            h = self.dropout(block(h))

        # Reduce price dimension: (B, hid, T, bins) → (B, 1, T, out_levels)
        reg_raw = F.adaptive_avg_pool2d(self.reg_head(h), (h.shape[2], self.out_levels))
        cls_raw = F.adaptive_avg_pool2d(self.cls_head(h), (h.shape[2], self.out_levels))

        # (B, 1, T, levels) → (B, T, levels)
        reg = F.softplus(reg_raw.squeeze(1))
        cls = cls_raw.squeeze(1)
        return reg, cls
