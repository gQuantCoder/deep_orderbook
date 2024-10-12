import asyncio
from typing import AsyncGenerator, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the Causal Convolution layer to ensure causality over the time dimension
class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=(1,1)):
        super(CausalConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation
        )

    def forward(self, x):
        # x shape: (BatchSize, Channels, TimeDimension, PriceDimension)
        pad_t = (self.kernel_size[0] - 1) * self.dilation[0]
        pad_p = (self.kernel_size[1] - 1) * self.dilation[1] // 2  # Symmetric padding in price dimension
        x = F.pad(x, (pad_p, pad_p, pad_t, 0))  # Pad (Left, Right, Top, Bottom)
        return self.conv(x)

# Define the Temporal Convolutional Network model
class TCNModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TCNModel, self).__init__()
        self.conv1 = CausalConv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(3, 3),
            dilation=(1, 1)
        )
        self.conv2 = CausalConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            dilation=(2, 1)
        )
        self.conv3 = CausalConv2d(
            in_channels=64,
            out_channels=output_channels,
            kernel_size=(3, 3),
            dilation=(4, 1)
        )

    def forward(self, x):
        # x shape: (BatchSize, Channels, TimeDimension, PriceDimension)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # Adjust PriceDimension to match target (e.g., 32)
        x = F.adaptive_avg_pool2d(x, (x.shape[2], 32))
        return x

# Trainer class to handle training logic
class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    async def train(self, data_loader):
        self.model.train()
        async for books_array, time_levels in data_loader:
            # Convert numpy arrays to torch tensors and move to the device
            books_tensor = torch.tensor(books_array, dtype=torch.float32, device=self.device)
            time_levels_tensor = torch.tensor(time_levels, dtype=torch.float32, device=self.device)

            # Rearrange dimensions to match PyTorch convention
            # From (TimeDimension, PriceDimension, FeatureDimension)
            # To (BatchSize, Channels, TimeDimension, PriceDimension)
            books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, FeatureDimension, TimeDimension, PriceDimension)
            time_levels_tensor = time_levels_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, ValueDimension, TimeDimension, PriceDimension)

            # Forward pass
            output = self.model(books_tensor)

            # Compute loss
            loss = self.criterion(output, time_levels_tensor)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss
            print(f'Loss: {loss.item()}')

# The asynchronous data generator as provided
async def iter_shapes() -> AsyncGenerator[Tuple[np.ndarray, np.ndarray], None]:
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed
    from deep_orderbook.replayer import ParquetReplayer
    from deep_orderbook.shaper import ArrayShaper

    shaper = ArrayShaper(zoom_frac=0.25)
    MARKETS = ["ETH-USD"]
    replayer = ParquetReplayer('data', date_regexp='2024-08-06')
    async with CoinbaseFeed(
        markets=MARKETS,
        replayer=replayer,
    ) as feed:
        async for onesec in feed.one_second_iterator(max_samples=200):
            books_array = await shaper.make_arr3d(onesec.symbols[MARKETS[0]])
            time_levels = await shaper.build_time_level_trade()
            yield books_array, time_levels

# Main function to set up and run the training
async def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    input_channels = 3  # FeatureDimension of books_array
    output_channels = 1  # ValueDimension of time_levels

    # Initialize model, optimizer, and loss function
    model = TCNModel(input_channels, output_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Create the trainer
    trainer = Trainer(model, optimizer, criterion, device)

    # Start training
    await trainer.train(iter_shapes())

if __name__ == '__main__':
    asyncio.run(main())
