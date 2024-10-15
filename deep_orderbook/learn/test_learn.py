import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import AsyncGenerator, Tuple
import numpy as np
import matplotlib.pyplot as plt

from deep_orderbook.config import ReplayConfig, ShaperConfig, TrainConfig


# Define a Residual Block with dilated convolutions
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        padding = (
            (kernel_size[0] - 1) * dilation[0],  # Time dimension padding for causality
            (kernel_size[1] - 1)
            * dilation[1]
            // 2,  # Symmetric padding for Price dimension
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
        # Causal padding for time dimension and symmetric padding for price dimension
        pad_t = (
            self.padding[1],
            self.padding[1],
            self.padding[0],
            0,
        )  # Left, Right, Top, Bottom
        x = F.pad(x, pad_t)  # Pad (Left, Right, Top, Bottom)
        out = self.relu(self.conv1(x))
        out = F.pad(out, pad_t)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


# Define the deeper Temporal Convolutional Network model
class TCNModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        num_levels = 8
        levels = [input_channels] + [64] * num_levels
        dilations = [
            (2**i, 1) for i in range(num_levels)
        ]  # Exponential dilation in time dimension
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
        # x shape: (BatchSize, Channels, TimeDimension, PriceDimension)
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.final_conv(out)
        # Adjust PriceDimension to match target (e.g., 32)
        out = F.adaptive_avg_pool2d(out, (out.shape[2], 32))
        return out


# Trainer class to handle training logic and predictions
class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, books_array: np.ndarray, time_levels: np.ndarray):
        self.model.train()
        # Convert numpy arrays to torch tensors and move to the device
        books_tensor = torch.tensor(
            books_array, dtype=torch.float32, device=self.device
        )
        time_levels_tensor = torch.tensor(
            time_levels, dtype=torch.float32, device=self.device
        )

        # Rearrange dimensions to match PyTorch convention
        # From (TimeDimension, PriceDimension, FeatureDimension)
        # To (BatchSize, Channels, TimeDimension, PriceDimension)
        books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(
            0
        )  # Shape: (1, FeatureDimension, TimeDimension, PriceDimension)
        time_levels_tensor = time_levels_tensor.permute(2, 0, 1).unsqueeze(
            0
        )  # Shape: (1, ValueDimension, TimeDimension, PriceDimension)

        # Forward pass
        output = self.model(books_tensor)

        # Compute loss
        loss = self.criterion(output, time_levels_tensor)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the loss and the output
        return loss.item(), output.detach().cpu().numpy()

    def predict(self, books_array: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            books_tensor = torch.tensor(
                books_array, dtype=torch.float32, device=self.device
            )
            books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(0)
            output = self.model(books_tensor)
            predictions = output.cpu().numpy()
            return predictions.squeeze()  # Remove batch dimension if needed


# Function to train and predict
async def train_and_predict(
    config: TrainConfig, replay_config: ReplayConfig, shaper_config: ShaperConfig
):
    from deep_orderbook.shaper import iter_shapes_t2l

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}')

    # Model parameters
    input_channels = 3  # FeatureDimension of books_array
    output_channels = 1  # ValueDimension of time_levels

    # Initialize model, optimizer, and loss function
    model = TCNModel(input_channels, output_channels)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Create the trainer
    trainer = Trainer(model, optimizer, criterion, device)

    # Lists to store losses and predictions for plotting
    losses = []

    # Iterate over data
    epoch_left = config.epochs
    while epoch_left > 0:
        epoch_left -= 1
        async for books_array, time_levels, pxar in iter_shapes_t2l(
            replay_config=replay_config, shaper_config=shaper_config
        ):
            time_levels[time_levels > 0.1] = 1
            # Perform a training step
            loss, prediction = trainer.train_step(
                books_array[: 256 - 64], time_levels[: 256 - 64]
            )
            losses.append(loss)
            prediction = trainer.predict(books_array)

            yield books_array, time_levels, pxar, prediction, loss


# Main function to run the script
async def main():
    train_config = TrainConfig()
    replay_config = ReplayConfig()
    shaper_config = ShaperConfig()
    async for books_array, time_levels, pxar, prediction, loss in train_and_predict(
        train_config, replay_config, shaper_config
    ):
        print(f'{loss=}')


if __name__ == '__main__':
    asyncio.run(main())
