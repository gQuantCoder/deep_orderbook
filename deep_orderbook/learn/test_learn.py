import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import AsyncGenerator, Tuple
import numpy as np
import matplotlib.pyplot as plt

from deep_orderbook.config import ReplayConfig, ShaperConfig, TrainConfig
from deep_orderbook.learn.tcn import TCNModel
from deep_orderbook.learn.trainer import Trainer


# Function to train and predict
async def train_and_predict(
    config: TrainConfig,
    replay_config: ReplayConfig,
    shaper_config: ShaperConfig,
    test_config: ReplayConfig,
):
    from deep_orderbook.shaper import iter_shapes_t2l

    # Model parameters
    input_channels = 3  # FeatureDimension of books_array
    output_channels = 1  # ValueDimension of time_levels

    # Initialize model, optimizer, and loss function
    model = TCNModel(input_channels, output_channels)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Create the trainer
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_config=config,
        replay_config=replay_config,
        shaper_config=shaper_config,
    )
    # Start data loading workers
    trainer.start_data_loading()

    # Lists to store losses and predictions for plotting
    losses = []

    # Iterate over data
    epoch_left = config.epochs
    while epoch_left > 0:
        epoch_left -= 1
        async for books_array, time_levels, pxar in iter_shapes_t2l(
            replay_config=test_config,
            shaper_config=shaper_config.but(only_full_arrays=False),
        ):
            time_levels[time_levels > 0.02] = 1
            # Perform a training step
            loss, prediction = trainer.train_step()
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
