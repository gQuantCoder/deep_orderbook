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
from deep_orderbook.utils import logger


# Function to train and predict
async def train_and_predict(
    config: TrainConfig,
    replay_config: ReplayConfig,
    shaper_config: ShaperConfig,
    test_config: ReplayConfig,
):
    from deep_orderbook.shaper import iter_shapes_t2l

    logger.warning(f"replay_config: {replay_config.num_files()=}")

    # Model parameters
    input_channels = 3  # FeatureDimension of books_array
    output_channels = 1  # ValueDimension of time_levels

    # Initialize model, optimizer, and loss function
    model = TCNModel(input_channels, output_channels, num_levels=config.num_levels)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Create the trainer
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_config=config,
        replay_config=replay_config,
        shaper_config=shaper_config.but(only_full_arrays=True),
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
            logger.debug(f"Queue size: {trainer.data_queue.qsize()}")
            try:
                loss, prediction = trainer.train_step()
                losses.append(loss)
                prediction = trainer.predict(books_array)

                yield books_array, time_levels, pxar, prediction, loss
            except Exception as e:
                print(f"Exception in training: {e}")


# Main function to run the script
async def main():
    from tqdm.auto import tqdm

    train_config = TrainConfig()
    replay_config = ReplayConfig(date_regexp='2024-09')
    shaper_config = ShaperConfig()
    test_config = replay_config
    bar = tqdm(
        train_and_predict(train_config, replay_config, shaper_config, test_config)
    )
    async for books_array, time_levels, pxar, prediction, loss in bar:
        bar.set_description(f'{loss=:.4f}')


if __name__ == '__main__':
    logger.setLevel('INFO')
    asyncio.run(main())
