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

    logger.warning(f"[Training] Starting training with {replay_config.num_files()=}")

    # Model parameters
    input_channels = 3  # FeatureDimension of books_array
    output_channels = 1  # ValueDimension of time_levels

    # Initialize model, optimizer, and loss function
    model = TCNModel(
        input_channels,
        output_channels,
        num_levels=config.num_levels,
        num_side_lvl=shaper_config.num_side_lvl,
        target_side_width=shaper_config.look_ahead_side_width,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    logger.info("[Training] Model initialized, starting trainer setup")

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
    logger.info("[Training] Data loading workers started")

    # Lists to store losses and predictions for plotting
    losses: list[float] = []
    samples_processed = 0

    # Iterate over data
    epoch_left = config.epochs
    while epoch_left > 0:
        epoch_left -= 1
        logger.info(f"[Training] Starting epoch {config.epochs - epoch_left}/{config.epochs}")
        epoch_samples = 0
        
        async for books_array, time_levels, pxar in iter_shapes_t2l(
            replay_config=test_config,
            shaper_config=shaper_config.but(only_full_arrays=False),
        ):
            logger.debug(f"[Training] Queue size before train step: {trainer.data_queue.qsize()}")
            try:
                # Get training loss, test loss and prediction using the test data
                train_loss, test_loss, prediction = trainer.train_step(test_data=(books_array, time_levels, pxar))
                if train_loss is None:
                    logger.warning("[Training] train_step returned None, queue might be empty")
                    continue
                
                samples_processed += 1
                epoch_samples += 1
                if epoch_samples % 10 == 0:
                    logger.info(f"[Training] Processed {epoch_samples} samples in current epoch {config.epochs - epoch_left}, total: {samples_processed}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")

                yield books_array, time_levels, pxar, prediction, train_loss, test_loss
            except Exception as e:
                logger.error(f"[Training] Exception in training: {e}")
                continue
        
        logger.info(f"[Training] Completed epoch {config.epochs - epoch_left} with {epoch_samples} samples")

# Main function to run the script
async def main():
    from tqdm.auto import tqdm
    from deep_orderbook.utils import make_handlers
    # logger.setLevel('DEBUG')
    # change logging file to train.log
    line_handler, noline_handler = make_handlers('train.log')
    logger.addHandler(line_handler)
    # clear the log file
    with open('train.log', 'w') as f:
        f.truncate()

    train_config = TrainConfig(
        num_workers=1,
        batch_size=16,
        data_queue_size=512,
        num_levels=8,
        learning_rate=0.0001,
        epochs=10,
    )
    replay_config = ReplayConfig(
        markets=["ETH-USD"],  # , "BTC-USD", "ETH-BTC"],
        date_regexp='2024-11-06T0*',  # 1-06T',
        data_dir='/media/photoDS216/crypto/',
        every="1000ms",
    )
    shaper_config = ShaperConfig(
        only_full_arrays=True,
        view_bips=20,
        num_side_lvl=8,
        look_ahead=32,
        look_ahead_side_bips=10,
        look_ahead_side_width=4,
        rolling_window_size=1024,
        window_stride=8,
        # use_cache=False,
    )
    test_config = replay_config.but(date_regexp='2024-11-06T0*')

    # Define your asynchronous function to update the figure

    bar = tqdm(train_and_predict(
        config=train_config,
        replay_config=replay_config,
        shaper_config=shaper_config,
        test_config=test_config,
    ))
    async for books_arr, t2l, pxar, prediction, train_loss, test_loss in bar:
        bar.set_description(f'{train_loss=:.4f}, {test_loss=:.4f}')


if __name__ == '__main__':
    logger.setLevel('INFO')
    asyncio.run(main())
