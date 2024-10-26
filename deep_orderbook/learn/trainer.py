# trainer.py

import torch
import numpy as np
import queue
import threading
from deep_orderbook.learn.data_loader import DataLoaderWorker
from deep_orderbook.config import ReplayConfig, ShaperConfig, TrainConfig
from deep_orderbook.utils import logger


class Trainer:
    """Trainer class to handle training and prediction."""

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_config: TrainConfig,
        replay_config: ReplayConfig,
        shaper_config: ShaperConfig,
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'{self.device=}')
        self.model = model.to(self.device)
        self.data_queue = queue.Queue(maxsize=train_config.data_queue_size)
        self.train_config = train_config
        self.replay_config = replay_config
        self.shaper_config = shaper_config
        self.workers = []

    def start_data_loading(self):
        """Starts data loading workers."""
        num_workers = self.train_config.num_workers
        for _ in range(num_workers):
            data_loader_worker = DataLoaderWorker(
                data_queue=self.data_queue,
                replay_config=self.replay_config,
                shaper_config=self.shaper_config,
            )
            data_loader_worker.start()
            self.workers.append(data_loader_worker)

    def train_step(self):
        """Performs a training step using a batch of data from the queue."""
        books_array_list = []
        time_levels_list = []
        pxar_list = []
        batch_size = self.train_config.batch_size
        while len(books_array_list) < batch_size:
            try:
                books_array, time_levels, pxar = self.data_queue.get(timeout=120)
                books_array_list.append(books_array)
                time_levels_list.append(time_levels)
                pxar_list.append(pxar)
            except queue.Empty:
                if len(books_array_list) == 0:
                    print("Data queue is empty. Waiting for data...")
                    return None
                else:
                    print(
                        "Not enough samples for a full batch. Proceeding with available samples."
                    )
                    break

        # Preprocess data
        for i in range(len(books_array_list)):
            time_levels_list[i][time_levels_list[i] > 0.02] = 1

        time_steps_used = (
            self.shaper_config.rolling_window_size - self.shaper_config.look_ahead
        )

        self.model.train()

        # Stack samples into batches
        books_tensor = torch.stack(
            [
                torch.tensor(books_array[:time_steps_used], dtype=torch.float32)
                for books_array in books_array_list
            ],
            dim=0,
        ).to(self.device)

        time_levels_tensor = torch.stack(
            [
                torch.tensor(time_levels[:time_steps_used], dtype=torch.float32)
                for time_levels in time_levels_list
            ],
            dim=0,
        ).to(self.device)

        # Rearrange dimensions to match PyTorch convention
        books_tensor = books_tensor.permute(
            0, 3, 1, 2
        )  # (BatchSize, Channels, Time, Price)
        time_levels_tensor = time_levels_tensor.permute(0, 3, 1, 2)

        # Forward pass
        output = self.model(books_tensor)

        # Compute loss
        loss = self.criterion(output, time_levels_tensor)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return first sample for visualization
        return (
            loss.item(),
            output[0].detach().cpu().numpy(),
            # books_array_list[0],
            # time_levels_list[0],
            # pxar_list[0],
        )

    def predict(self, books_array: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            books_tensor = torch.tensor(
                books_array, dtype=torch.float32, device=self.device
            )
            books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(0)
            output = self.model(books_tensor)
            predictions = output.detach().cpu().numpy()
        return predictions.squeeze()

    def save_model(self, filepath='trained_model.pth'):
        """Saves the trained model."""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath='trained_model.pth'):
        """Loads a trained model."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


def main():
    # Example usage of Trainer
    from deep_orderbook.learn.tcn import TCNModel
    import torch.optim as optim
    import torch.nn as nn
    import asyncio
    from deep_orderbook.config import ReplayConfig, ShaperConfig

    # Model parameters
    input_channels = 3
    output_channels = 1

    # Initialize model, optimizer, and loss function
    model = TCNModel(input_channels, output_channels, num_levels=4)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Configurations
    train_config = TrainConfig(
        num_workers=8, batch_size=32, data_queue_size=512, num_levels=8
    )
    replay_config = ReplayConfig(
        date_regexp='2024-0', data_dir='/media/photoDS216/crypto/'
    )
    shaper_config = ShaperConfig(only_full_arrays=True)

    # Create the trainer
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_config=train_config,
        replay_config=replay_config,
        shaper_config=shaper_config,
    )

    # Start data loading workers
    trainer.start_data_loading()

    # Training loop
    num_training_steps = 100
    while trainer.data_queue.qsize() < train_config.data_queue_size:
        logger.info(f"Waiting for data. Queue size: {trainer.data_queue.qsize()}")
        asyncio.run(asyncio.sleep(5))

    for step in range(num_training_steps):
        result = trainer.train_step()
        if result is not None:
            loss, prediction = result
            logger.warning(
                f"Training step {step + 1}/{num_training_steps}, queue: {trainer.data_queue.qsize()}, Loss: {loss}"
            )
            # print(f"Training step {step + 1}/{num_training_steps}, Loss: {loss}")
        else:
            asyncio.run(asyncio.sleep(1))

    # Save the trained model
    trainer.save_model('trained_model.pth')
    print("Model training completed and saved.")


if __name__ == '__main__':
    logger.setLevel('INFO')
    main()
