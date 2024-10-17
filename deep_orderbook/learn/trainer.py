# trainer.py

import torch
import numpy as np
import queue
import threading
from deep_orderbook.learn.data_loader import DataLoaderWorker
from deep_orderbook.config import ReplayConfig, ShaperConfig


class Trainer:
    """Trainer class to handle training and prediction."""

    def __init__(
        self, model, optimizer, criterion, device, replay_config, shaper_config
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_queue = queue.Queue(
            maxsize=1000
        )  # Queue is a member of the Trainer class
        self.replay_config = replay_config
        self.shaper_config = shaper_config
        self.workers = []

    def start_data_loading(self, num_workers=1):
        """Starts data loading workers."""
        for _ in range(num_workers):
            data_loader_worker = DataLoaderWorker(
                data_queue=self.data_queue,
                replay_config=self.replay_config,
                shaper_config=self.shaper_config,
            )
            data_loader_worker.start()
            self.workers.append(data_loader_worker)

    def train_step(self):
        """Performs a training step using data from the queue."""
        try:
            # Get data from queue
            books_array, time_levels, pxar = self.data_queue.get(timeout=30)
            # Preprocess data
            time_levels[time_levels > 0.02] = 1
            time_cut = (
                self.shaper_config.time_accumulate - self.shaper_config.look_ahead
            )

            self.model.train()
            # Convert numpy arrays to torch tensors and move to the device
            books_tensor = torch.tensor(
                books_array[:time_cut], dtype=torch.float32, device=self.device
            )
            t2l_tensor = torch.tensor(
                time_levels[:time_cut], dtype=torch.float32, device=self.device
            )

            # Rearrange dimensions to match PyTorch convention
            books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(0)
            t2l_tensor = t2l_tensor.permute(2, 0, 1).unsqueeze(0)

            # Forward pass
            output = self.model(books_tensor)

            # Compute loss
            loss = self.criterion(output, t2l_tensor)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return (
                loss.item(),
                output.detach().cpu().numpy(),
                # books_array,
                # time_levels,
                # pxar,
            )
        except queue.Empty:
            print("Data queue is empty.")
            return None

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


def main():
    # Example usage of Trainer
    from deep_orderbook.learn.tcn import TCNModel
    import torch.optim as optim
    import torch.nn as nn
    import asyncio
    from deep_orderbook.config import ReplayConfig, ShaperConfig

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Model parameters
    input_channels = 3
    output_channels = 1

    # Initialize model, optimizer, and loss function
    model = TCNModel(input_channels, output_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Configurations
    replay_config = ReplayConfig(date_regexp="2024-0")
    shaper_config = ShaperConfig(only_full_arrays=True)

    # Create the trainer
    trainer = Trainer(model, optimizer, criterion, device, replay_config, shaper_config)

    # Start data loading workers
    trainer.start_data_loading(num_workers=2)

    # Training loop
    for _ in range(1000):  # For example, perform 10 training steps
        result = trainer.train_step()
        if result is not None:
            loss, prediction = result
            print(f"Training loss: {loss}")
        else:
            # Wait for data to be available
            print("Waiting for data...")
            asyncio.run(asyncio.sleep(1))


if __name__ == '__main__':
    main()
