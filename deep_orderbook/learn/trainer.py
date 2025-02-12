# trainer.py

import torch
import numpy as np
from queue import Queue
from deep_orderbook.learn.data_loader import DataLoaderWorker
from deep_orderbook.config import ReplayConfig, ShaperConfig, TrainConfig
from deep_orderbook.utils import logger
import time
import os
import sys

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
        self.data_queue = Queue(maxsize=train_config.data_queue_size)
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
            worker = data_loader_worker.start()
            self.workers.append(data_loader_worker)

    def train_step(self, test_data=None):
        """Performs a training step using a batch of data from the queue and optionally computes test loss.
        
        Args:
            test_data: Optional tuple of (books_array, time_levels, pxar) for computing test loss
            
        Returns:
            Tuple of (train_loss, test_loss, prediction) where test_loss is None if no test_data provided
        """
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
            except Exception as e:
                if len(books_array_list) == 0:
                    logger.warning(f"Data queue is empty. Waiting for data... {e}")
                    return None, None, None
                else:
                    logger.warning(
                        "Not enough samples for a full batch. Proceeding with available samples."
                    )
                    break

        time_steps_used = (
            self.shaper_config.rolling_window_size - self.shaper_config.look_ahead
        )

        # Training step
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

        # Compute training loss
        train_loss = self.criterion(output, time_levels_tensor)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        # Compute test loss if test data provided
        test_loss = None
        prediction = None
        if test_data is not None:
            test_books_array, test_time_levels, _ = test_data
            test_loss, prediction = self.compute_test_loss(test_books_array, test_time_levels)

        # Return losses and prediction
        return (
            train_loss.item(),
            test_loss,
            output[0].detach().cpu().numpy() if prediction is None else prediction
        )

    def compute_test_loss(self, books_array: np.ndarray, time_levels: np.ndarray) -> tuple[float, np.ndarray]:
        """Computes test loss and predictions on test data.
        
        Args:
            books_array: Input test data
            time_levels: Target test data
            
        Returns:
            Tuple of (test_loss, prediction)
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare test data
            test_input = torch.from_numpy(books_array).float().to(self.device)
            test_target = torch.from_numpy(time_levels).float().to(self.device)
            
            # Add batch dimension and rearrange if needed
            if test_input.dim() == 3:
                test_input = test_input.unsqueeze(0)
                test_input = test_input.permute(0, 3, 1, 2)
            if test_target.dim() == 3:
                test_target = test_target.unsqueeze(0)
                test_target = test_target.permute(0, 3, 1, 2)
            
            # Forward pass
            test_prediction = self.model(test_input)
            test_loss = self.criterion(test_prediction, test_target).item()
            
            # Get prediction in numpy format
            prediction = test_prediction[0].cpu().numpy()
            
        return test_loss, prediction

    def predict(self, books_array: np.ndarray) -> np.ndarray:
        """Makes a prediction on the input data.
        
        This method is kept for backward compatibility and simple inference.
        For training with test loss computation, use train_step with test_data.
        """
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

    def cleanup(self):
        """Stop all workers and cleanup resources."""
        for worker in self.workers:
            worker.stop()
        self.workers = []


def main():
    # Example usage of Trainer
    from deep_orderbook.learn.tcn import TCNModel
    import torch.optim as optim
    import torch.nn as nn
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

    # Wait for the data queue to have at least one batch
    while trainer.data_queue.qsize() < trainer.train_config.batch_size:
        logger.info(f"Waiting for data. Queue size: {trainer.data_queue.qsize()}")
        time.sleep(1)

    for step in range(num_training_steps):
        result = trainer.train_step()
        if result is not None:
            train_loss, test_loss, prediction = result
            logger.warning(
                f"Training step {step + 1}/{num_training_steps}, queue: {trainer.data_queue.qsize()}, Train Loss: {train_loss}, Test Loss: {test_loss}"
            )
        else:
            time.sleep(1)

    # Save the trained model
    trainer.save_model('trained_model.pth')
    print("Model training completed and saved.")


if __name__ == '__main__':
    logger.setLevel('INFO')
    main()
