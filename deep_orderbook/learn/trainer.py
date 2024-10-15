# trainer.py

import torch
import numpy as np


class Trainer:
    """Trainer class to handle training and prediction."""

    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, books_array: np.ndarray, time_levels: np.ndarray):
        self.model.train()
        # Convert numpy arrays to torch tensors and move to the device
        books_tensor = torch.tensor(books_array, dtype=torch.float32, device=self.device)
        time_levels_tensor = torch.tensor(
            time_levels, dtype=torch.float32, device=self.device
        )

        # Rearrange dimensions to match PyTorch convention
        books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(0)
        time_levels_tensor = time_levels_tensor.permute(2, 0, 1).unsqueeze(0)

        # Forward pass
        output = self.model(books_tensor)

        # Compute loss
        loss = self.criterion(output, time_levels_tensor)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), output.detach().cpu().numpy()

    def predict(self, books_array: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            books_tensor = torch.tensor(books_array, dtype=torch.float32, device=self.device)
            books_tensor = books_tensor.permute(2, 0, 1).unsqueeze(0)
            output = self.model(books_tensor)
            predictions = output.cpu().numpy()
        return predictions.squeeze()
