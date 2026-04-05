import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_orderbook.learn.positional_encoding import PositionalEncoding
from deep_orderbook.utils import logger
import asyncio
from pathlib import Path
from tqdm.auto import tqdm
from deep_orderbook.utils import make_handlers
from deep_orderbook.visu import Visualizer
from deep_orderbook.strategy import Strategy
from deep_orderbook.config import TrainConfig, ReplayConfig, ShaperConfig
import torch.optim as optim
from typing import AsyncGenerator, Tuple, Optional
import numpy as np
from deep_orderbook.learn.trainer import Trainer


class TimeSeriesTransformer(nn.Module):
    """A pure attention-based model for time series prediction.
    
    This model uses a stack of transformer encoder layers to process time series data.
    It maintains causality through causal attention masks and includes both temporal
    and spatial attention mechanisms.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 4,
        num_side_lvl: int = 4,
        target_side_width: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 10000  # Added parameter for longer sequences
    ) -> None:
        """Initialize the TimeSeriesTransformer.
        
        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            num_side_lvl (int): Number of price levels per side
            target_side_width (int): Target number of price levels per side
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.target_side_width = target_side_width
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Conv2d(
            input_channels, d_model, kernel_size=1
        )
        
        # Positional encoding with specified max_seq_len
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_seq_len  # Pass max_seq_len to PositionalEncoding
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Conv2d(
            d_model, output_channels, kernel_size=1
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        logger.warning(
            f"TimeSeriesTransformer initialized with {num_layers} layers, "
            f"{nhead} heads, and {d_model} dimensions"
        )

    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate a causal attention mask.
        
        Args:
            size (int): Size of the sequence
            
        Returns:
            torch.Tensor: Causal mask where future positions are masked
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.input_projection.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time, price]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Initial projection
        x = self.input_projection(x)  # [batch, d_model, time, price]
        
        # Process each price level independently
        b, c, t, h = x.shape
        # Reshape to [batch * price, time, d_model]
        x = x.permute(0, 3, 2, 1).reshape(b * h, t, c)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Generate causal mask
        mask = self.generate_causal_mask(t)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)
        
        # Reshape back to [batch, d_model, time, price]
        x = x.reshape(b, h, t, c).permute(0, 3, 2, 1)
        
        # Final projection
        x = self.output_projection(x)
        
        # Ensure output size matches target width
        x = F.adaptive_avg_pool2d(x, (x.shape[2], 2 * self.target_side_width))
        
        return x

async def main() -> None:
    from tqdm.auto import tqdm
    from deep_orderbook.utils import make_handlers
    from deep_orderbook.visu import Visualizer
    from deep_orderbook.strategy import Strategy
    from deep_orderbook.config import TrainConfig, ReplayConfig, ShaperConfig
    import torch.optim as optim

    # Setup logging
    line_handler, noline_handler = make_handlers('pure_attention_test.log')
    logger.addHandler(line_handler)
    with open('pure_attention_test.log', 'w') as f:
        f.truncate()

    # Configuration
    train_config = TrainConfig(
        num_workers=1,
        batch_size=8,
        data_queue_size=512,
        num_levels=4,
        learning_rate=0.0001,
        epochs=10,
        criterion="StructuredT2L",
        loss_focus_last_step=True,
        save_checkpoint_mins=5.0,
        checkpoint_dir=Path("checkpoints_pure_attention"),
    )
    replay_config = ReplayConfig(
        markets=["ETH-USD"],
        date_regexp='2024-11-06T0*',
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
    )
    test_config = replay_config.but(date_regexp='2024-11-06T0*')

    # Model parameters
    input_channels = 3  # FeatureDimension of books_array
    output_channels = 1  # ValueDimension of time_levels

    # Initialize model
    model = TimeSeriesTransformer(
        input_channels=input_channels,
        output_channels=output_channels,
        d_model=64,
        nhead=4,
        num_layers=4,
        num_side_lvl=shaper_config.num_side_lvl,
        target_side_width=shaper_config.look_ahead_side_width,
        dropout=0.1,
        max_seq_len=10000
    )
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = nn.MSELoss()

    # Create trainer
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_config=train_config,
        replay_config=replay_config,
        shaper_config=shaper_config.but(only_full_arrays=True),
        gradient_accumulation_steps=4
    )

    # Try to load latest checkpoint
    trainer.load_latest_checkpoint()
    trainer.start_data_loading()

    # Training loop with visualization
    bar = tqdm(
        train_and_predict(
            trainer=trainer,
            test_config=test_config,
            shaper_config=shaper_config,
        )
    )
    vis = Visualizer()
    strategy = Strategy(threshold=0.3)

    async for books_arr, t2l, pxar, pred_t2l, train_loss, test_loss in bar:
        bar.set_description(f'{train_loss=:.4f}, {test_loss=:.4f}')
        
        gt_pnl, pos, gt_up_prox, gt_down_prox = strategy.compute_pnl(pxar, t2l)
        pred_pnl, pred_pos, pred_up_prox, pred_down_prox = strategy.compute_pnl(
            pxar, pred_t2l
        )

        vis.add_loss(train_loss, test_loss)
        vis.update(
            books_z_data=books_arr,
            level_reach_z_data=t2l,
            bidask=pxar,
            pred_t2l=pred_t2l,
            gt_pnl=gt_pnl,
            pred_pnl=pred_pnl,
            positions=pos,
            pred_positions=pred_pos,
            up_proximity=gt_up_prox,
            down_proximity=gt_down_prox,
            pred_up_proximity=pred_up_prox,
            pred_down_proximity=pred_down_prox,
        )

async def train_and_predict(
    trainer: Trainer,
    test_config: ReplayConfig,
    shaper_config: ShaperConfig,
) -> AsyncGenerator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float, float], None]:
    """Train the model and yield predictions.
    
    Args:
        trainer: The trainer instance
        test_config: Test configuration
        shaper_config: Shaper configuration
        
    Yields:
        Tuple of (books_array, time_levels, pxar, prediction, train_loss, test_loss)
        where prediction may be None if test step fails
    """
    from deep_orderbook.shaper import iter_shapes_t2l
    
    samples_processed = trainer.total_samples_processed
    epoch_left = trainer.train_config.epochs - trainer.current_epoch
    
    while epoch_left > 0:
        epoch_left -= 1
        trainer.current_epoch = trainer.train_config.epochs - epoch_left
        logger.info(f"[Training] Starting epoch {trainer.current_epoch}/{trainer.train_config.epochs}")
        epoch_samples = 0

        async for books_array, time_levels, pxar in iter_shapes_t2l(
            replay_config=test_config,
            shaper_config=shaper_config.but(only_full_arrays=False),
        ):
            try:
                result = trainer.train_step(test_data=(books_array, time_levels, pxar))
                if result is None:
                    continue
                train_loss, test_loss, prediction = result
                if test_loss is None:
                    continue

                samples_processed += 1
                epoch_samples += 1
                yield books_array, time_levels, pxar, prediction, train_loss, test_loss
                
            except Exception as e:
                logger.error(f"[Training] Exception in training: {e}")
                continue

        logger.info(f"[Training] Completed epoch {trainer.current_epoch} with {epoch_samples} samples")
        trainer.save_checkpoint()

if __name__ == '__main__':
    logger.setLevel('INFO')
    asyncio.run(main()) 