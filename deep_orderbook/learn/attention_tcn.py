import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_orderbook.learn.tcn import TCNModel
from deep_orderbook.learn.positional_encoding import PositionalEncoding
from deep_orderbook.utils import logger
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Tuple
import numpy as np
from deep_orderbook.learn.trainer import Trainer
from deep_orderbook.config import ReplayConfig, ShaperConfig


class AttentionTCN(nn.Module):
    """Temporal Convolutional Network with Self-Attention.
    
    This model combines the TCN's ability to process sequential data efficiently
    with self-attention to capture long-range dependencies. It maintains causality
    through both the TCN's dilated convolutions and a causal mask in the attention
    mechanism.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_levels: int = 4,
        num_side_lvl: int = 4,
        target_side_width: int = 4,
        attention_heads: int = 4,
        attention_dim: int = 32,  # Must be divisible by attention_heads
        dropout: float = 0.1,
        max_seq_len: int = 10000
    ) -> None:
        """Initialize the AttentionTCN model.
        
        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            num_levels (int): Number of TCN levels
            num_side_lvl (int): Number of price levels per side
            target_side_width (int): Target number of price levels per side
            attention_heads (int): Number of attention heads
            attention_dim (int): Dimension of attention mechanism (must be divisible by attention_heads)
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # TCN backbone
        self.tcn = TCNModel(
            input_channels=input_channels,
            output_channels=attention_dim,  # Changed to attention_dim
            num_levels=num_levels,
            num_side_lvl=num_side_lvl,
            target_side_width=target_side_width
        )
        
        # Store the target width for output sizing
        self.target_side_width = target_side_width
        
        # Separate temporal and spatial attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Positional encoding for temporal dimension
        self.pos_encoder = PositionalEncoding(
            d_model=attention_dim,
            dropout=dropout,
            max_len=max_seq_len
        )
        
        # Layer normalization
        self.temporal_norm = nn.LayerNorm(attention_dim)
        
        # Final projection to output channels
        self.output_projection = nn.Conv2d(
            attention_dim, output_channels, kernel_size=1
        )
        
        # Calculate and log the receptive field
        self.receptive_length = self.tcn.receptive_length
        logger.warning(f"AttentionTCN receptive length (timesteps): {self.receptive_length}")

    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate a causal attention mask.
        
        Args:
            size (int): Size of the sequence
            
        Returns:
            torch.Tensor: Causal mask where future positions are masked
        """
        # Create upper triangular matrix (1s above diagonal, 0s on and below)
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        # Convert 1s to -inf (will be masked in attention) and 0s to 0 (will be attended to)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.temporal_attention.in_proj_weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # TCN processing
        tcn_out = self.tcn(x)  # [batch, channels, time, price]
        b, c, t, h = tcn_out.shape
        
        # Process each price level independently through temporal attention
        # Reshape to [batch * price, time, channels]
        temporal_in = tcn_out.permute(0, 3, 2, 1).reshape(b * h, t, c)
        
        # Add positional encoding
        temporal_in = self.pos_encoder(temporal_in)
        
        # Apply layer normalization
        temporal_in = self.temporal_norm(temporal_in)
        
        # Generate causal mask for temporal attention
        temporal_mask = self.generate_causal_mask(t)
        
        # Apply temporal attention
        temporal_out, _ = self.temporal_attention(
            temporal_in, temporal_in, temporal_in,
            attn_mask=temporal_mask,
            need_weights=False
        )
        
        # Reshape back to [batch, channels, time, price]
        out = temporal_out.reshape(b, h, t, c).permute(0, 3, 2, 1)
        
        # Project to output channels
        out = self.output_projection(out)
        
        # Ensure output size matches target width
        out = F.adaptive_avg_pool2d(out, (out.shape[2], 2 * self.target_side_width))
        
        return out

async def main() -> None:
    from tqdm.auto import tqdm
    from deep_orderbook.utils import make_handlers
    from deep_orderbook.visu import Visualizer
    from deep_orderbook.strategy import Strategy
    from deep_orderbook.config import TrainConfig, ReplayConfig, ShaperConfig
    import torch.optim as optim

    # Setup logging
    line_handler, noline_handler = make_handlers('attention_tcn_test.log')
    logger.addHandler(line_handler)
    with open('attention_tcn_test.log', 'w') as f:
        f.truncate()

    # Configuration
    train_config = TrainConfig(
        num_workers=1,
        batch_size=16,
        data_queue_size=512,
        num_levels=8,
        learning_rate=0.0001,
        epochs=10,
        criterion="StructuredT2L",
        loss_focus_last_step=True,
        save_checkpoint_mins=5.0,
        checkpoint_dir=Path("checkpoints_attention_tcn"),
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
    model = AttentionTCN(
        input_channels=input_channels,
        output_channels=output_channels,
        num_levels=train_config.num_levels,
        num_side_lvl=shaper_config.num_side_lvl,
        target_side_width=shaper_config.look_ahead_side_width,
        attention_heads=4,
        attention_dim=32,
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
) -> AsyncGenerator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float], None]:
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