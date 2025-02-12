import numpy as np
from typing import Tuple
import asyncio


class Strategy:
    def __init__(self, side_width: int = 4, threshold: float = 0.5):
        """Initialize the strategy.

        Args:
            side_width: Number of price levels on each side
            threshold: Threshold for the level_proximity prediction to trigger a position.
                      Higher threshold means requiring stronger prediction of imminent crossing.
        """
        self.side_width = side_width
        self.threshold = threshold
        self.position = 0  # Current position: 0 = flat, 1 = long

    def should_get_long(self, level_proximity: np.ndarray) -> bool:
        """Decide whether to enter a long position based on level proximity predictions.

        Args:
            level_proximity: Array of shape (2*side_width, 1) containing predicted level proximities.
                           Higher values mean the level is expected to be reached sooner.

        Returns:
            bool: True if should enter long position
        """
        # Get the predictions for upward price movements (second half of array)
        up_predictions = level_proximity[self.side_width:, 0]
        # Get the predictions for downward price movements (first half of array)
        down_predictions = level_proximity[:self.side_width, 0]

        # Calculate average predicted proximity for up and down moves
        avg_up_proximity = np.mean(up_predictions)
        avg_down_proximity = np.mean(down_predictions)

        # Enter long if:
        # 1. Up moves are predicted to be more proximate than down moves
        # 2. Up move proximity is stronger than our threshold
        return avg_up_proximity > avg_down_proximity and avg_up_proximity > self.threshold

    def should_get_flat(self, level_proximity: np.ndarray) -> bool:
        """Decide whether to exit to flat based on level proximity predictions.

        Args:
            level_proximity: Array of shape (2*side_width, 1) containing predicted level proximities.
                           Higher values mean the level is expected to be reached sooner.

        Returns:
            bool: True if should exit to flat
        """
        # Get the predictions for upward price movements (second half of array)
        up_predictions = level_proximity[self.side_width:, 0]
        # Get the predictions for downward price movements (first half of array)
        down_predictions = level_proximity[:self.side_width, 0]

        # Calculate average predicted proximity for up and down moves
        avg_up_proximity = np.mean(up_predictions)
        avg_down_proximity = np.mean(down_predictions)

        # Exit long if:
        # 1. Down moves are predicted to be more proximate than up moves
        # 2. Down move proximity is stronger than our threshold
        return avg_down_proximity > avg_up_proximity and avg_down_proximity > self.threshold

    def compute_pnl(
        self, prices: np.ndarray, level_proximity_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PnL based on price movements and level proximity predictions.

        Args:
            prices: Array of shape (T, 2) containing [bid, ask] prices over time
            level_proximity_pred: Array of shape (T, 2*side_width, 1) containing predicted level proximities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (pnl_array, position_array)
            - pnl_array: Array of shape (T,) containing cumulative PnL at each timestep
            - position_array: Array of shape (T,) containing position at each timestep
        """
        T = len(prices)
        pnl = np.zeros(T)
        positions = np.zeros(T)
        entry_price = 0.0
        self.position = 0

        for t in range(T):
            # Store current position
            positions[t] = self.position

            # Get mid price
            mid_price = (prices[t, 0] + prices[t, 1]) / 2

            # Update PnL if we have a position
            if t > 0:
                if self.position == 1:  # If long
                    pnl[t] = pnl[t - 1] + (
                        mid_price - prices[t - 1, 1]
                    )  # Use ask price for entry
                else:
                    pnl[t] = pnl[t - 1]

            # Get current level proximity predictions
            curr_lp = level_proximity_pred[t]

            # Update position based on predictions
            if self.position == 0 and self.should_get_long(curr_lp):
                self.position = 1
                entry_price = prices[t, 1]  # Enter at ask
            elif self.position == 1 and self.should_get_flat(curr_lp):
                self.position = 0
                # Add exit PnL
                pnl[t] += prices[t, 0] - entry_price  # Exit at bid

        return pnl, positions


async def main():
    from deep_orderbook.config import ReplayConfig, ShaperConfig
    from deep_orderbook.shaper import iter_shapes_t2l

    replay_conf = ReplayConfig(
        markets=["ETH-USD"],  # , "BTC-USD", "ETH-BTC"],
        data_dir='/media/photoDS216/crypto/',
        date_regexp='2024-11-06T*',
        max_samples=-1,
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

    strategy = Strategy(threshold=0.2)
    async for books_array, t2l_array, pxar in iter_shapes_t2l(
        replay_config=replay_conf,
        shaper_config=shaper_config,
    ):
        pnl, positions = strategy.compute_pnl(pxar, t2l_array)
        # print(t2l_array)
        print(positions)
        print(pnl)
        # break


if __name__ == "__main__":
    asyncio.run(main())
