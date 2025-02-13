import numpy as np
from typing import Tuple, List
import asyncio


class Strategy:
    def __init__(
        self,
        threshold: float = 0.5,
        reentry_cooldown_steps: int = 20,  # Number of steps to wait before re-entering after exit
    ):
        """Initialize the strategy.

        Args:
            threshold: Threshold for the level_proximity prediction to trigger a position.
                      Higher threshold means requiring stronger prediction of imminent crossing.
            reentry_cooldown_steps: Number of steps to wait after closing before re-entering
        """
        self.threshold = threshold
        self.reentry_cooldown_steps = reentry_cooldown_steps
        self.position = 0  # Current position: 0 = flat, 1 = long
        self.steps_since_exit = 0  # Steps since last exit
        self.recent_up_proximities: List[float] = []  # Recent up proximity signals
        self.recent_down_proximities: List[float] = []  # Recent down proximity signals

    def should_get_long(self, level_proximity: np.ndarray) -> bool:
        """Decide whether to enter a long position based on level proximity predictions.

        Args:
            level_proximity: Array of shape (2*side_width, 1) containing predicted level proximities.
                           Higher values mean the level is expected to be reached sooner.

        Returns:
            bool: True if should enter long position
        """
        # Don't enter if we haven't waited long enough since last exit
        if self.steps_since_exit < self.reentry_cooldown_steps:
            return False

        # Get the predictions for upward price movements (second half of array)
        up_predictions = level_proximity[self.side_width :, 0]
        # Get the predictions for downward price movements (first half of array)
        down_predictions = level_proximity[: self.side_width, 0]

        # Use max values instead of means to be more conservative
        up_proximity = np.max(up_predictions)
        down_proximity = np.max(down_predictions)
        # print(level_proximity)
        # print(f"{up_proximity=}, {down_proximity=}")

        # Enter long if:
        # 1. Up moves are predicted to be significantly more proximate than down moves
        # 2. Up move proximity is strong in absolute terms
        # 3. Down moves are not too threatening
        return (
            up_proximity > self.threshold  # Strong absolute signal
            # and up_proximity > 2.0 * down_proximity  # Much stronger than down signals
            and down_proximity < 0.5
        )  # No strong contrary signals

    def should_get_flat(self, level_proximity: np.ndarray) -> bool:
        """Decide whether to exit to flat based on level proximity predictions.

        Args:
            level_proximity: Array of shape (2*side_width, 1) containing predicted level proximities.
                           Higher values mean the level is expected to be reached sooner.

        Returns:
            bool: True if should exit to flat
        """
        # Get the predictions for upward price movements (second half of array)
        up_predictions = level_proximity[self.side_width :, 0]
        # Get the predictions for downward price movements (first half of array)
        down_predictions = level_proximity[: self.side_width, 0]

        # Use max values instead of means
        up_proximity = np.max(up_predictions)
        down_proximity = np.max(down_predictions)

        # Exit long if:
        # 1. Down moves are predicted to be stronger than up moves
        # 2. Up signal has weakened significantly
        # 3. Or down signal is strong in absolute terms
        return not self.should_get_long(level_proximity) and (
            # down_proximity > up_proximity  # Down stronger than up
            # up_proximity < self.threshold * 0.5  # Up signal weakened
            down_proximity > self.threshold
        )  # Strong down signal

    def compute_proximities(self, level_proximity: np.ndarray) -> tuple[float, float]:
        """Compute up and down proximities from level proximity predictions.
        
        Args:
            level_proximity: Array of shape (2*side_width, 1) containing predicted level proximities.
            
        Returns:
            Tuple[float, float]: (up_proximity, down_proximity)
        """
        up_predictions = level_proximity[self.side_width:, 0]
        down_predictions = level_proximity[:self.side_width, 0]
        
        up_proximity = np.mean(up_predictions)
        down_proximity = np.mean(down_predictions)
        
        return up_proximity, down_proximity

    def compute_pnl(
        self, prices: np.ndarray, level_proximity_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute PnL based on price movements and level proximity predictions.

        Args:
            prices: Array of shape (T, 2) containing [bid, ask] prices over time
            level_proximity_pred: Array of shape (T, 2*side_width, 1) containing predicted level proximities.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (pnl_array, position_array, up_proximities, down_proximities)
            - pnl_array: Array of shape (T,) containing cumulative PnL at each timestep
            - position_array: Array of shape (T,) containing position at each timestep
            - up_proximities: Array of shape (T,) containing up proximity signals
            - down_proximities: Array of shape (T,) containing down proximity signals
        """
        T = len(prices)
        pnl = np.zeros(T)
        positions = np.zeros(T)
        up_proximities = np.zeros(T)
        down_proximities = np.zeros(T)
        entry_price = 0.0
        self.position = 0
        self.steps_since_exit = self.reentry_cooldown_steps  # Start ready to trade
        self.side_width = level_proximity_pred.shape[1] // 2

        for t in range(T):
            # Store current position
            positions[t] = self.position

            # Get current level proximity predictions and compute proximities
            curr_lp = level_proximity_pred[t]
            up_proximities[t], down_proximities[t] = self.compute_proximities(curr_lp)

            # First copy previous PnL
            if t > 0:
                pnl[t] = pnl[t - 1]

            # Update position based on predictions
            if self.position == 0 and self.should_get_long(curr_lp):
                self.position = 1
                entry_price = prices[t, 1]  # Enter at ask
                # Add entry cost to PnL (mark to market at bid)
                pnl[t] += prices[t, 0] - entry_price
            elif self.position == 1 and self.should_get_flat(curr_lp):
                self.position = 0
                self.steps_since_exit = 0  # Reset cooldown counter

            # If we're in a position, track the changes in bid price
            elif self.position == 1 and t > 0:  # If long and not first timestep
                pnl[t] += prices[t, 0] - prices[t - 1, 0]  # Track changes in bid price

            # Update cooldown counter if we're flat
            if self.position == 0:
                self.steps_since_exit += 1

        return pnl, positions, up_proximities, down_proximities


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
        pnl, positions, up_proximities, down_proximities = strategy.compute_pnl(pxar, t2l_array)
        # print(t2l_array)
        print(positions)
        print(pnl)
        # break


if __name__ == "__main__":
    asyncio.run(main())
