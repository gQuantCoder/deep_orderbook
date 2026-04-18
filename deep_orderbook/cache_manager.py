import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from deep_orderbook.config import ReplayConfig, ShaperConfig

if TYPE_CHECKING:
    from deep_orderbook.shaper import ArrayShaper


class ArrayCollector:
    """Collects arrays for a single file and handles caching."""

    def __init__(self, cache: "ArrayCache", data_file: Path | None = None):
        self.cache = cache
        self.current_file = data_file
        self.all_books: list[np.ndarray] = []
        self.all_prices: list[np.ndarray] = []
        self.window_counter = 0

    def add_arrays(self, books: np.ndarray, prices: np.ndarray, stride: int) -> bool:
        """Add arrays and return True when stride is reached."""
        self.all_books.append(books)
        self.all_prices.append(prices)
        self.window_counter += 1

        if self.window_counter >= stride:
            self.window_counter = 0
            return True
        return False

    def get_window(self, window_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the last window_size arrays."""
        window_books = np.stack(self.all_books[-window_size:])
        window_prices = np.stack(self.all_prices[-window_size:])
        return window_books, window_prices

    def has_full_window(self, size: int) -> bool:
        """Check if we have enough samples for a full window."""
        return len(self.all_books) >= size

    async def cache_arrays(
        self,
        shaper_config: ShaperConfig,
        shaper: "ArrayShaper",
        replay_config: ReplayConfig,
    ) -> None:
        """Cache the collected arrays if we have any."""
        if self.current_file and len(self.all_books) > 0:
            await self.cache.cache_complete_arrays(
                self.current_file,
                shaper_config,
                replay_config,
                self.all_books,
                self.all_prices,
                shaper,
            )

    def reset(self, new_file: Path | None = None) -> None:
        """Reset the collector for a new file."""
        self.current_file = new_file
        self.all_books = []
        self.all_prices = []
        self.window_counter = 0
        if new_file:
            logger.info(f"Processing new file: {new_file}")


class ArrayCache:
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def create_collector(self, data_file: Path | None = None) -> ArrayCollector:
        """Create a new array collector."""
        return ArrayCollector(self, data_file)

    def _get_config_hash(self, shaper_config: ShaperConfig, replay_config: ReplayConfig) -> str:
        """Create a unique hash for the shaper config parameters that affect array generation"""
        market = replay_config.markets[0].replace("-", "")
        return f"{market}_{replay_config.every}_vb{shaper_config.view_bips:02d}_lv{shaper_config.num_side_lvl:02d}_la{shaper_config.look_ahead:03d}_lasb{shaper_config.look_ahead_side_bips:02d}_lalv{shaper_config.look_ahead_side_width:02d}"

    def _get_cache_path(self, data_file: Path, shaper_config: ShaperConfig, replay_config: ReplayConfig) -> Path:
        """Generate cache file path based on input file and config.

        Creates a hierarchical structure: {cache_dir}/{config_hash}/{year}-{month}/{filename}.npz
        """
        config_hash = self._get_config_hash(shaper_config, replay_config)

        # Try to extract date from filename, assuming format contains year-month
        try:
            # filenames are timestamps: 2026-04-18T15-00-00 → year_month = 2026-04
            date_parts = next((p for p in data_file.stem.split("_") if "-" in p), "").split("-")
            year_month = f"{date_parts[0]}-{date_parts[1]}"
        except (IndexError, ValueError):
            year_month = "unknown"

        # Create the hierarchical directory structure
        cache_subdir = self.cache_dir / config_hash / year_month
        cache_subdir.mkdir(parents=True, exist_ok=True)

        cache_name = f"{data_file.stem}.npz"
        logger.info(f"Getting cache path for {data_file}: {cache_subdir}/{cache_name}")
        return cache_subdir / cache_name

    def load_cached(
        self, data_file: Path, shaper_config: ShaperConfig, replay_config: ReplayConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Load cached arrays if they exist"""
        cache_path = self._get_cache_path(data_file, shaper_config, replay_config)
        logger.info(f"Loading cached arrays from {cache_path}")
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path) as data:
                books_array = data["books_array"]
                time_levels = data["time_levels"]
                prices_array = data["prices_array"]
            logger.debug(f"Loaded cached arrays of length {len(books_array)} from {cache_path}")
            return books_array, time_levels, prices_array
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def save_to_cache(
        self,
        data_file: Path,
        shaper_config: ShaperConfig,
        replay_config: ReplayConfig,
        books_array: np.ndarray,
        time_levels: np.ndarray,
        prices_array: np.ndarray,
    ) -> None:
        """Save processed arrays to cache"""
        cache_path = self._get_cache_path(data_file, shaper_config, replay_config)
        np.savez_compressed(
            cache_path,
            books_array=books_array,
            time_levels=time_levels,
            prices_array=prices_array,
        )
        logger.info(f"Saved {len(books_array)} timesteps to {cache_path}")

    async def cache_complete_arrays(
        self,
        data_file: Path,
        shaper_config: ShaperConfig,
        replay_config: ReplayConfig,
        all_books: list[np.ndarray],
        all_prices: list[np.ndarray],
        shaper: "ArrayShaper",
    ) -> None:
        """Cache a complete set of arrays for a file, computing time_levels at the end."""
        if len(all_books) == 0:
            return

        try:
            full_books_array = np.stack(all_books)
            full_prices_array = np.stack(all_prices)

            # Compute time_levels for the entire array at once
            shaper.prices_array = full_prices_array
            full_time_levels = await shaper.build_time_level_trade()

            self.save_to_cache(
                data_file,
                shaper_config,
                replay_config,
                full_books_array,
                full_time_levels,
                full_prices_array,
            )
            logger.info(f"Successfully cached {len(full_books_array)} timesteps from {data_file}")
        except Exception as e:
            logger.error(f"Failed to cache data for {data_file}: {e}")

    # def clear_cache(self, older_than_days: Optional[int] = None) -> None:
    #     """Clear old cache files"""
    #     if older_than_days is not None:
    #         import time

    #         current_time = time.time()
    #         for cache_file in self.cache_dir.glob("*.npz"):
    #             if (current_time - cache_file.stat().st_mtime) > (
    #                 older_than_days * 86400
    #             ):
    #                 cache_file.unlink()
    #     else:
    #         for cache_file in self.cache_dir.glob("*.npz"):
    #             cache_file.unlink()


async def cache_manager_main() -> None:
    from tqdm.auto import tqdm

    from deep_orderbook.config import ReplayConfig, ShaperConfig
    from deep_orderbook.shaper import iter_shapes_t2l

    # replay_conf = ReplayConfig(
    #     markets=["ETH-USD"],  # , "BTC-USD", "ETH-BTC"],
    #     data_dir='/media/photoDS216/crypto/',
    #     date_regexp='2024-11*',
    #     max_samples=-1,
    #     every="1000ms",
    # )
    # shaper_config = ShaperConfig(
    #     only_full_arrays=False,
    #     view_bips=20,
    #     num_side_lvl=8,
    #     look_ahead=32,
    #     look_ahead_side_bips=10,
    #     look_ahead_side_width=4,
    #     rolling_window_size=256,
    # )

    replay_conf = ReplayConfig(
        markets=["ETH-USD"],  # , "BTC-USD", "ETH-BTC"],
        data_dir=Path("/media/photoDS216/crypto/"),
        date_regexp="2025-02-01T01*",
        max_samples=-1,
        every="100ms",
    )
    shaper_config = ShaperConfig(
        only_full_arrays=False,
        view_bips=5,
        num_side_lvl=8,
        look_ahead=32,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=1024,
        window_stride=8,
    )

    print(f"Configs: \n{shaper_config}\n{replay_conf}")
    async for books_array, t2l_array, pxar in tqdm(
        iter_shapes_t2l(
            replay_config=replay_conf,
            shaper_config=shaper_config,
        )
    ):
        pass


if __name__ == "__main__":
    asyncio.run(cache_manager_main())
