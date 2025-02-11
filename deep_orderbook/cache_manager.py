import hashlib
import json
from pathlib import Path
import numpy as np
import asyncio
from typing import Optional, Tuple, List
from deep_orderbook.config import ShaperConfig
from deep_orderbook.utils import logger


class ArrayCollector:
    """Collects arrays for a single file and handles caching."""

    def __init__(self, cache: 'ArrayCache', data_file: Optional[Path] = None):
        self.cache = cache
        self.current_file = data_file
        self.all_books: List[np.ndarray] = []
        self.all_prices: List[np.ndarray] = []
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

    def get_window(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the last window_size arrays."""
        window_books = np.stack(self.all_books[-window_size:])
        window_prices = np.stack(self.all_prices[-window_size:])
        return window_books, window_prices

    def has_full_window(self, size: int) -> bool:
        """Check if we have enough samples for a full window."""
        return len(self.all_books) >= size

    async def cache_arrays(self, shaper_config: ShaperConfig, shaper) -> None:
        """Cache the collected arrays if we have any."""
        if self.current_file and len(self.all_books) > 0:
            await self.cache.cache_complete_arrays(
                self.current_file,
                shaper_config,
                self.all_books,
                self.all_prices,
                shaper,
            )

    def reset(self, new_file: Optional[Path] = None) -> None:
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

    def create_collector(self, data_file: Optional[Path] = None) -> ArrayCollector:
        """Create a new array collector."""
        return ArrayCollector(self, data_file)

    def _get_config_hash(self, shaper_config: ShaperConfig) -> str:
        """Create a unique hash for the shaper config parameters that affect array generation"""
        relevant_params = {
            'zoom_frac': shaper_config.zoom_frac,
            'num_side_lvl': shaper_config.num_side_lvl,
            'rolling_window_size': 256, # fix. should not have been there but affected the hashes
            'look_ahead': shaper_config.look_ahead,
            'look_ahead_side_bips': shaper_config.look_ahead_side_bips,
            'look_ahead_side_width': shaper_config.look_ahead_side_width,
        }
        config_str = json.dumps(relevant_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _get_cache_path(self, data_file: Path, shaper_config: ShaperConfig) -> Path:
        """Generate cache file path based on input file and config"""
        config_hash = self._get_config_hash(shaper_config)
        cache_name = f"{data_file.stem}_{config_hash}.npz"
        logger.info(f"Getting cache path for {data_file}: {cache_name}")  
        return self.cache_dir / cache_name

    def load_cached(
        self, data_file: Path, shaper_config: ShaperConfig
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load cached arrays if they exist"""
        cache_path = self._get_cache_path(data_file, shaper_config)
        # logger.info(f"Loading cached arrays from {cache_path}")
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path) as data:
                books_array = data['books_array']
                time_levels = data['time_levels']
                prices_array = data['prices_array']
            logger.debug(f"Loaded cached arrays of length {len(books_array)} from {cache_path}")
            return books_array, time_levels, prices_array
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def save_to_cache(
        self,
        data_file: Path,
        shaper_config: ShaperConfig,
        books_array: np.ndarray,
        time_levels: np.ndarray,
        prices_array: np.ndarray,
    ) -> None:
        """Save processed arrays to cache"""
        cache_path = self._get_cache_path(data_file, shaper_config)
        np.savez_compressed(
            cache_path,
            books_array=books_array,
            time_levels=time_levels,
            prices_array=prices_array,
        )

    async def cache_complete_arrays(
        self,
        data_file: Path,
        shaper_config: ShaperConfig,
        all_books: list[np.ndarray],
        all_prices: list[np.ndarray],
        shaper,
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
                full_books_array,
                full_time_levels,
                full_prices_array,
            )
            logger.info(
                f"Successfully cached {len(full_books_array)} timesteps from {data_file}"
            )
        except Exception as e:
            logger.error(f"Failed to cache data for {data_file}: {e}")

    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear old cache files"""
        if older_than_days is not None:
            import time

            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.npz"):
                if (current_time - cache_file.stat().st_mtime) > (
                    older_than_days * 86400
                ):
                    cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.npz"):
                cache_file.unlink()


async def cache_manager_main():
    from deep_orderbook.config import ReplayConfig, ShaperConfig
    from deep_orderbook.shaper import iter_shapes_t2l
    from tqdm.auto import tqdm
    replay_conf = ReplayConfig(
        markets=["ETH-USD"],  # , "BTC-USD", "ETH-BTC"],
        data_dir='/media/photoDS216/crypto/',
        date_regexp='2024-10-2*',
        max_samples=-1,
        every="1000ms",
    )
    shaper_config = ShaperConfig(
        only_full_arrays=False,
        zoom_frac=0.002,
        num_side_lvl=8,
        look_ahead=32,
        look_ahead_side_bips=10,
        look_ahead_side_width=4,
        rolling_window_size=256,
    )

    print(f"Configs: \n{shaper_config}\n{replay_conf}")
    async for books_array, t2l_array, pxar in tqdm(iter_shapes_t2l(
        replay_config=replay_conf,
        shaper_config=shaper_config,
    )):
        pass


if __name__ == "__main__":
    asyncio.run(cache_manager_main())
