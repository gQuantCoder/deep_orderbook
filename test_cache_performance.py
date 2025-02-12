import asyncio
import time
from pathlib import Path
import numpy as np

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l
from deep_orderbook.cache_manager import ArrayCache
from deep_orderbook.utils import logger


async def measure_performance(
    use_cache: bool = True, clear_cache: bool = False, stride: int = 32
):
    """Measure performance with and without cache"""
    replay_config = ReplayConfig(
        data_dir=Path("/media/photoDS216/crypto/"),
        date_regexp="2024-08-08T03-55*",
        # max_samples=256,  # Reduced to 500 samples
    )
    shaper_config = ShaperConfig(
        rolling_window_size=64,
        num_side_lvl=8,
        window_stride=stride,  # Use provided stride
        only_full_arrays=False,
        use_cache=use_cache,
    )

    print(f"{shaper_config=}")

    # Clear cache if requested (should only be done once at the start)
    if clear_cache:
        cache = ArrayCache()
        cache.clear_cache()
        logger.info("Cleared existing cache")

    # Debug: Print cache directory and files
    cache = ArrayCache()
    print(f"\nCache directory: {cache.cache_dir}")
    print(f"Cache directory exists: {cache.cache_dir.exists()}")
    print(f"Cache files before: {list(cache.cache_dir.glob('*.npz'))}")

    start_time = time.time()
    count = 0
    nan_count = 0

    async for books_array, time_levels, pxar in iter_shapes_t2l(
        replay_config=replay_config,
        shaper_config=shaper_config,
    ):
        count += 1
        if np.isnan(pxar).any():
            nan_count += 1
        if count % 50 == 0:  # More frequent logging
            logger.info(f"Processed {count} samples (NaNs: {nan_count})...")
        if replay_config.max_samples > 0 and count >= replay_config.max_samples:
            break

    # Debug: Print cache files after processing
    print(f"Cache files after: {list(cache.cache_dir.glob('*.npz'))}")

    end_time = time.time()
    duration = end_time - start_time

    return count, duration, nan_count


async def main():
    # Test different stride values
    strides = [1, 32]

    for stride in strides:
        print(f"\nTesting with stride = {stride}")
        print("=" * 60)

        # Clear cache at the start of each stride test
        cache = ArrayCache()
        # cache.clear_cache()
        logger.info("Cleared existing cache at start")

        # Run with cache (should create cache)
        logger.info("\nRunning with cache (first time, creating cache)...")
        count_no_cache, time_no_cache, nans_no_cache = await measure_performance(
            use_cache=False, stride=stride
        )

        # Run with cache again (should use existing cache)
        logger.info("\nRunning with cache (second time, using cached data)...")
        count_cache2, time_cache2, nans_cache2 = await measure_performance(
            use_cache=True, stride=stride
        )

        print(f"  - NaN windows: {nans_no_cache}")
        print(f"\nWith cache (second run):")
        print(
            f"  - {count_cache2} samples in {time_cache2:.2f}s ({count_cache2/time_cache2:.2f} samples/s)"
        )
        print(f"  - NaN windows: {nans_cache2}")
        if count_cache2 > 0:
            print(f"Cache speedup: {time_no_cache/time_cache2:.2f}x faster")
            print(f"Data reduction: {stride}x fewer samples")
        else:
            print("No valid comparison possible - no samples in cached run")


if __name__ == "__main__":
    asyncio.run(main())
