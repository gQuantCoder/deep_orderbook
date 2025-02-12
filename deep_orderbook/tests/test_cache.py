import pytest
import numpy as np
from pathlib import Path
import shutil
import asyncio

from deep_orderbook.cache_manager import ArrayCache
from deep_orderbook.config import ShaperConfig, ReplayConfig
from deep_orderbook.shaper import iter_shapes_t2l


@pytest.fixture
def test_cache_dir(tmp_path):
    """Create a temporary directory for cache files"""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    yield cache_dir
    # Cleanup after tests
    shutil.rmtree(cache_dir)


@pytest.fixture
def array_cache(test_cache_dir):
    """Create an ArrayCache instance with test directory"""
    return ArrayCache(cache_dir=test_cache_dir)


def test_config_hash_consistency(array_cache):
    """Test that same configs produce same hashes and different configs produce different hashes"""
    config1 = ShaperConfig(view_bips=40, num_side_lvl=32)
    config2 = ShaperConfig(view_bips=40, num_side_lvl=32)
    config3 = ShaperConfig(view_bips=50, num_side_lvl=32)

    hash1 = array_cache._get_config_hash(config1)
    hash2 = array_cache._get_config_hash(config2)
    hash3 = array_cache._get_config_hash(config3)

    assert hash1 == hash2, "Same configs should produce same hashes"
    assert hash1 != hash3, "Different configs should produce different hashes"


def test_cache_path_generation(array_cache):
    """Test cache path generation"""
    config = ShaperConfig()
    data_file = Path("test_data.parquet")
    
    cache_path = array_cache._get_cache_path(data_file, config)
    assert cache_path.parent == array_cache.cache_dir
    assert cache_path.suffix == ".npz"
    assert data_file.stem in str(cache_path)


def test_save_and_load_cache(array_cache):
    """Test saving and loading arrays from cache"""
    config = ShaperConfig()
    data_file = Path("test_data.parquet")
    
    # Create test arrays
    books_array = np.random.rand(100, 64, 3)
    time_levels = np.random.rand(100, 32, 1)
    prices_array = np.random.rand(100, 2)
    
    # Save to cache
    array_cache.save_to_cache(data_file, config, books_array, time_levels, prices_array)
    
    # Load from cache
    loaded = array_cache.load_cached(data_file, config)
    assert loaded is not None
    
    loaded_books, loaded_times, loaded_prices = loaded
    np.testing.assert_array_equal(loaded_books, books_array)
    np.testing.assert_array_equal(loaded_times, time_levels)
    np.testing.assert_array_equal(loaded_prices, prices_array)


def test_clear_cache(array_cache):
    """Test cache clearing functionality"""
    config = ShaperConfig()
    
    # Create multiple cache files
    for i in range(3):
        data_file = Path(f"test_data_{i}.parquet")
        books_array = np.random.rand(10, 64, 3)
        time_levels = np.random.rand(10, 32, 1)
        prices_array = np.random.rand(10, 2)
        array_cache.save_to_cache(data_file, config, books_array, time_levels, prices_array)
    
    # Verify files exist
    assert len(list(array_cache.cache_dir.glob("*.npz"))) == 3
    
    # Clear cache
    array_cache.clear_cache()
    
    # Verify files are gone
    assert len(list(array_cache.cache_dir.glob("*.npz"))) == 0


@pytest.mark.asyncio
async def test_iter_shapes_with_cache():
    """Test the integration of caching with iter_shapes_t2l"""
    # Setup test config
    replay_config = ReplayConfig(
        data_dir=Path("/media/photoDS216/crypto/"),  # Use your actual data directory
        date_regexp="2024-08-04",  # Use a date you know exists
        max_samples=10  # Limit samples for testing
    )
    shaper_config = ShaperConfig(
        rolling_window_size=32,  # Smaller window for testing
        num_side_lvl=16,  # Fewer levels for testing
        use_cache=True,
    )
    
    # First run - should create cache
    first_run_data = []
    async for books_array, time_levels, pxar in iter_shapes_t2l(
        replay_config=replay_config,
        shaper_config=shaper_config,
    ):
        first_run_data.append((books_array.copy(), time_levels.copy(), pxar.copy()))
        if len(first_run_data) >= 5:  # Limit data for testing
            break
    
    # Second run - should use cache
    second_run_data = []
    async for books_array, time_levels, pxar in iter_shapes_t2l(
        replay_config=replay_config,
        shaper_config=shaper_config,
    ):
        second_run_data.append((books_array.copy(), time_levels.copy(), pxar.copy()))
        if len(second_run_data) >= 5:  # Limit data for testing
            break
    
    # Compare results
    assert len(first_run_data) == len(second_run_data)
    for (books1, times1, px1), (books2, times2, px2) in zip(first_run_data, second_run_data):
        np.testing.assert_array_almost_equal(books1, books2)
        np.testing.assert_array_almost_equal(times1, times2)
        np.testing.assert_array_almost_equal(px1, px2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 