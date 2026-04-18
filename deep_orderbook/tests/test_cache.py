import asyncio
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from deep_orderbook.cache_manager import ArrayCache
from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


def _default_replay(markets: list[str] = ["BTC-USD"]) -> ReplayConfig:
    return ReplayConfig(markets=markets)


@pytest.fixture
def test_cache_dir(tmp_path: Path) -> Iterator[Path]:
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    yield cache_dir
    shutil.rmtree(cache_dir)


@pytest.fixture
def array_cache(test_cache_dir: Path) -> ArrayCache:
    return ArrayCache(cache_dir=test_cache_dir)


def test_config_hash_consistency(array_cache: ArrayCache) -> None:
    """Same config → same hash; different config → different hash."""
    replay = _default_replay()
    config1 = ShaperConfig(view_bips=40, num_side_lvl=32)
    config2 = ShaperConfig(view_bips=40, num_side_lvl=32)
    config3 = ShaperConfig(view_bips=50, num_side_lvl=32)

    assert array_cache._get_config_hash(config1, replay) == array_cache._get_config_hash(config2, replay)
    assert array_cache._get_config_hash(config1, replay) != array_cache._get_config_hash(config3, replay)


def test_config_hash_includes_market(array_cache: ArrayCache) -> None:
    """Different markets must produce different hashes (the market-in-hash bug)."""
    config = ShaperConfig()
    btc_hash = array_cache._get_config_hash(config, _default_replay(["BTC-USD"]))
    eth_hash = array_cache._get_config_hash(config, _default_replay(["ETH-USD"]))
    assert btc_hash != eth_hash, "BTC and ETH must not share a cache namespace"
    assert "BTCUSD" in btc_hash
    assert "ETHUSD" in eth_hash


def test_cache_path_generation(array_cache: ArrayCache) -> None:
    """Cache path is inside cache_dir, has .npz extension, and includes the stem."""
    config = ShaperConfig()
    replay = _default_replay()
    data_file = Path("2026-04-18T10-00-00.parquet")

    cache_path = array_cache._get_cache_path(data_file, config, replay)
    assert cache_path.suffix == ".npz"
    assert data_file.stem in str(cache_path)
    assert str(array_cache.cache_dir) in str(cache_path)
    # hierarchical: cache_dir / hash / year-month / stem.npz
    assert cache_path.parent.parent.parent == array_cache.cache_dir


def test_cache_path_year_month_extraction(array_cache: ArrayCache) -> None:
    """Year-month is correctly extracted from timestamp filenames."""
    config = ShaperConfig()
    replay = _default_replay()
    path = array_cache._get_cache_path(Path("2026-04-18T10-00-00.parquet"), config, replay)
    assert "2026-04" in str(path)


def test_cache_path_unknown_date(array_cache: ArrayCache) -> None:
    """Files without parseable dates fall back to 'unknown' rather than crashing."""
    config = ShaperConfig()
    replay = _default_replay()
    path = array_cache._get_cache_path(Path("nodatefile.parquet"), config, replay)
    assert "unknown" in str(path)


def test_save_and_load_cache(array_cache: ArrayCache) -> None:
    """Round-trip: saved arrays are loaded back byte-identical."""
    config = ShaperConfig()
    replay = _default_replay()
    data_file = Path("2026-01-01T00-00-00.parquet")

    books = np.random.rand(100, 16, 3).astype(np.float32)
    t2l = np.random.rand(100, 8, 1).astype(np.float32)
    prices = np.random.rand(100, 2)

    array_cache.save_to_cache(data_file, config, replay, books, t2l, prices)

    loaded = array_cache.load_cached(data_file, config, replay)
    assert loaded is not None
    loaded_books, loaded_t2l, loaded_prices = loaded
    np.testing.assert_array_equal(loaded_books, books)
    np.testing.assert_array_equal(loaded_t2l, t2l)
    np.testing.assert_array_equal(loaded_prices, prices)


def test_cache_miss_returns_none(array_cache: ArrayCache) -> None:
    """load_cached returns None when no cache file exists."""
    result = array_cache.load_cached(
        Path("2099-01-01T00-00-00.parquet"), ShaperConfig(), _default_replay()
    )
    assert result is None


def test_clear_cache(array_cache: ArrayCache) -> None:
    """clear_cache removes all .npz files and returns the correct count."""
    config = ShaperConfig()
    replay = _default_replay()

    for i in range(3):
        data_file = Path(f"2026-0{i+1}-01T00-00-00.parquet")
        array_cache.save_to_cache(data_file, config, replay,
                                   np.zeros((10, 16, 3)), np.zeros((10, 8, 1)), np.zeros((10, 2)))

    assert len(list(array_cache.cache_dir.rglob("*.npz"))) == 3
    removed = array_cache.clear_cache()
    assert removed == 3
    assert len(list(array_cache.cache_dir.rglob("*.npz"))) == 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_iter_shapes_with_cache() -> None:
    """Cold run then warm run produce identical frames (cache round-trip)."""
    replay_config = ReplayConfig(
        data_dir=Path("/media/photoDS216/crypto/"),
        date_regexp="2024-08-04",
        max_samples=10,
    )
    shaper_config = ShaperConfig(rolling_window_size=32, num_side_lvl=16, use_cache=True)

    first_run = []
    async for books, t2l, px in iter_shapes_t2l(replay_config, shaper_config):
        first_run.append((books.copy(), t2l.copy(), px.copy()))
        if len(first_run) >= 5:
            break

    second_run = []
    async for books, t2l, px in iter_shapes_t2l(replay_config, shaper_config):
        second_run.append((books.copy(), t2l.copy(), px.copy()))
        if len(second_run) >= 5:
            break

    assert len(first_run) == len(second_run)
    for (b1, t1, p1), (b2, t2, p2) in zip(first_run, second_run):
        np.testing.assert_array_almost_equal(b1, b2)
        np.testing.assert_array_almost_equal(t1, t2)
        np.testing.assert_array_almost_equal(p1, p2)
