"""Continuity + pipeline-guard smoke tests.

These tests exercise the replayer -> feed -> shaper path on real 2026
parquet data (mounted at ``/mnt/data/repos/gaelreinaudi/crypto``). If the
mount is not reachable the tests skip with a clear reason so the rest of
the test suite still runs in isolated CI environments.

The slow, I/O-heavy tests are gated behind the ``not slow`` pytest
selector via the ``@pytest.mark.slow`` marker (see ``pytest.ini``).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.pipeline_guards import (
    aggregate_per_window_strategy_result,
    assert_image_meaningful,
    assert_non_overlapping,
    dt_seconds_from_every,
    select_non_overlapping_indices,
)
from deep_orderbook.shaper import iter_shapes_t2l


CRYPTO_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")


def _two_adjacent_2026_files() -> list[Path]:
    if not CRYPTO_DIR.exists():
        return []
    files = sorted(CRYPTO_DIR.glob("2026-04-16T1[6-7]-00-*.parquet"))
    return files[:2]


def _default_configs(rolling: int = 2048, look_ahead: int = 128) -> tuple[ShaperConfig, dict]:
    shaper = ShaperConfig(
        only_full_arrays=True,
        view_bips=5,
        num_side_lvl=8,
        look_ahead=look_ahead,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=rolling,
        window_stride=8,
        use_cache=True,
        save_cache=True,
    )
    return shaper, {"every": "100ms"}


# ---------- fast unit tests (no I/O) ----------


def test_dt_seconds_from_every_parses_common_units() -> None:
    assert dt_seconds_from_every("100ms") == pytest.approx(0.1)
    assert dt_seconds_from_every("1000ms") == pytest.approx(1.0)
    assert dt_seconds_from_every("1s") == pytest.approx(1.0)
    assert dt_seconds_from_every("250") == pytest.approx(0.25)  # default ms
    assert dt_seconds_from_every("1m") == pytest.approx(60.0)


def test_dt_seconds_from_every_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        dt_seconds_from_every("nope")
    with pytest.raises(TypeError):
        dt_seconds_from_every(100)  # type: ignore[arg-type]


def test_assert_image_meaningful_passes_for_new_default() -> None:
    shaper = ShaperConfig(rolling_window_size=2048, look_ahead=128)
    replay = ReplayConfig(every="100ms")
    geom = assert_image_meaningful(shaper, replay)
    assert geom["verdict"] == "ok"
    assert geom["context_seconds"] == pytest.approx(204.8)
    assert geom["context_to_horizon_ratio"] == pytest.approx(16.0)


def test_assert_image_meaningful_fails_for_legacy_microburst() -> None:
    shaper = ShaperConfig(rolling_window_size=256, look_ahead=64)
    replay = ReplayConfig(every="100ms")
    with pytest.raises(ValueError, match="assert_image_meaningful failed"):
        assert_image_meaningful(shaper, replay)


def test_assert_image_meaningful_allows_explicit_microburst_override() -> None:
    shaper = ShaperConfig(rolling_window_size=256, look_ahead=64)
    replay = ReplayConfig(every="100ms")
    geom = assert_image_meaningful(shaper, replay, allow_microburst=True)
    assert geom["verdict"] == "microburst_override"


def test_assert_non_overlapping_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="assert_non_overlapping failed"):
        assert_non_overlapping(n_windows=10, window_size=2048, stride=8)


def test_assert_non_overlapping_accepts_disjoint() -> None:
    assert_non_overlapping(n_windows=10, window_size=2048, stride=2048)


def test_select_non_overlapping_indices_picks_disjoint_subset() -> None:
    idx = select_non_overlapping_indices(stride=8, rolling_window=2048, n_windows=512)
    assert idx[0] == 0
    diffs = np.diff(np.asarray(idx))
    assert (diffs == 256).all(), f"expected step=256, got {diffs[:5]}"
    assert idx[-1] + (2048 // 8) <= 512 + (2048 // 8)


def test_select_non_overlapping_indices_passthrough_when_already_disjoint() -> None:
    idx = select_non_overlapping_indices(stride=2048, rolling_window=2048, n_windows=4)
    assert idx == [0, 1, 2, 3]


def test_aggregate_per_window_strategy_result_is_additive() -> None:
    rng = np.random.default_rng(7)
    N, T, K = 3, 64, 4
    px = np.stack(
        [np.cumsum(rng.normal(scale=0.01, size=(T, 2)), axis=0) + 100.0 for _ in range(N)]
    ).astype(np.float32)
    pred = rng.uniform(size=(N, T, 2 * K)).astype(np.float32)

    def fake_strategy(px_w, pred_w, **_kwargs):
        # deterministic: pnl = len * 0.5, two trades per window
        pnl = np.linspace(0.0, 0.5, num=px_w.shape[0], dtype=np.float32)
        pos = np.zeros(px_w.shape[0], dtype=np.int8)
        return {
            "final_pnl": float(pnl[-1]),
            "pnl": pnl,
            "positions": pos,
            "trade_count": 2,
            "avg_hold_steps": 3.0,
            "market_time_fraction": 0.1,
            "direction": "long",
        }

    out = aggregate_per_window_strategy_result(px, pred, fake_strategy)
    assert out["aggregation"] == "per_window"
    assert out["n_windows"] == N
    assert out["trade_count"] == 2 * N
    assert out["final_pnl"] == pytest.approx(0.5 * N)
    assert out["pnl"].shape == (N * T,)
    # stitched pnl must be monotonically non-decreasing here and end at final_pnl
    assert out["pnl"][-1] == pytest.approx(0.5 * N, rel=1e-5)
    assert (np.diff(out["pnl"]) >= -1e-6).all()


def test_aggregate_per_window_rejects_shape_mismatch() -> None:
    px = np.zeros((2, 10, 2), dtype=np.float32)
    pred = np.zeros((3, 10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="N="):
        aggregate_per_window_strategy_result(px, pred, lambda *a, **k: {})


# ---------- slow I/O tests over real parquet files ----------


async def _collect(
    replay_cfg: ReplayConfig,
    shaper_cfg: ShaperConfig,
    n_samples: int = 8,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    books_list: list[np.ndarray] = []
    t2l_list: list[np.ndarray] = []
    px_list: list[np.ndarray] = []
    async for books, t2l, px in iter_shapes_t2l(replay_cfg, shaper_cfg, live=False):
        books_list.append(np.asarray(books))
        t2l_list.append(np.asarray(t2l))
        px_list.append(np.asarray(px))
        if len(books_list) >= n_samples:
            break
    return books_list, t2l_list, px_list


@pytest.mark.slow
def test_cross_file_continuity_no_nan_prices() -> None:
    files = _two_adjacent_2026_files()
    if len(files) < 2:
        pytest.skip(f"need 2 adjacent 2026 parquet files under {CRYPTO_DIR}")

    shaper_cfg, replay_overrides = _default_configs()

    per_file_window_counts: list[int] = []
    total_books: list[np.ndarray] = []
    total_px: list[np.ndarray] = []
    for f in files:
        replay_cfg = ReplayConfig(
            markets=["BTC-USD"],
            one_path=f,
            data_dir=f.parent,
            date_regexp=f.stem,
            max_samples=-1,
            **replay_overrides,
        )
        books_list, _t2l, px_list = asyncio.run(
            _collect(replay_cfg, shaper_cfg, n_samples=4)
        )
        per_file_window_counts.append(len(books_list))
        total_books.extend(books_list)
        total_px.extend(px_list)

    assert sum(per_file_window_counts) == len(total_books)
    for px in total_px:
        assert np.isfinite(px).all(), "NaN/inf in price window"
        assert px.shape[0] == shaper_cfg.rolling_window_size
        # bid must be <= ask for every timestep in every window
        assert (px[:, 0] <= px[:, 1] + 1e-6).all()


@pytest.mark.slow
def test_window_internal_continuity_is_monotonic_in_price_support() -> None:
    files = _two_adjacent_2026_files()
    if not files:
        pytest.skip(f"need at least 1 2026 parquet file under {CRYPTO_DIR}")

    shaper_cfg, replay_overrides = _default_configs()
    f = files[0]
    replay_cfg = ReplayConfig(
        markets=["BTC-USD"],
        one_path=f,
        data_dir=f.parent,
        date_regexp=f.stem,
        max_samples=-1,
        **replay_overrides,
    )
    _books, _t2l, px = asyncio.run(_collect(replay_cfg, shaper_cfg, n_samples=2))
    assert px, "shaper yielded zero windows"
    mid = px[0].mean(axis=1)
    diffs = np.abs(np.diff(mid))
    pct = diffs / np.maximum(mid[:-1], 1e-6)
    # At 100ms cadence, tick-to-tick BTC price changes must stay well under 1%.
    assert float(pct.max()) < 0.01, (
        f"suspiciously large 100ms price jump inside a single window: "
        f"max |dPx|/Px = {float(pct.max()):.4f}"
    )
