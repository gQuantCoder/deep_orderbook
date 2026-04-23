"""exp44 — broad April stress-filter short replay with frozen 2D mapper setup.

Hypothesis
----------
If the earlier evening short branch was really a stress-conditioned downside continuation
phenomenon rather than a fragile clock-time pocket, then a broader April 2026 train pool
spanning multiple intraday phases should still produce tradable short routes once we keep
only present-time high-stress windows (realized move/range/vol/book impulse) on the next
chronological evening holdout.

Single main mutation vs exp42/exp43:
- replace hour-of-day regime slicing with a broader April train pool + observable stress filter
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep short-side replay and trigger sweep

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/exp44_april_broad_stress_short_replay.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from scripts.exp16_batch_h64_tcn import DEFAULT_SHAPER_CONFIG_2026
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")

DEFAULT_TRAIN_FILES = [
    DATA_DIR / "2026-04-14T00-00-01.parquet",
    DATA_DIR / "2026-04-14T06-00-00.parquet",
    DATA_DIR / "2026-04-14T13-00-00.parquet",
    DATA_DIR / "2026-04-14T18-00-00.parquet",
    DATA_DIR / "2026-04-15T00-00-00.parquet",
    DATA_DIR / "2026-04-15T06-00-00.parquet",
    DATA_DIR / "2026-04-15T12-00-00.parquet",
    DATA_DIR / "2026-04-15T18-00-10.parquet",
    DATA_DIR / "2026-04-16T00-00-00.parquet",
    DATA_DIR / "2026-04-16T06-00-01.parquet",
    DATA_DIR / "2026-04-16T12-00-01.parquet",
    DATA_DIR / "2026-04-16T18-00-01.parquet",
    DATA_DIR / "2026-04-17T00-00-01.parquet",
    DATA_DIR / "2026-04-17T06-00-01.parquet",
    DATA_DIR / "2026-04-17T12-00-00.parquet",
    DATA_DIR / "2026-04-17T18-00-11.parquet",
    DATA_DIR / "2026-04-18T00-00-01.parquet",
    DATA_DIR / "2026-04-18T06-00-00.parquet",
    DATA_DIR / "2026-04-18T12-00-00.parquet",
    DATA_DIR / "2026-04-18T18-00-00.parquet",
]
DEFAULT_TEST_FILES = [DATA_DIR / "2026-04-19T20-00-00.parquet"]

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


async def _run() -> None:
    await exp22_main(
        variants=["regonly_l1_2d_h8"],
        train_files=DEFAULT_TRAIN_FILES,
        test_files=DEFAULT_TEST_FILES,
        label="exp44_april_broad_stress_short_replay_cpu",
        directions=["short"],
        n_samples_per_file=None,
        load_every="1000ms",
        shaper_config_override=SHAPER,
    )


if __name__ == "__main__":
    asyncio.run(_run())
