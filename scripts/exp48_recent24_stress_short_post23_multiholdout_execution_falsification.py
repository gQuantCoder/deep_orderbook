"""exp48 — broaden execution falsification across the first two post-23:00 holdouts for the recent stress-conditioned short branch.

Hypothesis
----------
If exp47's failure was primarily a threshold / selectivity issue rather than a dead short branch,
then keeping the mapper, train pool, and stress filter fixed while advancing to the first two later
chronological holdouts (2026-04-22T00:00:00 and 2026-04-22T01:00:00) should separate:
- no-trade failure at strict fast thresholds (`q95_*`), from
- bad-fill / one-trade failure in relaxed fast routes (`q90_*`),
and at least one neighboring holdout should still convert the honest map into non-negative short PnL.

Single main mutation vs exp47:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep the observable present-time stress filter and short-side replay fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- broaden the holdout test from one file to the next two chronological post-23:00 files

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp48_recent24_stress_short_post23_multiholdout_execution_falsification.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from scripts.exp16_batch_h64_tcn import DEFAULT_SHAPER_CONFIG_2026
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")

DEFAULT_TRAIN_FILES = [
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
    DATA_DIR / "2026-04-19T00-00-00.parquet",
    DATA_DIR / "2026-04-19T06-00-00.parquet",
    DATA_DIR / "2026-04-19T12-00-00.parquet",
    DATA_DIR / "2026-04-19T18-00-00.parquet",
    DATA_DIR / "2026-04-20T00-00-04.parquet",
    DATA_DIR / "2026-04-20T06-00-00.parquet",
    DATA_DIR / "2026-04-20T12-00-00.parquet",
    DATA_DIR / "2026-04-20T18-00-01.parquet",
    DATA_DIR / "2026-04-21T00-00-00.parquet",
    DATA_DIR / "2026-04-21T06-00-00.parquet",
    DATA_DIR / "2026-04-21T12-00-00.parquet",
    DATA_DIR / "2026-04-21T18-00-10.parquet",
]

HOLDOUTS = [
    DATA_DIR / "2026-04-22T00-00-00.parquet",
    DATA_DIR / "2026-04-22T01-00-00.parquet",
]

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


async def _run_one(holdout: Path) -> None:
    holdout_tag = holdout.stem.replace(":", "").replace("-", "")
    label = f"exp48_recent24_stress_short_post23_multiholdout_execution_falsification_{holdout_tag}"
    await exp22_main(
        variants=["regonly_l1_2d_h8"],
        train_files=DEFAULT_TRAIN_FILES,
        test_files=[holdout],
        label=label,
        directions=["short"],
        n_samples_per_file=None,
        load_every="1000ms",
        shaper_config_override=SHAPER,
    )


async def _run() -> None:
    for holdout in HOLDOUTS:
        await _run_one(holdout)


if __name__ == "__main__":
    asyncio.run(_run())
