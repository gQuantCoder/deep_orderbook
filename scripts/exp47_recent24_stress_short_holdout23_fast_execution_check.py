"""exp47 — next-chronological holdout check for the recent 24-file stress-conditioned short branch.

Hypothesis
----------
If exp46's recent-2026 stress-conditioned short branch is real and the shift toward
faster execution is structural rather than accidental, then keeping the mapper / regime
filter fixed while moving the holdout one hour later to 2026-04-21T23:00:00 should
still yield positive short PnL with honest dense-map quality, and the best surviving
route should stay in or near the fast family (q90/q95, persistence=1, max_hold=48).

Single main mutation vs exp46:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep the observable present-time stress filter and short-side replay fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- move the chronological holdout from 2026-04-21T22:00:00 to 2026-04-21T23:00:00

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp47_recent24_stress_short_holdout23_fast_execution_check.py
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

HOLDOUT = DATA_DIR / "2026-04-21T23-00-00.parquet"

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


async def _run() -> None:
    holdout_tag = HOLDOUT.stem.replace(":", "").replace("-", "")
    label = f"exp47_recent24_stress_short_holdout23_fast_execution_check_{holdout_tag}"
    await exp22_main(
        variants=["regonly_l1_2d_h8"],
        train_files=DEFAULT_TRAIN_FILES,
        test_files=[HOLDOUT],
        label=label,
        directions=["short"],
        n_samples_per_file=None,
        load_every="1000ms",
        shaper_config_override=SHAPER,
    )


if __name__ == "__main__":
    asyncio.run(_run())
