"""exp46 — broadened-train later-holdout scale check for the April stress-conditioned short branch.

Hypothesis
----------
If exp45's promising short branch is truly stress-conditioned and not just a local April 19 pocket,
then keeping the same frozen mapper family / cadence / geometry / stress filter while broadening the
2026 April training pool to more days should still yield non-dead short routes on later chronological
April evening holdouts.

Single main mutation vs exp45:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep short-side replay and the same exp22 stress filter / trigger sweep
- broaden the train pool from Apr 14-18 to Apr 14-21 (00/06/12/18 files)
- test later chronological evening holdouts on 2026-04-21T21:00:01 and 2026-04-21T22:00:00

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp46_april_stress_short_scale_check.py
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

LATER_HOLDOUTS = [
    DATA_DIR / "2026-04-21T21-00-01.parquet",
    DATA_DIR / "2026-04-21T22-00-00.parquet",
]

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


async def _run_one(holdout: Path) -> None:
    holdout_tag = holdout.stem.replace(":", "").replace("-", "")
    label = f"exp46_april_stress_short_scale_check_{holdout_tag}"
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
    for holdout in LATER_HOLDOUTS:
        await _run_one(holdout)


if __name__ == "__main__":
    asyncio.run(_run())
