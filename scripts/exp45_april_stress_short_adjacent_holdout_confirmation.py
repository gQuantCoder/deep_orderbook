"""exp45 — adjacent-holdout confirmation for the April broad-stress short branch.

Hypothesis
----------
If exp44 found a real stress-conditioned downside continuation branch rather than a one-file pocket,
then the same frozen mapper setup and broad April stress filter should keep producing non-dead short
activity on the next adjacent chronological April evening holdouts.

Single main mutation vs exp44:
- keep mapper / cadence / geometry / stress filter fixed
- change only the holdout file, replaying the same setup on adjacent holdouts
- test chronological confirmation on 2026-04-19T21:00:00 and 2026-04-19T22:00:00

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp45_april_stress_short_adjacent_holdout_confirmation.py
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

ADJACENT_HOLDOUTS = [
    DATA_DIR / "2026-04-19T21-00-00.parquet",
    DATA_DIR / "2026-04-19T22-00-00.parquet",
]

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


async def _run_one(holdout: Path) -> None:
    holdout_tag = holdout.stem.replace(":", "").replace("-", "")
    label = f"exp45_april_stress_short_adjacent_holdout_confirmation_{holdout_tag}"
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
    for holdout in ADJACENT_HOLDOUTS:
        await _run_one(holdout)


if __name__ == "__main__":
    asyncio.run(_run())
