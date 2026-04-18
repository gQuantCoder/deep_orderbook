"""exp25 - test whether 500ms cadence reduces redundant-sample overfit.

This is the second queued mutation after exp24_smallnet_earlystop. It keeps the
smaller-net h32 mapper family but changes the replay cadence and compensated
geometry so wall-clock context and horizon remain meaningful:
- every: 100ms -> 500ms
- rolling_window_size: 2048 -> 512
- look_ahead: 128 -> 32
- window_stride: 8 -> 2

All honesty guards, per-window PnL aggregation, artifact generation, and DB
registration still flow through scripts.exp22_trigger_sweep.main.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from deep_orderbook.config import ShaperConfig
from scripts.exp16_batch_h64_tcn import DEFAULT_SHAPER_CONFIG_2026
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
DEFAULT_EVERY = "500ms"
DEFAULT_TRAIN_FILES = [
    str(DATA_DIR / "2026-04-15T14-00-10.parquet"),
    str(DATA_DIR / "2026-04-15T15-00-11.parquet"),
    str(DATA_DIR / "2026-04-15T16-00-10.parquet"),
]
DEFAULT_TEST_FILES = [str(DATA_DIR / "2026-04-16T17-00-10.parquet")]
DEFAULT_VARIANTS = [
    "l1_evt005_pw2_h32",
    "precision_evt005_pw2_thr010_h32",
    "regonly_wd1e3_h32",
    "regonly_huber_thr010_h32",
]


def build_exp25_shaper_config() -> ShaperConfig:
    return DEFAULT_SHAPER_CONFIG_2026.but(
        rolling_window_size=512,
        look_ahead=32,
        window_stride=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="exp25_timing_500ms")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--train-files", nargs="*", default=DEFAULT_TRAIN_FILES)
    parser.add_argument("--test-files", nargs="*", default=DEFAULT_TEST_FILES)
    parser.add_argument(
        "--directions",
        nargs="*",
        default=["long"],
        choices=["long", "short"],
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=800,
        help="Cap windows per parquet file for RAM safety on 32GB machines.",
    )
    args = parser.parse_args()

    asyncio.run(
        exp22_main(
            args.variants,
            [Path(p) for p in args.train_files],
            [Path(p) for p in args.test_files],
            args.label,
            args.directions,
            n_samples_per_file=args.max_windows_per_file,
            load_every=DEFAULT_EVERY,
            shaper_config_override=build_exp25_shaper_config(),
        )
    )
