"""exp23 - Honest long-window trigger sweep on fresh 2026 BTC data.

This script is the successor of ``scripts/exp22_trigger_sweep.py`` and
is intentionally a thin specialization of it:

* points at ``/mnt/data/repos/gaelreinaudi/crypto`` for fresh 2026 BTC
  recordings (instead of the older 2025 archive);
* inherits the fix that replaces the ``reshape(-1, ...)`` ghost-timeline
  backtest with a per-window aggregation (via
  ``deep_orderbook.pipeline_guards.aggregate_per_window_strategy_result``);
* uses the new meaningful image geometry (``rolling_window_size=2048``,
  ``look_ahead=128``) that ``load_file_windows`` now applies by default;
* enforces the executable sanity gates at load time (the
  ``assert_image_meaningful`` call inside ``load_file_windows``).

All the heavy lifting (training loop, artifact writing, trigger search,
DB registration) is reused from exp22 via ``scripts.exp22_trigger_sweep.main``,
so exp23 cannot silently drift from exp22's honesty fixes.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts.exp22_trigger_sweep import main as exp22_main


DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")

DEFAULT_TRAIN_FILES = [
    str(DATA_DIR / "2026-04-15T14-00-10.parquet"),
    str(DATA_DIR / "2026-04-15T15-00-11.parquet"),
    str(DATA_DIR / "2026-04-15T16-00-10.parquet"),
]
DEFAULT_TEST_FILES = [str(DATA_DIR / "2026-04-16T17-00-10.parquet")]

DEFAULT_VARIANTS = [
    "l1_evt005_pw2_h64",
    "precision_evt005_pw2_thr010",
    "regonly_wd1e3",
    "regonly_huber_thr010",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="exp23_longwindow_2026_btc")
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
        "--n-samples-per-file",
        type=int,
        default=1500,
        dest="n_samples_per_file",
        help=(
            "Max rolling-frame samples per parquet file. At rolling=2048/stride=8 a full "
            "hourly file yields ~4200 frames (~800MB raw stack); 1500 keeps RAM safe on "
            "32GB boxes while still using ~20 min of shaped context per file."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        exp22_main(
            args.variants,
            [Path(p) for p in args.train_files],
            [Path(p) for p in args.test_files],
            args.label,
            args.directions,
            n_samples_per_file=args.n_samples_per_file,
        )
    )
