"""exp24 - smaller net + early stop follow-on from exp23 long-window BTC baseline.

Implements the first queued post-exp23 mutation from
`experiments/notes/exp24_exp28_queue_postexp23_20260418T030000Z.md`:
- hidden dim 32
- epochs 2
- trunk dropout 0.2
- weight_decay 1e-3

This is intentionally a thin wrapper over the honest exp23/exp22 trigger-sweep
path so it inherits the pipeline guards, long-window geometry, per-window PnL
aggregation, artifact protocol, and friction table.
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
    "l1_evt005_pw2_h32",
    "precision_evt005_pw2_thr010_h32",
    "regonly_wd1e3_h32",
    "regonly_huber_thr010_h32",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="exp24_smallnet_earlystop")
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
        )
    )
