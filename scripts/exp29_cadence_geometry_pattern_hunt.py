"""exp29 — Cadence × price-bin geometry pattern hunt (scientist handoff).

Hypothesis
----------
Microstructure signals may live at different time scales. 100 ms captures
fine queue dynamics but is redundant; 500 ms–1 s may align better with
human-relevant moves; wider ``view_bips`` spreads bins in dollar space for
the same tick cadence.

This script runs **sequential** trigger sweeps (each produces its own JSON,
PNGs, DB rows) with compensated geometry so wall-clock context stays
meaningful (see table below).

Presets (dt = bucket duration)
-------------------------------
| slug           | every   | rolling | look_ahead | stride | view_bips | ~context @ cadence |
|----------------|---------|---------|------------|--------|-----------|---------------------|
| ms100_baseline | 100ms   | 2048    | 128        | 8      | 5         | ~205 s (default 2026)|
| ms500_mid      | 500ms   | 512     | 32         | 2      | 5         | ~256 s |
| ms1000_slow    | 1000ms  | 1024    | 64         | 4      | 5         | ~17 min |
| ms1000_wide    | 1000ms  | 512     | 32         | 2      | 20        | ~8.5 min, wider $ bins |

Network: ``TinyTCN`` from ``deep_orderbook.learn.causal_tcn`` (``n_blocks>=10``,
RF>=1024 ticks) — configured via variant dict; do not shrink RF below 1024.

Usage
-----
PYTHONPATH=. python scripts/exp29_cadence_geometry_pattern_hunt.py \\
  --n-samples-per-file 250 --variants regonly_huber_thr010

For a fuller search, add more ``--variants`` from ``btc_search_lab.BATCH_VARIANTS``.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts.exp16_batch_h64_tcn import DEFAULT_SHAPER_CONFIG_2026
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
DEFAULT_TRAIN_FILES = [
    str(DATA_DIR / "2026-04-15T14-00-10.parquet"),
    str(DATA_DIR / "2026-04-15T15-00-11.parquet"),
    str(DATA_DIR / "2026-04-15T16-00-10.parquet"),
]
DEFAULT_TEST_FILES = [str(DATA_DIR / "2026-04-16T17-00-10.parquet")]

PRESETS: list[dict] = [
    {
        "slug": "ms100_baseline",
        "every": "100ms",
        "shaper": None,
        "note": "2026 default: fine time, vb=5, ~205s context",
    },
    {
        "slug": "ms500_mid",
        "every": "500ms",
        "shaper": DEFAULT_SHAPER_CONFIG_2026.but(
            rolling_window_size=512,
            look_ahead=32,
            window_stride=2,
        ),
        "note": "500ms: less redundant samples, same ~4min-scale context as 2048@100ms/8",
    },
    {
        "slug": "ms1000_slow",
        "every": "1000ms",
        "shaper": DEFAULT_SHAPER_CONFIG_2026.but(
            rolling_window_size=1024,
            look_ahead=64,
            window_stride=4,
        ),
        "note": "1s buckets: minute-scale context, 64s horizon",
    },
    {
        "slug": "ms1000_wide",
        "every": "1000ms",
        "shaper": DEFAULT_SHAPER_CONFIG_2026.but(
            rolling_window_size=512,
            look_ahead=32,
            window_stride=2,
            view_bips=20,
            look_ahead_side_bips=10,
        ),
        "note": "1s + wide price bins (vb=20) for book-edge pathology probe",
    },
]


async def _run_presets(
    *,
    variants: list[str],
    train_files: list[Path],
    test_files: list[Path],
    directions: list[str],
    n_samples_per_file: int | None,
    label_prefix: str,
) -> None:
    for preset in PRESETS:
        slug = preset["slug"]
        label = f"{label_prefix}__{slug}"
        override = preset["shaper"]
        await exp22_main(
            variants,
            train_files,
            test_files,
            label,
            directions,
            n_samples_per_file=n_samples_per_file,
            load_every=preset["every"],
            shaper_config_override=override,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-prefix", default="exp29_cadence_geometry_pattern_hunt")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["regonly_huber_thr010", "precision_evt005_pw2_thr010"],
        help="Subset of btc_search_lab batch variants (default: 2 for screening speed).",
    )
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
        default=250,
        dest="n_samples_per_file",
        help="Screening cap; raise for serious runs once a cadence wins.",
    )
    args = parser.parse_args()

    asyncio.run(
        _run_presets(
            variants=args.variants,
            train_files=[Path(p) for p in args.train_files],
            test_files=[Path(p) for p in args.test_files],
            directions=args.directions,
            n_samples_per_file=args.n_samples_per_file,
            label_prefix=args.label_prefix,
        )
    )
