"""exp30 — 2D model architecture baseline on ms1000_slow geometry.

Hypothesis
----------
All previous experiments (exp23–exp29) used ``TinyTCN`` — a 1-D Conv1d model
operating on a flattened (T, bins×ch=48) feature vector.  That architecture is
**structurally incapable** of learning cross-bin patterns (e.g., bid depth rising
at best bid while ask thins: the canonical imbalance signal for microstructure).

``OrderBook2DTCN`` (``deep_orderbook/learn/tcn.py``) uses 2-D dilated causal
convolutions that slide across BOTH the time axis AND the price-bin axis, making
cross-bin spatial patterns learnable for the first time.

This experiment is a pure single-factor comparison:

| Factor            | exp29_ms1000_slow         | exp30 (this run)              |
|-------------------|---------------------------|-------------------------------|
| Cadence           | 1000 ms                   | 1000 ms (same)                |
| rolling / la      | 1024 / 64                 | 1024 / 64 (same)              |
| Data files        | same 3-train / 1-test     | same                          |
| Variants          | regonly_huber_thr010      | same                          |
| Model             | TinyTCN 1D (TOMBSTONED)   | **OrderBook2DTCN 2D** ← ONLY change |
| RF (ticks)        | 31 (was broken)           | 2047 (num_levels=10)          |
| hidden_dim        | 96 (Conv1d hid=96)        | 8 (Conv2d hid=8, ~equiv ops) |

Expected direction: better F1 / precision and lower rmse_ratio vs exp29_ms1000_slow
because the 2D model can detect spatial imbalance patterns the 1D model was blind to.

If the 2D model delivers worse metrics, the next step is to inspect the prediction
maps and check whether the 2D weights are collapsing to uniform bin responses (sign
of misconfigured residual scale or bad initialisation).

Usage
-----
PYTHONPATH=. python scripts/exp30_2d_model_ms1000_slow.py
PYTHONPATH=. python scripts/exp30_2d_model_ms1000_slow.py --n-samples-per-file 500 --directions long short
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts.exp16_batch_h64_tcn import DEFAULT_SHAPER_CONFIG_2026
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")

# Identical file split to exp29_ms1000_slow — controlled comparison
DEFAULT_TRAIN_FILES = [
    str(DATA_DIR / "2026-04-15T14-00-10.parquet"),
    str(DATA_DIR / "2026-04-15T15-00-11.parquet"),
    str(DATA_DIR / "2026-04-15T16-00-10.parquet"),
]
DEFAULT_TEST_FILES = [str(DATA_DIR / "2026-04-16T17-00-10.parquet")]

# Identical shaper geometry to exp29 ms1000_slow
MS1000_SLOW_SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


async def _run(
    *,
    variants: list[str],
    train_files: list[Path],
    test_files: list[Path],
    directions: list[str],
    n_samples_per_file: int | None,
    label: str,
) -> None:
    await exp22_main(
        variants,
        train_files,
        test_files,
        label,
        directions,
        n_samples_per_file=n_samples_per_file,
        load_every="1000ms",
        shaper_config_override=MS1000_SLOW_SHAPER,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="exp30_2d_model_ms1000_slow")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["regonly_huber_2d_h8", "regonly_l1_2d_h8", "precision_evt_2d_h8"],
        help="2D-appropriate variants (small hidden_dim). Conv2d FLOPs scale with bins×hid^2.",
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
        help="Screening cap — same as exp29 for direct comparison.",
    )
    args = parser.parse_args()

    asyncio.run(
        _run(
            variants=args.variants,
            train_files=[Path(p) for p in args.train_files],
            test_files=[Path(p) for p in args.test_files],
            directions=args.directions,
            n_samples_per_file=args.n_samples_per_file,
            label=args.label,
        )
    )
