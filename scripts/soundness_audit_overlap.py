import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


def _row_hashes(arr: np.ndarray) -> list[str]:
    return [hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in arr]


def _boundary_similarity(last_train: np.ndarray, first_test: np.ndarray) -> dict:
    # Compare the final train window with the first test window.
    # If rolling windows overlap heavily, many rows are identical (or near-identical).
    min_rows = min(last_train.shape[0], first_test.shape[0])
    a = last_train[-min_rows:]
    b = first_test[:min_rows]
    exact = np.all(a == b, axis=1)
    close = np.max(np.abs(a - b), axis=1) < 1e-9
    return {
        "rows_compared": int(min_rows),
        "exact_match_rows": int(exact.sum()),
        "exact_match_rate": float(exact.mean()),
        "near_match_rows": int(close.sum()),
        "near_match_rate": float(close.mean()),
    }


async def main() -> None:
    replay_conf = ReplayConfig(
        markets=["ETH-USD"],
        data_dir=Path("/media/photoDS216/crypto/"),
        date_regexp="2025-02-18T11*",
        max_samples=-1,
        every="100ms",
    )
    shaper_config = ShaperConfig(
        only_full_arrays=True,
        view_bips=5,
        num_side_lvl=8,
        look_ahead=64,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=256,
        window_stride=8,
        use_cache=False,
        save_cache=False,
    )

    Xw = []
    async for books_array, _, _ in iter_shapes_t2l(replay_conf, shaper_config, live=False):
        Xw.append(books_array.reshape(books_array.shape[0], -1).astype(np.float64))
        if len(Xw) >= 120:
            break

    split = int(len(Xw) * 0.75)
    X_train = np.concatenate(Xw[:split], axis=0)
    X_test = np.concatenate(Xw[split:], axis=0)

    train_hashes = set(_row_hashes(X_train))
    test_hashes = _row_hashes(X_test)
    dup = sum(1 for h in test_hashes if h in train_hashes)

    boundary = _boundary_similarity(Xw[split - 1], Xw[split])

    out = {
        "windows_total": len(Xw),
        "windows_train": split,
        "windows_test": len(Xw) - split,
        "rows_train": int(X_train.shape[0]),
        "rows_test": int(X_test.shape[0]),
        "duplicate_test_rows_seen_in_train": dup,
        "duplicate_rate_test_vs_train": float(dup / max(1, len(test_hashes))),
        "train_unique_rows": len(train_hashes),
        "test_unique_rows": len(set(test_hashes)),
        "boundary_similarity": boundary,
    }

    out_path = Path("experiments/results/soundness_audit_overlap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
