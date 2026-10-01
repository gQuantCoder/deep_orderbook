from __future__ import annotations

from math import ceil, log1p
from typing import Sequence

import numpy as np


def _mid_prices(prices_window: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices_window, dtype=np.float32)
    if prices.ndim != 2 or prices.shape[1] < 2:
        raise ValueError(f"expected prices window with shape [T,2+], got {prices.shape}")
    return prices[:, :2].mean(axis=1)


def score_window_eventfulness(prices_window: np.ndarray, books_window: np.ndarray) -> dict:
    mid = _mid_prices(prices_window)
    books = np.asarray(books_window, dtype=np.float32)
    if books.ndim < 2:
        raise ValueError(f"expected books window with at least 2 dims, got {books.shape}")

    mean_mid = max(float(np.mean(mid)), 1e-6)
    abs_return_bps = abs(float(mid[-1] - mid[0])) / mean_mid * 1e4
    range_bps = float(np.max(mid) - np.min(mid)) / mean_mid * 1e4
    diff = np.diff(mid)
    realized_vol_bps = float(np.std(diff)) / mean_mid * 1e4 if diff.size else 0.0
    book_std = float(np.std(books))
    book_impulse = float(np.std(np.diff(books[..., 0], axis=0))) if books.shape[0] > 1 and books.shape[-1] > 0 else 0.0

    score = (
        0.40 * log1p(abs_return_bps)
        + 0.30 * log1p(range_bps)
        + 0.20 * log1p(realized_vol_bps)
        + 0.07 * log1p(book_std)
        + 0.03 * log1p(book_impulse)
    )
    return {
        "score": float(score),
        "features": {
            "abs_return_bps": float(abs_return_bps),
            "range_bps": float(range_bps),
            "realized_vol_bps": float(realized_vol_bps),
            "book_std": float(book_std),
            "book_impulse": float(book_impulse),
        },
    }


def rank_eventful_windows(
    prices_windows: Sequence[np.ndarray],
    books_windows: Sequence[np.ndarray],
) -> list[dict]:
    if len(prices_windows) != len(books_windows):
        raise ValueError("prices_windows and books_windows must have the same length")
    ranked = []
    for idx, (prices_window, books_window) in enumerate(zip(prices_windows, books_windows)):
        item = score_window_eventfulness(prices_window, books_window)
        ranked.append({"index": idx, **item})
    ranked.sort(key=lambda item: (item["score"], item["features"]["abs_return_bps"], item["features"]["range_bps"]), reverse=True)
    return ranked


def select_eventful_window_indices(
    prices_windows: Sequence[np.ndarray],
    books_windows: Sequence[np.ndarray],
    top_fraction: float = 0.35,
    min_count: int = 8,
    max_count: int | None = None,
) -> list[int]:
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    ranked = rank_eventful_windows(prices_windows, books_windows)
    if not ranked:
        return []
    desired = max(min_count, ceil(len(ranked) * top_fraction))
    if max_count is not None:
        desired = min(desired, max_count)
    desired = min(desired, len(ranked))
    return [item["index"] for item in ranked[:desired]]
