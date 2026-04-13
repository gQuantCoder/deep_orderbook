from pathlib import Path
from typing import Iterable

import numpy as np


def choose_latest_parquet(directories: Iterable[Path]) -> Path:
    latest: Path | None = None
    latest_key: tuple[float, str] = (float('-inf'), '')
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob('*.parquet'):
            key = (path.stat().st_mtime, path.name)
            if key > latest_key:
                latest = path
                latest_key = key
    if latest is None:
        raise FileNotFoundError('No parquet files found in candidate directories')
    return latest


def richness_gate(books_array, target_array) -> dict:
    books = np.asarray(books_array, dtype=np.float32)
    target = np.asarray(target_array, dtype=np.float32)

    books_signal = float(np.std(books))
    books_peak = float(np.max(np.abs(books))) if books.size else 0.0
    target_active_ratio = float(np.mean(target > 1e-4)) if target.size else 0.0
    target_peak = float(np.max(target)) if target.size else 0.0

    usable = True
    reason = 'ok'
    if books_signal < 1e-3 or target_active_ratio < 1e-4 or target_peak < 1e-4:
        usable = False
        reason = 'flat_or_empty'
    elif books_peak > 1e3:
        usable = False
        reason = 'likely_saturated'

    return {
        'usable': usable,
        'reason': reason,
        'books_std': books_signal,
        'books_abs_peak': books_peak,
        'target_active_ratio': target_active_ratio,
        'target_peak': target_peak,
    }
