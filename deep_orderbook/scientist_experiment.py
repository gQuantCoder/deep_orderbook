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


def choose_walkforward_parquets(directory: Path, train_count: int = 3, test_count: int = 1) -> tuple[list[Path], list[Path]]:
    files = sorted(directory.glob('*.parquet'))
    needed = train_count + test_count
    if len(files) < needed:
        raise FileNotFoundError(f'Need at least {needed} parquet files in {directory}, found {len(files)}')
    selected = files[-needed:]
    train = selected[:train_count]
    test = selected[train_count:]
    return train, test


def apply_prediction_cap(pred: np.ndarray, cap_value: float) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=np.float32), 0.0, float(cap_value))


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
