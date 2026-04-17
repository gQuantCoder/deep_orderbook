from __future__ import annotations

from pathlib import Path

from deep_orderbook.scientist_experiment import choose_walkforward_parquets


def resolve_train_test_files(
    *,
    data_dir: Path,
    explicit_train_files: list[Path] | None,
    explicit_test_files: list[Path] | None,
    train_count: int,
    test_count: int,
) -> tuple[list[Path], list[Path]]:
    if explicit_train_files or explicit_test_files:
        if not explicit_train_files or not explicit_test_files:
            raise ValueError("explicit_train_files and explicit_test_files must both be provided together")
        missing = [p for p in [*explicit_train_files, *explicit_test_files] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing explicit parquet files: {missing}")
        return list(explicit_train_files), list(explicit_test_files)
    return choose_walkforward_parquets(data_dir, train_count=train_count, test_count=test_count)


def summarize_dataset_scale(
    *,
    train_files: list[Path],
    test_files: list[Path],
    train_windows_before_filter: int,
    train_windows_after_filter: int,
    test_windows_before_filter: int,
    test_windows_after_filter: int,
    rolling_window_size: int,
    target_levels: int,
    max_windows_per_file: int | None,
) -> dict:
    return {
        "train_file_count": len(train_files),
        "test_file_count": len(test_files),
        "train_files": [str(p) for p in train_files],
        "test_files": [str(p) for p in test_files],
        "train_windows_before_filter": train_windows_before_filter,
        "train_windows_after_filter": train_windows_after_filter,
        "test_windows_before_filter": test_windows_before_filter,
        "test_windows_after_filter": test_windows_after_filter,
        "rolling_window_size": rolling_window_size,
        "target_levels": target_levels,
        "train_timesteps": train_windows_after_filter * rolling_window_size,
        "test_timesteps": test_windows_after_filter * rolling_window_size,
        "train_target_pixels": train_windows_after_filter * rolling_window_size * target_levels,
        "test_target_pixels": test_windows_after_filter * rolling_window_size * target_levels,
        "max_windows_per_file": max_windows_per_file,
    }


def choose_training_device(*, prefer_cuda: bool = True, cuda_available: bool) -> str:
    if prefer_cuda and cuda_available:
        return "cuda"
    return "cpu"
