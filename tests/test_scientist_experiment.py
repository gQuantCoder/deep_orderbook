from pathlib import Path

import numpy as np

from deep_orderbook.scientist_experiment import (
    apply_prediction_cap,
    choose_latest_parquet,
    choose_walkforward_parquets,
    richness_gate,
)


def test_choose_latest_parquet_prefers_newest_file(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    old_file = a / "2025-01-01T00-00-00.parquet"
    new_file = b / "2025-01-01T01-00-00.parquet"
    old_file.write_text("x")
    new_file.write_text("y")

    chosen = choose_latest_parquet([a, b])

    assert chosen == new_file


def test_choose_walkforward_parquets_splits_train_and_test_files(tmp_path: Path) -> None:
    files = []
    for name in [
        "2026-04-14T00-00-01.parquet",
        "2026-04-14T01-00-10.parquet",
        "2026-04-14T02-00-01.parquet",
        "2026-04-14T03-00-00.parquet",
    ]:
        path = tmp_path / name
        path.write_text("x")
        files.append(path)

    train, test = choose_walkforward_parquets(tmp_path, train_count=3, test_count=1)

    assert train == files[:3]
    assert test == [files[3]]
    assert set(train).isdisjoint(test)


def test_richness_gate_rejects_flat_and_accepts_active_arrays() -> None:
    flat_books = [[[0.0] * 3 for _ in range(16)] for _ in range(64)]
    flat_target = [[[0.0] for _ in range(8)] for _ in range(64)]
    flat = richness_gate(flat_books, flat_target)
    assert flat["usable"] is False
    assert flat["reason"] == "flat_or_empty"

    active_books = [[[0.0, 0.0, 0.0] for _ in range(16)] for _ in range(64)]
    active_target = [[[0.0] for _ in range(8)] for _ in range(64)]
    for t in range(10, 30):
        active_books[t][6][0] = 3.0
        active_books[t][9][1] = -3.0
        active_target[t][3][0] = 0.8
    active = richness_gate(active_books, active_target)
    assert active["usable"] is True
    assert active["target_active_ratio"] > 0.0


def test_apply_prediction_cap_clips_extreme_values() -> None:
    pred = np.array([[0.1, 10.0], [5.0, 100.0]], dtype=np.float32)
    clipped = apply_prediction_cap(pred, cap_value=6.0)
    assert clipped.max() == 6.0
    assert clipped[0, 0] == pred[0, 0]
