from pathlib import Path

from deep_orderbook.scientist_experiment import choose_latest_parquet, richness_gate


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
