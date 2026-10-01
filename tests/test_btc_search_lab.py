from pathlib import Path
import struct
import zlib

import numpy as np

from deep_orderbook.btc_search_lab import (
    compute_png_quality_stats,
    list_batch_variant_names,
    list_event_filtered_suite_25,
    rank_variant_results,
)
from deep_orderbook.event_selection import (
    score_window_eventfulness,
    select_eventful_window_indices,
)


def _write_png(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError("expected HxWx3/4 uint8 image")
    color_type = 2 if arr.shape[-1] == 3 else 6
    raw_rows = b"".join(b"\x00" + arr[y].tobytes() for y in range(arr.shape[0]))
    compressed = zlib.compress(raw_rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", arr.shape[1], arr.shape[0], 8, color_type, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(payload)


def test_batch_variant_registry_has_expected_size() -> None:
    names = list_batch_variant_names()
    assert len(names) >= 25
    assert len(set(names)) == len(names)
    assert "regonly_huber_thr010" in names
    assert "large_h160_evt002_lr8e4" in names

    suite = list_event_filtered_suite_25()
    assert len(suite) == 25
    assert len(set(suite)) == 25
    assert set(suite).issubset(set(names))
    assert "l1_evt005_pw2_thr006" in suite
    assert "regonly_huber_thr014" in suite


def test_compute_png_quality_stats_flags_usable_nonflat_image(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:16, :16, 0] = 255
    image[16:, 16:, 1] = 153
    image[:, 8:12, 2] = 76
    path = tmp_path / "mixed.png"
    _write_png(path, image)

    stats = compute_png_quality_stats(path)

    assert stats["usable"] is True
    assert stats["near_black_fraction"] < 0.95
    assert stats["near_white_fraction"] < 0.95
    assert stats["gray_std"] > 0.05


def test_compute_png_quality_stats_flags_flat_black_image(tmp_path: Path) -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    path = tmp_path / "black.png"
    _write_png(path, image)

    stats = compute_png_quality_stats(path)

    assert stats["usable"] is False
    assert stats["reason"] == "mostly_black"


def test_rank_variant_results_prefers_balanced_metrics() -> None:
    ranked = rank_variant_results(
        [
            {
                "variant_name": "pretty_but_bad",
                "metrics": {"f1": 0.20, "precision": 0.08, "rmse": 1.2},
                "zero_baseline": {"rmse": 0.5},
                "image_quality": {"usable": True, "gray_std": 0.22},
                "pnl": {"fixed_slice": {"prediction_final": 12.0}},
            },
            {
                "variant_name": "balanced",
                "metrics": {"f1": 0.18, "precision": 0.12, "rmse": 0.55},
                "zero_baseline": {"rmse": 0.5},
                "image_quality": {"usable": True, "gray_std": 0.18},
                "pnl": {"fixed_slice": {"prediction_final": 4.0}},
            },
        ]
    )

    assert ranked[0]["variant_name"] == "balanced"
    assert ranked[0]["route_score"] > ranked[1]["route_score"]



def test_score_window_eventfulness_prefers_large_move_and_activity() -> None:
    quiet_prices = np.column_stack([np.full(16, 100.0), np.full(16, 100.2)])
    quiet_books = np.zeros((16, 8, 3), dtype=np.float32)

    active_books = quiet_books.copy()
    active_books[:, :, 0] = np.linspace(-3.0, 3.0, 16, dtype=np.float32)[:, None]

    move_prices = np.column_stack([
        np.linspace(100.0, 101.5, 16),
        np.linspace(100.2, 101.7, 16),
    ])

    quiet = score_window_eventfulness(quiet_prices, quiet_books)
    book_active = score_window_eventfulness(quiet_prices, active_books)
    moving = score_window_eventfulness(move_prices, active_books)

    assert quiet["score"] < book_active["score"] < moving["score"]
    assert moving["features"]["abs_return_bps"] > 100.0



def test_select_eventful_window_indices_returns_top_scored_windows() -> None:
    prices = []
    books = []
    for idx in range(5):
        base = 100.0 + idx
        move = idx * 0.4
        prices.append(np.column_stack([
            np.linspace(base, base + move, 16),
            np.linspace(base + 0.2, base + 0.2 + move, 16),
        ]))
        arr = np.zeros((16, 8, 3), dtype=np.float32)
        arr[:, :, 0] = np.linspace(-idx, idx, 16, dtype=np.float32)[:, None]
        books.append(arr)

    selected = select_eventful_window_indices(prices, books, top_fraction=0.4, min_count=2)

    assert selected == [4, 3]
