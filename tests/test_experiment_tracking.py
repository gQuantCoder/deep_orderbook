import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from deep_orderbook.experiment_tracking import register_experiment_run, save_map_preview


def test_register_experiment_run_persists_summary_row(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite3"
    json_path = tmp_path / "result.json"
    png_path = tmp_path / "preview.png"
    json_path.write_text("{}")
    png_path.write_bytes(b"png")

    row_id = register_experiment_run(
        db_path=db_path,
        experiment="exp05_tcn_structured_loss_compare",
        variant="h64_baseline_tcn",
        timestamp_utc="2026-04-13 12:34:56Z",
        git_commit="abc1234",
        symbol="ETH-USD",
        cadence="100ms",
        look_ahead=64,
        model="TinyTCN",
        result_json_path=json_path,
        picture_path=png_path,
        metrics={
            "rmse": 0.12,
            "mae": 0.03,
            "precision": 0.11,
            "recall": 0.22,
            "f1": 0.15,
            "side_acc_on_signals": 0.55,
        },
        notes="test row",
    )

    assert row_id > 0
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT experiment, variant, git_commit, cadence, look_ahead, model, result_json_path, picture_path, rmse, precision, f1, side_acc_on_signals, metrics_json, notes FROM experiment_runs"
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == "exp05_tcn_structured_loss_compare"
    assert row[1] == "h64_baseline_tcn"
    assert row[2] == "abc1234"
    assert row[3] == "100ms"
    assert row[4] == 64
    assert row[5] == "TinyTCN"
    assert row[6] == str(json_path)
    assert row[7] == str(png_path)
    assert row[8] == 0.12
    assert row[9] == 0.11
    assert row[10] == 0.15
    assert row[11] == 0.55
    assert json.loads(row[12])["recall"] == 0.22
    assert row[13] == "test row"


def test_save_map_preview_writes_nonempty_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    y_true = np.zeros((64, 8), dtype=np.float32)
    y_pred = np.zeros((64, 8), dtype=np.float32)
    y_true[10:20, 2:5] = 0.8
    y_pred[12:24, 2:5] = 0.5

    out_path = tmp_path / "preview.png"
    save_map_preview(y_true, y_pred, out_path, pred_title="Predicted future movement")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_map_preview_accepts_windowed_3d_arrays(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    y_true = np.zeros((3, 64, 8), dtype=np.float32)
    y_pred = np.zeros((3, 64, 8), dtype=np.float32)
    y_true[1, 10:20, 2:5] = 0.8
    y_pred[1, 12:24, 2:5] = 0.5

    out_path = tmp_path / "preview_3d.png"
    save_map_preview(y_true, y_pred, out_path, pred_title="Predicted future movement")

    assert out_path.exists()
    assert out_path.stat().st_size > 0
