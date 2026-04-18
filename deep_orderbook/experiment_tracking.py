import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EXPERIMENT_DB = Path("experiments/experiment_runs.sqlite3")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_experiment_db(db_path: Path = DEFAULT_EXPERIMENT_DB) -> Path:
    _ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                experiment TEXT NOT NULL,
                variant TEXT NOT NULL,
                git_commit TEXT,
                symbol TEXT,
                cadence TEXT,
                look_ahead INTEGER,
                model TEXT,
                result_json_path TEXT,
                picture_path TEXT,
                rmse REAL,
                mae REAL,
                precision REAL,
                recall REAL,
                f1 REAL,
                side_acc_on_signals REAL,
                metrics_json TEXT NOT NULL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_runs_lookup ON experiment_runs (experiment, variant, timestamp_utc)"
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def register_experiment_run(
    *,
    db_path: Path = DEFAULT_EXPERIMENT_DB,
    experiment: str,
    variant: str,
    timestamp_utc: str,
    git_commit: str | None,
    symbol: str | None,
    cadence: str | None,
    look_ahead: int | None,
    model: str | None,
    result_json_path: Path | None,
    picture_path: Path | None,
    metrics: dict[str, Any],
    notes: str | None = None,
) -> int:
    ensure_experiment_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO experiment_runs (
                timestamp_utc,
                experiment,
                variant,
                git_commit,
                symbol,
                cadence,
                look_ahead,
                model,
                result_json_path,
                picture_path,
                rmse,
                mae,
                precision,
                recall,
                f1,
                side_acc_on_signals,
                metrics_json,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp_utc,
                experiment,
                variant,
                git_commit,
                symbol,
                cadence,
                look_ahead,
                model,
                str(result_json_path) if result_json_path else None,
                str(picture_path) if picture_path else None,
                metrics.get("rmse"),
                metrics.get("mae"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("side_acc_on_signals"),
                json.dumps(metrics, sort_keys=True),
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def save_map_preview(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    true_title: str = "True future movement",
    pred_title: str = "Predicted future movement",
) -> Path:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
    if y_pred.ndim == 3:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    t = min(520, int(y_true.shape[0]), int(y_pred.shape[0]))
    ys = np.asarray(y_true[:t]).T
    ps = np.asarray(y_pred[:t]).T

    ys_show = np.sqrt(np.clip(ys, 0.0, None))
    ps_show = np.sqrt(np.clip(ps, 0.0, None))
    vmax = max(float(np.percentile(ys_show, 99.5)), float(np.percentile(ps_show, 99.5)), 1e-3)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    axes[0].imshow(ys_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[0].set_title(true_title)
    axes[1].imshow(ps_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[1].set_title(pred_title)
    fig.tight_layout(h_pad=0.4)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path
