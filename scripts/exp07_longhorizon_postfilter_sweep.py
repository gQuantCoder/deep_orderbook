import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


def metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1e-4) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    yb = yt > thr
    pb = yp > thr
    tp = int(np.sum(yb & pb))
    fp = int(np.sum((~yb) & pb))
    fn = int(np.sum(yb & (~pb)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    side = y_true.shape[1] // 2
    pred_sig = y_pred.max(axis=1) > thr
    if np.any(pred_sig):
        pred_up = y_pred[pred_sig, side:].max(axis=1)
        pred_dn = y_pred[pred_sig, :side].max(axis=1)
        true_up = y_true[pred_sig, side:].max(axis=1)
        true_dn = y_true[pred_sig, :side].max(axis=1)
        pred_side = (pred_up >= pred_dn).astype(np.int64)
        true_side = (true_up >= true_dn).astype(np.int64)
        side_acc = float(np.mean(pred_side == true_side))
    else:
        side_acc = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "event_rate_true": float(np.mean(yb)),
        "event_rate_pred": float(np.mean(pb)),
        "side_acc_on_signals": side_acc,
    }


async def load_rows(look_ahead: int = 64, max_windows: int = 120):
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
        look_ahead=look_ahead,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=256,
        window_stride=8,
        use_cache=False,
        save_cache=False,
    )

    Xw, Yw = [], []
    async for books_array, level_prox, _ in iter_shapes_t2l(replay_conf, shaper_config, live=False):
        Xw.append(books_array.reshape(books_array.shape[0], -1).astype(np.float64))
        Yw.append(level_prox[:, :, 0].astype(np.float64))
        if len(Xw) >= max_windows:
            break

    split = int(len(Xw) * 0.75)
    X_train = np.concatenate(Xw[:split], axis=0)
    Y_train = np.concatenate(Yw[:split], axis=0)
    X_test = np.concatenate(Xw[split:], axis=0)
    Y_test = np.concatenate(Yw[split:], axis=0)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd
    return X_train, Y_train, X_test, Y_test


def filter_predictions(pred: np.ndarray, row_gate: float, side_margin: float):
    keep = pred.max(axis=1) >= row_gate
    out = np.zeros_like(pred)
    if not np.any(keep):
        return out, 0.0

    kept_idx = np.where(keep)[0]
    pred_keep = pred[kept_idx]
    side = pred.shape[1] // 2
    up = pred_keep[:, side:].max(axis=1)
    dn = pred_keep[:, :side].max(axis=1)
    side_keep = np.abs(up - dn) >= side_margin
    out[kept_idx[side_keep]] = pred_keep[side_keep]
    return out, float(np.mean(side_keep)) if side_keep.size else 0.0


def score_row(m):
    side = m["metrics"]["side_acc_on_signals"]
    if np.isnan(side):
        side = -1.0
    # prioritize actionable quality over dense rmse
    return (m["metrics"]["f1"], m["metrics"]["precision"], side, -m["metrics"]["rmse"])


async def main():
    tau = 1e-4
    Xtr, Ytr, Xte, Yte = await load_rows(look_ahead=64, max_windows=120)

    active_w = (Ytr.max(axis=1) > tau).astype(np.float64)
    sample_weight = 1.0 + 10.0 * active_w

    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=6,
            learning_rate=0.06,
            max_iter=180,
            random_state=7,
        )
    )
    model.fit(Xtr, Ytr, sample_weight=sample_weight)
    pred_raw = np.clip(model.predict(Xte), 0.0, None)

    base = metrics(Yte, pred_raw, thr=tau)

    row_gates = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3, 1.8e-3, 2.5e-3]
    side_margins = [0.0, 5e-4, 1e-3, 2e-3, 3.5e-3, 5e-3, 8e-3]

    rows = []
    for rg in row_gates:
        for sm in side_margins:
            pred_f, side_keep = filter_predictions(pred_raw, row_gate=rg, side_margin=sm)
            m = metrics(Yte, pred_f, thr=tau)
            rows.append({
                "row_gate": rg,
                "side_margin": sm,
                "side_keep_within_row_gate": side_keep,
                "metrics": m,
            })

    best = sorted(rows, key=score_row, reverse=True)[0]

    constrained = [
        r for r in rows
        if r["metrics"]["precision"] >= 0.16
        and r["metrics"]["recall"] >= 0.55
        and (not np.isnan(r["metrics"]["side_acc_on_signals"]))
        and r["metrics"]["side_acc_on_signals"] >= 0.52
    ]
    best_constrained = sorted(constrained, key=score_row, reverse=True)[0] if constrained else None

    side_floor = base["side_acc_on_signals"]
    constrained_side = [
        r for r in rows
        if r["metrics"]["precision"] >= base["precision"]
        and (not np.isnan(r["metrics"]["side_acc_on_signals"]))
        and r["metrics"]["side_acc_on_signals"] >= side_floor
    ]
    best_side_preserving = sorted(constrained_side, key=score_row, reverse=True)[0] if constrained_side else None

    out = {
        "experiment": "exp07_longhorizon_postfilter_sweep",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "look_ahead": 64,
        "base_single_stage": base,
        "best_unconstrained": best,
        "best_constrained": best_constrained,
        "best_side_preserving": best_side_preserving,
        "num_candidates": len(rows),
        "top10": sorted(rows, key=score_row, reverse=True)[:10],
        "all_candidates": sorted(rows, key=score_row, reverse=True),
    }

    out_path = Path("experiments/results/exp07_longhorizon_postfilter_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
