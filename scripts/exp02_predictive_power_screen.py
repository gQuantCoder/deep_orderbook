import asyncio
import json
from pathlib import Path
import numpy as np

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


def ridge_fit(X: np.ndarray, Y: np.ndarray, l2: float, sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Return W for Y ~= X @ W using ridge closed form."""
    if sample_weight is None:
        XtX = X.T @ X
        XtY = X.T @ Y
    else:
        sw = sample_weight.reshape(-1, 1)
        Xw = X * sw
        XtX = X.T @ Xw
        XtY = X.T @ (Y * sw)
    d = XtX.shape[0]
    W = np.linalg.solve(XtX + l2 * np.eye(d), XtY)
    return W


def metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1e-4) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    sst = float(np.sum((yt - yt.mean()) ** 2))
    sse = float(np.sum((yp - yt) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")

    yb = yt > thr
    pb = yp > thr
    tp = int(np.sum(yb & pb))
    fp = int(np.sum((~yb) & pb))
    fn = int(np.sum(yb & (~pb)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "event_threshold": thr,
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_rate_true": float(np.mean(yb)),
        "event_rate_pred": float(np.mean(pb)),
    }


async def main() -> None:
    replay_conf = ReplayConfig(
        markets=["ETH-USD"],
        data_dir=Path('/media/photoDS216/crypto/'),
        date_regexp='2025-02-18T11*',
        max_samples=-1,
        every='100ms',
    )
    shaper_config = ShaperConfig(
        only_full_arrays=True,
        view_bips=5,
        num_side_lvl=8,
        look_ahead=32,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=256,
        window_stride=8,
        use_cache=False,
        save_cache=False,
    )

    max_windows = 140
    windows = 0
    X_windows: list[np.ndarray] = []
    Y_windows: list[np.ndarray] = []

    async for books_array, level_prox, _pxar in iter_shapes_t2l(
        replay_config=replay_conf,
        shaper_config=shaper_config,
        live=False,
    ):
        x = books_array[:, :, :].reshape(books_array.shape[0], -1)  # [T, 48]
        y = level_prox[:, :, 0]  # [T, 8]
        X_windows.append(x.astype(np.float64))
        Y_windows.append(y.astype(np.float64))
        windows += 1
        if windows >= max_windows:
            break

    if windows < 20:
        raise RuntimeError(f"Too few windows: {windows}")

    split = int(windows * 0.75)
    X_train = np.concatenate(X_windows[:split], axis=0)
    Y_train = np.concatenate(Y_windows[:split], axis=0)
    X_test = np.concatenate(X_windows[split:], axis=0)
    Y_test = np.concatenate(Y_windows[split:], axis=0)

    # feature standardization (train stats only)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    # Baseline 0 predictor
    pred0 = np.zeros_like(Y_test)

    # Unweighted ridge (MSE-like)
    W_mse = ridge_fit(Xtr, Y_train, l2=1.0, sample_weight=None)
    pred_mse = Xte @ W_mse

    # Weighted ridge (sparse-aware proxy)
    active_row = (Y_train.max(axis=1) > 1e-4).astype(np.float64)
    weights = 1.0 + 12.0 * active_row
    W_w = ridge_fit(Xtr, Y_train, l2=1.0, sample_weight=weights)
    pred_w = Xte @ W_w

    m0 = metrics(Y_test, pred0)
    m1 = metrics(Y_test, pred_mse)
    m2 = metrics(Y_test, pred_w)

    out = {
        "windows_analyzed": windows,
        "train_windows": split,
        "test_windows": windows - split,
        "samples_train": int(X_train.shape[0]),
        "samples_test": int(X_test.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "target_dim": int(Y_train.shape[1]),
        "models": {
            "zero_baseline": m0,
            "ridge_mse": m1,
            "ridge_weighted_sparse": m2,
        },
        "delta_vs_zero": {
            "ridge_mse_rmse_gain_pct": float((m0["rmse"] - m1["rmse"]) / (m0["rmse"] + 1e-12) * 100),
            "ridge_weighted_rmse_gain_pct": float((m0["rmse"] - m2["rmse"]) / (m0["rmse"] + 1e-12) * 100),
            "ridge_mse_f1_minus_zero": float(m1["event_f1"] - m0["event_f1"]),
            "ridge_weighted_f1_minus_zero": float(m2["event_f1"] - m0["event_f1"]),
            "weighted_vs_mse_f1": float(m2["event_f1"] - m1["event_f1"]),
        },
    }

    out_path = Path('experiments/results/exp02_predictive_power_screen.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == '__main__':
    asyncio.run(main())
