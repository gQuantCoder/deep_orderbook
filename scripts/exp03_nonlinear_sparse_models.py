import asyncio
import json
from pathlib import Path
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


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
    X_windows: list[np.ndarray] = []
    Y_windows: list[np.ndarray] = []

    async for books_array, level_prox, _ in iter_shapes_t2l(
        replay_config=replay_conf,
        shaper_config=shaper_config,
        live=False,
    ):
        X_windows.append(books_array.reshape(books_array.shape[0], -1).astype(np.float64))
        Y_windows.append(level_prox[:, :, 0].astype(np.float64))
        if len(X_windows) >= max_windows:
            break

    split = int(len(X_windows) * 0.75)
    X_train = np.concatenate(X_windows[:split], axis=0)
    Y_train = np.concatenate(Y_windows[:split], axis=0)
    X_test = np.concatenate(X_windows[split:], axis=0)
    Y_test = np.concatenate(Y_windows[split:], axis=0)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    tau = 1e-4
    sample_weight = 1.0 + 12.0 * (Y_train.max(axis=1) > tau).astype(np.float64)

    # Zero baseline
    pred0 = np.zeros_like(Y_test)

    # MLP baseline (nonlinear)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=512,
        max_iter=60,
        random_state=7,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=6,
        verbose=False,
    )
    mlp.fit(Xtr, Y_train)
    pred_mlp = np.clip(mlp.predict(Xte), 0.0, None)

    # Tree model (unweighted)
    hgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss='squared_error',
            max_depth=6,
            learning_rate=0.08,
            max_iter=120,
            random_state=7,
        )
    )
    hgb.fit(Xtr, Y_train)
    pred_hgb = np.clip(hgb.predict(Xte), 0.0, None)

    # Tree model (sparse-weighted)
    hgb_w = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss='squared_error',
            max_depth=6,
            learning_rate=0.08,
            max_iter=120,
            random_state=7,
        )
    )
    hgb_w.fit(Xtr, Y_train, sample_weight=sample_weight)
    pred_hgb_w = np.clip(hgb_w.predict(Xte), 0.0, None)

    m0 = metrics(Y_test, pred0, thr=tau)
    m_mlp = metrics(Y_test, pred_mlp, thr=tau)
    m_hgb = metrics(Y_test, pred_hgb, thr=tau)
    m_hgb_w = metrics(Y_test, pred_hgb_w, thr=tau)

    out = {
        "windows_analyzed": len(X_windows),
        "train_windows": split,
        "test_windows": len(X_windows) - split,
        "samples_train": int(Xtr.shape[0]),
        "samples_test": int(Xte.shape[0]),
        "feature_dim": int(Xtr.shape[1]),
        "target_dim": int(Y_train.shape[1]),
        "models": {
            "zero_baseline": m0,
            "mlp_relu": m_mlp,
            "hgb": m_hgb,
            "hgb_sparse_weighted": m_hgb_w,
        },
        "delta_vs_zero": {
            "mlp_rmse_gain_pct": float((m0['rmse'] - m_mlp['rmse']) / (m0['rmse'] + 1e-12) * 100.0),
            "hgb_rmse_gain_pct": float((m0['rmse'] - m_hgb['rmse']) / (m0['rmse'] + 1e-12) * 100.0),
            "hgb_w_rmse_gain_pct": float((m0['rmse'] - m_hgb_w['rmse']) / (m0['rmse'] + 1e-12) * 100.0),
            "mlp_f1_minus_zero": float(m_mlp['event_f1'] - m0['event_f1']),
            "hgb_f1_minus_zero": float(m_hgb['event_f1'] - m0['event_f1']),
            "hgb_w_f1_minus_zero": float(m_hgb_w['event_f1'] - m0['event_f1']),
        },
    }

    out_path = Path('experiments/results/exp03_nonlinear_sparse_models.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == '__main__':
    asyncio.run(main())
