import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


async def load_windows(max_windows: int = 36):
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
        look_ahead=64,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=256,
        window_stride=8,
        use_cache=False,
        save_cache=False,
    )

    Xw, Yw, Pxw = [], [], []
    async for books_array, level_prox, pxar in iter_shapes_t2l(replay_conf, shaper_config, live=False):
        Xw.append(books_array.astype(np.float64))
        Yw.append(level_prox[:, :, 0].astype(np.float64))
        Pxw.append(pxar.astype(np.float64))
        if len(Xw) >= max_windows:
            break

    if len(Xw) < 8:
        raise RuntimeError(f"Not enough windows loaded: {len(Xw)}")

    return Xw, Yw, Pxw


def _best_activity_start(true_seq: np.ndarray, horizon: int) -> int:
    # pick the slice with the highest future movement energy, so map isn't all dark
    per_t = true_seq.sum(axis=1)
    if per_t.size <= horizon:
        return 0
    kernel = np.ones(horizon, dtype=np.float64)
    score = np.convolve(per_t, kernel, mode='valid')
    return int(np.argmax(score))


def _robust_symmetric_limit(arr: np.ndarray, q: float = 99.0) -> float:
    lim = float(np.percentile(np.abs(arr), q))
    return max(lim, 1e-6)


async def main():
    Xw, Yw, Pxw = await load_windows(max_windows=36)

    split = int(len(Xw) * 0.75)
    X_train_seq = np.concatenate(Xw[:split], axis=0)
    Y_train_seq = np.concatenate(Yw[:split], axis=0)
    X_test_seq = np.concatenate(Xw[split:], axis=0)
    Y_test_seq = np.concatenate(Yw[split:], axis=0)
    PX_test_seq = np.concatenate(Pxw[split:], axis=0)

    X_train = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_test = X_test_seq.reshape(X_test_seq.shape[0], -1)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss='squared_error',
            max_depth=5,
            learning_rate=0.06,
            max_iter=120,
            random_state=7,
        )
    )
    model.fit(X_train, Y_train_seq)
    pred = np.clip(model.predict(X_test), 0.0, None)

    t = min(520, X_test_seq.shape[0])
    start = _best_activity_start(Y_test_seq, t)

    px = PX_test_seq[start:start + t]
    mid = px.mean(axis=1)

    # Input book pressure maps
    books_pressure = X_test_seq[start:start + t, :, 0].T
    imbalance = X_test_seq[start:start + t, :, 2].T  # signed; usually <= 0 in this pipeline

    true_map = Y_test_seq[start:start + t].T
    pred_map = pred[start:start + t].T

    # Dynamic range handling to avoid "all black" panels
    books_lim = _robust_symmetric_limit(books_pressure, q=99.0)
    imb_lim = _robust_symmetric_limit(imbalance, q=99.0)

    # Gamma lift for low-level signals in movement maps
    true_show = np.sqrt(np.clip(true_map, 0.0, None))
    pred_show = np.sqrt(np.clip(pred_map, 0.0, None))
    out_vmax = max(float(np.percentile(true_show, 99.5)), float(np.percentile(pred_show, 99.5)), 1e-3)

    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)

    x = np.arange(t)
    axes[0].plot(x, px[:, 0], color='green', linewidth=1.0)
    axes[0].plot(x, px[:, 1], color='red', linewidth=1.0)
    axes[0].plot(x, mid, color='orange', linewidth=1.0)

    axes[1].imshow(
        books_pressure,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        vmin=-books_lim,
        vmax=books_lim,
    )

    axes[2].imshow(
        imbalance,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
        vmin=-imb_lim,
        vmax=imb_lim,
    )

    axes[3].imshow(
        true_show,
        aspect='auto',
        origin='lower',
        cmap='turbo',
        vmin=0.0,
        vmax=out_vmax,
    )

    axes[4].imshow(
        pred_show,
        aspect='auto',
        origin='lower',
        cmap='turbo',
        vmin=0.0,
        vmax=out_vmax,
    )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.tight_layout(h_pad=0.35)

    out_path = Path('experiments/results/replay_readme_style_prediction.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(out_path.resolve())


if __name__ == '__main__':
    asyncio.run(main())
