import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


async def load_windows(max_windows: int = 48):
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

    Xw, Yw = [], []
    async for books_array, level_prox, _ in iter_shapes_t2l(replay_conf, shaper_config, live=False):
        Xw.append(books_array.astype(np.float64))
        Yw.append(level_prox[:, :, 0].astype(np.float64))
        if len(Xw) >= max_windows:
            break

    if len(Xw) < 4:
        raise RuntimeError(f"Not enough windows loaded: {len(Xw)}")

    return Xw, Yw


async def main():
    Xw, Yw = await load_windows(max_windows=48)

    split = int(len(Xw) * 0.75)
    X_train_seq = np.concatenate(Xw[:split], axis=0)
    Y_train_seq = np.concatenate(Yw[:split], axis=0)
    X_test_seq = np.concatenate(Xw[split:], axis=0)
    Y_test_seq = np.concatenate(Yw[split:], axis=0)

    X_train = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_test = X_test_seq.reshape(X_test_seq.shape[0], -1)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss='squared_error',
            max_depth=6,
            learning_rate=0.06,
            max_iter=180,
            random_state=7,
        )
    )
    model.fit(X_train, Y_train_seq)
    pred = np.clip(model.predict(X_test), 0.0, None)

    i = min(120, X_test_seq.shape[0] - 1)
    data_img = X_test_seq[max(0, i - 64): i + 1, :]
    true_row = Y_test_seq[i:i+1, :]
    pred_row = pred[i:i+1, :]

    out_path = Path('experiments/results/sim_data_prediction_snapshot.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

    im0 = axes[0].imshow(data_img, aspect='auto', origin='lower', cmap='coolwarm')
    axes[0].set_title('Simulation data slice (from /media/photoDS216/crypto)')
    axes[0].set_ylabel('Recent timesteps')
    axes[0].set_xlabel('Flattened book features')
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)

    vmax = max(float(true_row.max()), float(pred_row.max()), 1e-6)
    im1 = axes[1].imshow(true_row, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax)
    axes[1].set_title('True future map row (look_ahead=64)')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Price-level bins')
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)

    im2 = axes[2].imshow(pred_row, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax)
    axes[2].set_title('Predicted future map row')
    axes[2].set_ylabel('Pred')
    axes[2].set_xlabel('Price-level bins')
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

    fig.suptitle('Deep Orderbook simulation: data + prediction snapshot', fontsize=13)
    fig.savefig(out_path, dpi=160)
    print(out_path.resolve())


if __name__ == '__main__':
    asyncio.run(main())
