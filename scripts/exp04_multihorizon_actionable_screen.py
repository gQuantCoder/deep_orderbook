import asyncio
import json
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

    # Side-direction metric: split levels in half (down/up).
    side = y_true.shape[1] // 2
    pred_sig = y_pred.max(axis=1) > thr
    if np.any(pred_sig):
        pred_up = y_pred[pred_sig, side:].max(axis=1)
        pred_dn = y_pred[pred_sig, :side].max(axis=1)
        true_up = y_true[pred_sig, side:].max(axis=1)
        true_dn = y_true[pred_sig, :side].max(axis=1)
        pred_side = (pred_up >= pred_dn).astype(np.int64)
        true_side = (true_up >= true_dn).astype(np.int64)
        side_acc_on_signals = float(np.mean(pred_side == true_side))
    else:
        side_acc_on_signals = float('nan')

    return {
        'rmse': rmse,
        'mae': mae,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'event_rate_true': float(np.mean(yb)),
        'event_rate_pred': float(np.mean(pb)),
        'side_acc_on_signals': side_acc_on_signals,
    }


async def load_data_for_lookahead(look_ahead: int, max_windows: int = 120):
    replay_conf = ReplayConfig(
        markets=['ETH-USD'],
        data_dir=Path('/media/photoDS216/crypto/'),
        date_regexp='2025-02-18T11*',
        max_samples=-1,
        every='100ms',
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

    if len(Xw) < 20:
        raise RuntimeError(f'too few windows for look_ahead={look_ahead}: {len(Xw)}')

    split = int(len(Xw) * 0.75)
    X_train = np.concatenate(Xw[:split], axis=0)
    Y_train = np.concatenate(Yw[:split], axis=0)
    X_test = np.concatenate(Xw[split:], axis=0)
    Y_test = np.concatenate(Yw[split:], axis=0)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd
    return Xtr, Y_train, Xte, Y_test, len(Xw), split


async def run_one_horizon(look_ahead: int, tau: float = 1e-4) -> dict:
    Xtr, Ytr, Xte, Yte, n_windows, split = await load_data_for_lookahead(look_ahead)

    zero_pred = np.zeros_like(Yte)
    m_zero = metrics(Yte, zero_pred, thr=tau)

    active_w = (Ytr.max(axis=1) > tau).astype(np.float64)
    sample_weight = 1.0 + 10.0 * active_w

    grid = [
        {'name': 'hgb_d4_lr008', 'max_depth': 4, 'learning_rate': 0.08, 'max_iter': 140, 'weighted': False},
        {'name': 'hgb_d6_lr006_w', 'max_depth': 6, 'learning_rate': 0.06, 'max_iter': 180, 'weighted': True},
    ]

    rows = []
    for cfg in grid:
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss='squared_error',
                max_depth=cfg['max_depth'],
                learning_rate=cfg['learning_rate'],
                max_iter=cfg['max_iter'],
                random_state=7,
            )
        )
        if cfg['weighted']:
            model.fit(Xtr, Ytr, sample_weight=sample_weight)
        else:
            model.fit(Xtr, Ytr)

        pred = np.clip(model.predict(Xte), 0.0, None)
        m = metrics(Yte, pred, thr=tau)
        rows.append({'cfg': cfg, 'metrics': m})

    # Prefer event F1 and side accuracy while keeping RMSE bounded.
    def score(row):
        side_acc = row['metrics']['side_acc_on_signals']
        side_acc = -1.0 if np.isnan(side_acc) else side_acc
        return (row['metrics']['f1'], side_acc, -row['metrics']['rmse'])

    best = sorted(rows, key=score, reverse=True)[0]
    best_m = best['metrics']

    return {
        'look_ahead': look_ahead,
        'windows': n_windows,
        'train_windows': split,
        'test_windows': n_windows - split,
        'zero': m_zero,
        'candidates': rows,
        'best': best,
        'delta_vs_zero': {
            'f1_minus_zero': float(best_m['f1'] - m_zero['f1']),
            'precision_minus_zero': float(best_m['precision'] - m_zero['precision']),
            'side_acc_minus_zero': float((best_m['side_acc_on_signals'] if not np.isnan(best_m['side_acc_on_signals']) else 0.0)),
            'rmse_gain_pct': float((m_zero['rmse'] - best_m['rmse']) / (m_zero['rmse'] + 1e-12) * 100.0),
        },
    }


async def main():
    tau = 1e-4
    look_aheads = [8, 16, 32, 64]

    results = []
    for la in look_aheads:
        results.append(await run_one_horizon(la, tau=tau))

    # Aggregate actionable view.
    by_la = {r['look_ahead']: r for r in results}
    short = by_la[8]['best']['metrics']
    mid = by_la[32]['best']['metrics']
    long = by_la[64]['best']['metrics']

    actionable_summary = {
        'short_horizon_f1_la8': short['f1'],
        'mid_horizon_f1_la32': mid['f1'],
        'long_horizon_f1_la64': long['f1'],
        'short_precision_la8': short['precision'],
        'mid_precision_la32': mid['precision'],
        'long_precision_la64': long['precision'],
        'short_side_acc_la8': short['side_acc_on_signals'],
        'mid_side_acc_la32': mid['side_acc_on_signals'],
        'long_side_acc_la64': long['side_acc_on_signals'],
    }

    out = {
        'experiment': 'exp04_multihorizon_actionable_screen',
        'event_threshold': tau,
        'look_aheads': look_aheads,
        'results': results,
        'actionable_summary': actionable_summary,
    }

    out_path = Path('experiments/results/exp04_multihorizon_actionable_screen.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f'\nSaved: {out_path.resolve()}')


if __name__ == '__main__':
    asyncio.run(main())
