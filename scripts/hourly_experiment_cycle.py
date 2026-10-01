import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l

FLAG = Path('experiments/INTERESTING_FOUND.flag')
JOURNAL = Path('experiments/hourly_journal.md')
OUTDIR = Path('experiments/results/hourly')


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
        side_acc = float('nan')

    return {
        'rmse': rmse,
        'mae': mae,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'event_rate_true': float(np.mean(yb)),
        'event_rate_pred': float(np.mean(pb)),
        'side_acc_on_signals': side_acc,
    }


async def load_data(look_ahead: int, max_windows: int = 120):
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


async def run_horizon(look_ahead: int, tau: float = 1e-4) -> dict:
    Xtr, Ytr, Xte, Yte, n_windows, split = await load_data(look_ahead=look_ahead, max_windows=120)

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

    def score(row):
        side = row['metrics']['side_acc_on_signals']
        side = -1.0 if np.isnan(side) else side
        return (row['metrics']['f1'], row['metrics']['precision'], side, -row['metrics']['rmse'])

    best = sorted(rows, key=score, reverse=True)[0]

    return {
        'look_ahead': look_ahead,
        'windows': n_windows,
        'train_windows': split,
        'test_windows': n_windows - split,
        'zero': m_zero,
        'candidates': rows,
        'best': best,
    }


async def main():
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')

    if FLAG.exists():
        print(f"[{now}] STOPPED: interesting result already found ({FLAG.read_text().strip()})")
        return

    look_aheads = [16, 32, 64]
    tau = 1e-4
    runs = [await run_horizon(la, tau=tau) for la in look_aheads]
    by_la = {r['look_ahead']: r for r in runs}

    mid = by_la[32]['best']['metrics']
    long = by_la[64]['best']['metrics']

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {
        'timestamp_utc': now,
        'look_aheads': look_aheads,
        'runs': runs,
        'best_mid': by_la[32]['best'],
        'best_long': by_la[64]['best'],
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_path = OUTDIR / f'run_{stamp}.json'
    out_path.write_text(json.dumps(out, indent=2))

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    if not JOURNAL.exists():
        JOURNAL.write_text('# Hourly experiment journal\n\n')

    with JOURNAL.open('a') as f:
        f.write(f"## {now}\n")
        for la in look_aheads:
            best = by_la[la]['best']
            bm = best['metrics']
            zm = by_la[la]['zero']
            f.write(
                f"- la={la}: best={best['cfg']['name']} | "
                f"f1={bm['f1']:.4f}, precision={bm['precision']:.4f}, recall={bm['recall']:.4f}, "
                f"side_acc={bm['side_acc_on_signals'] if not np.isnan(bm['side_acc_on_signals']) else 'nan'}, "
                f"rmse={bm['rmse']:.5f}, zero_rmse={zm['rmse']:.5f}\n"
            )
        f.write(f"- Artifact: {out_path}\n\n")

    interesting = (
        mid['f1'] >= 0.18 and
        mid['precision'] >= 0.10 and
        long['f1'] >= 0.26 and
        long['precision'] >= 0.16 and
        (not np.isnan(long['side_acc_on_signals'])) and long['side_acc_on_signals'] >= 0.58 and
        long['rmse'] <= by_la[64]['zero']['rmse'] * 1.50
    )

    if interesting:
        FLAG.write_text(
            f"{now} :: multi-horizon-interesting :: "
            f"mid_f1={mid['f1']:.4f}, mid_p={mid['precision']:.4f}, "
            f"long_f1={long['f1']:.4f}, long_p={long['precision']:.4f}, "
            f"long_side_acc={long['side_acc_on_signals']:.4f}, long_rmse={long['rmse']:.5f}"
        )
        print(f"ALERT INTERESTING: {FLAG.read_text().strip()} | artifact={out_path}")
    else:
        print(
            "heartbeat: "
            f"mid(f1={mid['f1']:.4f},p={mid['precision']:.4f}) "
            f"long(f1={long['f1']:.4f},p={long['precision']:.4f},side={long['side_acc_on_signals']}) "
            f"artifact={out_path}"
        )


if __name__ == '__main__':
    asyncio.run(main())
