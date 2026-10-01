import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.experiment_tracking import register_experiment_run, save_map_preview
from deep_orderbook.learn.losses import StructuredT2LLoss
from deep_orderbook.shaper import iter_shapes_t2l


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TinyTCN(nn.Module):
    def __init__(self, in_ch: int, hid: int, out_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(CausalConv1d(in_ch, hid, 3, 1), nn.GELU())
        self.b2 = nn.Sequential(CausalConv1d(hid, hid, 3, 2), nn.GELU())
        self.b3 = nn.Sequential(CausalConv1d(hid, hid, 3, 4), nn.GELU())
        self.reg_head = CausalConv1d(hid, out_ch, 1, 1)
        self.cls_head = CausalConv1d(hid, out_ch, 1, 1)

    def forward(self, x_bt_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_bt_f.transpose(1, 2)
        h = self.b1(x)
        h = h + self.b2(h)
        h = h + self.b3(h)
        reg_raw = self.reg_head(h).transpose(1, 2)
        cls_logits = self.cls_head(h).transpose(1, 2)
        reg = F.softplus(reg_raw)
        return reg, cls_logits


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

    # Side metric on flattened per-timestep maps.
    yt_rows = y_true.reshape(-1, y_true.shape[-1])
    yp_rows = y_pred.reshape(-1, y_pred.shape[-1])
    side = yt_rows.shape[-1] // 2
    pred_sig = yp_rows.max(axis=1) > thr
    if np.any(pred_sig):
        pred_up = yp_rows[pred_sig, side:].max(axis=1)
        pred_dn = yp_rows[pred_sig, :side].max(axis=1)
        true_up = yt_rows[pred_sig, side:].max(axis=1)
        true_dn = yt_rows[pred_sig, :side].max(axis=1)
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


async def load_windows(look_ahead: int, max_windows: int = 120):
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
        Xw.append(books_array.reshape(books_array.shape[0], -1).astype(np.float32))
        Yw.append(level_prox[:, :, 0].astype(np.float32))
        if len(Xw) >= max_windows:
            break

    if len(Xw) < 30:
        raise RuntimeError(f'too few windows loaded for look_ahead={look_ahead}: {len(Xw)}')

    split = int(len(Xw) * 0.75)
    X_train = np.stack(Xw[:split], axis=0)
    Y_train = np.stack(Yw[:split], axis=0)
    X_test = np.stack(Xw[split:], axis=0)
    Y_test = np.stack(Yw[split:], axis=0)

    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, Y_train, X_test, Y_test


def train_one(
    mode: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    epochs: int = 8,
    batch_size: int = 8,
    tau: float = 1e-4,
) -> tuple[dict, np.ndarray]:
    torch.manual_seed(7)
    np.random.seed(7)

    device = torch.device('cpu')
    model = TinyTCN(in_ch=X_train.shape[-1], hid=96, out_ch=Y_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    base_reg = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))

    if mode == 'baseline':
        criterion_map = base_reg
        lam_evt = 0.25
    elif mode == 'structured':
        criterion_map = StructuredT2LLoss(
            base_loss=nn.SmoothL1Loss(),
            pointwise_weight=1.0,
            updown_rank_weight=0.35,
            monotonic_weight=0.15,
            rank_margin=0.05,
            focus_last_step=True,
        )
        lam_evt = 0.20
    else:
        raise ValueError(f'unknown mode: {mode}')

    xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(Y_train).to(device)
    xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(Y_test).to(device)

    n = xtr.shape[0]
    curve = []

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        model.train()
        total = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = xtr[idx]
            yb = ytr[idx]

            reg, cls_logits = model(xb)
            evt_target = (yb > tau).float()

            if mode == 'structured':
                reg_4d = reg.unsqueeze(1)
                yb_4d = yb.unsqueeze(1)
                loss_reg = criterion_map(reg_4d, yb_4d)
            else:
                loss_reg = criterion_map(reg, yb)

            loss_evt = bce(cls_logits, evt_target)
            loss = loss_reg + lam_evt * loss_evt

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())

        model.eval()
        with torch.no_grad():
            pred_te, _ = model(xte)
            m = metrics(yte.cpu().numpy(), pred_te.cpu().numpy(), thr=tau)

        curve.append({
            'epoch': ep,
            'train_loss': total / max(1, (n + batch_size - 1) // batch_size),
            'test_f1': m['f1'],
            'test_precision': m['precision'],
            'test_rmse': m['rmse'],
            'test_side_acc': m['side_acc_on_signals'],
        })

    with torch.no_grad():
        pred_te, _ = model(xte)
        pred_np = pred_te.cpu().numpy()

    final = metrics(Y_test, pred_np, thr=tau)
    return {
        'mode': mode,
        'metrics': final,
        'train_curve': curve,
    }, pred_np


async def run_horizon(look_ahead: int) -> dict:
    X_train, Y_train, X_test, Y_test = await load_windows(look_ahead, max_windows=120)

    zero = metrics(Y_test, np.zeros_like(Y_test))
    base, base_pred = train_one('baseline', X_train, Y_train, X_test, Y_test)
    structured, structured_pred = train_one('structured', X_train, Y_train, X_test, Y_test)

    return {
        'look_ahead': look_ahead,
        'samples': {
            'train_windows': int(X_train.shape[0]),
            'test_windows': int(X_test.shape[0]),
            'timesteps_per_window': int(X_train.shape[1]),
            'feature_dim': int(X_train.shape[2]),
            'target_dim': int(Y_train.shape[2]),
        },
        'zero_baseline': zero,
        'baseline_tcn': base,
        'structured_tcn': structured,
        'artifacts_preview_payload': {
            'y_true': Y_test,
            'baseline_pred': base_pred,
            'structured_pred': structured_pred,
        },
        'delta_structured_vs_baseline': {
            'f1': float(structured['metrics']['f1'] - base['metrics']['f1']),
            'precision': float(structured['metrics']['precision'] - base['metrics']['precision']),
            'side_acc': float(
                (structured['metrics']['side_acc_on_signals'] if not np.isnan(structured['metrics']['side_acc_on_signals']) else 0.0)
                - (base['metrics']['side_acc_on_signals'] if not np.isnan(base['metrics']['side_acc_on_signals']) else 0.0)
            ),
            'rmse_gain_pct': float(
                (base['metrics']['rmse'] - structured['metrics']['rmse'])
                / (base['metrics']['rmse'] + 1e-12)
                * 100.0
            ),
        },
    }


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def register_result_variants(
    *,
    timestamp_utc: str,
    git_commit: str | None,
    stamped_json_path: Path,
    stamped_picture_paths: dict[str, Path],
    results: list[dict],
) -> list[int]:
    row_ids: list[int] = []
    for result in results:
        look_ahead = result['look_ahead']
        for mode_key, variant_suffix in (("baseline_tcn", "baseline_tcn"), ("structured_tcn", "structured_tcn")):
            metrics_payload = result[mode_key]['metrics']
            row_ids.append(
                register_experiment_run(
                    experiment='exp05_tcn_structured_loss_compare',
                    variant=f"h{look_ahead}_{variant_suffix}",
                    timestamp_utc=timestamp_utc,
                    git_commit=git_commit,
                    symbol='ETH-USD',
                    cadence='100ms',
                    look_ahead=look_ahead,
                    model='TinyTCN',
                    result_json_path=stamped_json_path,
                    picture_path=stamped_picture_paths[f"h{look_ahead}_{variant_suffix}"],
                    metrics=metrics_payload,
                    notes='auto-registered from exp05_tcn_structured_loss_compare',
                )
            )
    return row_ids


async def main():
    ts = datetime.now(timezone.utc)
    timestamp_utc = ts.strftime('%Y-%m-%d %H:%M:%SZ')
    stamp = ts.strftime('%Y%m%dT%H%M%SZ')
    git_commit = current_git_commit()

    res32 = await run_horizon(32)
    res64 = await run_horizon(64)

    picture_dir = Path('experiments/pictures')
    stamped_picture_paths = {
        'h32_baseline_tcn': picture_dir / f'exp05_tcn_structured_loss_compare_h32_baseline_tcn_{stamp}.png',
        'h32_structured_tcn': picture_dir / f'exp05_tcn_structured_loss_compare_h32_structured_tcn_{stamp}.png',
        'h64_baseline_tcn': picture_dir / f'exp05_tcn_structured_loss_compare_h64_baseline_tcn_{stamp}.png',
        'h64_structured_tcn': picture_dir / f'exp05_tcn_structured_loss_compare_h64_structured_tcn_{stamp}.png',
    }

    save_map_preview(
        res32['artifacts_preview_payload']['y_true'],
        res32['artifacts_preview_payload']['baseline_pred'],
        stamped_picture_paths['h32_baseline_tcn'],
        pred_title='Predicted future movement (h32 baseline TCN)',
    )
    save_map_preview(
        res32['artifacts_preview_payload']['y_true'],
        res32['artifacts_preview_payload']['structured_pred'],
        stamped_picture_paths['h32_structured_tcn'],
        pred_title='Predicted future movement (h32 structured TCN)',
    )
    save_map_preview(
        res64['artifacts_preview_payload']['y_true'],
        res64['artifacts_preview_payload']['baseline_pred'],
        stamped_picture_paths['h64_baseline_tcn'],
        pred_title='Predicted future movement (h64 baseline TCN)',
    )
    save_map_preview(
        res64['artifacts_preview_payload']['y_true'],
        res64['artifacts_preview_payload']['structured_pred'],
        stamped_picture_paths['h64_structured_tcn'],
        pred_title='Predicted future movement (h64 structured TCN)',
    )

    for res in (res32, res64):
        res.pop('artifacts_preview_payload', None)

    out = {
        'experiment': 'exp05_tcn_structured_loss_compare',
        'timestamp_utc': timestamp_utc,
        'git_commit': git_commit,
        'horizons': [32, 64],
        'results': [res32, res64],
        'artifacts': {key: str(path) for key, path in stamped_picture_paths.items()},
        'promising': {
            'h32': bool(
                res32['structured_tcn']['metrics']['f1'] >= 0.20
                and res32['structured_tcn']['metrics']['precision'] >= 0.12
            ),
            'h64': bool(
                res64['structured_tcn']['metrics']['f1'] >= 0.26
                and res64['structured_tcn']['metrics']['precision'] >= 0.16
                and (not np.isnan(res64['structured_tcn']['metrics']['side_acc_on_signals']))
                and res64['structured_tcn']['metrics']['side_acc_on_signals'] >= 0.58
            ),
        },
    }

    out_dir = Path('experiments/results')
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped_json_path = out_dir / f'exp05_tcn_structured_loss_compare_{stamp}.json'
    latest_json_path = out_dir / 'exp05_tcn_structured_loss_compare.json'
    payload = json.dumps(out, indent=2)
    stamped_json_path.write_text(payload)
    latest_json_path.write_text(payload)

    row_ids = register_result_variants(
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        stamped_json_path=stamped_json_path,
        stamped_picture_paths=stamped_picture_paths,
        results=out['results'],
    )
    out['experiment_db_row_ids'] = row_ids
    payload = json.dumps(out, indent=2)
    stamped_json_path.write_text(payload)
    latest_json_path.write_text(payload)

    print(payload)
    print(f'\nSaved: {stamped_json_path.resolve()}')
    print(f'Pictures: {picture_dir.resolve()}')


if __name__ == '__main__':
    asyncio.run(main())
