import asyncio
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_orderbook.config import ReplayConfig, ShaperConfig
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
    def __init__(self, in_ch: int = 48, hid: int = 96, out_ch: int = 8):
        super().__init__()
        self.b1 = nn.Sequential(CausalConv1d(in_ch, hid, 3, 1), nn.GELU())
        self.b2 = nn.Sequential(CausalConv1d(hid, hid, 3, 2), nn.GELU())
        self.b3 = nn.Sequential(CausalConv1d(hid, hid, 3, 4), nn.GELU())
        self.reg_head = CausalConv1d(hid, out_ch, 1, 1)
        self.cls_head = CausalConv1d(hid, out_ch, 1, 1)

    def forward(self, x_bt_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F] -> [B, F, T]
        x = x_bt_f.transpose(1, 2)
        h = self.b1(x)
        h = h + self.b2(h)
        h = h + self.b3(h)
        reg_raw = self.reg_head(h).transpose(1, 2)  # [B, T, C]
        cls_logits = self.cls_head(h).transpose(1, 2)
        reg = F.softplus(reg_raw)  # target is non-negative
        return reg, cls_logits


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1e-4) -> dict:
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


async def load_windows(max_windows: int = 140):
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

    Xw: list[np.ndarray] = []
    Yw: list[np.ndarray] = []
    async for books_array, level_prox, _ in iter_shapes_t2l(
        replay_config=replay_conf,
        shaper_config=shaper_config,
        live=False,
    ):
        Xw.append(books_array.reshape(books_array.shape[0], -1).astype(np.float32))
        Yw.append(level_prox[:, :, 0].astype(np.float32))
        if len(Xw) >= max_windows:
            break
    return Xw, Yw


async def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    Xw, Yw = await load_windows(max_windows=140)
    if len(Xw) < 30:
        raise RuntimeError(f"Too few windows loaded: {len(Xw)}")

    split = int(len(Xw) * 0.75)
    X_train = np.stack(Xw[:split], axis=0)  # [B,T,F]
    Y_train = np.stack(Yw[:split], axis=0)  # [B,T,C]
    X_test = np.stack(Xw[split:], axis=0)
    Y_test = np.stack(Yw[split:], axis=0)

    # robust-ish standardization using train stats only
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    device = torch.device('cpu')
    model = TinyTCN(in_ch=X_train.shape[-1], hid=96, out_ch=Y_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))

    xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(Y_train).to(device)
    xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(Y_test).to(device)

    epochs = 10
    bs = 8
    lam_evt = 0.25
    tau = 1e-4

    train_log = []
    n = xtr.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        model.train()
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = xtr[idx]
            yb = ytr[idx]
            reg, cls_logits = model(xb)
            evt_target = (yb > tau).float()
            loss_reg = huber(reg, yb)
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
            m = event_metrics(yte.cpu().numpy(), pred_te.cpu().numpy(), thr=tau)

        train_log.append({
            "epoch": ep,
            "train_loss": total / max(1, (n + bs - 1) // bs),
            "test_rmse": m["rmse"],
            "test_event_f1": m["event_f1"],
        })

    with torch.no_grad():
        pred_te, _ = model(xte)
        final_m = event_metrics(yte.cpu().numpy(), pred_te.cpu().numpy(), thr=tau)

    zero_m = event_metrics(Y_test, np.zeros_like(Y_test), thr=tau)

    out = {
        "windows_analyzed": len(Xw),
        "train_windows": split,
        "test_windows": len(Xw) - split,
        "samples_train": int(X_train.shape[0] * X_train.shape[1]),
        "samples_test": int(X_test.shape[0] * X_test.shape[1]),
        "feature_dim": int(X_train.shape[2]),
        "target_dim": int(Y_train.shape[2]),
        "model": {
            "name": "TinyTCN",
            "hidden": 96,
            "epochs": epochs,
            "batch_size": bs,
            "loss": "SmoothL1 + 0.25*BCEWithLogits(pos_weight=8)",
            "event_threshold": tau,
        },
        "metrics": {
            "zero_baseline": zero_m,
            "tcn_sparse_loss": final_m,
        },
        "delta_vs_zero": {
            "rmse_gain_pct": float((zero_m["rmse"] - final_m["rmse"]) / (zero_m["rmse"] + 1e-12) * 100.0),
            "f1_minus_zero": float(final_m["event_f1"] - zero_m["event_f1"]),
        },
        "train_curve": train_log,
    }

    out_path = Path('experiments/results/exp03_tcn_sparse_loss.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == '__main__':
    asyncio.run(main())
