import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.experiment_tracking import register_experiment_run
from deep_orderbook.btc_variant_configs import get_variant_config
from deep_orderbook.scientist_experiment import apply_prediction_cap, choose_walkforward_parquets, richness_gate
from deep_orderbook.shaper import iter_shapes_t2l
from deep_orderbook.strategy import Strategy


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
        x = x_bt_f.transpose(1, 2)
        h = self.b1(x)
        h = h + self.b2(h)
        h = h + self.b3(h)
        reg_raw = self.reg_head(h).transpose(1, 2)
        cls_logits = self.cls_head(h).transpose(1, 2)
        reg = F.softplus(reg_raw)
        return reg, cls_logits


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1e-4) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
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
    return {
        "rmse": rmse,
        "mae": mae,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "event_rate_true": float(np.mean(yb)),
        "event_rate_pred": float(np.mean(pb)),
    }


def _best_activity_start(true_seq: np.ndarray, horizon: int) -> int:
    per_t = true_seq.sum(axis=1)
    if per_t.size <= horizon:
        return 0
    kernel = np.ones(horizon, dtype=np.float64)
    score = np.convolve(per_t, kernel, mode="valid")
    return int(np.argmax(score))


def _entries_exits(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = positions.astype(int)
    enter = np.where((pos[1:] == 1) & (pos[:-1] == 0))[0] + 1
    exit_ = np.where((pos[1:] == 0) & (pos[:-1] == 1))[0] + 1
    return enter, exit_


async def load_file_windows(one_path: Path, market: str, max_windows: int = 48):
    replay_conf = ReplayConfig(
        markets=[market],
        one_path=one_path,
        data_dir=one_path.parent,
        date_regexp=one_path.stem,
        max_samples=-1,
        every="100ms",
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
        Xw.append(books_array.astype(np.float32))
        Yw.append(level_prox[:, :, 0].astype(np.float32))
        Pxw.append(pxar.astype(np.float32))
        if len(Xw) >= max_windows:
            break
    return replay_conf, shaper_config, Xw, Yw, Pxw


def save_precheck(precheck_path: Path, prices: np.ndarray, books: np.ndarray, target: np.ndarray, title_suffix: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.8), sharex=True)
    x = np.arange(prices.shape[0])
    mid = prices.mean(axis=1)
    axes[0].plot(x, prices[:, 0], color="green", linewidth=1.0, label="bid")
    axes[0].plot(x, prices[:, 1], color="red", linewidth=1.0, label="ask")
    axes[0].plot(x, mid, color="orange", linewidth=0.8, label="mid")
    axes[0].set_title(f"Precheck price path — {title_suffix}")
    axes[0].legend(loc="upper left")

    books_show = books[:, :, 0].T
    lim = max(float(np.percentile(np.abs(books_show), 99.0)), 1e-6)
    axes[1].imshow(books_show, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1].set_title("Precheck books channel-0")

    target_show = np.sqrt(np.clip(target.T, 0.0, None))
    vmax = max(float(np.percentile(target_show, 99.5)), 1e-3)
    axes[2].imshow(target_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[2].set_title("Precheck target time2level")

    fig.tight_layout(h_pad=0.35)
    precheck_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(precheck_path, dpi=170)
    plt.close(fig)


def save_dashboard(
    out_path: Path,
    prices: np.ndarray,
    books_array: np.ndarray,
    true_map: np.ndarray,
    pred_map: np.ndarray,
    train_losses: list[float],
    test_losses: list[float],
    gt_pnl: np.ndarray,
    pred_pnl: np.ndarray,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=False)
    x = np.arange(prices.shape[0])
    mid = prices.mean(axis=1)

    gt_enter, gt_exit = _entries_exits(gt_pos)
    pr_enter, pr_exit = _entries_exits(pred_pos)

    axes[0].plot(x, prices[:, 0], color="green", linewidth=1.0, label="Bid")
    axes[0].plot(x, prices[:, 1], color="red", linewidth=1.0, label="Ask")
    axes[0].plot(x, mid, color="orange", linewidth=0.8, label="Mid")
    if gt_enter.size:
        axes[0].scatter(gt_enter, prices[gt_enter, 1], marker="^", color="green", s=35, label="Omniscient Entry")
    if gt_exit.size:
        axes[0].scatter(gt_exit, prices[gt_exit, 0], marker="v", color="green", s=35, label="Omniscient Exit")
    if pr_enter.size:
        axes[0].scatter(pr_enter, prices[pr_enter, 1], marker="^", color="blue", s=25, label="Prediction Entry")
    if pr_exit.size:
        axes[0].scatter(pr_exit, prices[pr_exit, 0], marker="v", color="red", s=25, label="Prediction Exit")
    axes[0].set_title(f"Bid and Ask Price Levels — {title_suffix}")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)

    books_show = books_array[:, :, 0].T
    lim = max(float(np.percentile(np.abs(books_show), 99.0)), 1e-6)
    axes[1].imshow(books_show, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1].set_title("Books")

    true_show = np.sqrt(np.clip(true_map.T, 0.0, None))
    pred_show = np.sqrt(np.clip(pred_map.T, 0.0, None))
    vmax = max(float(np.percentile(true_show, 99.5)), float(np.percentile(pred_show, 99.5)), 1e-3)
    axes[2].imshow(true_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[2].set_title("Level Proximity")
    axes[3].imshow(pred_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[3].set_title("Prediction")

    axes[4].plot(np.arange(1, len(train_losses) + 1), train_losses, color="blue", label="Training Loss")
    axes[4].plot(np.arange(1, len(test_losses) + 1), test_losses, color="red", label="Test Loss")
    axes[4].set_title("Training Loss vs Test Loss")
    axes[4].legend(loc="upper right")

    axes[5].plot(gt_pnl, color="green", label="Omniscient PnL")
    axes[5].plot(pred_pnl, color="red", label="Prediction PnL")
    axes[5].set_title("Omniscient PnL vs Prediction PnL")
    axes[5].legend(loc="upper left")

    fig.tight_layout(h_pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


async def main():
    cfg = get_variant_config("holdout_regression_only")
    torch.manual_seed(7)


    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    timestamp_utc = ts.strftime("%Y-%m-%d %H:%M:%SZ")
    git_commit = current_git_commit()

    data_dir = Path("/mnt/data/repos/gaelreinaudi/crypto")
    train_files, test_files = choose_walkforward_parquets(data_dir, train_count=3, test_count=1)
    test_file = test_files[0]

    replay_conf_test, shaper_config, X_test_w, Y_test_w, Px_test_w = await load_file_windows(test_file, "BTC-USD", max_windows=48)
    if len(X_test_w) < 12:
        raise RuntimeError(f"Too few test windows loaded from {test_file}: {len(X_test_w)}")

    X_train_w, Y_train_w = [], []
    train_file_names = []
    for train_file in train_files:
        train_file_names.append(train_file.name)
        _rc, _sc, Xw, Yw, _Pxw = await load_file_windows(train_file, "BTC-USD", max_windows=48)
        X_train_w.extend(Xw)
        Y_train_w.extend(Yw)

    if len(X_train_w) < 24:
        raise RuntimeError(f"Too few train windows loaded from {train_files}: {len(X_train_w)}")

    precheck_index = max(range(len(Y_test_w)), key=lambda i: float(np.sum(Y_test_w[i])))
    precheck_path = Path(f"experiments/pictures/exp15_multifile_btc_regonly_precheck_{stamp}.png")
    save_precheck(precheck_path, Px_test_w[precheck_index], X_test_w[precheck_index], Y_test_w[precheck_index], test_file.name)
    richness = richness_gate(X_test_w[precheck_index], Y_test_w[precheck_index][..., None])

    notes_dir = Path("experiments/notes")
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes_path = notes_dir / f"exp15_multifile_btc_regonly_{stamp}.md"
    result_path = Path(f"experiments/results/exp15_multifile_btc_regonly_{stamp}.json")
    dashboard_best_path = Path(f"experiments/pictures/exp15_multifile_btc_regonly_best_{stamp}.png")
    dashboard_fixed_path = Path(f"experiments/pictures/exp15_multifile_btc_regonly_fixed_{stamp}.png")

    if not richness["usable"]:
        notes = (
            "# exp13 multifile btc walkforward — rejected before training\n\n"
            f"- timestamp: {timestamp_utc}\n"
            f"- test parquet: `{test_file}`\n"
            "- hypothesis: walk-forward holdout on a different parquet should provide a more honest BTC generalization test.\n"
            f"- observation: richness gate failed with reason `{richness['reason']}`\n"
            f"- precheck image: `{precheck_path}`\n"
            "- decision: reject slice before training\n"
        )
        notes_path.write_text(notes)
        payload = {
            "experiment": "exp15_multifile_btc_regonly",
            "timestamp_utc": timestamp_utc,
            "git_commit": git_commit,
            "train_files": [str(p) for p in train_files],
            "test_file": str(test_file),
            "hypothesis": "walk-forward holdout on a different parquet should provide a more honest BTC generalization test.",
            "richness_gate": richness,
            "artifacts": {"precheck_png": str(precheck_path), "notes_md": str(notes_path)},
            "decision": "rejected_before_training",
        }
        result_path.write_text(json.dumps(payload, indent=2))
        register_experiment_run(
            experiment="exp15_multifile_btc_regonly",
            variant="rejected_precheck",
            timestamp_utc=timestamp_utc,
            git_commit=git_commit,
            symbol="BTC-USD",
            cadence="100ms",
            look_ahead=64,
            model="none",
            result_json_path=result_path,
            picture_path=precheck_path,
            metrics=richness,
            notes="rejected before training due to richness gate",
        )
        print(json.dumps(payload, indent=2))
        return

    X_train = np.stack([x.reshape(x.shape[0], -1) for x in X_train_w], axis=0)
    Y_train = np.stack(Y_train_w, axis=0)
    X_test = np.stack([x.reshape(x.shape[0], -1) for x in X_test_w], axis=0)
    Y_test = np.stack(Y_test_w, axis=0)
    PX_test = np.stack(Px_test_w, axis=0)
    books_test = np.stack(X_test_w, axis=0)

    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    device = torch.device("cpu")
    model = TinyTCN(in_ch=X_train.shape[-1], hid=96, out_ch=Y_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["pos_weight"], device=device))
    tau = 1e-4

    xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(Y_train).to(device)
    xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(Y_test).to(device)

    train_losses, test_losses = [], []
    epochs = 6
    bs = 8
    for _ep in range(epochs):
        perm = torch.randperm(xtr.shape[0])
        model.train()
        total = 0.0
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i : i + bs]
            xb = xtr[idx]
            yb = ytr[idx]
            reg, cls_logits = model(xb)
            evt_target = (yb > tau).float()
            loss_reg = huber(reg, yb)
            loss_evt = bce(cls_logits, evt_target)
            loss = loss_reg + cfg["event_loss_weight"] * loss_evt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        train_losses.append(total / max(1, (xtr.shape[0] + bs - 1) // bs))
        model.eval()
        with torch.no_grad():
            pred_te, cls_te = model(xte)
            evt_target = (yte > tau).float()
            test_loss = float((huber(pred_te, yte) + cfg["event_loss_weight"] * bce(cls_te, evt_target)).item())
        test_losses.append(test_loss)

    model.eval()
    with torch.no_grad():
        pred_test = model(xte)[0].cpu().numpy()

    pred_cap_value = float(np.percentile(Y_train, 99.5))
    pred_test = apply_prediction_cap(pred_test, cap_value=pred_cap_value)

    metrics = event_metrics(Y_test, pred_test, thr=tau)
    zero_metrics = event_metrics(Y_test, np.zeros_like(Y_test), thr=tau)

    Y_test_seq = Y_test.reshape(-1, Y_test.shape[-1])
    pred_seq = pred_test.reshape(-1, pred_test.shape[-1])
    px_seq = PX_test.reshape(-1, PX_test.shape[-1])
    books_seq = books_test.reshape(-1, books_test.shape[-2], books_test.shape[-1])

    horizon = min(520, Y_test_seq.shape[0])
    best_start = _best_activity_start(Y_test_seq, horizon)
    fixed_start = 0

    def slice_and_score(start: int, out_path: Path, label: str):
        end = start + horizon
        true_slice = Y_test_seq[start:end]
        pred_slice = pred_seq[start:end]
        px_slice = px_seq[start:end]
        books_slice = books_seq[start:end]
        gt_strategy = Strategy(threshold=0.2)
        pred_strategy = Strategy(threshold=cfg["trade_threshold"])
        gt_pnl, gt_pos, _, _ = gt_strategy.compute_pnl(px_slice, true_slice[..., None])
        pred_pnl, pred_pos, _, _ = pred_strategy.compute_pnl(px_slice, pred_slice[..., None])
        save_dashboard(
            out_path,
            prices=px_slice,
            books_array=books_slice,
            true_map=true_slice,
            pred_map=pred_slice,
            train_losses=train_losses,
            test_losses=test_losses,
            gt_pnl=gt_pnl,
            pred_pnl=pred_pnl,
            gt_pos=gt_pos,
            pred_pos=pred_pos,
            title_suffix=label,
        )
        return gt_pnl, pred_pnl

    gt_pnl_best, pred_pnl_best = slice_and_score(best_start, dashboard_best_path, "best activity test slice")
    gt_pnl_fixed, pred_pnl_fixed = slice_and_score(fixed_start, dashboard_fixed_path, "fixed first test slice")

    observations = [
        f"Richness gate passed on holdout file: books_std={richness['books_std']:.4f}, target_active_ratio={richness['target_active_ratio']:.6f}",
        f"Used different parquet files for training ({len(train_files)}) and testing (1 holdout)",
        f"Applied prediction cap at train q99.5={pred_cap_value:.4f}",
        f"Holdout metrics: f1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, rmse={metrics['rmse']:.5f}",
        f"Zero baseline holdout rmse={zero_metrics['rmse']:.5f}",
        f"Best-slice pnl: omniscient={float(gt_pnl_best[-1]):.5f}, prediction={float(pred_pnl_best[-1]):.5f}",
        f"Fixed-slice pnl: omniscient={float(gt_pnl_fixed[-1]):.5f}, prediction={float(pred_pnl_fixed[-1]):.5f}",
    ]

    rmse_ok = metrics["rmse"] <= zero_metrics["rmse"] * 1.20
    pnl_ok = pred_pnl_fixed[-1] > 0
    precision_ok = metrics["precision"] >= 0.10
    decision = "promising" if (rmse_ok and pnl_ok and precision_ok) else "not_promising_yet"

    hypothesis = cfg["hypothesis"]

    notes = "# exp15 multifile btc regonly\n\n" + "\n".join([
        f"- timestamp: {timestamp_utc}",
        f"- train files:",
        *[f"  - `{name}`" for name in train_file_names],
        f"- test file: `{test_file.name}`",
        f"- symbol: `BTC-USD`",
        f"- mutation: pure regression-style training (event loss weight 0) with multi-file holdout and prediction cap",
        f"- hypothesis: {hypothesis}",
        f"- precheck image: `{precheck_path}`",
        f"- best-slice dashboard: `{dashboard_best_path}`",
        f"- fixed-slice dashboard: `{dashboard_fixed_path}`",
        f"- result json: `{result_path}`",
        f"- observations:",
        *[f"  - {obs}" for obs in observations],
        f"- decision: {decision}",
        f"- next mutation: if precision is still weak, keep this multi-file split and change one factor that reduces event over-triggering rather than touching the split again.",
    ]) + "\n"
    notes_path.write_text(notes)

    payload = {
        "experiment": "exp15_multifile_btc_regonly",
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "train_files": [str(p) for p in train_files],
        "test_file": str(test_file),
        "replay_config_test": replay_conf_test.model_dump(mode="json"),
        "shaper_config": shaper_config.model_dump(mode="json"),
        "hypothesis": hypothesis,
        "richness_gate": richness,
        "mutation": {
            "type": "walkforward_multifile_holdout_plus_regression_only",
            "cap_value": pred_cap_value,
            "reference": "train_target_q99.5",
        },
        "metrics": metrics,
        "zero_baseline": zero_metrics,
        "pnl": {
            "best_slice": {"omniscient_final": float(gt_pnl_best[-1]), "prediction_final": float(pred_pnl_best[-1])},
            "fixed_slice": {"omniscient_final": float(gt_pnl_fixed[-1]), "prediction_final": float(pred_pnl_fixed[-1])},
        },
        "artifacts": {
            "precheck_png": str(precheck_path),
            "dashboard_best_png": str(dashboard_best_path),
            "dashboard_fixed_png": str(dashboard_fixed_path),
            "notes_md": str(notes_path),
        },
        "observations": observations,
        "decision": decision,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2))

    row_id = register_experiment_run(
        experiment="exp15_multifile_btc_regonly",
        variant="tinytcn_h64_btc_walkforward_regonly",
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        symbol="BTC-USD",
        cadence="100ms",
        look_ahead=64,
        model="TinyTCN",
        result_json_path=result_path,
        picture_path=dashboard_fixed_path,
        metrics={**metrics, "fixed_slice_prediction_final_pnl": float(pred_pnl_fixed[-1])},
        notes=f"notes_file={notes_path}",
    )
    payload["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {result_path}")
    print(f"Notes: {notes_path}")
    print(f"Best dashboard: {dashboard_best_path}")
    print(f"Fixed dashboard: {dashboard_fixed_path}")


if __name__ == "__main__":
    asyncio.run(main())
