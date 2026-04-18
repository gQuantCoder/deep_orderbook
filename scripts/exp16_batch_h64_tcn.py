import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_orderbook.btc_experiment_config import (
    choose_training_device,
    resolve_train_test_files,
    summarize_dataset_scale,
)
from deep_orderbook.btc_search_lab import (
    compute_png_quality_stats,
    get_batch_variant,
    list_batch_variant_names,
    rank_variant_results,
)
from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.event_selection import rank_eventful_windows, select_eventful_window_indices
from deep_orderbook.experiment_tracking import register_experiment_run
from deep_orderbook.pipeline_guards import assert_image_meaningful
from deep_orderbook.scientist_experiment import apply_prediction_cap, choose_walkforward_parquets, richness_gate
from deep_orderbook.shaper import iter_shapes_t2l
from deep_orderbook.strategy import Strategy


DEFAULT_SHAPER_CONFIG_2026 = ShaperConfig(
    only_full_arrays=True,
    view_bips=5,
    num_side_lvl=8,
    look_ahead=128,
    look_ahead_side_bips=5,
    look_ahead_side_width=4,
    rolling_window_size=2048,
    window_stride=8,
    use_cache=True,
    save_cache=True,
)


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


async def load_file_windows(
    one_path: Path,
    market: str,
    n_samples: int | None = 48,
    *,
    use_cache: bool = True,
    save_cache: bool = True,
    shaper_config: ShaperConfig | None = None,
    every: str = "100ms",
    allow_microburst: bool = False,
):
    """Shape a single parquet file into rolling-frame samples (books, t2l, prices).

    ``n_samples`` caps how many frames are returned (memory guard for large files).
    ``None`` means take all frames.  The underlying cache is always populated on
    first pass regardless of the cap — subsequent calls load instantly from .npz.

    Default geometry: ``DEFAULT_SHAPER_CONFIG_2026`` (rolling=2048, look_ahead=128).
    Pass a custom ``shaper_config`` or set ``allow_microburst=True`` for shorter windows.
    """
    if shaper_config is None:
        shaper_config = DEFAULT_SHAPER_CONFIG_2026.but(use_cache=use_cache, save_cache=save_cache)
    else:
        shaper_config = shaper_config.but(use_cache=use_cache, save_cache=save_cache)

    replay_conf = ReplayConfig(
        markets=[market],
        one_path=one_path,
        data_dir=one_path.parent,
        date_regexp=one_path.stem,
        max_samples=-1,
        every=every,
    )
    assert_image_meaningful(shaper_config, replay_conf, allow_microburst=allow_microburst)

    Xw, Yw, Pxw = [], [], []
    async for books_array, level_prox, pxar in iter_shapes_t2l(replay_conf, shaper_config, live=False):
        Xw.append(books_array.astype(np.float32))
        Yw.append(level_prox[:, :, 0].astype(np.float32))
        Pxw.append(pxar.astype(np.float32))
        if n_samples is not None and len(Xw) >= n_samples:
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


def filter_event_windows(
    X_windows: list[np.ndarray],
    Y_windows: list[np.ndarray],
    PX_windows: list[np.ndarray],
    *,
    top_fraction: float,
    min_count: int,
    max_count: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    selected = select_eventful_window_indices(PX_windows, X_windows, top_fraction=top_fraction, min_count=min_count, max_count=max_count)
    ranking = rank_eventful_windows(PX_windows, X_windows)
    summary = {
        "top_fraction": top_fraction,
        "selected_count": len(selected),
        "total_count": len(X_windows),
        "selected_indices": selected,
        "top_ranked": ranking[: min(5, len(ranking))],
    }
    return (
        [X_windows[idx] for idx in selected],
        [Y_windows[idx] for idx in selected],
        [PX_windows[idx] for idx in selected],
        summary,
    )



def regression_loss(pred: torch.Tensor, target: torch.Tensor, mode: str, active_reg_weight: float, tau: float) -> torch.Tensor:
    if mode == "huber":
        per = F.smooth_l1_loss(pred, target, reduction="none")
    elif mode == "mse":
        per = F.mse_loss(pred, target, reduction="none")
    elif mode == "l1":
        per = F.l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown reg_loss mode: {mode}")
    if active_reg_weight > 1.0:
        weights = 1.0 + (active_reg_weight - 1.0) * (target > tau).float()
        per = per * weights
    return per.mean()


def train_one_variant(
    variant_name: str,
    cfg: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    PX_test: np.ndarray,
    books_test: np.ndarray,
    out_dir: Path,
    stamp: str,
    train_files: list[Path],
    test_file: Path,
    git_commit: str | None,
    timestamp_utc: str,
    precheck_path: Path,
    richness: dict,
    variant_order: list[str],
    experiment_label: str,
    device: torch.device,
    data_scale: dict,
) -> dict:
    tau = 1e-4
    seed = 7 + variant_order.index(variant_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_n = (X_train - mu) / sd
    X_test_n = (X_test - mu) / sd

    variant_start = time.perf_counter()
    model = TinyTCN(in_ch=X_train_n.shape[-1], hid=cfg["hidden_dim"], out_ch=Y_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["pos_weight"], device=device))

    xtr = torch.from_numpy(X_train_n).to(device)
    ytr = torch.from_numpy(Y_train).to(device)
    xte = torch.from_numpy(X_test_n).to(device)
    yte = torch.from_numpy(Y_test).to(device)

    train_losses, test_losses = [], []
    bs = 8
    for _ep in range(cfg["epochs"]):
        perm = torch.randperm(xtr.shape[0])
        model.train()
        total = 0.0
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i : i + bs]
            xb = xtr[idx]
            yb = ytr[idx]
            reg, cls_logits = model(xb)
            evt_target = (yb > tau).float()
            loss_reg = regression_loss(reg, yb, cfg["reg_loss"], cfg["active_reg_weight"], tau)
            loss_evt = bce(cls_logits, evt_target) if cfg["event_loss_weight"] > 0 else torch.zeros((), device=device)
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
            loss_reg = regression_loss(pred_te, yte, cfg["reg_loss"], cfg["active_reg_weight"], tau)
            loss_evt = bce(cls_te, evt_target) if cfg["event_loss_weight"] > 0 else torch.zeros((), device=device)
            test_loss = float((loss_reg + cfg["event_loss_weight"] * loss_evt).item())
        test_losses.append(test_loss)

    model.eval()
    with torch.no_grad():
        pred_test = model(xte)[0].cpu().numpy()

    pred_cap_value = float(np.percentile(Y_train, cfg["prediction_cap_quantile"]))
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
    out_png = out_dir / f"{experiment_label}_{variant_name}_fixed_{stamp}.png"
    best_png = out_dir / f"{experiment_label}_{variant_name}_best_{stamp}.png"

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

    gt_pnl_best, pred_pnl_best = slice_and_score(best_start, best_png, f"{variant_name} best activity test slice")
    gt_pnl_fixed, pred_pnl_fixed = slice_and_score(fixed_start, out_png, f"{variant_name} fixed first test slice")
    image_quality = compute_png_quality_stats(out_png)

    rmse_ok = metrics["rmse"] <= zero_metrics["rmse"] * 1.20
    pnl_ok = pred_pnl_fixed[-1] > 0
    precision_ok = metrics["precision"] >= 0.10
    decision = "promising" if (rmse_ok and pnl_ok and precision_ok and image_quality["usable"]) else "not_promising_yet"

    result_path = Path(f"experiments/results/{experiment_label}_{variant_name}_{stamp}.json")
    notes_path = Path(f"experiments/notes/{experiment_label}_{variant_name}_{stamp}.md")
    observations = [
        f"Variant={variant_name}",
        f"Holdout metrics: f1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, rmse={metrics['rmse']:.5f}",
        f"Zero baseline rmse={zero_metrics['rmse']:.5f}",
        f"Image QC: usable={image_quality['usable']}, reason={image_quality['reason']}, gray_std={image_quality['gray_std']:.4f}, near_black={image_quality['near_black_fraction']:.3f}, near_white={image_quality['near_white_fraction']:.3f}",
        f"Best-slice pnl: omniscient={float(gt_pnl_best[-1]):.5f}, prediction={float(pred_pnl_best[-1]):.5f}",
        f"Fixed-slice pnl: omniscient={float(gt_pnl_fixed[-1]):.5f}, prediction={float(pred_pnl_fixed[-1]):.5f}",
    ]
    notes = "# exp16 batch variant\n\n" + "\n".join([
        f"- timestamp: {timestamp_utc}",
        f"- variant: `{variant_name}`",
        f"- train files:",
        *[f"  - `{p.name}`" for p in train_files],
        f"- test file: `{test_file.name}`",
        f"- config: `{json.dumps(cfg, sort_keys=True)}`",
        f"- precheck image: `{precheck_path}`",
        f"- fixed dashboard: `{out_png}`",
        f"- best dashboard: `{best_png}`",
        f"- result json: `{result_path}`",
        f"- observations:",
        *[f"  - {obs}" for obs in observations],
        f"- decision: {decision}",
    ]) + "\n"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(notes)

    payload = {
        "experiment": experiment_label,
        "variant_name": variant_name,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": str(device),
        "train_files": [str(p) for p in train_files],
        "test_file": str(test_file),
        "hypothesis": cfg["hypothesis"],
        "config": cfg,
        "richness_gate": richness,
        "data_scale": data_scale,
        "metrics": metrics,
        "zero_baseline": zero_metrics,
        "image_quality": image_quality,
        "pnl": {
            "best_slice": {"omniscient_final": float(gt_pnl_best[-1]), "prediction_final": float(pred_pnl_best[-1])},
            "fixed_slice": {"omniscient_final": float(gt_pnl_fixed[-1]), "prediction_final": float(pred_pnl_fixed[-1])},
        },
        "train_losses": train_losses,
        "test_losses": test_losses,
        "artifacts": {
            "precheck_png": str(precheck_path),
            "dashboard_fixed_png": str(out_png),
            "dashboard_best_png": str(best_png),
            "notes_md": str(notes_path),
        },
        "observations": observations,
        "decision": decision,
        "runtime_seconds": round(time.perf_counter() - variant_start, 3),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2))
    row_id = register_experiment_run(
        experiment=experiment_label,
        variant=variant_name,
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        symbol="BTC-USD",
        cadence="100ms",
        look_ahead=64,
        model="TinyTCN",
        result_json_path=result_path,
        picture_path=out_png,
        metrics={**metrics, "route_score_hint": 0.0, "fixed_slice_prediction_final_pnl": float(pred_pnl_fixed[-1])},
        notes=f"notes_file={notes_path}",
    )
    payload["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(payload, indent=2))
    return payload


async def main(
    variant_names: list[str] | None = None,
    experiment_label: str = "exp16_batch_h64_tcn",
    eventful_top_fraction: float | None = None,
    min_train_windows: int = 24,
    min_test_windows: int = 12,
    n_samples_per_file: int | None = 48,
    explicit_train_files: list[Path] | None = None,
    explicit_test_files: list[Path] | None = None,
    prefer_cuda: bool = True,
) -> None:
    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    timestamp_utc = ts.strftime("%Y-%m-%d %H:%M:%SZ")
    git_commit = current_git_commit()
    wall_start = time.perf_counter()

    data_dir = Path("/mnt/data/repos/gaelreinaudi/crypto")
    all_files = sorted(data_dir.glob("*.parquet"))
    train_files, test_files = resolve_train_test_files(
        data_dir=data_dir,
        explicit_train_files=explicit_train_files,
        explicit_test_files=explicit_test_files,
        train_count=3,
        test_count=1,
    )
    test_file = test_files[0]

    replay_conf_test, shaper_config, X_test_w, Y_test_w, Px_test_w = await load_file_windows(test_file, "BTC-USD", n_samples=n_samples_per_file)
    test_windows_before_filter = len(X_test_w)
    if len(X_test_w) < min_test_windows:
        raise RuntimeError(f"Too few test windows loaded from {test_file}: {len(X_test_w)}")

    event_filter_summary: dict | None = None
    if eventful_top_fraction is not None:
        X_test_w, Y_test_w, Px_test_w, test_event_summary = filter_event_windows(
            X_test_w,
            Y_test_w,
            Px_test_w,
            top_fraction=eventful_top_fraction,
            min_count=min_test_windows,
        )
        event_filter_summary = {
            "enabled": True,
            "top_fraction": eventful_top_fraction,
            "test": test_event_summary,
        }
        if len(X_test_w) < min_test_windows:
            raise RuntimeError(f"Event filter left too few test windows for {test_file}: {len(X_test_w)}")

    X_train_w, Y_train_w, Px_train_w = [], [], []
    train_file_pool = list(train_files)
    older_candidates = list(reversed(all_files[: max(0, len(all_files) - 4)]))
    expanded_with: list[str] = []
    for train_file in train_file_pool:
        _rc, _sc, Xw, Yw, Pxw = await load_file_windows(train_file, "BTC-USD", n_samples=n_samples_per_file)
        X_train_w.extend(Xw)
        Y_train_w.extend(Yw)
        Px_train_w.extend(Pxw)
    train_windows_before_filter = len(X_train_w)

    if eventful_top_fraction is not None:
        filtered_train = filter_event_windows(
            X_train_w,
            Y_train_w,
            Px_train_w,
            top_fraction=eventful_top_fraction,
            min_count=min_train_windows,
        )
        while len(filtered_train[0]) < min_train_windows and older_candidates:
            extra_file = older_candidates.pop(0)
            train_file_pool.insert(0, extra_file)
            expanded_with.append(extra_file.name)
            _rc, _sc, Xw, Yw, Pxw = await load_file_windows(extra_file, "BTC-USD", n_samples=n_samples_per_file)
            X_train_w.extend(Xw)
            Y_train_w.extend(Yw)
            Px_train_w.extend(Pxw)
            filtered_train = filter_event_windows(
                X_train_w,
                Y_train_w,
                Px_train_w,
                top_fraction=eventful_top_fraction,
                min_count=min_train_windows,
            )
        X_train_w, Y_train_w, Px_train_w, train_event_summary = filtered_train
        event_filter_summary["train"] = train_event_summary
        event_filter_summary["train"]["expanded_with_older_files"] = expanded_with
    if len(X_train_w) < min_train_windows:
        raise RuntimeError(f"Too few train windows loaded from {train_file_pool}: {len(X_train_w)}")

    if event_filter_summary:
        precheck_index = 0
        precheck_title = f"{test_file.name} event-ranked holdout"
    else:
        precheck_index = max(range(len(Y_test_w)), key=lambda i: float(np.sum(Y_test_w[i])))
        precheck_title = test_file.name
    precheck_path = Path(f"experiments/pictures/{experiment_label}_holdout_precheck_{stamp}.png")
    save_precheck(precheck_path, Px_test_w[precheck_index], X_test_w[precheck_index], Y_test_w[precheck_index], precheck_title)
    richness = richness_gate(X_test_w[precheck_index], Y_test_w[precheck_index][..., None])
    if not richness["usable"]:
        raise RuntimeError(f"Richness gate failed for holdout file {test_file}: {richness}")

    test_windows_after_filter = len(X_test_w)
    train_windows_after_filter = len(X_train_w)
    data_scale = summarize_dataset_scale(
        train_files=train_file_pool,
        test_files=test_files,
        train_windows_before_filter=train_windows_before_filter,
        train_windows_after_filter=train_windows_after_filter,
        test_windows_before_filter=test_windows_before_filter,
        test_windows_after_filter=test_windows_after_filter,
        rolling_window_size=shaper_config.rolling_window_size,
        target_levels=2 * shaper_config.look_ahead_side_width,
        n_samples_per_file=n_samples_per_file,
    )
    device = torch.device(choose_training_device(prefer_cuda=prefer_cuda, cuda_available=torch.cuda.is_available()))

    X_train = np.stack([x.reshape(x.shape[0], -1) for x in X_train_w], axis=0)
    Y_train = np.stack(Y_train_w, axis=0)
    X_test = np.stack([x.reshape(x.shape[0], -1) for x in X_test_w], axis=0)
    Y_test = np.stack(Y_test_w, axis=0)
    PX_test = np.stack(Px_test_w, axis=0)
    books_test = np.stack(X_test_w, axis=0)

    pictures_dir = Path("experiments/pictures")
    results = []
    variant_order = variant_names or list_batch_variant_names()
    for variant_name in variant_order:
        cfg = get_batch_variant(variant_name)
        payload = train_one_variant(
            variant_name=variant_name,
            cfg=cfg,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            PX_test=PX_test,
            books_test=books_test,
            out_dir=pictures_dir,
            stamp=stamp,
            train_files=train_file_pool,
            test_file=test_file,
            git_commit=git_commit,
            timestamp_utc=timestamp_utc,
            precheck_path=precheck_path,
            richness=richness,
            variant_order=variant_order,
            experiment_label=experiment_label,
            device=device,
            data_scale=data_scale,
        )
        if event_filter_summary is not None:
            payload["event_filter"] = event_filter_summary
            result_path = Path(f"experiments/results/{experiment_label}_{variant_name}_{stamp}.json")
            result_path.write_text(json.dumps(payload, indent=2))
        results.append(payload)
        print(f"DONE {variant_name} precision={payload['metrics']['precision']:.4f} f1={payload['metrics']['f1']:.4f} rmse={payload['metrics']['rmse']:.5f}")

    ranked = rank_variant_results(results)
    top = ranked[0]
    summary = {
        "experiment": experiment_label,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": str(device),
        "replay_config_test": replay_conf_test.model_dump(mode="json"),
        "shaper_config": shaper_config.model_dump(mode="json"),
        "train_files": [str(p) for p in train_file_pool],
        "test_file": str(test_file),
        "precheck_png": str(precheck_path),
        "richness_gate": richness,
        "data_scale": data_scale,
        "event_filter": event_filter_summary,
        "ranked_results": ranked,
        "best_variant": top["variant_name"],
        "best_route_score": top["route_score"],
        "status": "promising" if top["decision"] == "promising" else "not_promising_yet",
        "runtime_seconds_total": round(time.perf_counter() - wall_start, 3),
        "next_step": "hold mapper family fixed and calibrate event-trigger extraction on eventful windows if precision improves without RMSE blow-up",
    }
    summary_path = Path(f"experiments/results/{experiment_label}_{stamp}.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# exp16 batch h64 tcn search",
        "",
        f"- timestamp: {timestamp_utc}",
        f"- device: `{device}`",
        f"- train files: {', '.join(p.name for p in train_file_pool)}",
        f"- test file: `{test_file.name}`",
        f"- precheck image: `{precheck_path}`",
        f"- summary json: `{summary_path}`",
        f"- runtime_seconds_total: {summary['runtime_seconds_total']:.3f}",
        f"- train windows after filter: {data_scale['train_windows_after_filter']}",
        f"- test windows after filter: {data_scale['test_windows_after_filter']}",
        f"- train timesteps: {data_scale['train_timesteps']}",
        f"- test timesteps: {data_scale['test_timesteps']}",
        f"- max_windows_per_file: {data_scale['max_windows_per_file']}",
    ]
    if event_filter_summary is not None:
        md_lines.extend([
            f"- event filter: top_fraction={eventful_top_fraction:.2f}",
            f"- selected train windows: {event_filter_summary['train']['selected_count']} / {event_filter_summary['train']['total_count']}",
            f"- selected test windows: {event_filter_summary['test']['selected_count']} / {event_filter_summary['test']['total_count']}",
            f"- older fallback files added: {', '.join(event_filter_summary['train']['expanded_with_older_files']) or 'none'}",
        ])
    md_lines.extend([
        f"- top route: `{top['variant_name']}` score={top['route_score']:.4f}",
        f"- top metrics: precision={top['metrics']['precision']:.4f}, f1={top['metrics']['f1']:.4f}, rmse={top['metrics']['rmse']:.5f}, fixed pnl={top['pnl']['fixed_slice']['prediction_final']:.5f}",
        f"- top image qc: usable={top['image_quality']['usable']}, reason={top['image_quality']['reason']}, gray_std={top['image_quality']['gray_std']:.4f}",
        "",
        "## ranked variants",
    ])
    for idx, item in enumerate(ranked, start=1):
        md_lines.append(
            f"- {idx}. `{item['variant_name']}` score={item['route_score']:.4f} precision={item['metrics']['precision']:.4f} f1={item['metrics']['f1']:.4f} rmse={item['metrics']['rmse']:.5f} fixed_pnl={item['pnl']['fixed_slice']['prediction_final']:.5f} image={item['image_quality']['reason']}"
        )
    notes_path = Path(f"experiments/notes/{experiment_label}_{stamp}.md")
    notes_path.write_text("\n".join(md_lines) + "\n")

    print(json.dumps({"summary_json": str(summary_path), "summary_md": str(notes_path), "best_variant": top['variant_name']}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=None, help="subset of batch variant names to run")
    parser.add_argument("--label", default="exp16_batch_h64_tcn", help="artifact prefix / experiment label")
    parser.add_argument("--eventful-top-fraction", type=float, default=None, help="keep only the most eventful windows by realized move/activity score")
    parser.add_argument("--min-train-windows", type=int, default=24, help="minimum train windows after filtering")
    parser.add_argument("--min-test-windows", type=int, default=12, help="minimum test windows after filtering")
    parser.add_argument("--n-samples-per-file", type=int, default=48, dest="n_samples_per_file", help="max rolling-frame samples per parquet file; 0 = unlimited")
    parser.add_argument("--train-files", nargs="*", default=None, help="explicit train parquet file paths")
    parser.add_argument("--test-files", nargs="*", default=None, help="explicit test parquet file paths")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()
    n_samples = None if args.n_samples_per_file == 0 else args.n_samples_per_file
    train_files = [Path(p) for p in args.train_files] if args.train_files else None
    test_files = [Path(p) for p in args.test_files] if args.test_files else None
    asyncio.run(
        main(
            variant_names=args.variants,
            experiment_label=args.label,
            eventful_top_fraction=args.eventful_top_fraction,
            min_train_windows=args.min_train_windows,
            min_test_windows=args.min_test_windows,
            n_samples_per_file=n_samples,
            explicit_train_files=train_files,
            explicit_test_files=test_files,
            prefer_cuda=not args.cpu,
        )
    )
