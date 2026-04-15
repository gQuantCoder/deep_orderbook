import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deep_orderbook.btc_experiment_config import (
    choose_training_device,
    resolve_train_test_files,
    summarize_dataset_scale,
)
from deep_orderbook.btc_search_lab import get_batch_variant
from deep_orderbook.experiment_tracking import register_experiment_run
from deep_orderbook.strategy import Strategy
from deep_orderbook.strategy_search import evaluate_long_strategy, evaluate_short_strategy
from deep_orderbook.trigger_search import (
    build_friction_table,
    build_train_calibrated_strategy_grid,
    format_trigger_sweep_summary,
    score_strategy_result,
    summarize_route,
)
from scripts.exp16_batch_h64_tcn import (
    TinyTCN,
    current_git_commit,
    event_metrics,
    filter_event_windows,
    load_file_windows,
    regression_loss,
    save_dashboard,
    save_precheck,
)
from deep_orderbook.scientist_experiment import richness_gate

DEFAULT_TRAIN_FILES = [
    "/media/photoDS216/crypto/2025-03-10T14-00-32.parquet",
    "/media/photoDS216/crypto/2025-03-10T17-00-33.parquet",
    "/media/photoDS216/crypto/2025-03-11T00-00-11.parquet",
    "/media/photoDS216/crypto/2025-03-11T14-00-33.parquet",
    "/media/photoDS216/crypto/2025-03-11T15-00-33.parquet",
    "/media/photoDS216/crypto/2025-04-07T14-00-33.parquet",
    "/media/photoDS216/crypto/2025-04-07T15-00-33.parquet",
    "/media/photoDS216/crypto/2025-04-09T17-00-32.parquet",
]
DEFAULT_TEST_FILES = ["/media/photoDS216/crypto/2025-04-09T18-00-33.parquet"]
DEFAULT_VARIANTS = [
    "l1_evt005_pw2_h64",
    "precision_evt005_pw2_thr010",
    "regonly_wd1e3",
    "regonly_huber_thr010",
]


def _best_activity_start(true_seq: np.ndarray, horizon: int) -> int:
    per_t = true_seq.sum(axis=1)
    if per_t.size <= horizon:
        return 0
    kernel = np.ones(horizon, dtype=np.float64)
    score = np.convolve(per_t, kernel, mode="valid")
    return int(np.argmax(score))


async def main(variants: list[str], train_files: list[Path], test_files: list[Path], label: str, directions: list[str]) -> None:
    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    timestamp_utc = ts.strftime("%Y-%m-%d %H:%M:%SZ")
    git_commit = current_git_commit()
    wall_start = time.perf_counter()

    data_dir = train_files[0].parent
    train_files, test_files = resolve_train_test_files(
        data_dir=data_dir,
        explicit_train_files=train_files,
        explicit_test_files=test_files,
        train_count=3,
        test_count=1,
    )
    test_file = test_files[0]

    replay_conf_test, shaper_config, X_test_w, Y_test_w, Px_test_w = await load_file_windows(test_file, "BTC-USD", max_windows=None)
    test_windows_before = len(X_test_w)
    X_train_w, Y_train_w, Px_train_w = [], [], []
    for train_file in train_files:
        _rc, _sc, Xw, Yw, Pxw = await load_file_windows(train_file, "BTC-USD", max_windows=None)
        X_train_w.extend(Xw)
        Y_train_w.extend(Yw)
        Px_train_w.extend(Pxw)
    train_windows_before = len(X_train_w)

    X_test_w, Y_test_w, Px_test_w, test_event_summary = filter_event_windows(X_test_w, Y_test_w, Px_test_w, top_fraction=0.35, min_count=32)
    X_train_w, Y_train_w, Px_train_w, train_event_summary = filter_event_windows(X_train_w, Y_train_w, Px_train_w, top_fraction=0.35, min_count=128)

    precheck_path = Path(f"experiments/pictures/{label}_holdout_precheck_{stamp}.png")
    save_precheck(precheck_path, Px_test_w[0], X_test_w[0], Y_test_w[0], test_file.name)
    richness = richness_gate(X_test_w[0], Y_test_w[0][..., None])
    if not richness["usable"]:
        raise RuntimeError(f"Richness gate failed: {richness}")

    data_scale = summarize_dataset_scale(
        train_files=train_files,
        test_files=test_files,
        train_windows_before_filter=train_windows_before,
        train_windows_after_filter=len(X_train_w),
        test_windows_before_filter=test_windows_before,
        test_windows_after_filter=len(X_test_w),
        rolling_window_size=shaper_config.rolling_window_size,
        target_levels=2 * shaper_config.look_ahead_side_width,
        max_windows_per_file=None,
    )
    event_filter = {"enabled": True, "top_fraction": 0.35, "train": train_event_summary, "test": test_event_summary}

    X_train = np.stack([x.reshape(x.shape[0], -1) for x in X_train_w], axis=0)
    Y_train = np.stack(Y_train_w, axis=0)
    PX_train = np.stack(Px_train_w, axis=0)
    X_test = np.stack([x.reshape(x.shape[0], -1) for x in X_test_w], axis=0)
    Y_test = np.stack(Y_test_w, axis=0)
    PX_test = np.stack(Px_test_w, axis=0)
    books_test = np.stack(X_test_w, axis=0)

    device = torch.device(choose_training_device(prefer_cuda=True, cuda_available=torch.cuda.is_available()))
    pictures_dir = Path("experiments/pictures")
    all_routes = []

    for variant_index, variant_name in enumerate(variants):
        cfg = get_batch_variant(variant_name)
        seed = 71 + variant_index
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        mu = X_train.mean(axis=(0, 1), keepdims=True)
        sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = TinyTCN(in_ch=X_train_n.shape[-1], hid=cfg["hidden_dim"], out_ch=Y_train.shape[-1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["pos_weight"], device=device))
        xtr = torch.from_numpy(X_train_n).to(device)
        ytr = torch.from_numpy(Y_train).to(device)
        xte = torch.from_numpy(X_test_n).to(device)
        yte = torch.from_numpy(Y_test).to(device)
        tau = 1e-4
        bs = 16
        train_losses, test_losses = [], []
        start_variant = time.perf_counter()
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(xtr.shape[0], device=device)
            model.train()
            total = 0.0
            for i in range(0, xtr.shape[0], bs):
                idx = perm[i:i + bs]
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
                test_losses.append(float((loss_reg + cfg["event_loss_weight"] * loss_evt).item()))

        with torch.no_grad():
            pred_train = model(xtr)[0].detach().cpu().numpy()
            pred_test = model(xte)[0].detach().cpu().numpy()
        cap = float(np.percentile(Y_train, cfg["prediction_cap_quantile"]))
        pred_train = np.clip(pred_train, 0.0, cap)
        pred_test = np.clip(pred_test, 0.0, cap)

        metrics = event_metrics(Y_test, pred_test, thr=tau)
        zero_metrics = event_metrics(Y_test, np.zeros_like(Y_test), thr=tau)
        rmse_ratio = metrics["rmse"] / max(zero_metrics["rmse"], 1e-9)

        train_seq = pred_train.reshape(-1, pred_train.shape[-1])
        true_seq = Y_test.reshape(-1, Y_test.shape[-1])
        pred_seq = pred_test.reshape(-1, pred_test.shape[-1])
        px_seq = PX_test.reshape(-1, PX_test.shape[-1])
        books_seq = books_test.reshape(-1, books_test.shape[-2], books_test.shape[-1])
        gt_strategy = Strategy(threshold=0.2)
        gt_pnl, gt_pos, _, _ = gt_strategy.compute_pnl(px_seq, true_seq[..., None])

        strategy_grid = build_train_calibrated_strategy_grid(train_seq)
        for direction in directions:
            strategy_fn = evaluate_long_strategy if direction == "long" else evaluate_short_strategy
            for strat_cfg in strategy_grid:
                route = strategy_fn(px_seq, pred_seq, **{k: strat_cfg[k] for k in ["entry_threshold", "exit_threshold", "side_margin", "persistence", "cooldown", "max_hold"]})
                horizon = min(520, true_seq.shape[0])
                fixed_start = 0
                best_start = _best_activity_start(true_seq, horizon)
                def slice_eval(start: int):
                    end = start + horizon
                    return {
                        "prices": px_seq[start:end],
                        "books": books_seq[start:end],
                        "true": true_seq[start:end],
                        "pred": pred_seq[start:end],
                        "gt_pnl": gt_pnl[start:end],
                        "gt_pos": gt_pos[start:end],
                        "pred_pnl": route["pnl"][start:end],
                        "pred_pos": route["positions"][start:end],
                    }
                fixed = slice_eval(fixed_start)
                best = slice_eval(best_start)
                route_name = f"{variant_name}__{direction}__{strat_cfg['name']}"
                fixed_png = pictures_dir / f"{label}_{route_name}_fixed_{stamp}.png"
                best_png = pictures_dir / f"{label}_{route_name}_best_{stamp}.png"
                save_dashboard(fixed_png, fixed["prices"], fixed["books"], fixed["true"], fixed["pred"], train_losses, test_losses, fixed["gt_pnl"], fixed["pred_pnl"], fixed["gt_pos"], fixed["pred_pos"], f"{route_name} fixed slice")
                save_dashboard(best_png, best["prices"], best["books"], best["true"], best["pred"], train_losses, test_losses, best["gt_pnl"], best["pred_pnl"], best["gt_pos"], best["pred_pos"], f"{route_name} best slice")
                image_quality = {"usable": True, "reason": "ok"}
                final_pnl = float(route["final_pnl"])
                trade_count = int(route["trade_count"])
                route_score = score_strategy_result(route, precision=metrics["precision"], f1=metrics["f1"], rmse_ratio=rmse_ratio)
                decision = "promising" if (final_pnl > 0 and rmse_ratio <= 1.2 and metrics["precision"] >= 0.20) else "not_promising_yet"
                result = {
                    "experiment": label,
                    "variant_name": variant_name,
                    "direction": direction,
                    "strategy_name": strat_cfg["name"],
                    "timestamp_utc": timestamp_utc,
                    "git_commit": git_commit,
                    "device": str(device),
                    "train_files": [str(p) for p in train_files],
                    "test_file": str(test_file),
                    "config": cfg,
                    "strategy_config": strat_cfg,
                    "data_scale": data_scale,
                    "event_filter": event_filter,
                    "metrics": metrics,
                    "zero_baseline": zero_metrics,
                    "rmse_ratio": rmse_ratio,
                    "strategy_metrics": {
                        "final_pnl": final_pnl,
                        "trade_count": trade_count,
                        "avg_hold_steps": route["avg_hold_steps"],
                        "market_time_fraction": route["market_time_fraction"],
                        "fixed_slice_pnl": float(fixed["pred_pnl"][-1]),
                        "best_slice_pnl": float(best["pred_pnl"][-1]),
                        "omniscient_fixed_pnl": float(fixed["gt_pnl"][-1]),
                        "omniscient_best_pnl": float(best["gt_pnl"][-1]),
                    },
                    "artifacts": {
                        "precheck_png": str(precheck_path),
                        "dashboard_fixed_png": str(fixed_png),
                        "dashboard_best_png": str(best_png),
                    },
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "route_score": route_score,
                    "decision": decision,
                    "runtime_seconds": round(time.perf_counter() - start_variant, 3),
                }
                result_path = Path(f"experiments/results/{label}_{route_name}_{stamp}.json")
                note_path = Path(f"experiments/notes/{label}_{route_name}_{stamp}.md")
                note_path.write_text(
                    "\n".join(
                        [
                            f"# {label} / {route_name}",
                            "",
                            f"- timestamp: {timestamp_utc}",
                            f"- device: `{device}`",
                            f"- decision: `{decision}`",
                            f"- final_pnl: {final_pnl:.5f}",
                            f"- precision: {metrics['precision']:.4f}",
                            f"- f1: {metrics['f1']:.4f}",
                            f"- rmse_ratio: {rmse_ratio:.4f}",
                            f"- trades: {trade_count}",
                            f"- fixed_png: {fixed_png}",
                            f"- best_png: {best_png}",
                        ]
                    ) + "\n"
                )
                row_id = register_experiment_run(
                    experiment=label,
                    variant=route_name,
                    timestamp_utc=timestamp_utc,
                    git_commit=git_commit,
                    symbol="BTC-USD",
                    cadence="100ms",
                    look_ahead=int(pred_test.shape[-1]),
                    model=variant_name,
                    result_json_path=result_path,
                    picture_path=fixed_png,
                    metrics={
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "rmse": metrics["rmse"],
                        "zero_baseline_rmse": zero_metrics["rmse"],
                        "rmse_ratio": rmse_ratio,
                        "final_pnl": final_pnl,
                        "trade_count": trade_count,
                        "route_score": route_score,
                    },
                    notes=f"notes_file={note_path}",
                )
                result["experiment_db_row_id"] = row_id
                result_path.write_text(json.dumps(result, indent=2))
                all_routes.append(result)

        print(f"DONE {variant_name} trigger_routes={len(strategy_grid)} precision={metrics['precision']:.4f} f1={metrics['f1']:.4f} rmse_ratio={rmse_ratio:.4f}")

    ranked = sorted(all_routes, key=lambda x: (x["route_score"], x["strategy_metrics"]["final_pnl"]), reverse=True)
    top = ranked[0]
    best_by_score = summarize_route(top)
    best_by_pnl = summarize_route(max(ranked, key=lambda route: route["strategy_metrics"]["final_pnl"]))
    friction_table = build_friction_table(ranked[:10])
    summary = {
        "experiment": label,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": str(device),
        "train_files": [str(p) for p in train_files],
        "test_file": str(test_file),
        "data_scale": data_scale,
        "event_filter": event_filter,
        "route_count": len(all_routes),
        "ranked_routes": ranked,
        "best_route": {
            "variant_name": best_by_score["variant_name"],
            "direction": best_by_score["direction"],
            "strategy_name": best_by_score["strategy_name"],
            "route_score": best_by_score["route_score"],
            "final_pnl": best_by_score["final_pnl"],
            "precision": best_by_score["precision"],
            "f1": best_by_score["f1"],
            "rmse_ratio": best_by_score["rmse_ratio"],
            "trade_count": best_by_score["trade_count"],
            "pnl_per_trade": best_by_score["pnl_per_trade"],
        },
        "best_route_by_raw_pnl": {
            "variant_name": best_by_pnl["variant_name"],
            "direction": best_by_pnl["direction"],
            "strategy_name": best_by_pnl["strategy_name"],
            "route_score": best_by_pnl["route_score"],
            "final_pnl": best_by_pnl["final_pnl"],
            "precision": best_by_pnl["precision"],
            "f1": best_by_pnl["f1"],
            "rmse_ratio": best_by_pnl["rmse_ratio"],
            "trade_count": best_by_pnl["trade_count"],
            "pnl_per_trade": best_by_pnl["pnl_per_trade"],
        },
        "friction_table_top10": friction_table,
        "runtime_seconds_total": round(time.perf_counter() - wall_start, 3),
    }
    summary_path = Path(f"experiments/results/{label}_{stamp}.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    notes_path = Path(f"experiments/notes/{label}_{stamp}.md")
    lines = format_trigger_sweep_summary(
        label=label,
        timestamp_utc=timestamp_utc,
        device=str(device),
        data_scale=data_scale,
        runtime_seconds_total=summary["runtime_seconds_total"],
        ranked=ranked,
    )
    notes_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary_json": str(summary_path), "summary_md": str(notes_path), "best_route": best_by_score["label"], "best_route_by_raw_pnl": best_by_pnl["label"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="exp22_trigger_sweep")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--train-files", nargs="*", default=DEFAULT_TRAIN_FILES)
    parser.add_argument("--test-files", nargs="*", default=DEFAULT_TEST_FILES)
    parser.add_argument("--directions", nargs="*", default=["long"], choices=["long", "short"])
    args = parser.parse_args()
    asyncio.run(main(args.variants, [Path(p) for p in args.train_files], [Path(p) for p in args.test_files], args.label, args.directions))
