from __future__ import annotations

from collections.abc import AsyncGenerator

import numpy as np
import torch
from torch import nn

from deep_orderbook.btc_search_lab import get_batch_variant
from deep_orderbook.learn.tcn import OrderBook2DTCN, assert_min_receptive_field_2d
from deep_orderbook.research_core import ACTIVE_SHORT_ROUTE_NAMES
from deep_orderbook.trigger_search import build_train_calibrated_strategy_grid
from scripts.exp16_batch_h64_tcn import (
    DEFAULT_SHAPER_CONFIG_2026,
    filter_event_windows,
    load_file_windows,
    regression_loss,
)

# Canonical shaper for the current branch — matches exp64
SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


def _pick_watch_index(y_test: np.ndarray) -> int:
    return int(np.argmax(y_test.sum(axis=(1, 2))))


async def train_and_predict(
    config,
    replay_config,
    shaper_config,
    test_config,
    variant_name: str = "regonly_l1_2d_h8",
    market: str = "BTC-USD",
    train_top_fraction: float = 0.35,
    test_top_fraction: float = 0.35,
    train_n_samples_per_file: int | None = None,
    test_n_samples_per_file: int | None = None,
    display_every_batches: int = 10,
    move_test_window: bool = True,
) -> AsyncGenerator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float], None]:
    cfg = get_batch_variant(variant_name)

    train_files = replay_config.file_list()
    if not train_files:
        raise RuntimeError(f"No train files resolved from {replay_config}")

    # file_list() already handles one_path internally
    test_files = test_config.file_list()
    if not test_files:
        raise RuntimeError(f"No test files resolved from {test_config}")
    test_file = test_files[0]

    x_train_w, y_train_w, px_train_w = [], [], []
    for train_file in train_files:
        _rc, _sc, xw, yw, pxw = await load_file_windows(
            train_file,
            market,
            n_samples=train_n_samples_per_file,
            shaper_config=shaper_config,
            every=replay_config.every,
            allow_microburst=False,
        )
        x_train_w.extend(xw)
        y_train_w.extend(yw)
        px_train_w.extend(pxw)

    _rc_test, _sc_test, x_test_w, y_test_w, px_test_w = await load_file_windows(
        test_file,
        market,
        n_samples=test_n_samples_per_file,
        shaper_config=shaper_config,
        every=test_config.every,
        allow_microburst=False,
    )

    x_test_w, y_test_w, px_test_w, _ = filter_event_windows(
        x_test_w, y_test_w, px_test_w, top_fraction=test_top_fraction, min_count=32
    )
    x_train_w, y_train_w, px_train_w, _ = filter_event_windows(
        x_train_w, y_train_w, px_train_w, top_fraction=train_top_fraction, min_count=128
    )

    x_train = np.stack(x_train_w).astype(np.float32)
    y_train = np.stack(y_train_w).astype(np.float32)
    x_test = np.stack(x_test_w).astype(np.float32)
    y_test = np.stack(y_test_w).astype(np.float32)
    px_test = np.stack(px_test_w).astype(np.float32)

    mu = x_train.mean(axis=(0, 1), keepdims=True)
    sd = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_train_n = (x_train - mu) / sd
    x_test_n = (x_test - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrderBook2DTCN(
        bins=x_train_n.shape[2],
        in_ch=x_train_n.shape[3],
        hid=cfg.hidden_dim,
        out_levels=y_train.shape[-1],
        trunk_dropout=cfg.trunk_dropout,
        num_levels=cfg.n_blocks,
    ).to(device)
    assert_min_receptive_field_2d(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))

    xtr = torch.from_numpy(x_train_n).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    xte = torch.from_numpy(x_test_n).to(device)
    yte = torch.from_numpy(y_test).to(device)

    tau = 1e-4
    cap = float(np.percentile(y_train, cfg.prediction_cap_quantile))
    watch_idx = _pick_watch_index(y_test)
    update_count = 0

    def _display_index() -> int:
        if not move_test_window:
            return watch_idx
        return int((watch_idx + update_count) % len(y_test))

    def _eval_test() -> tuple[np.ndarray, float]:
        """Fast: only test-set inference. Called every display_every_batches."""
        model.eval()
        with torch.no_grad():
            pred_te, cls_te = model(xte)
            evt_te = (yte > tau).float()
            loss_reg = regression_loss(pred_te, yte, cfg.reg_loss, cfg.active_reg_weight, tau)
            loss_evt = bce(cls_te, evt_te) if cfg.event_loss_weight > 0 else torch.zeros((), device=device)
            test_loss = float((loss_reg + cfg.event_loss_weight * loss_evt).item())
        return np.clip(pred_te.detach().cpu().numpy(), 0.0, cap), test_loss

    def _calibrate_routes() -> list[dict]:
        """Slow: full train-set inference to calibrate strategy thresholds. Called once per epoch."""
        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, xtr.shape[0], config.batch_size):
                chunk, _ = model(xtr[i : i + config.batch_size])
                chunks.append(chunk.cpu().numpy())
        train_seq = np.clip(np.concatenate(chunks, axis=0), 0.0, cap).reshape(-1, chunks[0].shape[-1])
        return build_train_calibrated_strategy_grid(train_seq, route_names=ACTIVE_SHORT_ROUTE_NAMES)

    def _make_yield(pred_test: np.ndarray, train_loss: float, test_loss: float, route_configs: list[dict]) -> tuple:
        nonlocal update_count
        idx = _display_index()
        update_count += 1
        return (
            x_test[idx],
            y_test[idx][..., None],
            px_test[idx],
            pred_test[idx][..., None],
            train_loss,
            test_loss,
            route_configs,
        )

    cached_routes: list[dict] = []

    for _epoch in range(config.epochs):
        perm = torch.randperm(xtr.shape[0], device=device)
        model.train()
        total = 0.0
        batches_seen = 0
        for batches_seen, i in enumerate(range(0, xtr.shape[0], config.batch_size), start=1):
            idx = perm[i : i + config.batch_size]
            reg, cls_logits = model(xtr[idx])
            evt_target = (ytr[idx] > tau).float()
            loss_reg = regression_loss(reg, ytr[idx], cfg.reg_loss, cfg.active_reg_weight, tau)
            loss_evt = bce(cls_logits, evt_target) if cfg.event_loss_weight > 0 else torch.zeros((), device=device)
            loss = loss_reg + cfg.event_loss_weight * loss_evt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())

            if display_every_batches > 0 and batches_seen % display_every_batches == 0:
                pred_test, test_loss = _eval_test()
                yield _make_yield(pred_test, total / batches_seen, test_loss, cached_routes)
                model.train()

        pred_test, test_loss = _eval_test()
        cached_routes = _calibrate_routes()  # once per epoch — routes update after each full pass
        yield _make_yield(pred_test, total / batches_seen, test_loss, cached_routes)
