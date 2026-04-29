from __future__ import annotations

import itertools
import queue
import random
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deep_orderbook.btc_search_lab import BatchVariantConfig, get_batch_variant
from deep_orderbook.config import ReplayConfig, ShaperConfig, TrainConfig
from deep_orderbook.event_selection import score_window_eventfulness
from deep_orderbook.learn.data_loader import DataLoaderWorker
from deep_orderbook.learn.tcn import OrderBook2DTCN, assert_min_receptive_field_2d
from deep_orderbook.research_core import ACTIVE_CORE_SHORT_ROUTES, RouteSpec
from deep_orderbook.shaper import iter_shapes_t2l
from deep_orderbook.trigger_search import build_train_calibrated_strategy_grid
from deep_orderbook.utils import logger
from deep_orderbook.visu import set_window_label
from scripts.exp16_batch_h64_tcn import (
    DEFAULT_SHAPER_CONFIG_2026,
    filter_event_windows,
    regression_loss,
)

# Canonical shaper for the current branch — matches exp64
SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)

# Streaming hyperparameters (kept here, not in TrainConfig, to limit blast radius).
TRAIN_TOP_FRACTION = 0.35  # online eventfulness filter — match old filter_event_windows
SHUFFLE_BUFFER_SIZE = 2048  # ~ 2 minutes at 16 batches/s × 16/batch
WARMUP_WINDOWS = 256  # for normalization stats; large enough to be representative
EVENT_SCORE_WINDOW = 512  # rolling percentile horizon for the online filter


async def _load_test_reference(
    test_config: ReplayConfig,
    shaper_config: ShaperConfig,
    test_top_fraction: float,
    test_n_samples_per_file: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Stream ONE test file via ``iter_shapes_t2l``, honoring every field of ``test_config``.

    Iterates ``test_config.file_list()`` in order, picking the first file that yields windows
    after the eventfulness filter. The replay config is forwarded as-is (only ``one_path`` is
    bound per file) so ``markets``, ``every``, ``freq``, ``max_samples``, ``skip_until_time``
    and ``randomize`` all reach the replayer / feed exactly as the caller specified.
    """
    test_files = test_config.file_list()
    if not test_files:
        raise RuntimeError(f"No test files resolved from {test_config}")

    for test_file in test_files:
        per_file_config = test_config.but(one_path=test_file)
        xw: list[np.ndarray] = []
        yw: list[np.ndarray] = []
        pxw: list[np.ndarray] = []
        async for books, level_prox, pxar in iter_shapes_t2l(per_file_config, shaper_config, live=False):
            xw.append(books.astype(np.float32))
            yw.append(level_prox[:, :, 0].astype(np.float32))
            pxw.append(pxar.astype(np.float32))
            if test_n_samples_per_file is not None and len(xw) >= test_n_samples_per_file:
                break
        if not xw:
            continue
        xw, yw, pxw, _ = filter_event_windows(xw, yw, pxw, top_fraction=test_top_fraction, min_count=32)
        if xw:
            return (
                np.stack(xw).astype(np.float32),
                np.stack(yw).astype(np.float32),
                np.stack(pxw).astype(np.float32),
                test_file,
            )
    raise RuntimeError(f"No test windows produced from any of {len(test_files)} test files")


async def train_and_predict(
    config: TrainConfig,
    replay_config: ReplayConfig,
    shaper_config: ShaperConfig,
    test_config: ReplayConfig,
    variant_name: str = "regonly_l1_2d_h8",
    test_top_fraction: float = 0.35,
    test_n_samples_per_file: int | None = None,
    display_every_batches: int = 10,
    batches_per_epoch: int = 64,
    move_test_window: bool = True,
) -> AsyncGenerator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, list[RouteSpec]], None]:
    """Stream training data via ``DataLoaderWorker`` — preserves the four pieces that make SGD work.

    Architecture:
    * Test set: ONE filtered file pinned on GPU as the reference for test-loss + display.
    * Training stream: workers pick a random file each iteration AND set ``randomize=True`` so
      windows within a file arrive in shuffled order (cache-hit path only — see iter_shapes_t2l).
    * Online eventfulness filter: each incoming window is scored; only top ``TRAIN_TOP_FRACTION``
      relative to a rolling percentile reach the buffer. Same effect as exp16's
      ``filter_event_windows`` but stream-friendly.
    * Shuffle buffer: a ``deque`` of ``SHUFFLE_BUFFER_SIZE`` (x, y) pairs; every batch is
      ``random.sample(...)`` from it — restores SGD IID assumption killed by sequential streaming.
    * Normalisation: ``mu``/``sd`` computed from a warmup pass over training data (not test).
    * Calibration: rolling deque of recent train predictions seeds
      ``build_train_calibrated_strategy_grid`` — same routes as exp63/exp64.
    """
    cfg: BatchVariantConfig = get_batch_variant(variant_name)

    if replay_config.markets[:1] != test_config.markets[:1]:
        logger.warning(
            f"[config] train market {replay_config.markets[:1]} != test market {test_config.markets[:1]} "
            f"— normalisation stats and prediction scales will not transfer cleanly"
        )

    x_test, y_test, px_test, test_file = await _load_test_reference(
        test_config, shaper_config, test_top_fraction, test_n_samples_per_file
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrderBook2DTCN(
        bins=x_test.shape[2],
        in_ch=x_test.shape[3],
        hid=cfg.hidden_dim,
        out_levels=y_test.shape[-1],
        trunk_dropout=cfg.trunk_dropout,
        num_levels=cfg.n_blocks,
    ).to(device)
    assert_min_receptive_field_2d(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))

    # SGD's IID assumption demands shuffled batches. iter_shapes_t2l yields chronologically
    # within a file unless replay_config.randomize=True. We force it on for training (the user's
    # original randomize stays in effect for the test stream where determinism is the priority).
    train_stream_config = replay_config.but(randomize=True)
    if not replay_config.randomize:
        logger.info("[config] forcing replay_config.randomize=True on the training stream (SGD IID)")
    windows: queue.Queue = queue.Queue(maxsize=config.data_queue_size)
    workers = [
        DataLoaderWorker(
            data_queue=windows,
            replay_config=train_stream_config,
            shaper_config=shaper_config,
        )
        for _ in range(max(1, config.num_workers))
    ]
    for w in workers:
        w.start()

    shuffle_buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=SHUFFLE_BUFFER_SIZE)
    recent_scores: deque[float] = deque(maxlen=EVENT_SCORE_WINDOW)
    train_pred_buffer: deque[np.ndarray] = deque(maxlen=512)
    cached_routes: list[RouteSpec] = []
    display_idx_iter = itertools.cycle(range(len(y_test))) if move_test_window else itertools.repeat(0)

    def _ingest_one(timeout: float = 30.0) -> bool:
        """Pull one window, score it, push to buffer iff above the rolling eventfulness threshold."""
        try:
            bks, t2l, px = windows.get(timeout=timeout)
        except queue.Empty:
            logger.warning("[stream] data queue empty — workers stalled or no eligible files")
            return False
        score = score_window_eventfulness(px, bks)["score"]
        recent_scores.append(score)
        # During the first ~32 windows the percentile is unstable → accept everything.
        if len(recent_scores) < 32:
            keep = True
        else:
            threshold = float(np.percentile(np.fromiter(recent_scores, dtype=np.float32), 100.0 * (1.0 - TRAIN_TOP_FRACTION)))
            keep = score >= threshold
        if keep:
            shuffle_buffer.append((bks.astype(np.float32), t2l[:, :, 0].astype(np.float32)))
        return True

    def _refill_buffer(target: int) -> None:
        """Top up the shuffle buffer to at least ``target`` items, or until queue is dry."""
        while len(shuffle_buffer) < target:
            if not _ingest_one():
                break

    def _sample_batch() -> tuple[np.ndarray, np.ndarray]:
        chosen = random.sample(shuffle_buffer, config.batch_size)
        return np.stack([s[0] for s in chosen]), np.stack([s[1] for s in chosen])

    # === Warmup: fill buffer + compute normalization stats from TRAIN data ===
    logger.info(f"[warmup] filling shuffle buffer to {WARMUP_WINDOWS} windows")
    _refill_buffer(WARMUP_WINDOWS)
    if len(shuffle_buffer) < config.batch_size:
        for w in workers:
            w.stop()
        raise RuntimeError(
            f"warmup got only {len(shuffle_buffer)} windows after the eventfulness filter "
            f"— need at least {config.batch_size}. Check data files / shaper_config."
        )
    warmup_stack = np.stack([s[0] for s in shuffle_buffer])
    mu = warmup_stack.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd = (warmup_stack.std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)
    logger.info(
        f"[warmup] {len(shuffle_buffer)} windows, mu range [{mu.min():.4g}, {mu.max():.4g}], sd range [{sd.min():.4g}, {sd.max():.4g}]"
    )

    # Test set tensors — normalize on GPU with the same stats.
    xte = (torch.from_numpy(x_test).to(device) - mu_t) / sd_t
    yte = torch.from_numpy(y_test).to(device)

    tau = 1e-4
    cap = float(np.percentile(y_test, cfg.prediction_cap_quantile))

    def _eval_test() -> tuple[np.ndarray, float]:
        model.eval()
        with torch.no_grad():
            pred_te, cls_te = model(xte)
            evt_te = (yte > tau).float()
            loss_reg = regression_loss(pred_te, yte, cfg.reg_loss, cfg.active_reg_weight, tau)
            loss_evt = bce(cls_te, evt_te) if cfg.event_loss_weight > 0 else torch.zeros((), device=device)
            test_loss = float((loss_reg + cfg.event_loss_weight * loss_evt).item())
        return np.clip(pred_te.detach().cpu().numpy(), 0.0, cap), test_loss

    def _calibrate_routes() -> list[RouteSpec]:
        if not train_pred_buffer:
            return []
        stacked = np.concatenate(list(train_pred_buffer), axis=0)
        train_seq = np.clip(stacked, 0.0, cap).reshape(-1, stacked.shape[-1])
        # Pass the catalog so display_label/color/direction flow through to the notebook.
        return build_train_calibrated_strategy_grid(train_seq, routes=ACTIVE_CORE_SHORT_ROUTES)

    def _make_yield(pred_test: np.ndarray, train_loss: float, test_loss: float, route_configs: list[RouteSpec]) -> tuple:
        idx = next(display_idx_iter)
        set_window_label(f"{test_file.stem}   win {idx + 1}/{len(y_test)}")
        return (
            x_test[idx],
            y_test[idx][..., None],
            px_test[idx],
            pred_test[idx][..., None],
            train_loss,
            test_loss,
            route_configs,
        )

    try:
        for epoch in range(config.epochs):
            model.train()
            total = 0.0
            batches_seen = 0
            for _ in range(batches_per_epoch):
                _refill_buffer(min(SHUFFLE_BUFFER_SIZE, max(config.batch_size * 4, len(shuffle_buffer) + 1)))
                if len(shuffle_buffer) < config.batch_size:
                    logger.warning("[train] shuffle buffer underrun — workers can't keep up")
                    break

                xb_arr, yb_arr = _sample_batch()
                xb = (torch.from_numpy(xb_arr).to(device) - mu_t) / sd_t
                yb = torch.from_numpy(yb_arr).to(device)

                model.train()
                reg, cls_logits = model(xb)
                evt_target = (yb > tau).float()
                loss_reg = regression_loss(reg, yb, cfg.reg_loss, cfg.active_reg_weight, tau)
                loss_evt = bce(cls_logits, evt_target) if cfg.event_loss_weight > 0 else torch.zeros((), device=device)
                loss = loss_reg + cfg.event_loss_weight * loss_evt
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.item())
                batches_seen += 1

                train_pred_buffer.append(reg.detach().cpu().numpy())

                if display_every_batches > 0 and batches_seen % display_every_batches == 0:
                    pred_test, test_loss = _eval_test()
                    yield _make_yield(pred_test, total / batches_seen, test_loss, cached_routes)
                    model.train()

            if batches_seen == 0:
                continue  # degenerate epoch — don't yield garbage
            pred_test, test_loss = _eval_test()
            cached_routes = _calibrate_routes()
            logger.info(
                f"[epoch {epoch + 1}/{config.epochs}] avg_train_loss={total / batches_seen:.5g}, "
                f"test_loss={test_loss:.5g}, pred_max={pred_test.max():.4g}, pred_mean={pred_test.mean():.4g}, "
                f"buffer={len(shuffle_buffer)}, routes={len(cached_routes)}"
            )
            yield _make_yield(pred_test, total / batches_seen, test_loss, cached_routes)
    finally:
        for w in workers:
            w.stop()


# =============================================================================
# Smoke runner — proves the model actually learns and never predicts a black array.
# =============================================================================
async def _smoke_run(n_yields: int = 30) -> dict:
    """Run for a fixed number of yields against real local data; verify learning + non-zero output.

    Pass criteria:
        * test_loss decreases over the run (last vs first)
        * predictions are not all zero (max > 0)
        * predictions vary across yields (stddev of per-yield max > 0)
    """
    train_config = TrainConfig(
        num_workers=2,
        batch_size=8,
        data_queue_size=128,
        epochs=1,
        learning_rate=0.001,
    )
    replay_config = ReplayConfig(
        markets=["BTC-USD"],
        date_regexp="2026-04-14T0*",
        data_dir=Path("/mnt/data/repos/gaelreinaudi/crypto"),
        every="1000ms",
    )
    test_config = replay_config.but(randomize=False)

    test_losses: list[float] = []
    pred_maxes: list[float] = []
    pred_means: list[float] = []
    train_losses: list[float] = []

    gen = train_and_predict(
        config=train_config,
        replay_config=replay_config,
        shaper_config=SHAPER,
        test_config=test_config,
        display_every_batches=2,
        batches_per_epoch=n_yields * 2,  # so we get n_yields display events
        move_test_window=True,
    )
    count = 0
    async for _xb, _yb, _px, pred, train_loss, test_loss, routes in gen:
        count += 1
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        pred_maxes.append(float(pred.max()))
        pred_means.append(float(pred.mean()))
        sample_route = routes[0].name if routes else "(uncalibrated)"
        print(
            f"[smoke {count:>3}/{n_yields}] train={train_loss:.5g} test={test_loss:.5g} "
            f"pred_max={pred.max():.4g} pred_mean={pred.mean():.4g} routes={len(routes)} "
            f"first={sample_route}"
        )
        if count >= n_yields:
            break
    await gen.aclose()

    test_drop = test_losses[0] - test_losses[-1] if len(test_losses) >= 2 else 0.0
    pred_max_overall = max(pred_maxes) if pred_maxes else 0.0
    pred_max_std = float(np.std(pred_maxes)) if len(pred_maxes) > 1 else 0.0

    pass_loss = test_drop > 0.0
    pass_nonzero = pred_max_overall > 1e-6
    pass_varies = pred_max_std > 0.0

    report = {
        "yields": count,
        "first_test_loss": test_losses[0] if test_losses else None,
        "last_test_loss": test_losses[-1] if test_losses else None,
        "test_loss_drop": test_drop,
        "pred_max_overall": pred_max_overall,
        "pred_max_std": pred_max_std,
        "pass_loss_decreased": pass_loss,
        "pass_predictions_nonzero": pass_nonzero,
        "pass_predictions_vary": pass_varies,
        "ALL_PASS": pass_loss and pass_nonzero and pass_varies,
    }
    print("\n" + "=" * 72)
    print("SMOKE REPORT")
    print("=" * 72)
    for k, v in report.items():
        print(f"  {k:>30}: {v}")
    print("=" * 72)
    return report


if __name__ == "__main__":
    import asyncio
    import sys

    logger.setLevel("INFO")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    result = asyncio.run(_smoke_run(n_yields=n))
    sys.exit(0 if result["ALL_PASS"] else 1)
