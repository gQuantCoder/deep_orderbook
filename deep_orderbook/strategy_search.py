from __future__ import annotations

import numpy as np


def build_signal_features(pred_map: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(pred_map, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] % 2 != 0:
        raise ValueError(f"expected [T, 2*K] prediction map, got {arr.shape}")
    side_width = arr.shape[1] // 2
    down = arr[:, :side_width]
    up = arr[:, side_width:]
    down_max = down.max(axis=1)
    up_max = up.max(axis=1)
    return {
        "down_max": down_max,
        "up_max": up_max,
        "margin": up_max - down_max,
    }


def _evaluate_directional_strategy(
    prices: np.ndarray,
    pred_map: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
    side_margin: float,
    persistence: int,
    cooldown: int,
    max_hold: int,
    direction: str,
) -> dict:
    prices = np.asarray(prices, dtype=np.float32)
    if prices.ndim != 2 or prices.shape[1] < 2:
        raise ValueError(f"expected prices [T,2+], got {prices.shape}")
    feats = build_signal_features(pred_map)
    up_max = feats["up_max"]
    down_max = feats["down_max"]
    margin = feats["margin"]
    T = prices.shape[0]
    pnl = np.zeros(T, dtype=np.float32)
    positions = np.zeros(T, dtype=np.int8)
    trade_count = 0
    hold_steps = 0
    cooldown_left = 0
    in_pos = False
    persistent = 0
    hold_lengths: list[int] = []

    for t in range(T):
        if t > 0:
            pnl[t] = pnl[t - 1]

        if direction == "long":
            entry_ok = up_max[t] >= entry_threshold and margin[t] >= side_margin
        elif direction == "short":
            entry_ok = down_max[t] >= entry_threshold and margin[t] <= -side_margin
        else:
            raise ValueError(f"unsupported direction: {direction}")
        persistent = persistent + 1 if entry_ok else 0

        if in_pos and t > 0:
            if direction == "long":
                pnl[t] += float(prices[t, 0] - prices[t - 1, 0])
            else:
                pnl[t] += float(prices[t - 1, 1] - prices[t, 1])
            hold_steps += 1

        if in_pos:
            if direction == "long":
                exit_ok = down_max[t] >= exit_threshold or margin[t] <= 0 or hold_steps >= max_hold
            else:
                exit_ok = up_max[t] >= exit_threshold or margin[t] >= 0 or hold_steps >= max_hold
            if exit_ok:
                in_pos = False
                cooldown_left = cooldown
                hold_lengths.append(hold_steps)
                hold_steps = 0
        else:
            if cooldown_left > 0:
                cooldown_left -= 1
            elif persistent >= persistence:
                in_pos = True
                trade_count += 1
                hold_steps = 0
                if direction == "long":
                    pnl[t] += float(prices[t, 0] - prices[t, 1])
                else:
                    pnl[t] += float(prices[t, 0] - prices[t, 1])

        positions[t] = 1 if in_pos and direction == "long" else (-1 if in_pos else 0)

    return {
        "final_pnl": float(pnl[-1]) if T else 0.0,
        "pnl": pnl,
        "positions": positions,
        "trade_count": trade_count,
        "avg_hold_steps": float(np.mean(hold_lengths)) if hold_lengths else 0.0,
        "market_time_fraction": float(np.mean(np.abs(positions))) if T else 0.0,
        "features": feats,
        "direction": direction,
    }


def evaluate_long_strategy(
    prices: np.ndarray,
    pred_map: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
    side_margin: float,
    persistence: int,
    cooldown: int,
    max_hold: int,
) -> dict:
    return _evaluate_directional_strategy(
        prices,
        pred_map,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        side_margin=side_margin,
        persistence=persistence,
        cooldown=cooldown,
        max_hold=max_hold,
        direction="long",
    )


def evaluate_short_strategy(
    prices: np.ndarray,
    pred_map: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
    side_margin: float,
    persistence: int,
    cooldown: int,
    max_hold: int,
) -> dict:
    return _evaluate_directional_strategy(
        prices,
        pred_map,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        side_margin=side_margin,
        persistence=persistence,
        cooldown=cooldown,
        max_hold=max_hold,
        direction="short",
    )
