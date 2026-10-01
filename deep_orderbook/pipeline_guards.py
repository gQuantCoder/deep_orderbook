"""Executable sanity gates for the deep_orderbook research loop.

These guards exist to make the SKILL's prose rules enforceable in code.
Every experiment script should call them at the top of its data-loading
and backtesting paths. They are deliberately cheap so there is no excuse
to skip them.

Key invariants enforced here:

1. An order-book "image" must be long enough on the time axis to carry
   meaningful microstructure. A 25 s window at 100 ms cadence is not.
2. Overlapping rolling windows must never be stitched with ``reshape(-1, ...)``
   and fed to a PnL backtester: that creates fake price discontinuities
   at window boundaries and makes results numerically dishonest.
3. When a script already has overlapping windows (training-style stride),
   the backtest path must either (a) pick a non-overlapping subset, or
   (b) run the strategy per window and aggregate.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from deep_orderbook.config import ReplayConfig, ShaperConfig


_EVERY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m)?\s*$", re.IGNORECASE)


def dt_seconds_from_every(every: str) -> float:
    """Convert a replay cadence string like ``"100ms"``/``"1s"`` to seconds.

    Accepts integers/floats with optional ``ms`` / ``s`` / ``m`` suffix.
    Defaults to milliseconds when no unit is given.
    """
    if not isinstance(every, str):
        raise TypeError(f"every must be a string, got {type(every).__name__}")
    match = _EVERY_RE.match(every)
    if not match:
        raise ValueError(f"cannot parse ReplayConfig.every={every!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "ms").lower()
    if unit == "ms":
        return value / 1000.0
    if unit == "s":
        return value
    if unit == "m":
        return value * 60.0
    raise ValueError(f"unsupported time unit in every={every!r}")


def assert_image_meaningful(
    shaper_cfg: ShaperConfig,
    replay_cfg: ReplayConfig,
    *,
    min_context_seconds: float = 60.0,
    min_ratio: int = 8,
    allow_microburst: bool = False,
) -> dict[str, Any]:
    """Fail fast if the shaped image is too short to carry microstructure.

    Returns a dict with the measured geometry so callers can log it into
    their artifacts.

    Rules:
    - ``rolling_window_size * dt >= min_context_seconds``
    - ``rolling_window_size >= min_ratio * look_ahead``

    Set ``allow_microburst=True`` to downgrade the gate to a no-op. This
    is intentionally ugly so that experiments that opt out of the gate
    must say so out loud, and the opt-out shows up in the artifact log.
    """
    dt = dt_seconds_from_every(replay_cfg.every)
    rolling = int(shaper_cfg.rolling_window_size)
    look_ahead = int(shaper_cfg.look_ahead)
    context_seconds = rolling * dt
    ratio = rolling / max(look_ahead, 1)
    geometry = {
        "dt_seconds": dt,
        "rolling_window_size": rolling,
        "look_ahead": look_ahead,
        "context_seconds": context_seconds,
        "context_to_horizon_ratio": ratio,
        "min_context_seconds": min_context_seconds,
        "min_ratio": min_ratio,
        "allow_microburst": bool(allow_microburst),
    }
    if allow_microburst:
        geometry["verdict"] = "microburst_override"
        return geometry
    problems: list[str] = []
    if context_seconds < min_context_seconds:
        problems.append(
            f"context={context_seconds:.1f}s < min_context_seconds={min_context_seconds:.1f}s "
            f"(rolling_window_size={rolling} * dt={dt:.3f}s). "
            f"Bump rolling_window_size to at least {int(np.ceil(min_context_seconds / dt))}."
        )
    if ratio < min_ratio:
        problems.append(
            f"rolling_window_size/look_ahead={ratio:.2f} < min_ratio={min_ratio}. "
            f"Bump rolling_window_size to at least {min_ratio * look_ahead} or shrink look_ahead."
        )
    if problems:
        raise ValueError(
            "assert_image_meaningful failed:\n  - "
            + "\n  - ".join(problems)
            + "\n  If you really intend a microburst-scale experiment, "
            + "pass allow_microburst=True and log the reason in the artifact."
        )
    geometry["verdict"] = "ok"
    return geometry


def assert_non_overlapping(n_windows: int, window_size: int, stride: int) -> None:
    """Guard against stitching overlapping rolling windows into a 1D timeline.

    This must be called at the top of any backtest path that is about to
    flatten stacked windows of shape ``(N, window_size, ...)`` into a
    ``(N*window_size, ...)`` array and feed that to a PnL function.
    If the stride used to build those windows is smaller than the window
    size, the flattened series contains each real timestep ``window_size/stride``
    times with fake price jumps at every seam.
    """
    if stride < window_size:
        raise ValueError(
            f"assert_non_overlapping failed: stride={stride} < window_size={window_size}. "
            f"Flattening {n_windows} overlapping windows would create a fake-continuous "
            f"timeline where every real timestep appears ~{window_size // max(stride, 1)} "
            f"times. Either (a) use select_non_overlapping_indices to pick a disjoint "
            f"subset, or (b) call aggregate_per_window_strategy_result which runs the "
            f"strategy on each internally-continuous window independently."
        )


def select_non_overlapping_indices(
    stride: int,
    rolling_window: int,
    n_windows: int,
) -> list[int]:
    """Pick the indices of a disjoint window subset from a stride-N stacked load.

    If you loaded ``N`` windows with ``window_stride=stride`` and
    ``rolling_window_size=rolling_window``, every consecutive window
    shares ``rolling_window - stride`` timesteps with its neighbor.
    Taking every ``(rolling_window // stride)``-th window gives a
    disjoint subset suitable for honest backtesting.

    Returns at most ``n_windows // step`` indices. If ``stride >= rolling_window``
    the input is already non-overlapping and every index is returned.
    """
    if rolling_window <= 0 or stride <= 0:
        raise ValueError(
            f"rolling_window and stride must be positive, got rolling_window={rolling_window} "
            f"stride={stride}"
        )
    if stride >= rolling_window:
        return list(range(n_windows))
    step = max(rolling_window // stride, 1)
    return list(range(0, n_windows, step))


def _concat_pnl_curves(per_window: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-window PnL curves into one continuous equity curve.

    Each window carries its own local PnL starting from 0. To glue them
    into a single curve for plotting we shift each segment by the final
    value of the preceding concatenated segment.
    """
    if not per_window:
        return np.zeros(0, dtype=np.float32)
    pieces: list[np.ndarray] = []
    running = 0.0
    for curve in per_window:
        if curve.size == 0:
            continue
        shifted = np.asarray(curve, dtype=np.float32) + running
        pieces.append(shifted)
        running = float(shifted[-1])
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def aggregate_per_window_strategy_result(
    px_windows: np.ndarray,
    pred_windows: np.ndarray,
    strategy_fn: Callable[..., dict],
    **strat_kwargs: Any,
) -> dict:
    """Run a directional strategy per-window and aggregate the results.

    ``px_windows`` is expected to have shape ``(N, T, 2)`` (bid, ask per
    timestep). ``pred_windows`` has shape ``(N, T, 2*K)`` -- a predicted
    proximity map per window. ``strategy_fn`` is e.g.
    ``evaluate_long_strategy`` or ``evaluate_short_strategy`` from
    ``deep_orderbook.strategy_search``.

    Returns a dict shaped like a single-call strategy result so existing
    plotting and scoring keep working:
      * ``final_pnl`` - sum of per-window final PnL
      * ``pnl`` - equity curve stitched from per-window curves (plottable)
      * ``positions`` - concatenated per-window position arrays
      * ``trade_count`` - sum across windows
      * ``avg_hold_steps`` - size-weighted mean of per-window averages
      * ``market_time_fraction`` - weighted mean of per-window fractions
      * ``direction`` - echoed from the first non-empty window
      * ``per_window`` - per-window final PnL for diagnostics
    """
    px_windows = np.asarray(px_windows)
    pred_windows = np.asarray(pred_windows)
    if px_windows.ndim != 3 or px_windows.shape[-1] < 2:
        raise ValueError(f"px_windows must be (N, T, 2+), got {px_windows.shape}")
    if pred_windows.ndim != 3:
        raise ValueError(f"pred_windows must be (N, T, 2*K), got {pred_windows.shape}")
    if px_windows.shape[0] != pred_windows.shape[0]:
        raise ValueError(
            f"px_windows N={px_windows.shape[0]} != pred_windows N={pred_windows.shape[0]}"
        )
    if px_windows.shape[1] != pred_windows.shape[1]:
        raise ValueError(
            f"px_windows T={px_windows.shape[1]} != pred_windows T={pred_windows.shape[1]}"
        )

    per_final: list[float] = []
    per_curve: list[np.ndarray] = []
    per_positions: list[np.ndarray] = []
    total_trades = 0
    weighted_hold_num = 0.0
    weighted_hold_den = 0.0
    weighted_mkt_num = 0.0
    weighted_mkt_den = 0.0
    direction = str(strat_kwargs.get("direction", "")) or None

    for i in range(px_windows.shape[0]):
        px = np.asarray(px_windows[i], dtype=np.float32)
        pm = np.asarray(pred_windows[i], dtype=np.float32)
        route = strategy_fn(px, pm, **strat_kwargs)
        direction = direction or route.get("direction")
        final = float(route.get("final_pnl", 0.0))
        per_final.append(final)
        pnl_curve = np.asarray(route.get("pnl", np.zeros(px.shape[0], dtype=np.float32)))
        positions = np.asarray(route.get("positions", np.zeros(px.shape[0], dtype=np.int8)))
        per_curve.append(pnl_curve)
        per_positions.append(positions)
        total_trades += int(route.get("trade_count", 0))
        n = int(px.shape[0])
        weighted_hold_num += float(route.get("avg_hold_steps", 0.0)) * n
        weighted_hold_den += n
        weighted_mkt_num += float(route.get("market_time_fraction", 0.0)) * n
        weighted_mkt_den += n

    pnl_stitched = _concat_pnl_curves(per_curve)
    positions_stitched = (
        np.concatenate(per_positions, axis=0) if per_positions else np.zeros(0, dtype=np.int8)
    )
    return {
        "final_pnl": float(np.sum(per_final)),
        "pnl": pnl_stitched,
        "positions": positions_stitched,
        "trade_count": int(total_trades),
        "avg_hold_steps": float(weighted_hold_num / weighted_hold_den) if weighted_hold_den else 0.0,
        "market_time_fraction": float(weighted_mkt_num / weighted_mkt_den) if weighted_mkt_den else 0.0,
        "direction": direction,
        "per_window_final_pnl": [float(x) for x in per_final],
        "n_windows": int(px_windows.shape[0]),
        "aggregation": "per_window",
    }


__all__ = [
    "dt_seconds_from_every",
    "assert_image_meaningful",
    "assert_non_overlapping",
    "select_non_overlapping_indices",
    "aggregate_per_window_strategy_result",
]
