from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from deep_orderbook.strategy_search import build_signal_features


def _quantile(arr: np.ndarray, q: float, fallback: float) -> float:
    if arr.size == 0:
        return fallback
    return float(np.quantile(arr, q))


def build_train_calibrated_strategy_grid(train_pred_map: np.ndarray) -> list[dict]:
    feats = build_signal_features(train_pred_map)
    up = feats["up_max"]
    margin = feats["margin"]
    positive_margin = margin[margin > 0]
    q80 = _quantile(up, 0.80, 0.1)
    q90 = _quantile(up, 0.90, q80)
    q95 = _quantile(up, 0.95, q90)
    m60 = _quantile(positive_margin, 0.60, 0.0)
    m80 = _quantile(positive_margin, 0.80, m60)
    m90 = _quantile(positive_margin, 0.90, m80)
    return [
        {"name": "q80_p1_hold48", "entry_threshold": q80, "exit_threshold": q80, "side_margin": m60, "persistence": 1, "cooldown": 4, "max_hold": 48},
        {"name": "q80_p2_hold48", "entry_threshold": q80, "exit_threshold": q80, "side_margin": m60, "persistence": 2, "cooldown": 4, "max_hold": 48},
        {"name": "q80_p3_hold48", "entry_threshold": q80, "exit_threshold": q80, "side_margin": m60, "persistence": 3, "cooldown": 4, "max_hold": 48},
        {"name": "q90_p1_hold48", "entry_threshold": q90, "exit_threshold": q90, "side_margin": m60, "persistence": 1, "cooldown": 4, "max_hold": 48},
        {"name": "q90_p2_hold48", "entry_threshold": q90, "exit_threshold": q90, "side_margin": m80, "persistence": 2, "cooldown": 6, "max_hold": 48},
        {"name": "q90_p2_hold96", "entry_threshold": q90, "exit_threshold": q90, "side_margin": m80, "persistence": 2, "cooldown": 6, "max_hold": 96},
        {"name": "q95_p1_hold48", "entry_threshold": q95, "exit_threshold": q90, "side_margin": m80, "persistence": 1, "cooldown": 8, "max_hold": 48},
        {"name": "q95_p2_hold96", "entry_threshold": q95, "exit_threshold": q90, "side_margin": m90, "persistence": 2, "cooldown": 8, "max_hold": 96},
        {"name": "q90_persistence2_fast_exit", "entry_threshold": q90, "exit_threshold": q80, "side_margin": m80, "persistence": 2, "cooldown": 6, "max_hold": 24},
        {"name": "q80_persistence2_wide_margin", "entry_threshold": q80, "exit_threshold": q80, "side_margin": m90, "persistence": 2, "cooldown": 6, "max_hold": 48},
    ]


def score_strategy_result(route: dict, *, precision: float, f1: float, rmse_ratio: float) -> float:
    pnl = float(route.get("final_pnl", 0.0))
    trades = float(route.get("trade_count", 0.0))
    market_time = float(route.get("market_time_fraction", 0.0))
    return (pnl / 50.0) + 1.5 * precision + 1.2 * f1 - 0.05 * trades - 0.2 * market_time - max(rmse_ratio - 1.2, 0.0)


def _route_label(route: dict) -> str:
    return f"{route['variant_name']} / {route['direction']} / {route['strategy_name']}"


def summarize_route(route: dict) -> dict:
    strategy_metrics = route["strategy_metrics"]
    trades = int(strategy_metrics["trade_count"])
    pnl = float(strategy_metrics["final_pnl"])
    return {
        "variant_name": route["variant_name"],
        "direction": route["direction"],
        "strategy_name": route["strategy_name"],
        "label": _route_label(route),
        "route_score": float(route["route_score"]),
        "final_pnl": pnl,
        "trade_count": trades,
        "pnl_per_trade": (pnl / trades) if trades else 0.0,
        "precision": float(route["metrics"]["precision"]),
        "f1": float(route["metrics"]["f1"]),
        "rmse_ratio": float(route["rmse_ratio"]),
    }


def build_friction_table(ranked: list[dict], *, per_trade_costs: Iterable[float] = (1.0, 2.0, 5.0, 10.0), top_n: int | None = None) -> list[dict]:
    rows = ranked[:top_n] if top_n is not None else ranked
    costs = tuple(float(c) for c in per_trade_costs)
    table = []
    for route in rows:
        summary = summarize_route(route)
        table.append(
            {
                **summary,
                "adjusted_pnl": {
                    cost: summary["final_pnl"] - (summary["trade_count"] * cost)
                    for cost in costs
                },
            }
        )
    return table


def format_trigger_sweep_summary(
    *,
    label: str,
    timestamp_utc: str,
    device: str,
    data_scale: dict,
    runtime_seconds_total: float,
    ranked: list[dict],
) -> list[str]:
    best_by_score = summarize_route(ranked[0])
    best_by_pnl = summarize_route(max(ranked, key=lambda route: route["strategy_metrics"]["final_pnl"]))
    lines = [
        f"# {label}",
        "",
        f"- timestamp: {timestamp_utc}",
        f"- device: `{device}`",
        f"- route_count: {len(ranked)}",
        f"- train_timesteps: {data_scale['train_timesteps']}",
        f"- test_timesteps: {data_scale['test_timesteps']}",
        f"- runtime_seconds_total: {runtime_seconds_total:.3f}",
        f"- best route by score: `{best_by_score['label']}`",
        f"- best score pnl: {best_by_score['final_pnl']:.5f}",
        f"- best score precision: {best_by_score['precision']:.4f}",
        f"- best score f1: {best_by_score['f1']:.4f}",
        f"- best score rmse_ratio: {best_by_score['rmse_ratio']:.4f}",
        f"- best route by raw pnl: `{best_by_pnl['label']}`",
        f"- best raw pnl: {best_by_pnl['final_pnl']:.5f}",
        f"- best raw pnl/trade: {best_by_pnl['pnl_per_trade']:.5f}",
        "",
        "## top 10 routes",
    ]
    for idx, item in enumerate(ranked[:10], start=1):
        summary = summarize_route(item)
        lines.append(
            f"- {idx}. `{summary['label']}` pnl={summary['final_pnl']:.5f} pnl_per_trade={summary['pnl_per_trade']:.5f} precision={summary['precision']:.4f} f1={summary['f1']:.4f} rmse_ratio={summary['rmse_ratio']:.4f} trades={summary['trade_count']} score={summary['route_score']:.4f}"
        )
    return lines
