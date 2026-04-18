import numpy as np

from deep_orderbook.trigger_search import (
    build_friction_table,
    build_train_calibrated_strategy_grid,
    format_trigger_sweep_summary,
    score_strategy_result,
)


def test_build_train_calibrated_strategy_grid_returns_multiple_distinct_configs() -> None:
    pred = np.array(
        [
            [0.1, 0.2, 0.6, 0.7],
            [0.2, 0.1, 0.8, 0.9],
            [0.1, 0.1, 0.4, 0.5],
            [0.3, 0.2, 0.9, 1.1],
        ],
        dtype=np.float32,
    )
    grid = build_train_calibrated_strategy_grid(pred)
    names = [cfg["name"] for cfg in grid]
    assert len(grid) >= 8
    assert len(set(names)) == len(names)
    assert any(cfg["persistence"] > 1 for cfg in grid)


def test_score_strategy_result_rewards_positive_pnl_and_penalizes_churn() -> None:
    strong = score_strategy_result({"final_pnl": 10.0, "trade_count": 5, "market_time_fraction": 0.2}, precision=0.3, f1=0.4, rmse_ratio=1.0)
    weak = score_strategy_result({"final_pnl": -5.0, "trade_count": 20, "market_time_fraction": 0.8}, precision=0.3, f1=0.4, rmse_ratio=1.0)
    assert strong > weak


def test_format_trigger_sweep_summary_includes_direction_and_separate_raw_pnl_leader() -> None:
    ranked = [
        {
            "variant_name": "mapper_a",
            "direction": "short",
            "strategy_name": "q95_p2_hold96",
            "route_score": 6.0,
            "rmse_ratio": 1.03,
            "metrics": {"precision": 0.27, "f1": 0.41},
            "strategy_metrics": {"final_pnl": 516.625, "trade_count": 103},
        },
        {
            "variant_name": "mapper_a",
            "direction": "long",
            "strategy_name": "q80_p1_hold48",
            "route_score": 5.0,
            "rmse_ratio": 1.03,
            "metrics": {"precision": 0.27, "f1": 0.41},
            "strategy_metrics": {"final_pnl": 3180.203125, "trade_count": 3734},
        },
    ]

    lines = format_trigger_sweep_summary(
        label="exp30_validate_l1_holdout20_both",
        timestamp_utc="2026-04-15 17:31:44Z",
        device="cuda",
        data_scale={"train_timesteps": 3827456, "test_timesteps": 383232},
        runtime_seconds_total=48.24,
        ranked=ranked,
    )
    text = "\n".join(lines)

    assert "best route by score: `mapper_a / short / q95_p2_hold96`" in text
    assert "best route by raw pnl: `mapper_a / long / q80_p1_hold48`" in text
    assert "1. `mapper_a / short / q95_p2_hold96`" in text
    assert "2. `mapper_a / long / q80_p1_hold48`" in text


def test_build_friction_table_subtracts_per_trade_costs() -> None:
    ranked = [
        {
            "variant_name": "mapper_a",
            "direction": "long",
            "strategy_name": "q80_p1_hold48",
            "route_score": 5.0,
            "rmse_ratio": 1.03,
            "metrics": {"precision": 0.27, "f1": 0.41},
            "strategy_metrics": {"final_pnl": 100.0, "trade_count": 10},
        },
        {
            "variant_name": "mapper_a",
            "direction": "short",
            "strategy_name": "q95_p2_hold96",
            "route_score": 6.0,
            "rmse_ratio": 1.03,
            "metrics": {"precision": 0.27, "f1": 0.41},
            "strategy_metrics": {"final_pnl": 50.0, "trade_count": 5},
        },
    ]

    table = build_friction_table(ranked, per_trade_costs=(1.0, 5.0))

    assert table[0]["direction"] == "long"
    assert table[0]["pnl_per_trade"] == 10.0
    assert table[0]["adjusted_pnl"][1.0] == 90.0
    assert table[0]["adjusted_pnl"][5.0] == 50.0
    assert table[1]["adjusted_pnl"][5.0] == 25.0
