import numpy as np

from deep_orderbook.strategy_search import (
    build_signal_features,
    evaluate_long_strategy,
    evaluate_short_strategy,
)


def test_build_signal_features_extracts_up_down_max_and_margin() -> None:
    pred = np.array(
        [
            [0.1, 0.2, 0.4, 0.9],
            [0.3, 0.2, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    feats = build_signal_features(pred)
    assert np.allclose(feats["down_max"], [0.2, 0.3])
    assert np.allclose(feats["up_max"], [0.9, 0.8])
    assert np.allclose(feats["margin"], [0.7, 0.5])



def test_evaluate_long_strategy_requires_persistence_and_exits_on_down_signal() -> None:
    prices = np.array(
        [
            [100.0, 100.1],
            [100.2, 100.3],
            [100.4, 100.5],
            [100.6, 100.7],
            [100.1, 100.2],
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            [0.1, 0.1, 0.7, 0.8],
            [0.1, 0.1, 0.8, 0.9],
            [0.1, 0.1, 0.7, 0.85],
            [0.8, 0.9, 0.1, 0.1],
            [0.8, 0.9, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    res = evaluate_long_strategy(
        prices,
        pred,
        entry_threshold=0.75,
        exit_threshold=0.75,
        side_margin=0.4,
        persistence=2,
        cooldown=1,
        max_hold=10,
    )
    assert res["trade_count"] == 1
    assert res["positions"].max() == 1
    assert res["final_pnl"] > 0



def test_evaluate_long_strategy_stays_flat_on_noisy_low_margin_signal() -> None:
    prices = np.array([[100.0, 100.1], [100.0, 100.1], [100.1, 100.2]], dtype=np.float32)
    pred = np.array(
        [
            [0.3, 0.4, 0.45, 0.5],
            [0.4, 0.45, 0.5, 0.55],
            [0.45, 0.4, 0.48, 0.5],
        ],
        dtype=np.float32,
    )
    res = evaluate_long_strategy(
        prices,
        pred,
        entry_threshold=0.6,
        exit_threshold=0.6,
        side_margin=0.25,
        persistence=2,
        cooldown=1,
        max_hold=5,
    )
    assert res["trade_count"] == 0
    assert res["final_pnl"] == 0.0


def test_evaluate_short_strategy_profits_when_down_signal_precedes_selloff() -> None:
    prices = np.array(
        [
            [100.0, 100.1],
            [99.8, 99.9],
            [99.4, 99.5],
            [99.0, 99.1],
            [99.2, 99.3],
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            [0.8, 0.9, 0.1, 0.1],
            [0.85, 0.95, 0.1, 0.1],
            [0.8, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.8, 0.9],
            [0.1, 0.1, 0.85, 0.95],
        ],
        dtype=np.float32,
    )
    res = evaluate_short_strategy(
        prices,
        pred,
        entry_threshold=0.75,
        exit_threshold=0.75,
        side_margin=0.4,
        persistence=2,
        cooldown=1,
        max_hold=10,
    )
    assert res["trade_count"] == 1
    assert res["positions"].min() == -1
    assert res["final_pnl"] > 0
