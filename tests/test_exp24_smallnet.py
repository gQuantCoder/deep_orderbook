from pathlib import Path

import pytest

from deep_orderbook.learn.tcn import OrderBook2DTCN, assert_min_receptive_field_2d


def test_orderbook2dtcn_exposes_trunk_dropout() -> None:
    model = OrderBook2DTCN(bins=16, in_ch=3, hid=8, out_levels=8, num_levels=10, trunk_dropout=0.2)
    assert model.dropout.p == 0.2


def test_orderbook2dtcn_min_rf_guard_passes() -> None:
    model = OrderBook2DTCN(bins=16, in_ch=3, hid=8, out_levels=8, num_levels=10)
    assert_min_receptive_field_2d(model)  # must not raise


def test_orderbook2dtcn_min_rf_guard_rejects_small_model() -> None:
    model = OrderBook2DTCN(bins=16, in_ch=3, hid=8, out_levels=8, num_levels=3)
    with pytest.raises(ValueError, match="receptive field"):
        assert_min_receptive_field_2d(model)


def test_tinytcn_is_tombstoned() -> None:
    from deep_orderbook.learn.causal_tcn import TinyTCN
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        TinyTCN(in_ch=48, hid=96, out_ch=8)


def test_exp24_smallnet_script_exists() -> None:
    path = Path("scripts/exp24_smallnet_earlystop.py")
    assert path.exists(), f"missing wrapper script: {path}"
