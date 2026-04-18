from deep_orderbook.btc_variant_configs import get_variant_config, list_variant_names


def test_variant_registry_exposes_expected_names() -> None:
    names = list_variant_names()
    assert "holdout_capped_precision" in names
    assert "holdout_regression_only" in names


def test_get_variant_config_returns_expected_fields() -> None:
    cfg = get_variant_config("holdout_capped_precision")
    assert cfg["market"] == "BTC-USD"
    assert cfg["event_loss_weight"] < 0.25
    assert cfg["pos_weight"] < 8.0
    assert cfg["prediction_cap_quantile"] == 99.5
