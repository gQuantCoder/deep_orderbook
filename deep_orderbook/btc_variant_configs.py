VARIANTS: dict[str, dict] = {
    "holdout_capped_precision": {
        "market": "BTC-USD",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "prediction_cap_quantile": 99.5,
        "trade_threshold": 0.10,
        "hypothesis": "Reducing event-loss pressure and class weight should cut event spam and improve holdout precision while keeping capped image-to-image structure.",
    },
    "holdout_regression_only": {
        "market": "BTC-USD",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "prediction_cap_quantile": 99.5,
        "trade_threshold": 0.10,
        "hypothesis": "Pure image-to-image regression may preserve map geometry better than event-augmented training on sparse holdout data.",
    },
}


def list_variant_names() -> list[str]:
    return sorted(VARIANTS.keys())


def get_variant_config(name: str) -> dict:
    if name not in VARIANTS:
        raise KeyError(f"Unknown BTC variant: {name}")
    return dict(VARIANTS[name])
