from pathlib import Path

from deep_orderbook.config import ShaperConfig


def test_exp25_timing_script_exists() -> None:
    path = Path("scripts/exp25_timing_500ms.py")
    assert path.exists(), f"missing wrapper script: {path}"


def test_exp25_timing_wrapper_exposes_expected_geometry() -> None:
    from scripts.exp25_timing_500ms import DEFAULT_EVERY, DEFAULT_VARIANTS, build_exp25_shaper_config

    cfg = build_exp25_shaper_config()

    assert DEFAULT_EVERY == "500ms"
    assert DEFAULT_VARIANTS == [
        "l1_evt005_pw2_h32",
        "precision_evt005_pw2_thr010_h32",
        "regonly_wd1e3_h32",
        "regonly_huber_thr010_h32",
    ]
    assert isinstance(cfg, ShaperConfig)
    assert cfg.rolling_window_size == 512
    assert cfg.look_ahead == 32
    assert cfg.window_stride == 2
    assert cfg.look_ahead_side_bips == 5
    assert cfg.look_ahead_side_width == 4
