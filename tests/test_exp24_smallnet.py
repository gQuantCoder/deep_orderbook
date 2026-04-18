from pathlib import Path

from scripts.exp16_batch_h64_tcn import TinyTCN


def test_tinytcn_exposes_configured_trunk_dropout() -> None:
    model = TinyTCN(in_ch=16, hid=32, out_ch=8, trunk_dropout=0.2)
    assert model.trunk_dropout.p == 0.2


def test_exp24_smallnet_script_exists() -> None:
    path = Path("scripts/exp24_smallnet_earlystop.py")
    assert path.exists(), f"missing wrapper script: {path}"
