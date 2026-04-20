"""causal_tcn.py — legacy 1-D causal TCN utilities.

⛔ TinyTCN IS FORBIDDEN IN EXPERIMENT SCRIPTS ⛔
================================================
TinyTCN uses 1-D Conv1d on a FLATTENED (T, bins×channels) feature vector.
Flattening destroys every cross-bin spatial relationship in the order book.
A 1-D TCN cannot learn "bid depth rising at bin 7 while ask depth thins at
bin 8" — it only sees 48 independent scalars per tick.

Use ``OrderBook2DTCN`` from ``deep_orderbook.learn.tcn`` instead.
That model:
  • preserves the full 2-D book structure  (B, 3, T, bins)
  • uses Conv2d, learning cross-bin spatial patterns
  • has minimum-RF enforcement built in
  • matches exactly what learn.ipynb / Trainer / visu.py produce

CausalConv1d is kept here for internal library use only (e.g. attention
positional masking helpers).  Do NOT use it to build experiment models.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from deep_orderbook.utils import logger

# ---------------------------------------------------------------------------
# RF helpers — still used by tcn.py's assert_min_receptive_field_2d
# ---------------------------------------------------------------------------


def min_n_blocks_for_rf(target_rf: int, kernel: int = 3) -> int:
    """Return the minimum number of doubling-dilation blocks to reach target_rf."""
    n = math.ceil(math.log2((target_rf - 1) / (kernel - 1) + 1))
    return max(n, 1)


def assert_min_receptive_field(model: object, min_rf: int = 1024) -> None:
    """Hard gate: raise if model.receptive_field < min_rf ticks.

    Works on any model that exposes a ``.receptive_field`` attribute
    (both TinyTCN for backward compat and OrderBook2DTCN).
    """
    rf = getattr(model, "receptive_field", None)
    if rf is None:
        raise TypeError(f"{type(model).__name__} has no .receptive_field attribute")
    if rf < min_rf:
        raise ValueError(
            f"{type(model).__name__} receptive field {rf} ticks is below the "
            f"mandatory minimum of {min_rf} ticks.  "
            f"A model that cannot see 1024 ticks back cannot capture "
            f"order-book microstructure patterns.  Use OrderBook2DTCN."
        )


# ---------------------------------------------------------------------------
# CausalConv1d — internal building block, NOT for experiment models
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Left-padded dilated Conv1d — INTERNAL USE ONLY.

    Do not use this to build experiment-facing models.  It processes a 1-D
    sequence and cannot exploit the 2-D spatial structure of the order book.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


# ---------------------------------------------------------------------------
# TinyTCN — TOMBSTONED.  Raises on instantiation.
# ---------------------------------------------------------------------------


class TinyTCN(nn.Module):
    """⛔ DEPRECATED — FORBIDDEN FOR EXPERIMENT SCRIPTS.

    TinyTCN was a 1-D Conv1d model operating on flattened order-book features
    (T, bins×channels = T, 48).  It CANNOT learn cross-bin spatial patterns.
    Every attempt to use it in experiment scripts is a bug.

    Migration path
    --------------
    Replace every usage of TinyTCN with OrderBook2DTCN:

        # WRONG (destroyed)
        from deep_orderbook.learn.causal_tcn import TinyTCN
        model = TinyTCN(in_ch=48, hid=96, out_ch=8, n_blocks=11)
        X = books.reshape(T, -1)      # ← NEVER flatten

        # CORRECT
        from deep_orderbook.learn.tcn import OrderBook2DTCN, assert_min_receptive_field_2d
        model = OrderBook2DTCN(bins=16, in_ch=3, hid=8, out_levels=8, num_levels=10)
        assert_min_receptive_field_2d(model)
        # X stays as (B, T, bins, 3) — spatial structure preserved
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "\n\n"
            "⛔  TinyTCN is FORBIDDEN in experiment scripts.\n"
            "    It uses 1-D Conv1d on a flattened (T, 48) vector,\n"
            "    destroying every cross-bin spatial pattern in the order book.\n\n"
            "    Use OrderBook2DTCN from deep_orderbook.learn.tcn instead:\n\n"
            "        from deep_orderbook.learn.tcn import (\n"
            "            OrderBook2DTCN, assert_min_receptive_field_2d\n"
            "        )\n"
            "        model = OrderBook2DTCN(\n"
            "            bins=16, in_ch=3, hid=8, out_levels=8, num_levels=10\n"
            "        )\n"
            "        assert_min_receptive_field_2d(model)\n"
            "        # books tensor stays (B, T, bins, 3) — never reshape to (B, T, 48)\n"
        )

    # Stub so type checkers don't complain about missing attribute on the class
    receptive_field: int = 0
    n_blocks: int = 0

    def forward(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        raise RuntimeError("TinyTCN is tombstoned — see class docstring.")


logger.warning("causal_tcn.py loaded: TinyTCN is TOMBSTONED — use OrderBook2DTCN from deep_orderbook.learn.tcn")
