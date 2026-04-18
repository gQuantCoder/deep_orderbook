# Deep OrderBook Architecture Review (updated)

This file is a short top-level index. Detailed docs moved under `docs/architecture/` and `docs/quant/`.

## Canonical docs

- `docs/architecture/current-system.md`
- `docs/architecture/prediction-targets.md`
- `docs/quant/useful-future-prediction-plan.md`

## Core concept (confirmed)

The model is trained as image-to-image forecasting:
- input: correlated 2D order-book/trade tensor over time x levels
- output: correlated 2D TTL proximity tensor over time x levels (down + up together)

At each step, we forecast the future map from the current known state.
This keeps timing/path structure that scalar direction labels lose.

## Training update now in code

`TrainConfig.criterion = "StructuredT2L"` now enables a structured field loss:
- pointwise map fit
- up/down dominance ranking
- near/far level monotonicity regularization
- optional final-step focus (`loss_focus_last_step=True`) for actionable live decision timing
