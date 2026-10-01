# Scientific experiment search and logging for deep_orderbook

## What exists today

There is already a decent base for reproducible experiments:

- `scripts/hourly_experiment_cycle.py`
  - runs a small baseline sweep
  - writes machine-readable JSON to `experiments/results/hourly/`
  - appends summary lines to `experiments/hourly_journal.md`
- `scripts/replay_readme_style_prediction.py`
  - makes a 5-panel replay figure for visual inspection
  - already uses practical contrast guards:
    - percentile scaling for signed inputs
    - `sqrt()` gamma lift for sparse positive target/prediction maps
    - shared `vmax` across true/pred for fair comparison
  - writes `experiments/results/replay_readme_style_prediction.png`
- `scripts/exp08_purged_hurdle_side.py`
  - makes a true-vs-pred preview figure
  - writes `experiments/results/exp08_purged_hurdle_side_preview.png`
- `experiments/image_prediction_lab_log.md`
  - long-form scientific notes
- `experiments/hourly_journal.md`
  - short continuity log for ongoing runs

## What does NOT exist yet

There is no dedicated image-quality gate script that says, automatically:

- this image is too black
- this image is too saturated
- this image has too little structure to be useful

Right now, visual quality is handled mostly by plotting heuristics plus manual eyeballing.

That is fine for now, but we should add a lightweight QC check so every experiment produces:

- metrics JSON
- preview PNG
- image QC JSON

## Recommended experiment philosophy

Do not search everything at once.
That becomes noise.

Use a staged search:

1. Fix task semantics first
2. Sweep data semantics
3. Sweep model complexity
4. Sweep decision policy
5. Keep only promising regions

Scientific rule: change one primary factor at a time unless you are doing a deliberately designed coarse grid.

## Non-negotiable semantics

Every experiment must state explicitly:

- timestep / replay cadence
  - e.g. `100ms`, `250ms`, `1000ms`
- horizon / target definition
  - e.g. `look_ahead=16, 32, 64`
- symbol
  - e.g. `ETH-USD`, `BTC-USD`
- shaping config
  - `view_bips`, `num_side_lvl`, `look_ahead_side_bips`, `look_ahead_side_width`, `rolling_window_size`, `window_stride`
- model family
  - baseline tree / TCN / AttentionTCN / pure attention / other
- output semantics
  - full future 2D map, not collapsed scalar-only target
- decision rule
  - threshold, side rule, post-filter, trading conversion logic

If any of those are unclear, the result is not scientifically comparable.

## Practical search order

### Stage 1 — data/task sweep

First search the problem framing, not the fanciest model.

Recommended first axes:

- cadence: `100ms`, `250ms`, `1000ms`
- horizon: `16`, `32`, `64`
- symbol: `ETH-USD`, `BTC-USD`

Keep model simple here.
Use one stable baseline model only.

Goal:
- find where the signal is most tradable
- not where the model is most expressive

### Stage 2 — window and shaping sweep

Once a promising task regime exists, test:

- `rolling_window_size`: short / medium / long
- `window_stride`
- `view_bips`
- `look_ahead_side_bips`
- `look_ahead_side_width`

Goal:
- confirm the representation is helping
- avoid learning artifacts from a bad shaping choice

### Stage 3 — model complexity sweep

Only after stages 1 and 2:

- simple: baseline tree / shallow CNN / simple TCN
- medium: better TCN / AttentionTCN
- complex: pure attention / heavier sequence models

Rule:
- complexity must earn its keep by improving practical metrics, not just prettier training loss

### Stage 4 — trading rule sweep

Separate prediction quality from trading-rule quality.

Sweep:
- event threshold
- side threshold
- dominance margin
- post-filtering
- entry/exit conversion logic

Do not mix model changes and trading-rule changes in the same experiment unless the purpose is explicitly joint optimization.

## What to log for every experiment

Each experiment should write three artifacts.

### 1. Metrics JSON

Path pattern:
- `experiments/results/<experiment_name>_<timestamp>.json`

Required fields:
- timestamp
- git commit if available
- script name
- dataset slice
- symbol
- cadence
- horizon
- shaping config
- model config
- train/test split description
- metrics:
  - rmse
  - mae
  - precision
  - recall
  - f1
  - event_rate_true
  - event_rate_pred
  - side_acc_on_signals
- decision:
  - keep / reject / unclear
- short hypothesis
- next mutation

### 2. Preview PNG

Path pattern:
- `experiments/results/<experiment_name>_preview.png`

Minimum requirement:
- true map and predicted map must share the same scale
- figure must be readable without opening notebook state

### 3. Image QC JSON

Path pattern:
- `experiments/results/<experiment_name>_image_qc.json`

Suggested fields:
- mean_intensity
- std_intensity
- p01
- p99
- saturated_high_fraction
- near_zero_fraction
- active_pixel_fraction
- usable_for_eyeball: true/false
- notes

## Eyeball checklist

Every preview should pass this manual check:

### Hard fail

Reject the image if:
- almost everything is black and events are not visible
- almost everything is saturated and structure is blown out
- true and prediction use different scales without saying so
- predicted map is visually active everywhere with no sparse structure
- image is so tiny or blurry that structure cannot be compared

### Soft fail

Caution if:
- prediction only gets rough timing but no shape
- prediction is stripey/noisy while true map is compact
- the figure looks good only because scaling is too aggressive
- one panel is readable and the others are not

### Good enough

Accept for scientific comparison if:
- sparse events are visible
- true and pred share comparable contrast
- structure is visible at a glance
- panel is not mostly dead black
- panel is not globally clipped/saturated

## Simple automated image-QC gate to add

We should add one small script next.

Suggested file:
- `scripts/check_image_quality.py`

It should compute for each generated PNG:

- grayscale mean and std
- fraction of pixels below a low threshold
- fraction of pixels above a high threshold
- edge magnitude / gradient energy proxy
- panel-wise stats if the figure has multiple stacked panels

Practical default flags:
- `near_zero_fraction > 0.97` -> likely too black
- `saturated_high_fraction > 0.25` -> likely too saturated
- `std_intensity < small_threshold` -> likely too flat

This should not replace eyeballing.
It should only catch obviously bad plots automatically.

## Ranking experiments scientifically

Use a two-level decision rule.

### Level 1 — prediction quality

A run is promising only if it improves on:
- event precision / f1
- side accuracy on predicted signals
- dense error not exploding

### Level 2 — trade usefulness

A run is truly interesting only if better prediction quality survives conversion into trading logic:
- better directional correctness
- tolerable trigger rate
- stable behavior across multiple days
- not just one lucky replay slice

## Recommended scorecard

For each run, write:

- Hypothesis
- What changed
- What stayed fixed
- Result
- Decision
- Next mutation

Example:

- Hypothesis: `100ms` plus `look_ahead=32` on `ETH-USD` should produce cleaner actionable targets than `look_ahead=64`.
- Changed: `look_ahead 64 -> 32`
- Fixed: symbol, model, training split, threshold
- Result: better side accuracy, lower f1
- Decision: unclear
- Next mutation: keep `look_ahead=32`, test `BTC-USD`

## Minimal search plan to start from now

Recommended first matrix:

- symbols: `ETH-USD`, `BTC-USD`
- cadences: `100ms`, `250ms`, `1000ms`
- horizons: `16`, `32`, `64`
- one stable baseline model

That is 18 task settings.
Run those first before doing a serious network sweep.

After that, take the top 3 task settings and test:
- simple model
- medium model
- complex model

That is the cleanest way to avoid fooling ourselves.

## Bottom line

Use this workflow:

1. lock semantics
2. search task regime first
3. search model second
4. save JSON + PNG every run
5. append one short journal entry every run
6. add a tiny automated image-QC script, but keep human eyeballing

The current repo already has the beginnings of this.
What is missing is the explicit image-QC gate and a stricter experiment schema.
