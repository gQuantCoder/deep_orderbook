# Deep OrderBook scientist handoff protocol

Goal: find any repeatable pattern in the future-map prediction problem that looks tradable.

This is the handoff document a new scientist must read before running anything.
It is the compact explanation of what has been tried, what has been observed, what artifacts matter, and how to decide the next experiment.

## 1. Core objective

We are not optimizing for pretty losses.
We are searching for any pattern that appears tradable.

A candidate is only interesting if it shows some combination of:
- non-random visual structure in predicted `time2level` / proximity maps
- improved event precision / F1
- acceptable directional quality on triggered signals
- no catastrophic dense-map degradation
- some plausible conversion into trade logic

Important: a model that emits lots of events and gets a slightly better F1 while destroying side accuracy is not good enough.

## 2. Canonical state sources

Always read these in this order before trying something new:

1. `experiments/experiment_runs.sqlite3`
   - this is the lightweight experiment DB
   - each row stores experiment name, variant, timestamp, core metrics, JSON artifact path, and preview picture path
2. `experiments/pictures/`
   - preview images for eyeballing predicted maps
3. `experiments/image_prediction_lab_log.md`
   - long-form scientist notes and interpretation
4. `experiments/hourly_journal.md`
   - short continuity notes and queued next mutation
5. `experiments/results/*.json`
   - exact machine-readable outputs

If those disagree, prefer the timestamped JSON artifact, then the DB row, then the markdown summary.

## 3. Data sources and how to read them

### Historical replay data
Canonical replay data currently used for experiments:
- `/media/photoDS216/crypto/*.parquet`

Typical experiment slice used so far:
- symbol: `ETH-USD`
- cadence: `100ms`
- replay date pattern often around `2025-02-18T11*`

### Live recorder data
Current recorder process is running.
Observed process:
- `python deep_orderbook/consumers/recorder.py`

Observed write behavior from the code and current filesystem:
- active raw live files go to:
  - `deep_orderbook/data/L2/<timestamp>_update.jsonl`
  - `deep_orderbook/data/L2/<timestamp>_trades.jsonl`
- on hourly rollover, recorder merges those into parquet and writes to:
  - `../crypto/<timestamp>.parquet`

When launched from repo root, that means:
- active JSONL path:
  - `/mnt/data/repos/gaelreinaudi/deep_orderbook/data/L2/`
- hourly parquet output path:
  - `/mnt/data/repos/gaelreinaudi/crypto/`

Current live files observed:
- `/mnt/data/repos/gaelreinaudi/deep_orderbook/data/L2/2026-04-13T22-51-51_update.jsonl`
- `/mnt/data/repos/gaelreinaudi/deep_orderbook/data/L2/2026-04-13T22-51-51_trades.jsonl`

Practical implication:
- if we want freshest data before hour rollover, use the JSONL files
- if we want stable replayable new data after rollover, use the generated parquet in `../crypto/`

## 4. Problem semantics that must never drift

- Input is a structured order-book tensor, not a generic image.
- Output is a full future 2D map, not a scalar-only label.
- Causality matters.
- The right side is future-after-now.
- We care about tradable signal, not only dense reconstruction quality.
- Pictures are essential because intermediate `time2level` predictions can reveal whether the model is learning shape/timing structure or just emitting junk.

## 5. What has been learned so far

### Data / target structure
Observed earlier in the lab log:
- targets are sparse and event-like
- active ratio was around `~3.9%`
- target median is effectively near zero
- therefore plain dense MSE-style thinking is not enough

Interpretation:
- sparse-event behavior matters
- visual inspection matters
- a model can look numerically active but still be visually wrong or untradable

### Multi-horizon framing
Empirical result from prior screens:
- longer horizon `look_ahead=64` has looked more promising than shorter horizons for event quality
- short horizons were noisier under the tested model families

### Strongest non-deep baseline found so far
The strongest practical classical baseline so far has been the weighted HGB-style family around:
- `hgb_d6_lr006_w`

Typical long-horizon behavior seen in prior runs:
- `f1` around `0.22 - 0.24`
- `precision` around `0.13`
- `side_acc_on_signals` around `0.55 - 0.60`
- `rmse` around `0.029`

This is not amazing, but it is the practical bar that later experiments need to beat.

### TCN / structured-loss branch
Observed repeatedly:
- TCN variants can produce event activity, but they have not yet beaten the practical baseline
- structured-loss TCN often increases event spam and worsens RMSE
- visually, these models often do not yet look convincingly pattern-matched

Most recent TCN compare in the DB:
- experiment: `exp05_tcn_structured_loss_compare`
- DB rows 1-4
- pictures saved under `experiments/pictures/exp05_tcn_structured_loss_compare_*`

Latest long-horizon TCN snapshot:
- h64 baseline TCN:
  - `f1=0.0472`
  - `precision=0.0252`
  - `side_acc=0.1154`
  - `rmse=0.0377`
- h64 structured TCN:
  - `f1=0.0355`
  - `precision=0.0185`
  - `side_acc=0.8574`
  - `rmse=0.0673`

Interpretation:
- both are still bad in practical terms
- structured TCN side accuracy is misleading without the rest of the context because dense error and precision are poor
- neither is promising yet

### Hurdle branch
Observed repeatedly:
- hurdle-style gating can improve sparse-event F1 / precision
- but it keeps collapsing side accuracy and/or inflating RMSE
- therefore it has not cleared the practical tradability gate

Interpretation:
- do not trust F1 improvements alone
- any new candidate must be checked against side quality and visual map quality

## 6. Mandatory precheck when conditions changed

If data conditions changed, or fresh recorder data is being used, do not jump directly into training.
First run a short shaper-based richness precheck.

Minimum precheck protocol:
- run a small shaping pass on the intended data slice
- save a quick preview image of input + target maps
- inspect whether the slice is worth training on

Reject the slice before training if:
- books/input maps are permanently flat
- inputs are globally saturated / blown out
- target `time2level` map is effectively all black or empty for the inspected region
- there is no visible localized activity worth predicting

Practical rule:
- no richness -> no experiment

If the slice fails, log that failure and move to a better date / symbol / cadence / time block.

## 7. What every new experiment must produce

Every experiment must leave these artifacts behind.
No exceptions.

### A. Timestamped JSON artifact
Path:
- `experiments/results/<experiment_name>_<timestamp>.json`

Must include:
- timestamp
- git commit if available
- experiment name
- exact hypothesis
- what changed
- what stayed fixed
- symbol
- cadence
- horizon
- shaping config
- model config
- train/test split description
- metrics
- decision
- next mutation
- picture paths

### B. Preview pictures
Path:
- `experiments/pictures/<experiment_name>_<variant>_<timestamp>.png`

These are mandatory.
If a run has no picture, it is incomplete.

Minimum acceptable preview:
- true future map
- predicted future map
- same scale for true and pred
- readable enough to eyeball if prediction is learning timing / shape or just noise

Better preview when possible:
- bid/ask path
- books heatmap
- level proximity / target
- prediction map
- train/test loss
- simple pnl overlay

The learn-100ms notebook style is a good target shape for these pictures.

### C. Experiment DB row
Path:
- `experiments/experiment_runs.sqlite3`

At minimum, register:
- experiment
- variant
- timestamp
- core metrics
- JSON artifact path
- picture path

## 7. Mandatory eyeball protocol

Before calling a run interesting, look at the picture.

### Hard fail
Reject immediately if:
- prediction map is mostly black and shows no structure
- prediction map is saturated / blown out
- true and pred are on different scales without explanation
- prediction is active everywhere in a meaningless wash
- the map timing and shape obviously do not line up with true activity

### Soft fail
Be cautious if:
- prediction captures timing but not shape
- prediction is stripey/noisy while truth is compact
- prediction looks plausible numerically but visually wrong

### Positive signs
These are what we want to see:
- predicted activity appears where true activity later appears
- predicted blobs have some similar vertical placement and time extent
- repeated motifs look matched, not random
- map is sparse but not dead
- structure is visible without heroic scaling

## 8. Decision rule for the next experiment

When choosing what to try next:

1. read the latest DB rows and newest images
2. read the latest markdown handoff notes
3. ask: what failed?
   - no event signal?
   - event spam?
   - poor side accuracy?
   - visually wrong map geometry?
4. mutate one primary factor only unless running a deliberate coarse task sweep
5. state the mutation explicitly in the next artifact

Good mutation examples:
- hold data fixed, change only horizon
- hold model fixed, change only rolling window
- hold task fixed, change only model capacity
- hold prediction fixed, change only trading threshold

Bad mutation examples:
- change cadence, horizon, model, and threshold all at once
- run a new model without saving pictures
- run a new branch without logging what it was supposed to prove

## 9. The actual scientific loop

Use this every time.

1. Read state sources
   - DB
   - newest pictures
   - lab log
   - hourly journal
2. Summarize current best known candidate and current blocker
3. Write one explicit hypothesis
4. Run one experiment with one main mutation
5. Save timestamped JSON
6. Save one or more preview pictures
7. Register rows in SQLite DB
8. Append scientist handoff note to markdown
9. Decide one next mutation
10. Mark run as:
   - promising
   - not promising yet
   - dead end

## 10. Current practical status

Current status: not promising yet.

Reason:
- we do see some predictive signal in classical baselines
- we do not yet have a model whose map predictions look convincingly tradable while also surviving the metric checks
- TCN branch is not there yet
- hurdle branch improves sparse-event metrics but still looks practically weak

## 11. What a new scientist should try next

Priority order:
1. use freshest data once recorder rolls parquet into `../crypto/`
2. re-run task framing screens on fresh data
   - cadence
   - horizon
   - symbol
3. keep saving pictures for every serious run
4. only push deeper architectures after confirming the task regime is still favorable on fresher data

## 12. Non-negotiable reminder

The goal is not to prove a specific architecture is clever.
The goal is to find any pattern that appears tradable.

If a dumb model with good pictures and stable directional quality works, that is better than a fancy model with bad pictures and bad trading behavior.
