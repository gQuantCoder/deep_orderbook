# Deep OrderBook Large-Data Research Plan

> For Hermes: planning only. Next execution phase should follow the deep-orderbook-scientist-handoff-loop skill and log data scale/runtime explicitly in every artifact.

Goal: stop doing toy screening runs and run materially larger BTC event-reaction experiments on older and newer data, with explicit proof of dataset scale, runtime, and no future leakage.

Architecture:
- Use chronological multi-file holdouts, never random splits.
- Use present-window event conditioning only as an optional regime filter, never future target intensity.
- Separate two phases: (1) data reconnaissance / benchmark-slice selection, then (2) heavy training/testing on larger datasets with proper runtime accounting.

Tech stack:
- Existing repo scripts in `scripts/exp16_batch_h64_tcn.py`
- Event selection helpers in `deep_orderbook/event_selection.py`
- Replay/shaping stack in `deep_orderbook/shaper.py`
- Result logging in `experiments/results/`, `experiments/notes/`, `experiments/pictures/`, `experiments/experiment_runs.sqlite3`

---

## What data is available right now

### New local recent corpus
Path: `/mnt/data/repos/gaelreinaudi/crypto`
- 14 parquet files
- total size: about 0.18 GB
- largest files:
  - `2026-04-13T23-00-00.parquet` — 27.0 MB
  - `2026-04-14T00-00-01.parquet` — 17.9 MB
  - `2026-04-14T08-00-00.parquet` — 15.2 MB
  - `2026-04-14T01-00-10.parquet` — 14.7 MB
  - `2026-04-14T07-00-00.parquet` — 14.7 MB

This corpus is useful for recency, but it is not big enough by itself for a serious GPU-heavy research phase.

### Historical corpus
Path: `/media/photoDS216/crypto`
- 6540 parquet files
- total size: about 102.97 GB
- clearly the main source for real training scale

Largest / densest visible files:
- `TOO_BIG_2_024-08-05T00-00-00.parquet` — 835.1 MB
  - BTC rows: 79,873,648
  - BTC trade rows: 1,800,576
  - time span: `2024-08-05 00:00:00` -> `2024-08-05 20:49:50`
  - trade price range: 9232.82
- `2025-04-07T14-00-33.parquet` — 54.5 MB
  - BTC trade rows: 167,052
  - trade price range: 3439.29
- `2025-03-11T14-00-33.parquet` — 52.9 MB
  - BTC trade rows: 68,466
  - trade price range: 2353.86
- `2025-04-09T17-00-32.parquet` — 48.9 MB
  - BTC trade rows: 142,512
  - trade price range: 4712.38
- `2025-03-10T14-00-32.parquet` — 48.4 MB
  - BTC trade rows: 89,471
  - trade price range: 2360.66
- `2025-02-25T15-00-32.parquet` — 47.9 MB
  - BTC trade rows: 140,904
  - trade price range: 1813.82
- `2025-03-11T00-00-11.parquet` — 47.7 MB
  - BTC trade rows: 68,023
  - trade price range: 2641.65

Takeaway:
- there is plenty of older data
- there are both huge multi-hour files and many large one-hour files in volatile periods
- the next scientist should not pretend we are data-limited

---

## Current blocker

Current recent-data experiments proved this:
- event-filtered recent BTC windows are learnable
- event metrics improved under honest holdout
- but previous runs were still small and fast
- and the current bottleneck is partly data scale, partly trigger extraction

We need to answer two separate questions:
1. Does performance improve materially when training on much larger historical event-rich data?
2. Are the best candidates still failing because of mapper weakness, or because trigger extraction is weak?

---

## Rules for the next research phase

1. No future leakage in event selection.
   - Allowed: present-window abs return, range, realized vol, book std, book impulse.
   - Forbidden: future target intensity, future return, future pnl, best future slice selection for training.

2. Every serious run must log data scale.
   - files used
   - windows before/after filter
   - rolling window size
   - train/test timesteps
   - target pixels
   - whether any per-file cap was applied

3. Every serious run must log runtime.
   - start time
   - end time
   - wall-clock
   - if possible: shaping/load time vs train/eval time

4. Do not blindly use the newest file as holdout.
   - Run richness gate first.
   - If latest file is empty/dead after shaping, move back to latest usable chronological block.

5. Use the huge historical corpus intentionally.
   - especially the 835 MB `TOO_BIG_2_024-08-05T00-00-00.parquet`
   - and the high-activity March/April 2025 files listed above

---

## Research plan

### Phase 1: Build a real data reconnaissance table

Objective: produce a machine-readable inventory of candidate large/volatile BTC files before running the heavy experiments.

Files:
- Create: `scripts/exp19_data_recon.py`
- Create: `experiments/results/exp19_data_recon_<timestamp>.json`
- Create: `experiments/notes/exp19_data_recon_<timestamp>.md`

Tasks:
1. Scan `/media/photoDS216/crypto/*.parquet` for BTC-only aggregates.
2. For each file, compute at least:
   - file size
   - BTC row count
   - BTC trade row count
   - timestamp span
   - trade price std
   - trade price range
3. Rank files by:
   - size
   - trade count
   - volatility proxy
4. Output three candidate buckets:
   - giant files
   - volatile one-hour files
   - recent files for forward holdout
5. Save the ranked table and brief scientist notes.

Validation:
- confirm `TOO_BIG_2_024-08-05T00-00-00.parquet` appears near the top
- confirm several March/April 2025 files appear as high-activity candidates

### Phase 2: Remove toy caps from the training pipeline

Objective: make the experiment script honest about data scale and capable of larger training loads.

Files:
- Modify: `scripts/exp16_batch_h64_tcn.py`
- Modify: `tests/test_btc_search_lab.py`
- Possibly create: `tests/test_exp16_large_data_config.py`

Tasks:
1. Replace hardcoded `max_windows=48` behavior with CLI-configurable caps.
2. Add CLI flags such as:
   - `--max-windows-per-file`
   - `--max-train-files`
   - `--max-test-files`
   - `--train-date-regex` or explicit file list support
3. Default notes/json must record whether caps were used.
4. Add runtime timing instrumentation:
   - total shaping time
   - total training/eval time
   - total wall-clock time
5. Add data scale fields into every artifact.

Validation:
- tests pass
- artifact JSON shows caps explicitly
- artifact JSON shows timing explicitly

### Phase 3: Heavy benchmark suite on large historical event-rich data

Objective: run a materially larger benchmark, not another toy screen.

Recommended dataset shape:
- Training on either:
  - the huge `TOO_BIG_2_024-08-05T00-00-00.parquet`, or
  - 8-20 large volatile hourly files from Feb/Mar/Apr 2025
- Holdout on a later chronological volatile block not used in training
- Optional second holdout on recent 2026 data

Proposed experiment families:
1. Mapper geometry family
   - `l1_evt005_pw2_h64`
   - `l1_evt005_pw2`
   - `l1_evt005_pw2_short_e4`
   - `precision_evt005_pw2_thr010`
2. Balanced/PnL curiosity family
   - `regonly_wd1e3`
   - `regonly_huber_thr010`
   - `regonly_activew3`
3. Trigger-calibration family
   - threshold-only sweeps around the stronger candidates

Run design:
- use fewer variants than the last 25-run suite if needed, but dramatically more data
- spend compute on training, not just screening breadth

Validation:
- artifact must show much larger train timesteps than 13,056
- runtime must be materially longer than ~106 seconds if the run is truly heavy
- if still tiny/fast, call it out as a screening run, not a serious training run

### Phase 4: Trigger-only follow-up after heavy mapper run

Objective: stop conflating map quality with trading conversion.

Files likely:
- Create: `scripts/exp20_trigger_sweep.py`
- Create: `experiments/results/exp20_trigger_sweep_<timestamp>.json`

Tasks:
1. Freeze 1-3 top mapper candidates from Phase 3.
2. Run only trigger mutations:
   - threshold sweep
   - local-max extraction
   - side-aware extraction
3. Evaluate regime-split pnl, precision, F1, rmse ratio.

Validation:
- if event metrics stay high and pnl improves, trigger extraction was the blocker
- if pnl remains dead, mapper still lacks sufficient actionable structure

---

## Concrete first execution order

1. Run Phase 1 data reconnaissance on `/media/photoDS216/crypto`
2. Patch the experiment script so caps and timing are explicit and configurable
3. Select one giant-file path and one multi-file volatile path
4. Run one serious large-data benchmark on each
5. Compare against recent 2026 holdout
6. Only then do trigger-only sweeps

---

## Files likely to change

- `scripts/exp16_batch_h64_tcn.py`
- `scripts/exp19_data_recon.py`
- `scripts/exp20_trigger_sweep.py`
- `deep_orderbook/btc_search_lab.py`
- `deep_orderbook/event_selection.py`
- `tests/test_btc_search_lab.py`
- `tests/test_scientist_experiment.py`
- `experiments/image_prediction_lab_log.md`
- `experiments/hourly_journal.md`

---

## Verification checklist for future execution

- Richness gate passed for chosen holdout
- No future leakage in event filtering
- Data scale logged
- Runtime logged
- Train/test files listed exactly
- Per-file cap stated explicitly
- Preview pictures saved
- DB rows saved
- Summary note tells whether run was screening-scale or serious-scale

---

## Bottom line

What is available:
- a tiny recent corpus useful for recency checks
- a huge historical corpus over 100 GB, including one 835 MB monster file and many 45-55 MB volatile BTC hours

Research direction:
- stop pretending the recent 3-file capped run is enough
- use the older large files on purpose
- make scale and runtime impossible to hide in future artifacts
