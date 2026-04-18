# AGENTS.md

## Commands

```bash
# Lint (must pass before committing)
ruff check . && ruff format --check .

# Fast tests (run before every commit — ~40s)
pytest tests/ -m "not slow" --ignore=tests/test_receiver.py --ignore=tests/test_replayer.py

# Slow I/O tests (need real parquet files)
pytest tests/ -m slow

# Run an experiment
PYTHONPATH=. python scripts/expNN_<name>.py

# Warm the shaped-array cache for a file set
python -m deep_orderbook.cache_manager

# Record live data (rotates hourly to ../crypto/)
deepbook record
```

## Data paths (not in any other doc)

| What | Path |
|---|---|
| Live parquet (2026) | `/mnt/data/repos/gaelreinaudi/crypto/*.parquet` |
| Historical parquet NAS (2024–2025) | `/media/photoDS216/crypto/*.parquet` |
| Shaped-array cache | `cache/` at repo root — also symlinked at `/mnt/data/repos/gaelreinaudi/deep_orderbook/cache/` |
| Experiment DB | `experiments/experiment_runs.sqlite3` |
| Experiment results | `experiments/results/*.json` |

Each `.parquet` file contains **all 3 markets mixed** (`BTC-USD`, `ETH-USD`, `ETH-BTC`). `ReplayConfig.markets[0]` selects which one to use at replay time.

## Non-obvious patterns

### Config mutation: use `.but()`, never raw kwargs
```python
# correct
shaper_cfg = DEFAULT_SHAPER_CONFIG_2026.but(look_ahead=32)

# wrong — raw kwarg chains bypass config validation and logging
load_file_windows(path, market, look_ahead=32)
```

### Generator contract: `break` at the call site, not inside the generator
`iter_shapes_t2l` saves the `.npz` cache in its `finally` block — this fires on `break`, `return`, or exception. **Never** add sample-count limits inside the generator itself.
```python
async for books, t2l, px in iter_shapes_t2l(cfg, shaper_cfg):
    samples.append(...)
    if len(samples) >= n_samples:
        break  # finally runs → cache saved
```

### Naming that is not obvious from the code
- `n_samples` — count of rolling-frame ML training examples per file
- `rolling_window_size` — temporal context in ticks (2048 = ~3.4 min at 100ms)
- `look_ahead` — forecast horizon in ticks (128 = ~12.8 s)
- Do **not** call the sample count `max_windows` — that collides with the temporal window concept

### Cache directory format (market prefix is mandatory)
```
cache/{MARKET}_{every}_vb{view_bips}_lv{num_side_lvl}_la{look_ahead}_lasb{side_bips}_lalv{side_width}/
      {YYYY-MM}/
          {parquet_stem}.npz
```
Old dirs without market prefix (`100ms_vb20_...`) are orphaned legacy — inert, do not delete. `rolling_window_size` and `window_stride` are **not** part of the cache key; they are applied at load time.

### GPU memory cap
At `rolling_window_size=2048`, one full hourly file → ~4500 frames → ~17 GB tensor. Always use `n_samples` in `load_file_windows`. The experiment scripts push the entire dataset to GPU at once — `DataLoaderWorker` exists for streaming but is not yet wired into experiment scripts.

### PnL calculation
Never `reshape(-1, ...)` to flatten overlapping windows for backtesting. Use:
- `aggregate_per_window_strategy_result` for honest per-window PnL
- `select_non_overlapping_indices` for dashboard slicing

## Experiments: mandatory rules

- **Never edit** a past `scripts/expNN_*.py` — create `expNN+1_*.py`
- Every run **must** produce: `.json` artifact + PNG dashboards + DB row
- Every run **must** call `assert_image_meaningful()` from `pipeline_guards.py`
- Read `experiments/scientist_handoff_protocol.md` before writing any new experiment
- Current experiment state: `experiments/hourly_journal.md` → queued next mutation

## Boundaries

### ✅ Always allowed
- Read files, run lint, run fast tests, run a single test file

### ⚠️ Ask first
- Install or remove packages
- Delete `.npz` cache files or experiment artifacts
- Push to remote / open a PR

### 🚫 Never
- Commit anything in `credentials/`
- Edit or delete rows in `experiments/experiment_runs.sqlite3`
- Force push to `main`
- Run `tests/test_receiver.py` or `tests/test_replayer.py` — pre-existing import errors, unrelated to pipeline work
