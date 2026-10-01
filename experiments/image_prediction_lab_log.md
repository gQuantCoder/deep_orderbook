# Deep Orderbook Future-Image Prediction Lab Log

Last updated: 2026-04-12
Owner: Gael + Hermes

## 0) Problem gist (data, goal, constraints)

We are not doing generic video prediction. We are forecasting a structured future map from limit-order-book history.

Current tensors (from shaper pipeline):
- Input `books_array`: shape `[T, 2*num_side_lvl, 3]` (time × price-level bins × channels)
- 3 channels are transformed market microstructure signals (book/trade derived)
- Target `time_levels` / `level_prox`: shape `[T, 2*look_ahead_side_width, 1]`
- Encodes future time-to-level / level-proximity style signal
- Typical training window now: `T=256`, `num_side_lvl=8`, `look_ahead_side_width=4`
- So input is `[256,16,3]`, target is `[256,8,1]`

Empirical visual behavior (from live/replay captures):
- Input maps are dense and noisy/low-contrast.
- Target maps are sparse, high-contrast, event-like.

Implication:
- Pure pixel MSE is not enough.
- We need a model/loss combo that handles sparse targets and keeps causal temporal semantics.

Non-negotiable semantics for this repo:
- Predict from current `now` only.
- Right side = future-after-now.
- Keep full 2D map outputs (no scalar hit-first collapse).

## 1) Objective

Build a robust, causal predictor of future target maps from past input maps that:
1. Preserves full 2D target structure.
2. Improves event recall on sparse target activations.
3. Stays stable enough for iterative online/replay experimentation.

## 2) Smart experimental strategy (image-processing mindset)

We’ll run this as a disciplined sequence, not random model thrashing:

1. Data diagnostics first (distribution, sparsity, leakage checks).
2. Strong but simple baseline (CNN/TCN-style) with better loss.
3. Hybrid upgrade (TCN + causal attention) only if it beats baseline on meaningful metrics.
4. Calibrate and stress-test on multiple replay days.
5. Keep full audit trail in this file + machine-readable JSON artifacts.

Model priority (practical):
- Baseline: TCN/CNN family (DeepLOB/SimVP spirit)
- Main candidate: `AttentionTCN` (already present in untracked files)
- Heavy fallback: pure attention / PredRNN++ style only if needed

## 3) Planned experiment queue

- EXP-01: Target structure + sparsity quantification (replay).
- EXP-02: Baseline training with sparse-aware loss (MSE + weighted event term).
- EXP-03: `AttentionTCN` vs baseline head-to-head on identical windows.
- EXP-04: Ablation on `rolling_window_size` and `look_ahead`.
- EXP-05: Calibration + threshold policy for event detection.
- EXP-06: Multi-day replay robustness (regime drift check).

---

## EXP-01 — Target structure quantification (STARTED)

Status: DONE
Date: 2026-04-05

### Setup

Command used:

```bash
PYTHONPATH=. uv run \
--with numpy --with polars --with coinbase-advanced-py \
--with pydantic-settings --with tqdm --with rich \
python scripts/exp01_target_structure.py
```

Config:
- market: ETH-USD
- replay date regexp: `2025-02-18T11*`
- cadence: `100ms`
- rolling window: 256
- stride: 8
- look_ahead: 32
- levels: input 16 bins, target 8 bins
- analyzed windows: 80

Artifacts:
- JSON: `experiments/results/exp01_target_structure.json`
- Visual samples:
- `images/replay_sample_io.png`
- `images/live_sample_io.png`

### Key observations

From measured stats:
- Target median ~ `5e-9` (effectively zero floor)
- Target mean `0.00654`, std `0.0391`
- Target q99 `0.2103`, max `0.8419`
- Active ratio (`target > 1e-4`) = `0.03885` (~3.9%)

Interpretation:
- Target is strongly sparse/imbalanced.
- Most pixels are near-zero baseline; meaningful events are rare but can be high amplitude.
- This confirms we should avoid plain MSE-only optimization.

Input channel-0 stats (for scaling context):
- min/max ≈ `[-12.01, 11.95]`
- std ≈ `5.24`

Interpretation:
- Inputs have broad dynamic range; normalization strategy matters for stable training.

### Immediate hypotheses

H1) A sparse-aware loss (weighted event term or focal-style component) should improve event recall vs pure MSE.

H2) `AttentionTCN` should beat pure attention on stability/efficiency for this dataset shape and sparsity profile.

H3) Per-channel robust normalization (or clipped z-score) should reduce optimization noise.

## 4) Network + loss ideas to test (with expected predictive-power behavior)

1) TCN baseline (causal conv, no attention)
- Why: strongest simple baseline for local spatiotemporal structure.
- Loss to try:
- `Huber + λ_event * BCE(logit(target>τ))`
- optional monotonic/ranking regularizer already aligned with StructuredT2L idea.
- Expected: best stability, decent RMSE, moderate event recall.

2) AttentionTCN (TCN + causal temporal attention)
- Why: keep conv locality, add longer-range context.
- Loss to try:
- same as baseline + optional focal event term for sparsity.
- Expected: should improve event recall/F1 with manageable compute.

3) Pure attention transformer (current pure_attention.py family)
- Why: high capacity, but risk overfit/instability for this data density.
- Loss to try:
- same sparse-aware combo, but likely needs stronger regularization.
- Expected: only worth it if hybrid fails; not first choice.

4) Uncertainty head (later)
- Add mean + scale prediction (`μ, σ`) and optimize NLL + event term.
- Expected: better decision calibration, not necessarily lower RMSE.

---

## EXP-02 — Predictive-power screen (linear proxy, sparse-aware weighting)

Status: DONE
Date: 2026-04-05

### Purpose

Before full deep runs, quickly test if simple mappings have predictive power and whether sparse-weighting helps event detection.

### Setup

Command used:

```bash
PYTHONPATH=. uv run \
--with numpy --with polars --with coinbase-advanced-py \
--with pydantic-settings --with tqdm --with rich \
python scripts/exp02_predictive_power_screen.py
```

Artifact:
- `experiments/results/exp02_predictive_power_screen.json`

Data split:
- 140 windows total
- 105 train / 35 test (time-ordered)
- Samples: 26,880 train / 8,960 test timesteps
- Features per timestep: 48
- Targets per timestep: 8

### Models tested

- Zero baseline
- Ridge (MSE-like objective)
- Weighted ridge (sparse-aware row weighting)

### Outcome (predictive power)

Dense regression quality:
- Zero baseline RMSE: `0.0263`
- Ridge RMSE: `3.4312` (catastrophic; much worse)
- Weighted ridge RMSE: `4.5370` (even worse)

Event detection at threshold `1e-4`:
- Zero baseline F1: `0.0000` (predicts no events)
- Ridge F1: `0.0800`
- Weighted ridge F1: `0.0954`

Interpretation:
- Linear proxies are not viable for dense map regression here.
- Sparse-weighting helps recall/F1 a bit, but with huge false positives and terrible calibration.
- This supports using nonlinear causal nets with bounded outputs and sparse-aware losses.

### Practical conclusion from EXP-02

- Keep sparse-aware objective idea.
- Drop linear-style baselines for final modeling.
- Next experiment must be a real causal nonlinear net (TCN family), with:
- robust input normalization,
- bounded output head (avoid exploding predictions),
- dual metrics: dense error + event F1/precision/recall.

## EXP-03 — Nonlinear baseline screen (MLP + boosted trees, sparse weighting)

Status: DONE
Date: 2026-04-05

### Purpose

Run the next experiment with nonlinear models to measure predictive power beyond linear proxies and test sparse weighting impact.

### Notes on execution constraints

- Attempted to run a compact PyTorch TCN first.
- Blocker: installing torch via uv default pulled large CUDA artifacts and hit disk-space limits.
- Practical workaround used for this run: nonlinear scikit-learn models (MLP + HistGradientBoosting), still useful as fast predictive-power screen.

### Setup

Command used:

```bash
PYTHONPATH=. uv run \
--with numpy --with polars --with coinbase-advanced-py \
--with pydantic-settings --with tqdm --with rich --with scikit-learn \
python scripts/exp03_nonlinear_sparse_models.py
```

Artifact:
- `experiments/results/exp03_nonlinear_sparse_models.json`

### Outcome (predictive power)

Dense regression (RMSE, lower is better):
- Zero baseline: `0.02634`
- MLP: `1.47111` (bad)
- HGB: `0.02761` (~4.8% worse than zero)
- HGB sparse-weighted: `0.04332` (much worse)

Event detection (F1 @ 1e-4, higher is better):
- Zero: `0.0000`
- MLP: `0.0608`
- HGB: `0.1948`
- HGB sparse-weighted: `0.2090` (best F1 in this run)

Interpretation:
- Nonlinear models can recover event signal (F1 up to ~0.21) unlike zero baseline.
- But event gains currently come with dense-regression degradation.
- Sparse weighting improves recall/F1 slightly but hurts calibration and RMSE.

### Practical conclusion

- We do have predictive signal, but current objective/model balance is not yet good enough.
- Next should optimize for a Pareto point: better event F1 with bounded RMSE degradation.
- Candidate path remains: causal TCN / AttentionTCN with explicit multi-objective tuning and bounded output behavior.

### Next action (hourly loop enabled)

- Start hourly autonomous reflection/experiment cycle.
- Each cycle runs small nonlinear sweeps, logs artifacts, and emits alert only when “interesting” criteria are met.
- Criteria (current):
- F1 >= 0.30
- precision >= 0.20
- RMSE <= 1.20 * zero-baseline RMSE

## EXP-04 — Multi-horizon actionable screen (few vs many steps ahead)

Status: DONE
Date: 2026-04-05

### Purpose

Test the same image-map concept across multiple future depths (look_ahead 8/16/32/64) to find actionable intermediate targets, not just one fixed horizon.

### Setup

Command used:

```bash
PYTHONPATH=. uv run --python .venv/bin/python \
scripts/exp04_multihorizon_actionable_screen.py
```

Artifact:
- `experiments/results/exp04_multihorizon_actionable_screen.json`

Model family screened per horizon:
- HistGradientBoosting (unweighted)
- HistGradientBoosting (sparse-weighted)

### Outcome summary

Best per-horizon event quality (F1 / precision / side-acc-on-signals):
- look_ahead=8:  F1=0.0274, precision=0.0140, side_acc=0.3734
- look_ahead=16: F1=0.0905, precision=0.0474, side_acc=0.2421
- look_ahead=32: F1=0.1602, precision=0.0871, side_acc=0.3384
- look_ahead=64: F1=0.2253, precision=0.1286, side_acc=0.5600

Interpretation:
- Longer-horizon target (64) currently yields the strongest actionable classification-like behavior.
- Short horizon is still too noisy under this model/loss family.
- RMSE remains worse than zero baseline across horizons, so this is still event-signal-first, not robust dense-map regression yet.

### Practical decision

Use multi-horizon tracking going forward (16/32/64 at minimum), and gate “interesting found” on joint mid+long horizon precision/F1/side-accuracy, not on one scalar metric.

### Hourly loop update

`hourly_experiment_cycle.py` now runs multi-horizon probes (16, 32, 64), logs per-horizon metrics, and only sets stop flag when stricter joint criteria are met.

## EXP-06 — Hurdle map model (event gate + conditional map regression)

Status: IN PROGRESS (first mutation run completed)
Date: 2026-04-05

### Purpose

When structured TCN underperforms baseline on long-horizon precision/F1, test a 2-stage model:
1) predict whether any event should be emitted,
2) if yes, regress full 2D map conditionally.

### Run executed

Command used:

```bash
PYTHONPATH=. uv run --python .venv/bin/python scripts/exp06_tcn_hurdle_map.py
```

Artifacts:
- `experiments/results/exp06_tcn_hurdle_map_20260405T203023Z.json`
- `experiments/results/exp06_hurdle_mutation_state.json`

Mutation (one factor):
- `gate_threshold`: `0.50 -> 0.58`
- Rationale: cut false positives to improve precision/F1 at look_ahead=64.

### Outcome (look_ahead=64)

Reference (single-stage baseline in same run):
- F1: `0.2366`
- Precision: `0.1358`
- Side accuracy: `0.5323`
- RMSE: `0.02963`

Hurdle candidate:
- F1: `0.2471` (+0.0105)
- Precision: `0.1415` (+0.0058)
- Side accuracy: `0.3897` (-0.1427)
- RMSE: `0.03321` (+0.00358)

Interpretation:
- Precision/F1 improved, but directional quality and dense error degraded materially.
- This mutation is not yet a practical replacement for the current best long-horizon candidate.

### Decision + next mutation

Decision: **reject current hurdle variant** as primary candidate.

Next mutation (for next run, execute first):
- Keep hurdle architecture.
- Increase `gate_threshold` to `0.62` to suppress over-triggering and attempt to recover side accuracy/RMSE while preserving precision gains.


## EXP-06H — Hourly continuity run 2026-04-05 21:38:32Z

Status: DONE (not promising yet)

Hypothesis:
- Adding a side-dominance post-filter to hurdle outputs should recover directional quality while preserving the precision/F1 lift from event gating.

Mutation (single factor):
- Added `side_margin_postfilter=0.0015` with fixed `gate_threshold=0.62`, `pos_weight=7.0`, `reg_weight_scale=12.0`.

Artifacts:
- Baseline: `experiments/results/hourly/run_20260405T213416Z.json`
- Structured compare (timestamped): `experiments/results/exp05_tcn_structured_loss_compare_20260405T213832Z.json`
- Hurdle: `experiments/results/exp06_tcn_hurdle_map_20260405T213627Z.json`

Long-horizon (look_ahead=64) readout:
- Baseline best (hourly HGB): f1=0.2340, precision=0.1340, side_acc=0.5644, rmse=0.02999, zero_rmse=0.01770
- Structured TCN vs TCN baseline: f1 0.0355 vs 0.0472, precision 0.0185 vs 0.0252 (underperformed)
- Hurdle vs single-stage in same run: Δf1=+0.0263, Δprecision=+0.0166, Δside_acc=-0.1387, Δrmse=+0.00312

Decision:
- Reject current side-margin setting (0.0015). It did not gate anything (`side_keep_rate_within_gated=1.000`), so directional quality and RMSE issues remained.

Next mutation:
- Increase side margin only to `0.0045` (same gate threshold and class weight), then re-check side_acc and RMSE before touching other knobs.

### 2026-04-05 22:45:52Z hourly scientist update
- Baseline best @64 stayed `hgb_d6_lr006_w` (f1=0.2255, p=0.1285, side=0.5236, rmse=0.02979).
- Structured-loss remained below baseline @64 (f1=0.0330, p=0.0172; baseline-TCN f1=0.0472, p=0.0252).
- Hurdle side-margin=0.01 and subsequent gate_threshold=0.66 improved sparse precision/F1 vs single-stage but still failed practical gate due side_acc collapse and RMSE inflation.
- Artifacts: `run_20260405T224132Z.json`, `exp05_tcn_structured_loss_compare_20260405T224407Z.json`, `exp06_tcn_hurdle_map_20260405T224223Z.json`, `exp06_tcn_hurdle_map_20260405T224329Z.json`.
- Next mutation queued: reduce pos_weight 7.0 -> 6.0 with gate_threshold=0.66 and side_margin=0.01 fixed.

### 2026-04-05 23:52:20Z hourly scientist update
- Baseline best @64 stayed `hgb_d6_lr006_w` (f1=0.2330, p=0.1333, side=0.5516, rmse=0.02942).
- Structured-loss remained below baseline-TCN @64 (f1=0.0366 vs 0.0472; p=0.0191 vs 0.0252).
- Hurdle pos_weight=6.0 improved sparse precision/F1 vs single-stage but still failed practical gate due side_acc collapse and RMSE inflation (Δf1=+0.0101, Δp=+0.0065, Δside=-0.1915, Δrmse=+0.00251).
- Artifacts: `experiments/results/hourly/run_20260405T234847Z.json`, `experiments/results/exp05_tcn_structured_loss_compare_20260405T235220Z.json`, `experiments/results/exp06_tcn_hurdle_map_20260405T234933Z.json`.
- Next mutation queued: side_margin 0.0100 -> 0.0150 (single factor) with gate_threshold=0.66 and pos_weight=6.0 fixed.


### 2026-04-06 00:58:46Z hourly scientist update
- Baseline best @64 stayed `hgb_d6_lr006_w` (`f1=0.2274`, `p=0.1297`, `side=0.5831`, `rmse=0.02916`, `zero_rmse=0.01770`) from `experiments/results/hourly/run_20260406T005431Z.json`.
- Structured-loss remained below baseline-TCN @64 (`f1=0.0366` vs `0.0472`; `p=0.0191` vs `0.0252`) from `experiments/results/exp05_tcn_structured_loss_compare_20260406T005553Z.json`.
- Executed queued handoff mutation first: hurdle `side_margin=0.015` with `gate_threshold=0.66`, `pos_weight=6.0` fixed.
- Hurdle improved sparse-event quality but still failed practical gate due side-quality deficit: vs single-stage baseline `Δf1=+0.0458`, `Δp=+0.0315`, `Δside=-0.1441`, `Δrmse=+0.00073`; side filter keep-rate still high at `0.975`.
- Artifact: `experiments/results/exp06_tcn_hurdle_map_20260406T005628Z.json`.
- Decision: **not promising yet**.
- Next mutation queued: raise `gate_threshold` to `0.70` only (keep side_margin=0.015, pos_weight=6.0) to force stricter gating and target side-accuracy recovery.

## 2026-04-06 02:04:07Z hourly scientist update
- Baseline (la=64): f1=0.2360, p=0.1356, side_acc=0.6061, rmse=0.02901
- Structured TCN (la=64): f1=0.0346, p=0.0180 (fails baseline gate)
- Hurdle @ gate=0.70, pos_w=6, side_margin=0.015: f1=0.2540, p=0.1489, side_acc=0.3901, rmse=0.03138, keep_rate=0.969
- Outcome: sparse-event metrics improved but directional quality still unacceptable; practical status remains not promising yet.
- Next queued one-factor mutation: pos_weight 6.0 -> 5.0 with gate/side_margin fixed.



### 2026-04-12 18:37:20Z hourly scientist update
- Baseline best @64 stayed `hgb_d6_lr006_w` (`f1=0.2269`, `p=0.1294`, `side=0.5761`, `rmse=0.02952`, `zero_rmse=0.01770`) from `experiments/results/hourly/run_20260412T183559Z.json`.
- Structured-loss remained below baseline-TCN @64 (`f1=0.0356` vs `0.0472`; `p=0.0185` vs `0.0252`) from `experiments/results/exp05_tcn_structured_loss_compare_20260412T183644Z.json`.
- Executed queued handoff mutation first: hurdle `pos_weight=5.0` with `gate_threshold=0.70`, `side_margin=0.015`, `reg_weight_scale=12.0` fixed.
- Hurdle lost quality vs the prior hurdle checkpoint and still failed the practical gate: vs same-run single-stage baseline `Δf1=+0.0034`, `Δp=+0.0040`, `Δside=-0.1734`, `Δrmse=+0.00269`; side filter keep-rate remained very high at `0.977`.
- Artifact: `experiments/results/exp06_tcn_hurdle_map_20260412T183659Z.json`.
- Decision: **not promising yet**.
- Next mutation queued: lower `reg_weight_scale` `12.0 -> 8.0` only (keep gate_threshold=0.70, pos_weight=5.0, side_margin=0.015) to reduce conditional-map overshoot now that gate threshold is capped and class-weight reduction did not recover directional quality.

### 2026-04-12 19:43:04Z hourly scientist update
- Baseline best @64 stayed `hgb_d6_lr006_w` (`f1=0.2311`, `p=0.1321`, `side=0.5845`, `rmse=0.02910`, `zero_rmse=0.01770`) from `experiments/results/hourly/run_20260412T194221Z.json`.
- Structured-loss remained below baseline-TCN @64 (`f1=0.0338` vs `0.0472`; `p=0.0176` vs `0.0252`) from `experiments/results/exp05_tcn_structured_loss_compare_20260412T194333Z.json`.
- Executed queued handoff mutation first: hurdle `reg_weight_scale=8.0` with `gate_threshold=0.70`, `pos_weight=5.0`, `side_margin=0.015` fixed.
- Hurdle degraded further versus both the prior hurdle checkpoint and the same-run single-stage baseline: vs single-stage baseline `Δf1=-0.0019`, `Δp=+0.0013`, `Δside=-0.1910`, `Δrmse=+0.00627`; side filter keep-rate remained very high at `0.9797`.
- Artifact: `experiments/results/exp06_tcn_hurdle_map_20260412T194304Z.json`.
- Decision: **not promising yet**.
- Next mutation queued: lower `reg_weight_scale` `8.0 -> 6.0` only (keep gate threshold / class weight / side margin fixed) to finish the overshoot-softening sweep before abandoning this hurdle branch.

## 2026-04-13 scientist workflow update

Status: DONE

### What changed
- Added experiment DB: `experiments/experiment_runs.sqlite3`
- Added mandatory preview-picture output path: `experiments/pictures/`
- Added scientist handoff protocol: `experiments/scientist_handoff_protocol.md`
- Updated `scripts/exp05_tcn_structured_loss_compare.py` so it now:
  - writes timestamped JSON artifacts
  - saves preview PNGs for each horizon/model variant
  - registers each run in the experiment DB

### New TCN compare artifact
- JSON: `experiments/results/exp05_tcn_structured_loss_compare_20260413T225239Z.json`
- Pictures:
  - `experiments/pictures/exp05_tcn_structured_loss_compare_h32_baseline_tcn_20260413T225239Z.png`
  - `experiments/pictures/exp05_tcn_structured_loss_compare_h32_structured_tcn_20260413T225239Z.png`
  - `experiments/pictures/exp05_tcn_structured_loss_compare_h64_baseline_tcn_20260413T225239Z.png`
  - `experiments/pictures/exp05_tcn_structured_loss_compare_h64_structured_tcn_20260413T225239Z.png`
- DB rows inserted: ids `1..4`

### Practical interpretation
- The infrastructure is now better: every serious run can leave behind machine-readable metrics plus reviewable pictures.
- The latest TCN branch is still **not promising yet**.
- This makes future scientist handoff much less lossy: a new run can inspect the DB, the pictures, and the protocol before choosing the next mutation.

### Recorder note
- Live recorder currently writes active JSONL to `data/L2/`
- On hourly rollover it writes parquet to `../crypto/` relative to repo root
- Historical replay corpus remains `/media/photoDS216/crypto/`

## 2026-04-13 exp10 one-shot scientist run

Status: DONE

### Purpose
Run the new protocol once end-to-end:
1. precheck shaped data richness
2. save a precheck image
3. run one learning pass
4. save a dashboard with prediction / trade decisions / pnl
5. register the result in the experiment DB
6. save a scientist note file

### Data source
- fresh rolled parquet from recorder output: `/mnt/data/repos/gaelreinaudi/crypto/2026-04-13T22-51-51.parquet`

### Artifacts
- JSON: `experiments/results/exp10_scientist_once_20260413T231612Z.json`
- Precheck image: `experiments/pictures/exp10_scientist_once_precheck_20260413T231612Z.png`
- Dashboard image: `experiments/pictures/exp10_scientist_once_dashboard_20260413T231612Z.png`
- Notes: `experiments/notes/exp10_scientist_once_20260413T231612Z.md`
- DB row: `experiment_runs.id = 5`

### Richness gate
- Passed.
- books_std=`2.7849`
- books_abs_peak=`12.1190`
- target_active_ratio=`0.4409`
- target_peak=`2.2045`

Visual verdict from precheck:
- rich enough for training
- not flat
- not dead
- not globally saturated

### Quick TinyTCN run
- cadence: `100ms`
- horizon: `64`
- source: fresh recorder rollover parquet
- metrics:
  - f1=`0.5081`
  - precision=`0.4768`
  - recall=`0.5438`
  - rmse=`0.20647`
- zero baseline rmse=`0.20622`
- pnl:
  - omniscient=`12.09033`
  - prediction=`0.76025`

### Interpretation
- This was a useful protocol validation run, not a genuinely strong trading result.
- The shaper richness gate behaved correctly and selected a visually rich slice.
- The dashboard was successfully saved and is readable.
- Prediction quality is still not practically convincing:
  - prediction pnl is far below omniscient
  - rmse does not beat the zero baseline
  - visual inspection suggests the prediction map has some structure but is still weak / overfit-prone

### Decision
- **not promising yet**

### Next mutation
- Keep the new protocol.
- Next run should try the same flow on another fresh rolled parquet and compare symbols / thresholds, while continuing to require precheck images before training.

## 2026-04-13 exp11 second one-shot scientist run (BTC mutation)

Status: DONE

### Mutation from prior run
- Changed one primary factor only: symbol `ETH-USD` -> `BTC-USD`
- Kept the same fresh parquet source, cadence (`100ms`), horizon (`64`), model family (TinyTCN), and general protocol flow.

### Hypothesis
- After the weak ETH run, switching only the symbol to BTC-USD on the same fresh parquet may reveal cleaner map structure and more tradable behavior.

### Artifacts
- JSON: `experiments/results/exp11_scientist_once_btc_20260413T232646Z.json`
- Precheck image: `experiments/pictures/exp11_scientist_once_btc_precheck_20260413T232646Z.png`
- Dashboard image: `experiments/pictures/exp11_scientist_once_btc_dashboard_20260413T232646Z.png`
- Notes: `experiments/notes/exp11_scientist_once_btc_20260413T232646Z.md`
- DB row: `experiment_runs.id = 6`

### Richness gate
- Passed.
- books_std=`2.3890`
- books_abs_peak=`13.3866`
- target_active_ratio=`0.4185`
- target_peak=`69.9300`

Visual verdict from precheck:
- rich enough for training
- not flat
- not dead
- not globally saturated

### Outcome
- f1=`0.5873`
- precision=`0.4180`
- recall=`0.9872`
- rmse=`1596.30530`
- zero baseline rmse=`6.65414`
- omniscient pnl=`163.57812`
- prediction pnl=`45.67188`

### Interpretation
- This mutation did produce a more interesting-looking dashboard and higher raw prediction pnl than the ETH run.
- But it is **not** genuinely promising yet.
- The key blocker is catastrophic dense-map error: rmse exploded relative to the zero baseline.
- So the raw precision/F1 and pnl numbers are misleading if read alone.
- Visual read: some structure is present, but calibration / amplitude appears badly controlled.

### Decision
- **not promising yet**

### Next mutation
- Keep `BTC-USD`.
- Change one factor that directly targets output calibration / amplitude control before trusting this branch.

## 2026-04-14 exp12 third one-shot scientist run (BTC capped prediction)

Status: DONE

### Mutation from prior run
- Kept `BTC-USD`
- Changed one primary factor only: applied an inference-time prediction amplitude cap using the train-target q99.5 percentile
- Everything else kept conceptually the same: fresh recorder parquet, `100ms`, `look_ahead=64`, TinyTCN quick run, same protocol flow

### Hypothesis
- BTC showed interesting raw structure but catastrophic dense-map error; capping prediction amplitude to a train-derived percentile may preserve useful pattern timing while controlling blow-up.

### Artifacts
- JSON: `experiments/results/exp12_scientist_once_btc_capped_20260414T035312Z.json`
- Precheck image: `experiments/pictures/exp12_scientist_once_btc_capped_precheck_20260414T035312Z.png`
- Dashboard image: `experiments/pictures/exp12_scientist_once_btc_capped_dashboard_20260414T035312Z.png`
- Notes: `experiments/notes/exp12_scientist_once_btc_capped_20260414T035312Z.md`
- DB row: `experiment_runs.id = 7`

### Richness gate
- Passed.
- books_std=`2.4195`
- books_abs_peak=`13.3865`
- target_active_ratio=`0.1221`
- target_peak=`23.2547`

Visual verdict from precheck:
- usable for training
- not dead
- not globally saturated
- somewhat sparse, but acceptable

### Outcome
- prediction cap used: train q99.5 = `7.7531`
- f1=`0.1201`
- precision=`0.0639`
- recall=`1.0000`
- rmse=`0.51082`
- zero baseline rmse=`0.84282`
- omniscient pnl=`55.92188`
- prediction pnl=`48.21875`

### Interpretation
- This mutation fixed the catastrophic RMSE blow-up from the previous BTC run.
- The dashboard looks more controlled and visually more believable than the uncapped BTC version.
- But sparse-event precision is still weak and recall is too indiscriminate.
- So this is a real calibration improvement, not yet a tradable-model success.

### Decision
- **not promising yet**

### Next mutation
- Keep `BTC-USD` and the calibration mindset.
- Next single-factor change should target event over-triggering / precision recovery, not dense-map amplitude anymore.

## 2026-04-14 exp13 multi-file walk-forward BTC holdout run

Status: DONE

### Online validation tricks applied
- Switched from same-file train/test to a stricter walk-forward holdout split, consistent with online guidance on time-series leakage control and walk-forward validation.
- Kept `BTC-USD` because online microstructure guidance suggests larger-tick setups are often easier to forecast than smaller/noisier ones.

### Mutation from prior run
- Kept `BTC-USD`
- Kept capped prediction calibration
- Changed one primary validation factor: train and test now use different parquet files
  - train:
    - `2026-04-14T00-00-01.parquet`
    - `2026-04-14T01-00-10.parquet`
    - `2026-04-14T02-00-01.parquet`
  - test:
    - `2026-04-14T03-00-00.parquet`

### Hypothesis
- Walk-forward holdout on different parquet files, plus capped BTC predictions, should give a more honest read on whether the apparent pattern survives beyond one local file.

### Artifacts
- JSON: `experiments/results/exp13_multifile_btc_walkforward_20260414T041201Z.json`
- Precheck image: `experiments/pictures/exp13_multifile_btc_walkforward_precheck_20260414T041201Z.png`
- Best-slice dashboard: `experiments/pictures/exp13_multifile_btc_walkforward_best_20260414T041201Z.png`
- Fixed-slice dashboard: `experiments/pictures/exp13_multifile_btc_walkforward_fixed_20260414T041201Z.png`
- Notes: `experiments/notes/exp13_multifile_btc_walkforward_20260414T041201Z.md`
- DB row: `experiment_runs.id = 8`

### Outcome
- Holdout richness gate passed, but target activity was much sparser than prior same-file runs (`target_active_ratio=0.0366`).
- Holdout metrics:
  - f1=`0.1299`
  - precision=`0.0696`
  - recall=`0.9733`
  - rmse=`2.14880`
- Zero baseline holdout rmse=`0.44353`
- PnL on both best and fixed holdout slices:
  - omniscient=`15.0625`
  - prediction=`-3.6406`

### Interpretation
- The stricter multi-file holdout made the apparent pattern largely collapse out-of-sample.
- This strongly supports the suspicion that earlier same-file dashboards were flattering the model.
- Current BTC branch is still over-triggering events and failing the honest holdout test.

### Decision
- **not promising yet**

### Next mutation
- Keep the stricter multi-file validation.
- Next single-factor change should explicitly reduce event firing / improve precision on holdout, not relax the validation standard.

## 2026-04-14 exp14 / exp15 BTC batch debrief

Status: DONE

### Why this batch was run
- Holdout exp13 showed the same-file-looking BTC pattern mostly collapsed under different-parquet testing.
- New goal: keep the stricter walk-forward holdout and probe whether different training/loss styles can recover a better image-to-image signal.

### Online ideas used
- Walk-forward / different-file holdout remained mandatory to avoid flattering overlap leakage.
- Sparse-event / field-prediction literature suggests trying different loss emphasis rather than trusting one default objective.
- Two concrete hypotheses were tested:
  1. lower event-loss pressure may reduce event spam and improve precision
  2. pure image-to-image regression may preserve cleaner map geometry than event-augmented training

### Scripts created
- `scripts/exp14_multifile_btc_precision.py`
- `scripts/exp15_multifile_btc_regonly.py`

### Variant 1 — exp14 precision-focused BTC holdout
- JSON: `experiments/results/exp14_multifile_btc_precision_20260414T042753Z.json`
- Fixed dashboard: `experiments/pictures/exp14_multifile_btc_precision_fixed_20260414T042753Z.png`
- Notes: `experiments/notes/exp14_multifile_btc_precision_20260414T042753Z.md`
- DB row: `experiment_runs.id = 9`

Mutation:
- lower event loss weight (`0.05`)
- lower positive class weight (`2.0`)
- higher trade threshold (`0.10`)
- kept capped prediction and multi-file holdout

Outcome:
- f1=`0.1459`
- precision=`0.0788`
- recall=`0.9840`
- rmse=`2.06836`
- zero baseline rmse=`0.44353`
- fixed-slice prediction pnl=`29.30469`

Interpretation:
- Slightly better than exp13 on F1/precision/RMSE, but still far worse than zero-baseline RMSE.
- Prediction PnL looked unrealistically strong relative to omniscient, so realism remains questionable.
- Decision: **not promising yet**.

### Variant 2 — exp15 regression-only BTC holdout
- JSON: `experiments/results/exp15_multifile_btc_regonly_20260414T042842Z.json`
- Fixed dashboard: `experiments/pictures/exp15_multifile_btc_regonly_fixed_20260414T042842Z.png`
- Notes: `experiments/notes/exp15_multifile_btc_regonly_20260414T042842Z.md`
- DB row: `experiment_runs.id = 10`

Mutation:
- pure regression-style training (`event_loss_weight=0.0`)
- kept prediction cap and multi-file holdout

Outcome:
- f1=`0.1584`
- precision=`0.0876`
- recall=`0.8267`
- rmse=`2.10439`
- zero baseline rmse=`0.44353`
- fixed-slice prediction pnl=`9.42969`

Interpretation:
- Best precision/F1 of the three strict holdout BTC variants so far (`exp13/14/15`), and the dashboard looks more believable than exp14.
- Still fails the hard RMSE guard by a wide margin.
- Decision: **not promising yet**.

### Batch conclusion
Across the strict holdout BTC set:
- exp13 baseline capped holdout: f1=`0.1299`, precision=`0.0696`, rmse=`2.1488`
- exp14 precision-focused: f1=`0.1459`, precision=`0.0788`, rmse=`2.0684`
- exp15 regression-only: f1=`0.1584`, precision=`0.0876`, rmse=`2.1044`

So:
- regression-only currently looks like the most interesting of the strict-holdout variants
- but none are close to clearing the dense-error guard
- the main unresolved issue is still holdout generalization, not just pretty pictures

### Next mutation
- Keep:
  - BTC
  - multi-file walk-forward holdout
  - prediction cap
- Next single-factor experiment should target generalization / early overfit control directly:
  - earlier stopping / fewer epochs
  - or smaller hidden width

## 2026-04-14 exp16 12-variant BTC TCN search + best-route rerun

Status: DONE

### Why this batch was run
- The previous strict-holdout BTC variants had feature-rich images but weak honest generalization.
- New goal: search a broader 12-variant TCN loss/training grid while keeping the hard holdout fixed.
- Then rerun the best route once to see if the result is stable or just a lucky seed.

### Scripts created
- `scripts/exp16_batch_h64_tcn.py`
- `scripts/exp16_rerun_best_variant.py`
- `deep_orderbook/btc_search_lab.py`

### Search space
Kept fixed:
- cadence=`100ms`
- look_ahead=`64`
- symbol=`BTC-USD`
- walk-forward split:
  - train: `2026-04-14T00-00-01.parquet`, `2026-04-14T01-00-10.parquet`, `2026-04-14T02-00-01.parquet`
  - test: `2026-04-14T03-00-00.parquet`

Varied across 12 runs:
- regression loss: Huber / MSE / L1
- event loss weight: `0.0`, `0.02`, `0.05`, `0.10`, `0.25`
- class weighting: `1`, `2`, `4`, `8`
- hidden width: `64`, `96`, `128`, `160`
- epochs: `4`, `6`, `8`, `10`
- learning rate / weight decay
- one active-pixel weighted regression variant

### Automatic image QC added
Each fixed dashboard now logs:
- gray mean / std
- near-black fraction
- near-white fraction
- simple edge score
- usable vs reject

Practical eyeball rule stayed the same:
- not fully black
- not fully saturated
- enough contrast to compare target vs prediction

### Batch summary artifact
- JSON: `experiments/results/exp16_batch_h64_tcn_20260414T045203Z.json`
- Notes: `experiments/notes/exp16_batch_h64_tcn_20260414T045203Z.md`
- Precheck: `experiments/pictures/exp16_batch_h64_tcn_holdout_precheck_20260414T045203Z.png`

### Ranked outcome
Top route:
- variant=`l1_evt005_pw2`
- precision=`0.1829`
- recall=`0.1115`
- f1=`0.1386`
- rmse=`0.44350`
- zero-baseline rmse=`0.44353`
- fixed pnl=`0.0000`
- image qc: usable, gray_std=`0.3031`

Most important comparison:
- this was the first variant to get RMSE essentially back to zero-baseline while also clearing `precision > 0.10`
- but it did that by becoming too conservative to trade (`prediction pnl = 0`)

Other notable routes:
- `regonly_short_e4`: f1=`0.1669`, precision=`0.0911`, rmse=`1.0143`, fixed pnl=`-14.25`
- `small_h64_regonly`: f1=`0.1575`, precision=`0.0871`, rmse=`1.1727`, fixed pnl=`12.3281`
- baseline `exp13`-like route stayed weak: precision=`0.0696`, rmse=`2.1492`

### Best-route rerun
- JSON: `experiments/results/exp16_best_rerun_l1_evt005_pw2_20260414T045412Z.json`
- Fixed dashboard: `experiments/pictures/exp16_best_rerun_l1_evt005_pw2_fixed_20260414T045412Z.png`
- Best dashboard: `experiments/pictures/exp16_best_rerun_l1_evt005_pw2_best_20260414T045412Z.png`

Rerun metrics:
- precision=`0.1085`
- recall=`0.1580`
- f1=`0.1287`
- rmse=`0.44379`
- fixed pnl=`0.0000`
- image qc: usable, gray_std=`0.3041`

Interpretation:
- The rerun preserved the same story:
  - image stays feature rich and visually usable
  - RMSE remains near zero-baseline
  - precision stays above prior baseline levels
  - but trading still does not fire usefully
- So the top route looks real, but it is real as a conservative non-trading mapper, not yet as a tradable signal generator.

### Eyeball result
Manual/vision check of the rerun dashboards:
- not entirely black
- not completely saturated
- books and target panels are rich and plausible
- prediction panel is sparse and underactive rather than noisy or blown out
- this visually matches the zero-PnL behavior

### Decision
- **most promising route so far for honest structure preservation:** `l1_evt005_pw2`
- **most promising route for tradability:** still unresolved
- overall status: **not promising yet for trading**, but finally somewhat promising for stable holdout image quality

### Next mutation
Keep the winning route fixed as the geometry baseline:
- loss=`L1 + 0.05 * BCE`
- pos_weight=`2`
- hid=`96`
- epochs=`6`
- same multi-file holdout

Then mutate only the signal extraction layer, not the core mapper:
1. lower trade threshold calibration sweep on the L1 route
2. side-aware / local-max event extraction on prediction map
3. maybe a shallow second-stage trigger model on top of the conservative map

## EXP-17 — Event-filtered recent BTC holdout (DONE)

Status: DONE
Date: 2026-04-14

### Purpose

Test the user's new idea directly: train and evaluate only on the most eventful windows from the newest replay files, using present-time observable triggers rather than future-target cheating.

### Setup

Command used:

```bash
PYTHONPATH=. python scripts/exp16_batch_h64_tcn.py \
  --label exp17_event_filtered_h64_tcn \
  --eventful-top-fraction 0.35 \
  --min-train-windows 24 \
  --min-test-windows 12 \
  --variants l1_evt005_pw2 regonly_huber_thr010 precision_evt005_pw2_thr010 regonly_activew3
```

Data split:
- train files: `2026-04-14T07-00-00.parquet`, `2026-04-14T08-00-00.parquet`, `2026-04-14T09-00-00.parquet`
- test file: `2026-04-14T10-00-00.parquet`
- event filter: keep top 35% of windows by intrawindow eventfulness score
- eventfulness proxies used:
  - abs return bps
  - intrawindow range bps
  - realized vol of mid-price changes
  - book std
  - book impulse
- selected windows:
  - train: `51 / 144`
  - test: `17 / 48`
- older-file backfill needed: none

Artifacts:
- summary json: `experiments/results/exp17_event_filtered_h64_tcn_20260414T114251Z.json`
- notes: `experiments/notes/exp17_event_filtered_h64_tcn_20260414T114251Z.md`
- top fixed dashboard: `experiments/pictures/exp17_event_filtered_h64_tcn_l1_evt005_pw2_fixed_20260414T114251Z.png`
- top best dashboard: `experiments/pictures/exp17_event_filtered_h64_tcn_l1_evt005_pw2_best_20260414T114251Z.png`

### Outcome

Best variant: `l1_evt005_pw2`

Holdout metrics:
- rmse: `1.14068`
- zero-baseline rmse: `1.14359`
- precision: `0.2957`
- recall: `0.3697`
- f1: `0.3285`
- fixed-slice prediction pnl: `-0.02344`

Other routes:
- `regonly_huber_thr010`: precision `0.3208`, f1 `0.2310`, rmse `1.13593`, fixed pnl `0.00000`
- `precision_evt005_pw2_thr010`: precision `0.1472`, f1 `0.2477`, rmse `1.09895`, fixed pnl `0.93750`
- `regonly_activew3`: precision `0.1252`, f1 `0.2202`, rmse `1.08061`, fixed pnl `-16.21094`

### Interpretation

This branch did something real:
- precision and F1 improved a lot versus the earlier broad-slice strict holdout runs
- RMSE stayed near the zero baseline instead of blowing out
- image QC stayed usable

But it still failed the practical tradability test:
- best route PnL was basically flat/slightly negative
- vision check of the top dashboard showed partial timing alignment but underfiring / vertically collapsed predictions
- in plain English: the model sees some reaction timing in violent windows, but the trade-trigger extraction is still too weak to monetize it

### Practical conclusion from EXP-18

Event-window conditioning is worth keeping in the palette.
It is not a gimmick.
It improved honest event-quality metrics on recent data without RMSE collapse.

But it is not enough by itself.
Current mapper outputs are still too weak/collapsed for profitable triggering.

### Cache-system correction

Important correction for future scientists:
- the repo already had a proper shaper/cache path
- `deep_orderbook/shaper.py` + `deep_orderbook/cache_manager.py` already cache shaped arrays
- `ShaperConfig` defaults already enable `use_cache=True` and `save_cache=True`
- the slowdown in later custom experiment scripts came from the experiment code forcing cache off, not from missing infrastructure

Anti-wheel-reinvention rule:
- before writing new preprocessing loops, inspect existing cache behavior first
- for repeated trigger/strategy sweeps, reuse cached shaped arrays and preferably cached model predictions instead of replaying/parsing history every run
- if any script disables cache, that must be called out explicitly in its notes/json and justified

### Next mutation

Keep the event-window conditioning fixed and stop thrashing the mapper family for one step.
Next mutation should act only on signal extraction / trading conversion, for example:
- local-max trigger extraction from predicted maps
- side-aware trigger extraction
- threshold sweep on the event-filtered holdout
- or a shallow second-stage trigger model trained only on event-filtered windows

## EXP-18 — 25-run event-filtered suite on recent BTC (DONE)

Status: DONE
Date: 2026-04-14

### Purpose

Do a larger disciplined search on the recent event-filtered regime, with enough variants to separate mapper quality from trade-trigger behavior.

### Leakage check

The event filter remained leakage-safe:
- window ranking used only present-window observables
- specifically: mid-price abs return, intrawindow range, realized vol, book std, book impulse
- no future target intensity or future PnL was used to select windows
- holdout stayed chronological by parquet file (`07,08,09` train -> `10` test)

### Data scale and runtime accounting

This was not full-dataset heavy training.
It was a screened recent-data run.
That needs to be stated plainly.

Actual scale used:
- train parquet files: `3`
- test parquet files: `1`
- per-file load cap in script: `max_windows=48`
- train windows before filter: `144`
- train windows after filter: `51`
- test windows before filter: `48`
- test windows after filter: `17`
- rolling window size: `256`
- approximate train timesteps seen by the model: `13,056`
- approximate test timesteps seen by the model: `4,352`
- approximate train target pixels: `104,448`
- approximate test target pixels: `34,816`

Runtime observed for the 25-run suite:
- experiment timestamp: `2026-04-14 12:06:16Z`
- first per-variant artifact written: about `12:07:03Z`
- summary artifact written: about `12:08:02Z`
- total wall-clock runtime from experiment start to summary: about `106s`
- per-variant artifact span once training started: about `59s`

Interpretation:
- this is enough for a screening sweep
- it is not enough to claim the GPU should be working hard
- it is absolutely fair to call this a relatively small training run
- future scientists must state this explicitly so nobody mistakes a screening run for a serious full-data training pass

### Setup

Run label:
- `exp18_event_filtered_suite25`

Data:
- train files: `2026-04-14T07-00-00.parquet`, `2026-04-14T08-00-00.parquet`, `2026-04-14T09-00-00.parquet`
- test file: `2026-04-14T10-00-00.parquet`
- event filter: top 35% windows by present-time eventfulness score
- selected windows:
  - train: `51 / 144`
  - test: `17 / 48`
- older-file fallback needed: none

Suite size:
- 25 variants
- hypotheses spanned:
  - event-loss pressure
  - class positive weight
  - hidden width
  - epochs
  - learning rate / decay
  - active-pixel regression weighting
  - trade-threshold calibration

Artifacts:
- summary json: `experiments/results/exp18_event_filtered_suite25_20260414T120616Z.json`
- summary notes: `experiments/notes/exp18_event_filtered_suite25_20260414T120616Z.md`
- precheck: `experiments/pictures/exp18_event_filtered_suite25_holdout_precheck_20260414T120616Z.png`
- per-variant json/notes/pictures: `experiments/results|notes|pictures/exp18_event_filtered_suite25_*_20260414T120616Z.*`

### Ranked outcome

Top route by current route-score:
- `l1_evt005_pw2_h64`
- precision=`0.2861`
- f1=`0.3734`
- rmse=`1.14125`
- zero-baseline rmse≈`1.14359`
- fixed pnl=`-0.02344`

Other top event-quality routes:
- `precision_evt005_pw2_thr010`: precision=`0.2744`, f1=`0.3660`, rmse=`1.09526`, pnl=`0.0000`
- `l1_evt005_pw2_short_e4`: precision=`0.3059`, f1=`0.2772`, rmse=`1.14192`, pnl=`-0.02344`
- `l1_evt005_pw2`: precision=`0.2878`, f1=`0.2627`, rmse=`1.14212`, pnl=`-0.02344`
- `l1_evt005_pw2_long_e10`: precision=`0.2910`, f1=`0.2482`, rmse=`1.13212`, pnl=`-0.02344`

Best positive-PnL routes on this slice:
- `regonly_activew3`: pnl=`6.4375`, precision=`0.1272`, f1=`0.2232`, rmse ratio=`0.9237`
- `l1_evt005_pw2_h128`: pnl=`0.9453`, precision=`0.1775`, f1=`0.0793`, rmse ratio=`1.0000`
- `regonly_wd1e3`: pnl=`0.9375`, precision=`0.1881`, f1=`0.2999`, rmse ratio=`0.9812`
- `regonly_huber_thr010`: pnl=`0.9375`, precision=`0.1394`, f1=`0.2368`, rmse ratio=`0.9545`

### Interpretation

This 25-run sweep made the situation clearer.

What seems real:
- event-window conditioning consistently improves honest event metrics versus the older broad-slice strict holdouts
- many routes now stay at or below zero-baseline RMSE rather than blowing up
- several routes achieve materially better precision/F1 than the earlier broad-slice BTC holdouts

What still blocks deployment:
- the highest event-quality routes still underfire and do not monetize
- the best raw-PnL route (`regonly_activew3`) is not yet trustworthy as a trading candidate from one slice alone; it needs regime-split / neighboring-threshold confirmation
- visual check of the top-ranked route still shows partial structure alignment plus underfiring rather than full useful capture

Practical synthesis:
- there is now enough evidence that the event-filtered regime is learnable
- but the bottleneck has shifted from map reconstruction into signal extraction / execution conversion

### Decision

- keep event-window conditioning permanently in the research palette
- keep the recent-BTC event-filtered holdout as an honest benchmark regime
- do not call the top-ranked mapper tradable yet
- the most interesting follow-up candidates are now:
  - `l1_evt005_pw2_h64` for event geometry
  - `regonly_activew3` for raw PnL curiosity
  - `regonly_wd1e3` for balanced RMSE + positive PnL

### Next mutation

Stop doing large mapper sweeps for one step.
Keep the event-filtered dataset fixed and mutate only trigger extraction / execution logic, for example:
1. local-max extraction on predicted maps
2. side-aware trigger extraction
3. threshold sweep around the better balanced routes (`regonly_wd1e3`, `regonly_huber_thr010`, `l1_evt005_pw2_h64`)
4. regime-split confirmation of `regonly_activew3` before trusting its positive PnL

## 2026-04-17 - exp23_longwindow_2026_btc - honest trigger loop baseline on 2026 data

### Context

Introduces an executable sanity layer in `deep_orderbook/pipeline_guards.py` and makes every experiment script call it. Fixes a numerically dishonest PnL path in the trigger sweep: exp22 was flattening overlapping rolling windows with `reshape(-1, ...)` and feeding the resulting ghost timeline (each real timestep appearing `rolling_window_size / window_stride = 256` times, with synthetic jumps at every seam) to the strategy backtester. All exp22/exp34 PnL magnitudes up to this point should be treated as `deprecated_overlapping_pnl=True` and NOT compared directly to exp23 numbers.

Simultaneously lengthens the image geometry from `rolling=256, look_ahead=64` (25.6 s context at 100 ms cadence, 4x context/horizon ratio) to `rolling=2048, look_ahead=128` (204.8 s context, 16x ratio). The new geometry is now the default in `scripts/exp16_batch_h64_tcn.DEFAULT_SHAPER_CONFIG_2026` and is enforced at load time by `assert_image_meaningful` (`rolling * dt >= 60 s` AND `rolling >= 8 * look_ahead`).

Data source: `/mnt/data/repos/gaelreinaudi/crypto/` (fresh 2026-04 BTC-USD recordings). Training on 2026-04-15T14/15/16; walk-forward test on 2026-04-16T17.

### Result (recalibrated honest baseline, rolling=2048 / look_ahead=128, per-window PnL)

4 TCN variants, 10 trigger routes each, 40 artifacts total, `rmse_ratio` in [0.9949, 1.0049].

| Rank | Variant | Strategy | Precision | F1 | RMSE ratio | Final PnL | Trades | PnL/trade |
|---|---|---|---|---|---|---|---|---|
| best by score | regonly_huber_thr010 | q90_p2_hold48 | 0.3320 | 0.4951 | 1.0049 | 62.54 | 23 | 2.72 |
| best by raw PnL | precision_evt005_pw2_thr010 | q80_p2_hold48 | 0.3340 | 0.4977 | 0.9949 | 88.80 | 45 | 1.97 |

Full 40-row table lives in `experiments/results/exp23_longwindow_2026_btc_20260418T024330Z.json`.

### What is different vs prior event-filtered suite lab log

- RMSE ratios cluster at ~1.00 for the first time under an honest backtest. Before, exp22-era numbers were already near the zero baseline but the PnL was computed on a ghost timeline, so the low RMSE was honest while the PnL sign was potentially fake.
- Precision jumped from the 0.10-0.20 range of earlier BTC event-filtered runs to 0.33-0.39. The 16x longer context window is the most plausible cause: the TCN now sees 204.8 s of pre-event microstructure instead of 25.6 s.
- Each test-side backtest now runs on exactly 2 non-overlapping 2048-step windows (picked via `select_non_overlapping_indices(stride=8, rolling=2048, n_windows=280)`), which means PnL magnitudes here are NOT comparable in absolute units to older overlapping-stack numbers.

### What still blocks deployment

- Only 2 non-overlapping test windows per variant. A positive PnL on 2 windows is a recalibrated baseline, not yet a pattern.
- Per-file window cap of 800 (to stay under 32 GB RAM at rolling=2048). Full-hour walk-forward runs need either a higher-memory host or a streamed / float16 training accumulator (see Risks in the plan).
- No friction kill-test yet on the new honest numbers.

### Decision

- Keep the new `rolling=2048 / look_ahead=128` geometry and `pipeline_guards` in the permanent research palette.
- Treat exp23 as the new recalibrated baseline; do NOT promote any variant to tradable from it alone.
- Freeze the 4 variants and trigger grid; only vary the holdout.

### Next mutation

Run exp23 unchanged on the two neighbouring holdouts `2026-04-16T18` and `2026-04-16T19`. Rank routes by friction-adjusted `pnl_per_trade` at fixed costs 1 / 2 / 5 / 10 units per trade. Only if the same variant wins on BOTH neighbours after friction do we accept it as a real candidate and go back to varying mapper knobs.

## 2026-04-18 - post-exp23 analysis and queued mutations exp24..exp28

### Analysis performed (not a new training run)

Deep analysis of the exp23 artifacts (`experiments/results/exp23_longwindow_2026_btc_20260418T024330Z.json` + 40 per-route children, the `_fixed_*.png` dashboards for the top two routes, and the consolidated overview `experiments/pictures/exp23_deepanalysis_overview_20260418T024330Z.png`).

### Findings

- All 4 mapper variants overfit from epoch 1: training loss monotonically down, test loss monotonically up across 6 epochs. `rmse_ratio` clusters at 0.995-1.005 (at or WORSE than the zero-baseline on the test set).
- Eyeball verdict on the prediction heatmap: the model learns `where` intensity lives on the price axis, but NOT `when`. Predictions are a low-frequency smear of a visibly structured target map.
- Order-book pathology: `Books` panels show mass only at rows 0 and 15 (the far-edge bins); the inner 14 rows are empty. Most liquidity lives within +-1 bp and is being aliased into the edges by `view_bips=5 / num_side_lvl=8`.
- Friction kill-test on the exp23 top-6 routes: all cross zero between cost=1 and cost=2 per trade; all negative by cost=3; all below -100 at cost=10.
- Conclusion: exp23 did NOT find a tradable pattern. It established an honest recalibrated baseline and surfaced four candidate blockers.

### Queued next mutations (single-factor, in execution order)

1. `exp24_smallnet_earlystop` — TrainConfig only: epochs 6->2, hidden 64->32, dropout 0->0.2, weight_decay->1e-3. Directly targets the overfit.
2. `exp25_timing_500ms` — ReplayConfig cadence 100ms -> 500ms with compensated rolling=512 / la=32. Tests whether 100 ms is oversampled.
3. `exp26_horizon_la32` — ShaperConfig look_ahead 128 -> 32 (+`allow_microburst=True`). Tests whether the model can learn `when` at short horizon.
4. `exp27_binning_fine` — ShaperConfig view_bips 5->2 and num_side_lvl 8->16. Tests whether book-edge aliasing was the real bottleneck.
5. `exp28_loss_structured` — TrainConfig criterion Huber -> StructuredT2L with rank+monotonic terms. Tests whether the map is a rank problem rather than a magnitude one. This one bakes in the neighbour-holdout promotion step.

Each has explicit hypothesis, pass-criteria, kill-criteria, required artifacts, and cost estimate in `experiments/notes/exp24_exp28_queue_postexp23_20260418T030000Z.md`. Each obeys the SKILL's locked-in guards: `pipeline_guards`, per-window PnL, non-overlapping dashboards, friction kill-test, direction reported, both best-by-score and best-by-raw-PnL reported, richness-gate first on the primary test file.

### Decision for the queue

Execute exp24 first and block until done. Inherit its training config into exp25-exp28 only if exp24 passes. If all five kill, tag `queued_exit_branch: architectural` and stop sweeping the TCN with one-factor mutations; next scientist should change the architecture (second-stage trigger head, attention trunk, or per-level conv) rather than keep running loss/geometry tweaks on an overfit model.

