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
