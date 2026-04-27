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

### Practical conclusion from the 2026-04-22 exp46/exp47 pair
- The recent 24-file stress-conditioned short branch is still the current best practical candidate, but only as a **signal-quality branch**, not a settled execution recipe.
- exp46 (`2026-04-21T22-00-00` holdout) strengthened the branch by showing that the winning execution family could shift from persistence-heavy to faster `q90_p1_hold48` while remaining positive even under heavy friction.
- exp47 (`2026-04-21T23-00-00` holdout) then falsified the stronger execution claim one hour later:
  - dense/map quality actually improved a bit (`precision≈0.4829`, `f1≈0.6023`, `rmse_ratio≈0.9660`),
  - but `q95_*` fast routes produced **zero trades**,
  - and relaxing back toward `q90_p1_hold48` yielded a single losing trade (`-60.30`).
- Interpretation: the mapper still looks honest enough to keep, but the strategy layer is more fragile than exp46 alone suggested. This is no longer a simple “fast family works” story; it is a regime-within-regime execution fragility story.
- Scientist consequence:
  - do **not** retrain the mapper yet,
  - do **not** promote the current fast execution branch,
  - next mutation should broaden execution falsification across multiple later evening holdouts and explicitly separate (a) no-trade over-selectivity from (b) bad-fill / one-trade failure.

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

## 2026-04-18 - exp24_smallnet_earlystop - smaller net / early stop follow-on

### Context

This was the first queued post-exp23 mutation from `experiments/notes/exp24_exp28_queue_postexp23_20260418T030000Z.md`, run on the same honest 2026 BTC walk-forward slice after implementing the missing wrapper `scripts/exp24_smallnet_earlystop.py` and the minimal code support it needed:
- new h32 variants in `deep_orderbook.btc_search_lab`
- `TinyTCN(..., trunk_dropout=...)` support in `scripts/exp16_batch_h64_tcn.py`
- `scripts/exp22_trigger_sweep.py` passing `cfg.get("trunk_dropout", 0.0)` into the model

Sanity check before implementation was done with failing tests first, then green:
- `tests/test_btc_search_lab.py`
- `tests/test_exp24_smallnet.py`
- command: `pytest tests/test_btc_search_lab.py tests/test_exp24_smallnet.py -q`

### Exact mutation

Compared with exp23 baseline:
- hidden dim: 64/96 family -> 32 for all queued exp24 variants
- epochs: 6 -> 2
- trunk dropout: 0.0 -> 0.2
- weight decay: -> 1e-3

Variants run:
- `l1_evt005_pw2_h32`
- `precision_evt005_pw2_thr010_h32`
- `regonly_wd1e3_h32`
- `regonly_huber_thr010_h32`

Everything else stayed fixed:
- same train/test files (`2026-04-15T14/15/16` train, `2026-04-16T17` test)
- same honest long-window geometry (`rolling=2048`, `look_ahead=128`, `stride=8`)
- same event filter top_fraction=0.35
- same long-only trigger grid
- same per-window PnL aggregation and non-overlapping dashboard slice selection

### Artifact-backed results

Summary JSON:
- `experiments/results/exp24_smallnet_earlystop_20260418T040625Z.json`

Summary note:
- `experiments/notes/exp24_smallnet_earlystop_20260418T040625Z.md`

Representative dashboard:
- `experiments/pictures/exp24_smallnet_earlystop_precision_evt005_pw2_thr010_h32__long__q80_persistence2_wide_margin_fixed_20260418T040625Z.png`

Data scale:
- train windows before/after filter: 2400 -> 840
- test windows before/after filter: 800 -> 280
- rolling window: 2048
- train timesteps: 1,720,320
- test timesteps: 573,440
- runtime: 710.359s on cuda

Best by score:
- `precision_evt005_pw2_thr010_h32 / long / q80_persistence2_wide_margin`
- precision = 0.3277
- f1 = 0.4932
- rmse_ratio = 0.9468
- pnl = 89.125
- trades = 24
- pnl_per_trade = 3.71

Best by raw PnL:
- `precision_evt005_pw2_thr010_h32 / long / q80_p1_hold48`
- pnl = 122.4766
- trades = 76
- pnl_per_trade = 1.61

Variant-level mapper summary:
- `l1_evt005_pw2_h32`: precision 0.3300, f1 0.4898, rmse_ratio 0.9818
- `precision_evt005_pw2_thr010_h32`: precision 0.3277, f1 0.4932, rmse_ratio 0.9468
- `regonly_wd1e3_h32`: precision 0.3277, f1 0.4933, rmse_ratio 0.9496
- `regonly_huber_thr010_h32`: precision 0.3276, f1 0.4931, rmse_ratio 0.9501

### Visual / scientific read

The fixed dashboard is scientist-usable and no longer suffers from the old ghost-timeline discontinuities. The prediction map is still somewhat diffuse, but visually meaningful, with rough timing alignment to truth. The main structural change is that the dense-map error improved materially; all four h32 variants landed well below the exp23 ~1.00 RMSE-ratio cluster.

### Friction sanity

Top score route:
- +65.1 at cost=1/trade
- +41.1 at cost=2/trade
- -30.9 at cost=5/trade
- -150.9 at cost=10/trade

So the route still dies under moderate friction. This is not tradable yet.

### Decision

Call exp24 a pass as a recalibrated baseline improvement, not a trading success.

What improved:
- dense-map honesty improved a lot (`rmse_ratio` 0.9468-0.9818 vs exp23 ~0.995-1.005)
- raw honest PnL improved
- score-best churn remained more reasonable than the raw-PnL winner

What did not improve enough:
- precision/F1 did not clearly beat exp23
- the route still fails a realistic friction kill-test
- the original queue hypothesis about fully curing overfit is only partially supported

### Next mutation

Proceed to `exp25_timing_500ms` next.
Use the smaller-net training config as the inherited baseline unless we explicitly want to isolate cadence from exp23 instead.

## EXP-25 — 500 ms cadence with compensated geometry

Status: DONE
Date: 2026-04-18

### Purpose

Test whether 100 ms is simply oversampled for this branch. If adjacent frames are too redundant, moving to 500 ms should reduce memorization pressure without throwing away useful signal.

### What changed vs exp24

- `ReplayConfig.every: 100ms -> 500ms`
- `ShaperConfig.rolling_window_size: 2048 -> 512`
- `ShaperConfig.look_ahead: 128 -> 32`
- `ShaperConfig.window_stride: 8 -> 2`
- kept the exp24 smaller-net mapper family:
  - `l1_evt005_pw2_h32`
  - `precision_evt005_pw2_thr010_h32`
  - `regonly_wd1e3_h32`
  - `regonly_huber_thr010_h32`

Implementation work needed:
- added `scripts/exp25_timing_500ms.py`
- extended `scripts/exp22_trigger_sweep.py` so the honest trigger-sweep path accepts cadence/shaper overrides
- added tests in `tests/test_exp25_timing_500ms.py`
- verification run:
  - `pytest tests/test_btc_search_lab.py tests/test_exp24_smallnet.py tests/test_exp25_timing_500ms.py tests/test_continuity.py -q`
  - `24 passed`

### Data scale / runtime

- train files: 3
- test files: 1
- windows before filter: train `2400`, test `800`
- windows after filter: train `840`, test `280`
- rolling window: `512`
- look ahead: `32`
- train timesteps: `430,080`
- test timesteps: `143,360`
- total runtime: `276.434s`
- device: `cuda`
- cadence in artifact: `500ms`
- `dt_seconds_from_every`: `0.5`

Artifacts:
- JSON: `experiments/results/exp25_timing_500ms_20260418T155751Z.json`
- Notes: `experiments/notes/exp25_timing_500ms_20260418T155751Z.md`
- Representative fixed dashboard:
  - `experiments/pictures/exp25_timing_500ms_regonly_wd1e3_h32__long__q90_persistence2_fast_exit_fixed_20260418T155751Z.png`

### Main result

Best route by score:
- `regonly_wd1e3_h32 / long / q90_persistence2_fast_exit`
- pnl `82.3672`
- trades `10`
- pnl/trade `8.2367`
- precision `0.3187`
- f1 `0.4812`
- rmse_ratio `0.9806`

Best route by raw pnl:
- `regonly_wd1e3_h32 / long / q80_p1_hold48`
- pnl `82.9219`
- trades `25`
- pnl/trade `3.3169`
- precision `0.3187`
- f1 `0.4812`
- rmse_ratio `0.9806`

Friction sanity for the score-winner:
- cost 1/trade -> `72.3672`
- cost 2/trade -> `62.3672`
- cost 5/trade -> `32.3672`
- cost 10/trade -> `-17.6328`
- therefore cost 3/trade is still positive (`52.3672`)

Variant-level dense-map honesty:
- `l1_evt005_pw2_h32`: `rmse_ratio=0.9713`
- `precision_evt005_pw2_thr010_h32`: `rmse_ratio=0.9848`
- `regonly_wd1e3_h32`: `rmse_ratio=0.9806`
- `regonly_huber_thr010_h32`: `rmse_ratio=0.9646`

### What the run says

Good news:
- cadence coarsening did not kill signal quality
- precision stayed above `0.30`
- multiple variants stayed below `rmse_ratio < 0.99`
- at least one route survives a realistic intermediate friction check on the primary holdout

Bad news:
- the core overfit hypothesis did not actually clear
- test loss still increased on every variant, e.g. `regonly_wd1e3_h32`:
  - train: `2.4943 -> 2.2395`
  - test: `2.5544 -> 2.8206`
- so 500 ms cadence improved trade behavior on this holdout but did not fix the train/test generalization drift

### Decision

Call exp25 `mixed_not_promoted`.

This is not a dead branch, but it is also not a pass on the queue's main criterion. The result is strategy-positive and dense-map-honest, but cadence alone did not solve overfit.

### Next mutation

Proceed to `exp26_horizon_la32` next.
The next thing to test is whether shortening the horizon / target geometry improves timing/localization enough to reduce the remaining train/test drift.


---

## exp30_2d_model_ms1000_slow — 2026-04-19 UTC

**Single mutation**: model architecture 1D TinyTCN → 2D OrderBook2DTCN (hid=8, RF=2047 ticks).  
Geometry unchanged from exp29_ms1000_slow: every=1000ms, rolling=1024, look_ahead=64.

**Key result**:
- rmse_ratio: 0.980 (1D) → **0.844** (2D) — first time substantially beating zero baseline
- Best route: `regonly_l1_2d_h8 / long / q80_p3_hold48` pnl=+127.22, 3 trades, survives cost=10/trade
- L1 loss > Huber for 2D map geometry (sparsity-promoting matches the sparse t2l structure)
- **ALL 30 routes positive PnL** — unprecedented

**Decision**: very_promising, needs neighbor holdout (only 3 trades on primary test file).

### Next mutation

Validate `regonly_l1_2d_h8` mapper on adjacent holdout files (`2026-04-16T18`, `2026-04-16T19`).
If both pass at cost=3/trade, also try `hid=16` for more 2D capacity.

---

## exp38 / exp39 evening short-side continuation — 2026-04-20 UTC

Simple idea:
- stop retraining blindly
- keep the honest 2D mapper (`regonly_l1_2d_h8`)
- use several uncapped full evening files
- test whether the branch is directionally inverted / evening-liquidity biased by flipping execution to short

This was a market-structure hypothesis, not a metric trick:
- thinner evening liquidity may produce cleaner downside reactions
- the mapper may already be learning bearish pressure / imbalance structure
- long-only extraction may have been suppressing the signal

### exp38 — holdout20 short probe
- train files: 9 uncapped evening files across `2026-04-14` to `2026-04-16T19`
- test holdout: `2026-04-16T20-00-10.parquet`
- train windows: `4208 -> 1473` after event filter
- test windows: `249 -> 88`
- best route by score/raw pnl: `regonly_l1_2d_h8 / short / q95_p2_hold96`
- pnl `+41.13`, trades `2`, pnl/trade `20.57`
- precision `0.5365`, f1 `0.6565`, rmse_ratio `0.9471`
- friction table: `+39.13 @1`, `+37.13 @2`, `+31.13 @5`, `+21.13 @10`
- key comparison: prior long-side same-regime candidate was dead (`0 trades`, `0 pnl`)
- decision: promising adjacent-holdout continuation, but still fragile because only 2 trades

Artifacts:
- `experiments/results/exp38_evening_regime_fullfiles_l1_2d_h8_holdout20_shortprobe_20260420T113832Z.json`
- `experiments/notes/exp38_evening_regime_fullfiles_l1_2d_h8_holdout20_shortprobe_20260420T113832Z.md`

### exp39 — immediate adjacent holdout21 short probe
- same frozen mapper and same several-day evening regime framing
- test holdout: `2026-04-16T21-00-10.parquet`
- best route by score/raw pnl: `regonly_l1_2d_h8 / short / q80_p2_hold48`
- pnl `+247.32`, trades `26`, pnl/trade `9.51`
- precision `0.5616`, f1 `0.6652`, rmse_ratio `0.9338`
- friction table: `+221.32 @1`, `+195.32 @2`, `+117.32 @5`, `-12.68 @10`
- this is materially stronger than exp38 and survives meaningful friction

Artifacts:
- `experiments/results/exp39_evening_regime_fullfiles_l1_2d_h8_holdout21_shortprobe_20260420T114321Z.json`
- `experiments/notes/exp39_evening_regime_fullfiles_l1_2d_h8_holdout21_shortprobe_20260420T114321Z.md`

### What this says
Good:
- screenshots are from holdout test data, not train data
- several-day uncapped evening training + short-side execution is the first branch that turned the honest 2D mapper into a live-looking candidate
- rmse stayed honest on both holdouts
- adjacent holdout behavior improved rather than collapsing immediately

Bad / caution:
- visual map fit is still coarse, not tightly event-aligned
- train loss vs test loss still suggests overfit risk
- two holdouts is encouraging, not enough for promotion

### Decision
Call this branch `promising_short_evening_candidate_not_promoted`.
It is the best current idea/trick: freeze the mapper, exploit directional inversion in thinner evening liquidity, and validate on adjacent holdouts before touching architecture again.

### Exact next mutation
Do NOT retrain a new architecture yet.
Keep `regonly_l1_2d_h8` frozen, keep the evening-regime slice, keep short-only execution, and run the next adjacent holdouts:
- `2026-04-16T22-00-00.parquet`
- `2026-04-16T23-00-02.parquet`
using the same friction table (`1/2/5/10`).
Only if those fail do we change the extraction/training recipe.

---

## exp40 / exp41 adjacent evening short-side confirmation — 2026-04-20 UTC

### Idea iterated
Do not touch architecture yet.
Pressure-test the same frozen short evening branch on the next adjacent holdouts before claiming anything.

### exp40 — holdout22
- test holdout: `2026-04-16T22-00-00.parquet`
- train files: expanded uncapped evening train set through `2026-04-16T21`
- best route by score: `regonly_l1_2d_h8 / short / q90_p2_hold96`
- pnl `+144.40`, trades `7`, pnl/trade `20.63`
- best raw pnl: `q80_p3_hold48` -> `+161.68`
- precision `0.5459`, f1 `0.5860`, rmse_ratio `0.9962`
- friction for score winner: `+137.40 @1`, `+130.40 @2`, `+109.40 @5`, `+74.40 @10`
- verdict: still alive and robust. Metrics softened vs holdout21, but the branch clearly did not collapse.

Artifacts:
- `experiments/results/exp40_evening_regime_fullfiles_l1_2d_h8_holdout22_shortprobe_20260420T182439Z.json`
- `experiments/notes/exp40_evening_regime_fullfiles_l1_2d_h8_holdout22_shortprobe_20260420T182439Z.md`

### exp41 — holdout23
- initial GPU run hit `torch.OutOfMemoryError` after expanding the uncapped train span again
- correct response was NOT to reintroduce hidden data caps; reran on CPU with the same uncapped train/test setup
- test holdout: `2026-04-16T23-00-02.parquet`
- best route by score/raw pnl: `regonly_l1_2d_h8 / short / q90_p2_hold96`
- pnl `+45.07`, trades `6`, pnl/trade `7.51`
- precision `0.4535`, f1 `0.5289`, rmse_ratio `0.9748`
- friction for score winner: `+39.07 @1`, `+33.07 @2`, `+15.07 @5`, `-14.93 @10`
- verdict: weaker, but still positive and still alive at cost 5. The branch degraded, but it did not snap.

Artifacts:
- `experiments/results/exp41_evening_regime_fullfiles_l1_2d_h8_holdout23_shortprobe_cpu_20260420T183059Z.json`
- `experiments/notes/exp41_evening_regime_fullfiles_l1_2d_h8_holdout23_shortprobe_cpu_20260420T183059Z.md`

### What this new confirmation pass says
Good:
- the short evening branch is now positive on four adjacent holdouts (`T20`, `T21`, `T22`, `T23`)
- it stayed honest on RMSE throughout (`rmse_ratio` stayed below 1.0 on all four)
- it survived friction at cost 5 on all four holdouts
- the branch is therefore more than a cherry-picked screenshot

Caution:
- strength is uneven: `T21` and `T22` are much stronger than `T20` and especially `T23`
- precision/f1 drifted down on `T23`
- the branch still looks regime-sensitive rather than universally strong
- growing the uncapped train span can exceed 8 GB VRAM, so future confirmation runs may need CPU or a lower-memory execution path

### Decision
Upgrade the branch from `promising_short_evening_candidate_not_promoted` to `promising_regime_specific_short_candidate`.
Still not a universal signal. But now it has enough adjacent-holdout support to justify one new idea/trick before any architecture change.

### Exact next improvement / idea / trick
Freeze the mapper again.
Do NOT retrain a new model yet.
The next real mutation should be extraction-level and regime-aware:
- keep `regonly_l1_2d_h8`
- keep evening-only framing
- keep short direction
- narrow the trigger family around the winner `q90_p2_hold96`
- add one single-factor execution mutation: longer persistence / hold logic around downside continuation versus fast re-entry churn

Reason:
- across `T22` and `T23`, `q90_p2_hold96` was the most stable score-winner
- the branch already survives friction 5, so the smartest next test is to improve stability/churn, not change the map

---

## exp42 many-day April evening expansion — 2026-04-20 UTC

You were right to push on data scale.
This run stopped babying the branch and used much more April 2026 history.

### Setup
- train files: 20 uncapped evening files across `2026-04-14` through `2026-04-18`, hours `20:00` to `23:00`
- test holdout: `2026-04-19T20-00-00.parquet`
- mapper unchanged: `regonly_l1_2d_h8`
- direction unchanged: `short`
- no hidden sample caps
- CPU run (to preserve full span instead of shrinking data after GPU/VRAM issues)

### Data scale
- train windows: `11265 -> 3943` after event filter
- test windows: `643 -> 226`
- train timesteps: `4,037,632`
- test timesteps: `231,424`
- runtime: `1091.09s`

### Result
Best route:
- `regonly_l1_2d_h8 / short / q80_p1_hold48`
- pnl `+8.57`
- trades `1`
- precision `0.2113`
- f1 `0.0001`
- rmse_ratio `1.0000`

Friction:
- `+7.57 @1`
- `+6.57 @2`
- `+3.57 @5`
- `-1.43 @10`

### What this means
This broader April expansion mostly killed the branch.
The earlier evening short effect was real enough to show up on the local neighboring holdouts, but it does NOT look stable when scaled to much more April evening history in the same recipe.

So the correct conclusion is not “use less data.”
It is:
- more data exposed that the current evening-only short recipe is too local / brittle
- the branch should be demoted
- the next idea must change framing, not just add even more of the same regime slice

### Decision
Demote the branch from `promising_regime_specific_short_candidate` to `interesting_local_regime_effect_not_stable_under_broad_april_expansion`.

### Exact next mutation
Use much more April data again, but stop restricting to evening-only.
Next run should broaden the framing itself:
- all-hours or much wider-hours April train set
- April holdout test
- same frozen mapper first
- only then decide whether the mapper has any structural short edge left once regime overfitting is removed

---

## exp44 broad April stress-filter replay — 2026-04-22 UTC

This run executed the queued stress-filter follow-up without changing the mapper family.
Instead of slicing by clock hour, it trained on a broader April pool spanning midnight, morning, midday, and evening, then kept only present-time high-stress windows using the existing eventfulness score (abs return + range + realized vol + book std + book impulse).

### Setup
- train files: 20 uncapped April 2026 files across `2026-04-14` through `2026-04-18`
- intraday phases per day: roughly `00:00`, `06:00`, `12/13:00`, `18:00`
- test holdout: `2026-04-19T20-00-00.parquet`
- mapper family unchanged: `regonly_l1_2d_h8`
- direction unchanged: `short`
- cadence / geometry unchanged: `1000ms`, `rolling=1024`, `look_ahead=64`
- filter: top 35% by present-time stress/eventfulness, no hidden per-file caps
- CPU run

### Data scale
- 2026 coverage: all 21 files were April 2026 (`20` train + `1` holdout), no older backfill needed
- exact train days used: `2026-04-14` through `2026-04-18`
- exact holdout day used: `2026-04-19`
- total file count: `21`
- train windows: `12398 -> 4340` after stress filter
- test windows: `643 -> 226` after stress filter
- train timesteps: `4,444,160`
- test timesteps: `231,424`
- capped vs uncapped: uncapped
- run class: serious training / evaluation, not screening
- runtime: `336.05s`
- device: `cpu`

### Result
Best route:
- `regonly_l1_2d_h8 / short / q80_p1_hold48`
- pnl `+143.38`
- trades `10`
- pnl/trade `14.34`
- precision `0.5041`
- recall `0.7169`
- f1 `0.5920`
- rmse_ratio `0.9679`
- zero-baseline rmse `7.47357`
- model rmse `7.23399`

Friction:
- `+133.38 @1`
- `+123.38 @2`
- `+93.38 @5`
- `+43.38 @10`

### What this means
This is the strongest result since the local evening branch and it matters because the edge reappeared only after switching from hour-of-day slicing to observable stress conditioning.
That is a much more believable mechanism: short continuation appears when the book is already stressed, not merely because the clock says “evening”.

Still, do not overclaim.
This is one holdout, and the candidate still needs confirmation on later chronological April files before promotion. But unlike exp42/exp43, this branch has real recall, real trade count, positive dense metrics, and meaningful friction survival.

### Decision
`promising`, but not yet promoted to robust strategy.

### Exact next mutation
Keep the mapper frozen again. Do not do architecture churn.
Advance to the next later April holdouts (`2026-04-19T21-00-00.parquet`, then `2026-04-19T22-00-00.parquet`) with the SAME broad train pool + stress filter + short execution to see whether this survives adjacent chronology rather than just one evening file.

---

## exp43 overnight 4-phase April falsification — 2026-04-22 UTC

This executed the exact broad-framing follow-up from exp42 without changing the mapper.
The question was simple: was the short branch structural, or was it just hiding inside one cozy hour-of-day pocket?

### Setup
- train files: 20 uncapped April 2026 files across `2026-04-14` through `2026-04-18`, hours `00:00` to `03:00`
- test holdout: `2026-04-19T00-00-00.parquet`
- mapper unchanged: `regonly_l1_2d_h8`
- direction unchanged: `short`
- cadence / geometry unchanged: `1000ms`, `rolling=1024`, `look_ahead=64`
- event filter unchanged: top 35% of windows by present-time eventfulness, no hidden per-file caps
- CPU run

### Data scale
- 2026 coverage: all 21 files were April 2026 (`20` train + `1` holdout), no older backfill needed
- exact train days used: `2026-04-14`, `2026-04-15`, `2026-04-16`, `2026-04-17`, `2026-04-18`
- exact holdout day used: `2026-04-19`
- total file count: `21`
- train windows: `12821 -> 4488` after event filter
- test windows: `643 -> 226`
- train timesteps: `4,595,712`
- test timesteps: `231,424`
- capped vs uncapped: uncapped
- run class: serious training / evaluation, not screening
- runtime: `494.56s`
- device: `cpu`

### Result
Best route:
- `regonly_l1_2d_h8 / short / q80_p1_hold48`
- pnl `-0.00781`
- trades `1`
- pnl/trade `-0.00781`
- precision `0.0867`
- recall `0.0000238`
- f1 `0.0000476`
- rmse_ratio `1.0000`
- zero-baseline rmse `4.82557`
- model rmse `4.82557`

Friction:
- `-1.00781 @1`
- `-2.00781 @2`
- `-5.00781 @5`
- `-10.00781 @10`

### What this means
This overnight 4-phase falsification is dead.
The mapper stayed honest on dense error, but the strategy layer was effectively absent: one trade, near-zero recall, and negative pnl before friction.

That is useful information.
It means the earlier short-side branch was not a general April short edge that simply needed broader time coverage.
The signal appears tied to a narrower local regime, most plausibly thin-liquidity evening downside continuation, not to a robust all-hours / wider-hours short map.

### Decision
`not promising yet`.
Really: this specific broadened-framing branch is close to dead.
Do not promote it.

### Exact next mutation
Do not do architecture churn.
Do not keep sweeping hour-of-day slices blindly.
The next grounded mutation should keep April breadth and the frozen mapper, but change the filter from clock time to observable stress:
- broad April train pool
- present-time realized-vol / intrawindow range / book-impulse filter
- short-side replay on a later April holdout

That tests a mechanism a trader would actually say out loud:
short continuation may exist only when the book is stressed, not merely because it is a certain hour.
If that still gives near-zero recall and dead pnl, stop treating this mapper as a viable short-strategy branch.

---

## exp45 adjacent-holdout confirmation of the broad April stress short branch — 2026-04-22 UTC

This executed the exact queued confirmation after exp44.
No mapper retrain, no architecture churn, no new regime slicing trick.
The only mutation was chronological confirmation: run the same frozen broad-April stress-conditioned short replay on the next adjacent holdouts.

### Setup
- train files: same 20 uncapped April 2026 files used in exp44, spanning `2026-04-14` through `2026-04-18` across roughly `00:00`, `06:00`, `12/13:00`, and `18:00`
- holdouts: `2026-04-19T21-00-00.parquet` and `2026-04-19T22-00-00.parquet`
- mapper unchanged: `regonly_l1_2d_h8`
- direction unchanged: `short`
- cadence / geometry unchanged: `1000ms`, `rolling=1024`, `look_ahead=64`
- event filter unchanged: top 35% of windows by present-time stress score, uncapped loading
- CPU run

### Data scale
- 2026 coverage: all files were 2026; no older backfill needed
- exact train days used: `2026-04-14`, `2026-04-15`, `2026-04-16`, `2026-04-17`, `2026-04-18`
- exact holdout day used: `2026-04-19`
- total file count per serious run: `21` (`20` train + `1` holdout)
- train windows: `12398 -> 4340` after stress filter
- each holdout windows: `644 -> 226`
- train timesteps: `4,444,160`
- each holdout timesteps: `231,424`
- capped vs uncapped: uncapped
- run class: serious training / evaluation, not screening
- runtimes: `408.968s` for `21:00`, `438.006s` for `22:00`
- device: `cpu`

### Results
Holdout `21:00` best route:
- `regonly_l1_2d_h8 / short / q80_p1_hold48`
- pnl `+93.57`
- trades `11`
- pnl/trade `8.51`
- precision `0.5179`
- recall `0.7089`
- f1 `0.5985`
- rmse_ratio `0.9577`
- friction: `+82.57 @1`, `+71.57 @2`, `+38.57 @5`, `-16.43 @10`

Holdout `22:00` best route:
- `regonly_l1_2d_h8 / short / q90_p2_hold96`
- pnl `+95.41`
- trades `7`
- pnl/trade `13.63`
- precision `0.6089`
- recall `0.7974`
- f1 `0.6905`
- rmse_ratio `0.9528`
- friction: `+88.41 @1`, `+81.41 @2`, `+60.41 @5`, `+25.41 @10`

### What this means
This is the first honest adjacent-holdout confirmation that the exp44 branch was not just one lucky file.
The same frozen mapper survived the next two chronological evening holdouts with positive short PnL and honest dense-map quality.

That matters scientifically.
It upgrades the interpretation from `interesting one-holdout stress replay` to `credible regime-specific short branch`.
The mechanism story is now more believable: downside continuation is being captured when the book is already stressed, and that effect persists across multiple neighboring holdouts.

But there is still a real caution line.
This is not yet a robust all-regime strategy. Execution quality is clearly weaker than signal quality:
- the `21:00` holdout turns negative at `10` ticks friction,
- the `22:00` holdout still contains overtrading trigger families that lose despite the same underlying mapper metrics.

So the next bottleneck is not obviously representation learning. It is extraction / execution discipline.

### Decision
`promising`.
Not ready for live deployment, but clearly stronger than exp44 alone and strong enough to justify another frozen-mapper confirmation/broadening cycle before any retrain.

### Exact next mutation
Keep the mapper frozen again.
Do not do architecture churn yet.
Broaden the same stress-conditioned regime across more 2026 dates and test a later chronological evening holdout beyond `2026-04-19`, while lightly mutating execution around the surviving trigger family (`q80_p1_hold48` vs `q90_p2_hold96`) to reduce friction fragility.

## 2026-04-22 exp46 recent-24 later-holdout scale check

Status: DONE

### Why this run existed
The exp45 adjacent-holdout confirmation upgraded the stress-conditioned short branch from a curiosity to a credible regime-specific candidate.
The right next scientific question was not "can we make the mapper fancier?" but "does the same frozen mapper survive a later chronological evening holdout when trained on materially more recent 2026 data?"

### Mutation from prior run
- Kept the mapper frozen: `regonly_l1_2d_h8`
- Kept direction short-only
- Kept cadence / geometry fixed: `1000ms`, `rolling=1024`, `look_ahead=64`
- Kept the same present-time stress filter (top 35% by observable stress)
- Changed one practical factor only: broadened the train pool from 20 files (`2026-04-14`..`2026-04-18`) to 24 more recent files (`2026-04-16`..`2026-04-21`) and advanced the holdout to `2026-04-21T22-00-00`

Practical execution note:
- The originally planned 32-file uncapped scale-up was too large for this machine and died before artifact creation.
- Rather than fake success or silently downshift to a tiny cap, the completed run used the largest uncapped recent-2026 slice that finished honestly on CPU.

### Hypothesis
If the branch is structurally stress-conditioned and not just an April 19 pocket, then a later evening holdout on a broader recent-2026 train pool should still yield positive short PnL with `rmse_ratio < 1`, even if the best extraction policy shifts away from the prior persistence-heavy winner.

### Artifacts
- Summary JSON: `experiments/results/exp46_recent24_stress_short_scale_check_20260421T220000_20260422T171817Z.json`
- Best-route JSON: `experiments/results/exp46_recent24_stress_short_scale_check_20260421T220000_regonly_l1_2d_h8__short__q90_p1_hold48_20260422T171817Z.json`
- Summary note: `experiments/notes/exp46_recent24_stress_short_scale_check_20260421T220000_20260422T171817Z.md`
- Aligned continuity image: `experiments/pictures/exp46_recent24_stress_short_scale_check_20260421T220000_regonly_l1_2d_h8__short_slide_20260422T171817Z.png`
- Diagnostic fixed image: `experiments/pictures/exp46_recent24_stress_short_scale_check_20260421T220000_regonly_l1_2d_h8__short__q90_p1_hold48_fixed_20260422T171817Z.png`
- Diagnostic best image: `experiments/pictures/exp46_recent24_stress_short_scale_check_20260421T220000_regonly_l1_2d_h8__short__q90_p1_hold48_best_20260422T171817Z.png`
- DB rows: `experiment_runs.id = 782..791`

### Data scale
- 2026 coverage used: 24 train files from `2026-04-16T00-00-00` through `2026-04-21T18-00-10`; holdout `2026-04-21T22-00-00`
- Total file count: `25` (`24` train + `1` holdout), all from 2026
- Capped vs uncapped: uncapped (`max_windows_per_file=null`)
- Train windows: `14957 -> 5235` after stress filter
- Test windows: `642 -> 225` after stress filter
- Train timesteps: `5,360,640`
- Test timesteps: `230,400`
- Runtime: `480.998s` total wall-clock (`372.118s` for the best-route artifact)
- Device: `cpu`

### Results
Best route by score and raw PnL:
- `regonly_l1_2d_h8 / short / q90_p1_hold48`
- pnl `+138.27`
- trades `6`
- pnl/trade `23.05`
- precision `0.4588`
- recall `0.7418`
- f1 `0.5670`
- rmse_ratio `0.9563`
- friction: `+132.27 @1`, `+126.27 @2`, `+108.27 @5`, `+78.27 @10`

Execution sweep readout:
- Fast routes dominated: `q90_p1_hold48`, `q80_p2_hold48`, and `q80_p3_hold48` all stayed strongly positive.
- The old persistence-heavy family degraded badly here: `q90_p2_hold96` and neighbors turned negative despite the same mapper metrics.
- That means the representation may still be valid, but regime-specific execution extraction is changing over time.

### Interpretation
This is a meaningful confirmation, not a victory lap.

What held up:
- The branch survived a later chronological evening holdout.
- Dense-map quality stayed honest (`rmse_ratio < 1`).
- Best-route PnL exceeded the earlier Apr 19 22:00 confirmation in raw dollars and stayed positive even at `10` ticks friction.

What weakened:
- Precision and F1 were lower than exp45's strongest Apr 19 22:00 route.
- The previously strongest persistence-heavy trigger family failed here.

Scientific read:
- The mapper still appears to contain a real stress-conditioned downside signal.
- The bottleneck is increasingly extraction policy, not obviously representation learning.
- The surviving edge now looks more like a faster downside continuation / liquidation follow-through branch than a slower persistence-hold branch.

### Decision
`promising`.
Still regime-specific, still not live-ready, but the branch is now harder to dismiss as a local April 19 artifact.

### Exact next mutation
Keep the mapper frozen.
Do not retrain yet.
Run the same recent stress-conditioned framing on the next chronological evening holdout (`2026-04-21T23-00-00` or the next available comparable evening holdout) and mutate only fast execution around the new winner:
- `q90_p1_hold48`
- nearby higher-threshold fast-entry variants
- slightly wider side margins / fast exits

---

## exp48 post-23:00 multi-holdout execution falsification — 2026-04-22 UTC

This executed the exact queued follow-up from exp47 without changing the mapper.
The question was no longer whether the map was honest — exp47 already suggested it was.
The question was whether the execution failure at `2026-04-21T23:00:00` meant the branch was dead, or whether execution selectivity had simply become hour-sensitive inside the same stress regime.

### Setup
- train files: same 24 uncapped recent-2026 files used in exp46/exp47, spanning `2026-04-16` through `2026-04-21` across `00:00`, `06:00`, `12:00`, and `18:00`
- holdouts: `2026-04-22T00-00-00.parquet` and `2026-04-22T01-00-00.parquet`
- mapper unchanged: `regonly_l1_2d_h8`
- direction unchanged: `short`
- cadence / geometry unchanged: `1000ms`, `rolling=1024`, `look_ahead=64`
- event filter unchanged: top 35% of windows by present-time observable stress, uncapped loading
- CPU run

### Data scale
- 2026 coverage: all files were 2026; no older backfill needed
- exact train days used: `2026-04-16`, `2026-04-17`, `2026-04-18`, `2026-04-19`, `2026-04-20`, `2026-04-21`
- exact holdout day used: `2026-04-22`
- total unique file count across the mutation: `26` (`24` train + `2` holdouts)
- per serious run file count: `25` (`24` train + `1` holdout)
- train windows per run: `14957 -> 5235` after stress filter
- test windows: `641 -> 225` for `00:00`; `618 -> 217` for `01:00`
- train timesteps per run: `5,360,640`
- test timesteps: `230,400` for `00:00`; `222,208` for `01:00`
- capped vs uncapped: uncapped
- run class: serious training / evaluation, not screening
- runtime: `484.910s` for `00:00`, `485.471s` for `01:00`, about `970.381s` combined
- device: `cpu`

### Results
Holdout `00:00` best route:
- `regonly_l1_2d_h8 / short / q95_p1_hold48`
- pnl `+62.34`
- trades `2`
- pnl/trade `31.17`
- precision `0.4875`
- recall `0.7166`
- f1 `0.5802`
- rmse_ratio `0.9585`

Execution split on `00:00`:
- stricter fast family recovered: `q95_p1_hold48=+62.34`, `q95_p2_hold96=+58.97`
- persistence/side-margin routes were also positive but smaller
- relaxed fast family still failed: `q90_p1_hold48=-24.14`, `q90_p2_hold48=-15.23`

Holdout `01:00` best route:
- `regonly_l1_2d_h8 / short / q90_p1_hold48`
- pnl `+113.54`
- trades `6`
- pnl/trade `18.92`
- precision `0.4931`
- recall `0.7444`
- f1 `0.5932`
- rmse_ratio `0.9679`

Execution split on `01:00`:
- relaxed fast family came back strongly: `q90_p1_hold48=+113.54`
- stricter fast family weakened: `q95_p1_hold48=+13.05`, `q95_p2_hold96=-6.97`
- several neighbors remained positive, so this was not a one-route fluke

### What this means
This is the key scientific update from the run:
exp47 was not mapper death.
The short branch reactivated immediately on the next two chronological post-23:00 holdouts while keeping honest dense/map quality.

But the run also says something equally important:
there is still no single stable extraction family.
The optimal trigger flipped by hour:
- `00:00` preferred stricter high-threshold fast entries,
- `01:00` preferred the previously failing relaxed fast route.

That is exactly the kind of behavior that argues for conditional execution selection rather than immediate mapper retraining.
The representation still looks live; the extraction policy is the fragile layer.

### Decision
`promising` as a mapper + stress-regime branch.
Still not a fixed-policy strategy candidate.
Do not promote a universal `q90` or universal `q95` rule from this run.

### Exact next mutation
Keep the mapper frozen again.
Do not retrain yet.
Mutate only execution-policy selection:
- build a simple present-time observable gate that chooses between strict fast (`q95_p1_hold48`) and relaxed fast (`q90_p1_hold48`)
- use current-window stress / impulse intensity only
- validate that selector on the next later chronological holdout(s)

If that selector fails, *then* consider broader execution-family falsification or mapper refresh.

The goal is to test whether the fast extraction family, not just the mapper, survives multiple later holdouts with positive high-friction PnL.



## 2026-04-22 21:11:49Z exp49 selector validation

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Changed one practical layer only: execution selection between `q95_p1_hold48` and `q90_p1_hold48` using a single present-time observable (`top5 mean book_impulse` on the holdout event-filter ranking).
- Calibration reused exp48 holdouts `00:00` and `01:00`; serious validation moved to `02:00` and `03:00`.

### Hypothesis
- The post-23:00 execution flip is partly explainable by current-window stress intensity, so a simple book-impulse gate can choose the better fast short policy without retraining the mapper.

### Artifacts
- JSON: `experiments/results/exp49_recent24_stress_short_selector_validation_20260422T211149Z.json`
- Summary picture: `experiments/pictures/exp49_recent24_stress_short_selector_validation_selector_summary_20260422T211149Z.png`
- Notes: `experiments/notes/exp49_recent24_stress_short_selector_validation_20260422T211149Z.md`
- DB row: `experiment_runs.id = 842`

### Outcome
- Selector threshold: `0.448849`
- Selector total pnl on validation: `-105.66406`
- Fixed q95 total pnl: `-105.66406`
- Fixed q90 total pnl: `45.16406`
- Correct route selections: `1/2`
- Decision: `not promising yet`

### Interpretation
- The selector did not generalize cleanly enough on the next holdouts, so execution drift remains the blocker and the mapper should stay frozen while the execution layer is falsified further.

## 2026-04-22 22:34:51Z exp50 two-feature execution selector

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Changed one practical layer only: execution selection. The failed 1-feature selector was replaced by a 2-feature nearest-centroid gate using `top5 mean book_impulse` + `top5 mean range_bps`.
- Calibration reused existing holdouts `00:00`, `01:00`, `02:00`, `03:00`; serious validation moved to `04:00` and `05:00`.

### Hypothesis
- The frozen short branch maps different execution winners onto a coarse present-time stress geometry, so a simple observable 2-feature selector can choose among strict fast, relaxed fast, and persistence-fast-exit routes without changing the mapper.

### Artifacts
- JSON: `experiments/results/exp50_recent24_stress_short_two_feature_route_selector_20260422T223451Z.json`
- Summary picture: `experiments/pictures/exp50_recent24_stress_short_two_feature_route_selector_selector_summary_20260422T223451Z.png`
- Notes: `experiments/notes/exp50_recent24_stress_short_two_feature_route_selector_20260422T223451Z.md`
- DB row: `experiment_runs.id = 863`

### Outcome
- Selector total pnl on validation: `119.80469`
- Fixed route totals: `q95_p1_hold48=43.46, q90_p1_hold48=119.80, q90_persistence2_fast_exit=109.07`
- Correct route selections: `2/2`
- Decision: `promising`

### Interpretation
- The selector adds evidence that execution drift is at least partly observable in present-time stress geometry and may be addressable without mapper retraining.

## 2026-04-22 23:56:33Z exp51 frozen two-feature selector follow-up

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Kept the exp50 2-feature selector frozen as well: same scaler, same centroids, same candidate routes, same calibration holdouts `00:00`..`03:00`.
- Changed one practical layer only: validation scope. Serious validation moved forward to later chronological holdouts `06:00` and `07:00`.

### Hypothesis
- If the exp50 selector was real rather than a local tie with the best fixed route on one active holdout, it should continue to match or beat the best fixed route on later chronological holdouts without any refit.

### Artifacts
- JSON: `experiments/results/exp51_recent24_stress_short_two_feature_route_selector_followup_20260422T235633Z.json`
- Summary picture: `experiments/pictures/exp51_recent24_stress_short_two_feature_route_selector_followup_selector_summary_20260422T235633Z.png`
- Notes: `experiments/notes/exp51_recent24_stress_short_two_feature_route_selector_followup_20260422T235633Z.md`
- DB row: `experiment_runs.id = 884`

### Outcome
- Selector total pnl on validation: `-53.36719`
- Fixed route totals: `q95_p1_hold48=-53.37, q90_p1_hold48=51.89, q90_persistence2_fast_exit=48.03`
- Correct route selections: `0/2`
- Decision: `not promising yet`

### Interpretation
- The frozen selector did not stay ahead of fixed routes on later holdouts, so exp50 should be treated as incomplete evidence and execution fragility remains the active blocker.

## 2026-04-23 01:18:14Z exp52 frozen-route day/evening falsification

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Kept the exp50 2-feature selector frozen as well: same scaler, same centroids, same candidate routes, same calibration holdouts `00:00`..`03:00`.
- Changed one practical layer only: validation scope. Serious validation broadened to later day/evening holdouts `08:00`, `09:00`, `18:00`, and `19:00`.

### Hypothesis
- If the selector is real rather than an early-morning local fit, it should remain competitive with the best fixed route on a materially broader later-2026 day/evening validation set without any refit.

### Artifacts
- JSON: `experiments/results/exp52_recent24_stress_short_route_falsification_day_evening_20260423T011814Z.json`
- Summary picture: `experiments/pictures/exp52_recent24_stress_short_route_falsification_day_evening_selector_summary_20260423T011814Z.png`
- Notes: `experiments/notes/exp52_recent24_stress_short_route_falsification_day_evening_20260423T011814Z.md`
- DB row: `experiment_runs.id = 925`

### Outcome
- Selector total pnl on validation: `-73.95312`
- Fixed route totals: `q95_p1_hold48=2.03, q90_p1_hold48=22.84, q90_persistence2_fast_exit=-78.23`
- Correct route selections: `0/4`
- Decision: `not promising yet`

### Interpretation
- The frozen selector still lagged the best fixed routes on broader later holdouts, so selector geometry remains overfit and execution fragility remains the active blocker.

## 2026-04-23 02:45:30Z exp53 fixed-route later-evening robustness

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Stopped selector refits after exp52 failure.
- Changed one practical layer only: compared fixed execution families on adjacent later-evening holdouts `20:00`, `21:00`, `22:00`, `23:00`.

### Hypothesis
- If selector geometry was the overfit layer rather than the mapper, adjacent later-evening holdouts should reveal one least-fragile fixed route family without retraining the mapper.

### Artifacts
- JSON: `experiments/results/exp53_recent24_stress_short_fixed_route_robustness_late_evening_20260423T024530Z.json`
- Summary picture: `experiments/pictures/exp53_recent24_stress_short_fixed_route_robustness_late_evening_route_summary_20260423T024530Z.png`
- Notes: `experiments/notes/exp53_recent24_stress_short_fixed_route_robustness_late_evening_20260423T024530Z.md`
- DB row: `experiment_runs.id = 966`

### Outcome
- Route totals: `q95_p1_hold48=14.97, q90_p1_hold48=209.97, q90_persistence2_fast_exit=129.39, q95_p2_hold96=61.41, q90_p2_hold96=119.00`
- Champion: `q90_p1_hold48` with pnl `209.96875`, wins `2/4`, trades `34`
- Decision: `promising`

### Interpretation
- A fixed route family emerged as the least-fragile execution candidate on adjacent later-evening holdouts.
## 2026-04-23 04:18:17Z exp54 next-date friction retest

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Kept fixed-route evaluation; no selector refit and no mapper retrain.
- Changed one practical layer only: moved validation to the next chronological date block (`00:00`, `01:00`, `02:00`, `03:00`) and aggregated explicit higher-friction PnL.

### Hypothesis
- If late-evening `q90_p1_hold48` captured a real thin-book short branch, it should remain competitive on the next date block and stay positive under meaningful friction.

### Artifacts
- JSON: `experiments/results/exp54_recent24_stress_short_next_date_friction_retest_20260423T041817Z.json`
- Summary picture: `experiments/pictures/exp54_recent24_stress_short_next_date_friction_retest_route_and_friction_summary_20260423T041817Z.png`
- Notes: `experiments/notes/exp54_recent24_stress_short_next_date_friction_retest_20260423T041817Z.md`
- DB row: `experiment_runs.id = 1007`

### Outcome
- Raw route totals: `q95_p1_hold48=163.41, q90_p1_hold48=144.20, q90_persistence2_fast_exit=232.18, q95_p2_hold96=174.43, q90_p2_hold96=181.11`
- Friction totals: `q95_p1_hold48: c1=163.41, c2=163.41, c5=163.41, c10=163.41; q90_p1_hold48: c1=144.20, c2=144.20, c5=144.20, c10=144.20; q90_persistence2_fast_exit: c1=232.18, c2=232.18, c5=232.18, c10=232.18; q95_p2_hold96: c1=174.43, c2=174.43, c5=174.43, c10=174.43; q90_p2_hold96: c1=181.11, c2=181.11, c5=181.11, c10=181.11`
- Champion: `q90_persistence2_fast_exit` with raw pnl `232.17969`, cost-5 pnl `232.17969`, wins `2/4`, trades `24`
- Decision: `promising`

### Interpretation
- The fixed-route branch survived the date-roll and higher-friction retest well enough to stay in the serious-candidate set.
## 2026-04-23 05:47:22Z exp55 freshest 04:00 route broadening with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation; no selector refit and no mapper retrain.
- Changed one research layer only: broadened the execution family on the freshest available new 04:00 holdout and corrected friction accounting locally after exp54's route-file lookup was shown to miss per-route friction payloads.

### Hypothesis
- If the branch is real, persistence-oriented routes should remain competitive after corrected friction even if looser q80 fast routes win raw pnl.

### Artifacts
- JSON: `experiments/results/exp55_recent24_stress_short_0400_route_broadening_corrected_friction_20260423T054722Z.json`
- Summary picture: `experiments/pictures/exp55_recent24_stress_short_0400_route_broadening_corrected_friction_route_and_friction_summary_20260423T054722Z.png`
- Notes: `experiments/notes/exp55_recent24_stress_short_0400_route_broadening_corrected_friction_20260423T054722Z.md`
- DB row: `experiment_runs.id = 1018`

### Outcome
- Raw route totals: `q80_p1_hold48=-36.56, q95_p1_hold48=0.00, q90_p1_hold48=-30.81, q90_persistence2_fast_exit=4.97, q95_p2_hold96=0.00, q90_p2_hold96=6.19, q80_persistence2_wide_margin=0.00`
- Corrected friction totals: `q80_p1_hold48: c1=-44.56, c2=-52.56, c5=-76.56, c10=-116.56; q95_p1_hold48: c1=0.00, c2=0.00, c5=0.00, c10=0.00; q90_p1_hold48: c1=-33.81, c2=-36.81, c5=-45.81, c10=-60.81; q90_persistence2_fast_exit: c1=3.97, c2=2.97, c5=-0.03, c10=-5.03; q95_p2_hold96: c1=0.00, c2=0.00, c5=0.00, c10=0.00; q90_p2_hold96: c1=5.19, c2=4.19, c5=1.19, c10=-3.81; q80_persistence2_wide_margin: c1=0.00, c2=0.00, c5=0.00, c10=0.00`
- Champion: `q90_p2_hold96` with raw pnl `6.18750`, cost-5 pnl `1.18750`, trades `1`
- Decision: `not promising yet`

### Interpretation
- This run is a freshness/falsification checkpoint, not promotion evidence. It only matters insofar as it tells us which execution families remain alive on the newest available 2026 hour.
## 2026-04-23 07:04:38Z exp56 freshest 05:00 route continuation with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout from 04:00 to 05:00 with the same broadened route family.

### Hypothesis
- If the branch is real, one of the persistence2 routes should remain competitive after corrected friction on the next chronological holdout rather than reverting to empty/negative behavior.

### Artifacts
- JSON: `experiments/results/exp56_recent24_stress_short_0500_route_continuation_corrected_friction_20260423T070438Z.json`
- Summary picture: `experiments/pictures/exp56_recent24_stress_short_0500_route_continuation_corrected_friction_route_and_friction_summary_20260423T070438Z.png`
- Notes: `experiments/notes/exp56_recent24_stress_short_0500_route_continuation_corrected_friction_20260423T070438Z.md`
- DB row: `experiment_runs.id = 1029`

### Outcome
- Raw route totals: `q80_p1_hold48=-50.48, q95_p1_hold48=-10.70, q90_p1_hold48=-8.08, q90_persistence2_fast_exit=-19.48, q95_p2_hold96=-34.05, q90_p2_hold96=-41.48, q80_persistence2_wide_margin=-10.70`
- Corrected friction totals: `q80_p1_hold48: c1=-55.48, c2=-60.48, c5=-75.48, c10=-100.48; q95_p1_hold48: c1=-12.70, c2=-14.70, c5=-20.70, c10=-30.70; q90_p1_hold48: c1=-10.08, c2=-12.08, c5=-18.08, c10=-28.08; q90_persistence2_fast_exit: c1=-22.48, c2=-25.48, c5=-34.48, c10=-49.48; q95_p2_hold96: c1=-36.05, c2=-38.05, c5=-44.05, c10=-54.05; q90_p2_hold96: c1=-43.48, c2=-45.48, c5=-51.48, c10=-61.48; q80_persistence2_wide_margin: c1=-12.70, c2=-14.70, c5=-20.70, c10=-30.70`
- Champion: `q90_p1_hold48` with raw pnl `-8.07812`, cost-5 pnl `-18.07812`, trades `2`
- Decision: `not promising yet`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch on the very next 2026 hour.
## 2026-04-23 08:30:35Z exp57 fresh 06:00/07:00 route falsification with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout set from 05:00 to the next two chronological fresh files (06:00 and 07:00) with the same broadened route family.

### Hypothesis
- If the branch is real, the 05:00 fast-route stabilization should survive across the next two chronological fresh holdouts under corrected friction rather than collapsing back into route drift or all-negative cost-adjusted PnL.

### Artifacts
- JSON: `experiments/results/exp57_recent24_stress_short_0600_0700_route_falsification_corrected_friction_20260423T083035Z.json`
- Summary picture: `experiments/pictures/exp57_recent24_stress_short_0600_0700_route_falsification_corrected_friction_route_and_friction_summary_20260423T083035Z.png`
- Notes: `experiments/notes/exp57_recent24_stress_short_0600_0700_route_falsification_corrected_friction_20260423T083035Z.md`
- DB row: `experiment_runs.id = 1060`

### Outcome
- Raw route totals: `q80_p1_hold48=43.73, q95_p1_hold48=91.04, q90_p1_hold48=68.68, q90_persistence2_fast_exit=115.75, q95_p2_hold96=88.22, q90_p2_hold96=118.21, q80_persistence2_wide_margin=96.49`
- Corrected friction totals: `q80_p1_hold48: c1=26.73, c2=9.73, c5=-41.27, c10=-126.27; q95_p1_hold48: c1=88.04, c2=85.04, c5=76.04, c10=61.04; q90_p1_hold48: c1=59.68, c2=50.68, c5=23.68, c10=-21.32; q90_persistence2_fast_exit: c1=111.75, c2=107.75, c5=95.75, c10=75.75; q95_p2_hold96: c1=85.22, c2=82.22, c5=73.22, c10=58.22; q90_p2_hold96: c1=114.21, c2=110.21, c5=98.21, c10=78.21; q80_persistence2_wide_margin: c1=93.49, c2=90.49, c5=81.49, c10=66.49`
- Champion: `q90_p2_hold96` with raw pnl `118.21094`, cost-5 pnl `98.21094`, trades `4`
- Decision: `not promising yet`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch on the very next 2026 hour.
## 2026-04-23 09:55:03Z exp58 fresh 08:00 route continuation with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout from the 06:00/07:00 pair to the next chronological fresh file (08:00) with the same broadened route family.

### Hypothesis
- If the branch is real, the exp57 rebound should continue to express cost-adjusted short edge on the next chronological fresh holdout rather than collapsing back into zero-trade or all-negative cost-adjusted behavior.

### Artifacts
- JSON: `experiments/results/exp58_recent24_stress_short_0800_route_continuation_corrected_friction_20260423T095503Z.json`
- Summary picture: `experiments/pictures/exp58_recent24_stress_short_0800_route_continuation_corrected_friction_route_and_friction_summary_20260423T095503Z.png`
- Notes: `experiments/notes/exp58_recent24_stress_short_0800_route_continuation_corrected_friction_20260423T095503Z.md`
- DB row: `experiment_runs.id = 1071`

### Outcome
- Raw route totals: `q80_p1_hold48=74.23, q95_p1_hold48=7.48, q90_p1_hold48=-1.61, q90_persistence2_fast_exit=13.12, q95_p2_hold96=-5.19, q90_p2_hold96=-22.58, q80_persistence2_wide_margin=-37.04`
- Corrected friction totals: `q80_p1_hold48: c1=54.23, c2=34.23, c5=-25.77, c10=-125.77; q95_p1_hold48: c1=0.48, c2=-6.52, c5=-27.52, c10=-62.52; q90_p1_hold48: c1=-10.61, c2=-19.61, c5=-46.61, c10=-91.61; q90_persistence2_fast_exit: c1=3.12, c2=-6.88, c5=-36.88, c10=-86.88; q95_p2_hold96: c1=-10.19, c2=-15.19, c5=-30.19, c10=-55.19; q90_p2_hold96: c1=-30.58, c2=-38.58, c5=-62.58, c10=-102.58; q80_persistence2_wide_margin: c1=-44.04, c2=-51.04, c5=-72.04, c10=-107.04`
- Champion: `q80_p1_hold48` with raw pnl `74.23438`, cost-5 pnl `-25.76562`, trades `20`
- Decision: `not promising yet`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch on the very next 2026 hour.
## 2026-04-23 11:15:05Z exp59 fresh 09:00/10:00 route continuation with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout set from the 08:00 checkpoint to the next two chronological fresh files (09:00 and 10:00) with the same broadened route family.

### Hypothesis
- If the branch is real, it should reactivate on the next two chronological fresh holdouts rather than staying friction-dead after the weak 08:00 checkpoint.

### Artifacts
- JSON: `experiments/results/exp59_recent24_stress_short_0910_route_continuation_corrected_friction_20260423T111505Z.json`
- Summary picture: `experiments/pictures/exp59_recent24_stress_short_0910_route_continuation_corrected_friction_route_and_friction_summary_20260423T111505Z.png`
- Notes: `experiments/notes/exp59_recent24_stress_short_0910_route_continuation_corrected_friction_20260423T111505Z.md`
- DB row: `experiment_runs.id = 1092`

### Outcome
- Raw route totals: `q80_p1_hold48=-89.46, q95_p1_hold48=63.93, q90_p1_hold48=-32.96, q90_persistence2_fast_exit=42.13, q95_p2_hold96=-73.03, q90_p2_hold96=-39.94, q80_persistence2_wide_margin=-71.93`
- Corrected friction totals: `q80_p1_hold48: c1=-141.46, c2=-193.46, c5=-349.46, c10=-609.46; q95_p1_hold48: c1=42.93, c2=21.93, c5=-41.07, c10=-146.07; q90_p1_hold48: c1=-66.96, c2=-100.96, c5=-202.96, c10=-372.96; q90_persistence2_fast_exit: c1=12.13, c2=-17.87, c5=-107.87, c10=-257.87; q95_p2_hold96: c1=-85.03, c2=-97.03, c5=-133.03, c10=-193.03; q90_p2_hold96: c1=-58.94, c2=-77.94, c5=-134.94, c10=-229.94; q80_persistence2_wide_margin: c1=-90.93, c2=-109.93, c5=-166.93, c10=-261.93`
- Champion: `q95_p1_hold48` with raw pnl `63.92969`, cost-5 pnl `-41.07031`, trades `21`
- Decision: `not promising yet`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch across the next two fresh 2026 hours.
## 2026-04-26 23:21:03Z exp60 fresh 11:00/12:00 route continuation with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout set from the 09:00/10:00 checkpoint to the next two chronological fresh files (11:00 and 12:00) with the same broadened route family.

### Hypothesis
- If the branch is real, it should reactivate on the next two chronological fresh holdouts rather than staying friction-dead after the raw-positive but cost-5-dead 09:00/10:00 checkpoint.

### Artifacts
- JSON: `experiments/results/exp60_recent24_stress_short_1112_route_continuation_corrected_friction_20260426T232103Z.json`
- Summary picture: `experiments/pictures/exp60_recent24_stress_short_1112_route_continuation_corrected_friction_route_and_friction_summary_20260426T232103Z.png`
- Notes: `experiments/notes/exp60_recent24_stress_short_1112_route_continuation_corrected_friction_20260426T232103Z.md`
- DB row: `experiment_runs.id = 1113`

### Outcome
- Raw route totals: `q80_p1_hold48=76.15, q95_p1_hold48=-37.20, q90_p1_hold48=137.14, q90_persistence2_fast_exit=-68.19, q95_p2_hold96=-26.60, q90_p2_hold96=-28.60, q80_persistence2_wide_margin=-58.49`
- Corrected friction totals: `q80_p1_hold48: c1=63.15, c2=50.15, c5=11.15, c10=-53.85; q95_p1_hold48: c1=-43.20, c2=-49.20, c5=-67.20, c10=-97.20; q90_p1_hold48: c1=126.14, c2=115.14, c5=82.14, c10=27.14; q90_persistence2_fast_exit: c1=-74.19, c2=-80.19, c5=-98.19, c10=-128.19; q95_p2_hold96: c1=-29.60, c2=-32.60, c5=-41.60, c10=-56.60; q90_p2_hold96: c1=-32.60, c2=-36.60, c5=-48.60, c10=-68.60; q80_persistence2_wide_margin: c1=-61.49, c2=-64.49, c5=-73.49, c10=-88.49`
- Champion: `q90_p1_hold48` with raw pnl `137.14062`, cost-5 pnl `82.14062`, trades `11`
- Decision: `not promising yet`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch across the next two fresh 2026 hours.
## 2026-04-26 23:38:04Z exp62 fresh 13:00/14:00 route continuation with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout set from the 11:00/12:00 checkpoint to the next two chronological fresh files (13:00 and 14:00) with the same broadened route family.

### Hypothesis
- If the branch is real, it should stay alive on the next two chronological fresh holdouts rather than collapsing immediately after the positive-on-11:00/12:00 checkpoint.

### Artifacts
- JSON: `experiments/results/exp62_recent24_stress_short_1314_route_continuation_corrected_friction_20260426T233804Z.json`
- Summary picture: `experiments/pictures/exp62_recent24_stress_short_1314_route_continuation_corrected_friction_route_and_friction_summary_20260426T233804Z.png`
- Notes: `experiments/notes/exp62_recent24_stress_short_1314_route_continuation_corrected_friction_20260426T233804Z.md`
- DB row: `experiment_runs.id = 1134`

### Outcome
- Raw route totals: `q80_p1_hold48=-199.06, q95_p1_hold48=-148.30, q90_p1_hold48=-255.96, q90_persistence2_fast_exit=-172.74, q95_p2_hold96=-45.09, q90_p2_hold96=-189.96, q80_persistence2_wide_margin=-146.50`
- Corrected friction totals: `q80_p1_hold48: c1=-245.06, c2=-291.06, c5=-429.06, c10=-659.06; q95_p1_hold48: c1=-165.30, c2=-182.30, c5=-233.30, c10=-318.30; q90_p1_hold48: c1=-289.96, c2=-323.96, c5=-425.96, c10=-595.96; q90_persistence2_fast_exit: c1=-193.74, c2=-214.74, c5=-277.74, c10=-382.74; q95_p2_hold96: c1=-54.09, c2=-63.09, c5=-90.09, c10=-135.09; q90_p2_hold96: c1=-209.96, c2=-229.96, c5=-289.96, c10=-389.96; q80_persistence2_wide_margin: c1=-158.50, c2=-170.50, c5=-206.50, c10=-266.50`
- Champion: `q95_p2_hold96` with raw pnl `-45.09375`, cost-5 pnl `-90.09375`, trades `9`
- Decision: `not promising yet`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch across the next two fresh 2026 hours.
## 2026-04-27 03:27:26Z exp63 active-core continuation

Status: DONE

- Active routes: `['q95_p1_hold48', 'q90_p1_hold48', 'q90_p2_hold96', 'q90_persistence2_fast_exit']`
- Validation holdouts: `['/mnt/data/repos/gaelreinaudi/crypto/2026-04-23T15-00-10.parquet', '/mnt/data/repos/gaelreinaudi/crypto/2026-04-23T16-00-11.parquet']`
- Champion: `q90_persistence2_fast_exit` cost5 `91.17188`
- Decision: `not promising yet`
