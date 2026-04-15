# Hourly experiment journal

## 2026-04-05 08:57:46Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03858, f1=0.1760, precision=0.0970, recall=0.9439
- Artifact: experiments/results/hourly/run_20260405T085804Z.json

## 2026-04-05 09:58:55Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03845, f1=0.1911, precision=0.1060, recall=0.9688
- Artifact: experiments/results/hourly/run_20260405T095914Z.json

## 2026-04-05 10:59:31Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03797, f1=0.1890, precision=0.1055, recall=0.9073
- Artifact: experiments/results/hourly/run_20260405T105950Z.json

## 2026-04-05 12:00:08Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03858, f1=0.1820, precision=0.1007, recall=0.9474
- Artifact: experiments/results/hourly/run_20260405T120026Z.json

## 2026-04-05 13:00:55Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03845, f1=0.1777, precision=0.0978, recall=0.9688
- Artifact: experiments/results/hourly/run_20260405T130114Z.json

## 2026-04-05 14:01:37Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03880, f1=0.1789, precision=0.0984, recall=0.9777
- Artifact: experiments/results/hourly/run_20260405T140156Z.json

## 2026-04-05 15:02:16Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03838, f1=0.1673, precision=0.0916, recall=0.9652
- Artifact: experiments/results/hourly/run_20260405T150235Z.json

## 2026-04-05 16:02:57Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03850, f1=0.1653, precision=0.0906, recall=0.9474
- Artifact: experiments/results/hourly/run_20260405T160315Z.json

## 2026-04-05 17:03:34Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03833, f1=0.1823, precision=0.1014, recall=0.8993
- Artifact: experiments/results/hourly/run_20260405T170353Z.json

## 2026-04-05 18:04:06Z
- Best: hgb_d6_lr008
- Zero rmse=0.01722, f1=0.0000
- Best rmse=0.03850, f1=0.1701, precision=0.0942, recall=0.8752
- Artifact: experiments/results/hourly/run_20260405T180425Z.json

## 2026-04-05 18:39:49Z
- la=16: best=hgb_d6_lr006_w | f1=0.0829, precision=0.0435, recall=0.9008, side_acc=0.27441244620986427, rmse=0.03324, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1461, precision=0.0788, recall=0.9964, side_acc=0.3348409687773563, rmse=0.03877, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2335, precision=0.1337, recall=0.9209, side_acc=0.5935004902647429, rmse=0.02894, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T184039Z.json

## 2026-04-05 19:04:41Z
- la=16: best=hgb_d6_lr006_w | f1=0.0924, precision=0.0484, recall=1.0000, side_acc=0.23178807947019867, rmse=0.03377, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1586, precision=0.0861, recall=1.0000, side_acc=0.36449864498644985, rmse=0.03804, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2306, precision=0.1318, recall=0.9209, side_acc=0.5451845566397964, rmse=0.03011, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T190531Z.json

## 2026-04-05 20:30:23Z
- Continuity: last best long-horizon config was hgb_d6_lr006_w; last failed hypothesis was that structured TCN loss would beat baseline on long-horizon precision/F1 (it did not: 0.0176/0.0337 vs baseline 0.0252/0.0472).
- Hypothesis: a 2-stage hurdle model can increase long-horizon precision/F1 by explicitly gating events before conditional map regression.
- Mutation applied: exp06 gate_threshold=0.58 (targeted one-factor change; first hurdle mutation).
- Baseline cycle artifact: experiments/results/hourly/run_20260405T202818Z.json
- Structured compare artifact: experiments/results/exp05_tcn_structured_loss_compare.json
- Hurdle artifact: experiments/results/exp06_tcn_hurdle_map_20260405T203023Z.json
- Metrics delta vs previous long-horizon best (run_20260405T202818Z):
- single_stage baseline: f1 0.2357 -> 0.2366 (+0.0009), precision 0.1349 -> 0.1358 (+0.0009), side_acc 0.5826 -> 0.5323 (-0.0503), rmse 0.02947 -> 0.02963 (+0.00016)
- hurdle candidate: f1 0.2357 -> 0.2471 (+0.0114), precision 0.1349 -> 0.1415 (+0.0066), side_acc 0.5826 -> 0.3897 (-0.1929), rmse 0.02947 -> 0.03321 (+0.00374)
- Decision: reject current hurdle variant for production candidate (side accuracy and RMSE degradation are too large despite F1/precision lift).
- Next mutation (for next run, execute first): keep hurdle architecture, increase gate_threshold to 0.62 to reduce over-triggering and recover side accuracy/RMSE while preserving precision gains.

## 2026-04-05 20:26:19Z
- la=16: best=hgb_d6_lr006_w | f1=0.0896, precision=0.0469, recall=0.9858, side_acc=0.24416720674011666, rmse=0.03397, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1639, precision=0.0893, recall=0.9929, side_acc=0.3379216457419452, rmse=0.03820, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2347, precision=0.1347, recall=0.9081, side_acc=0.558974358974359, rmse=0.02955, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T202712Z.json

## 2026-04-05 20:27:25Z
- la=16: best=hgb_d6_lr006_w | f1=0.0957, precision=0.0503, recall=1.0000, side_acc=0.22150644202180378, rmse=0.03365, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1614, precision=0.0878, recall=1.0000, side_acc=0.3354564755838641, rmse=0.03811, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2357, precision=0.1349, recall=0.9337, side_acc=0.5825527750594156, rmse=0.02917, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T202818Z.json

## 2026-04-05 20:59:09Z
- Continuity: executed previously logged next mutation first (gate_threshold=0.62).
- Hypothesis: raising gate threshold from 0.58 to 0.62 should reduce over-triggering and recover side accuracy/RMSE while keeping precision gains.
- Mutation applied: exp06 gate_threshold=0.62 (pos_weight=8.0, reg_weight_scale=12.0).
- Hurdle artifact: experiments/results/exp06_tcn_hurdle_map_20260405T205909Z.json
- Delta vs single-stage baseline in same run:
- f1: 0.2478 -> 0.2440 (-0.0038)
- precision: 0.1435 -> 0.1403 (-0.0032)
- side_acc: 0.5224 -> 0.3964 (-0.1260)
- rmse: 0.02905 -> 0.03283 (+0.00377)
- Decision: reject hurdle @0.62 (still degrades side accuracy and RMSE; now also slightly worse precision/F1).
- Next mutation (for next run, execute first): keep hurdle architecture, reduce classifier_pos_weight from 8.0 to 7.0 to lower event over-triggering while preserving gate_threshold=0.62.

## 2026-04-05 21:00:41Z
- Continuity: executed previously logged next mutation first (classifier_pos_weight=7.0, gate_threshold held at 0.62).
- Hypothesis: lowering classifier positive weight should reduce false positives enough to improve precision/F1 while recovering some side quality.
- Mutation applied: exp06 classifier_pos_weight=7.0 (gate_threshold=0.62, reg_weight_scale=12.0).
- Hurdle artifact: experiments/results/exp06_tcn_hurdle_map_20260405T210041Z.json
- Delta vs single-stage baseline in same run:
- f1: 0.2351 -> 0.2651 (+0.0300)
- precision: 0.1347 -> 0.1546 (+0.0198)
- side_acc: 0.5532 -> 0.4086 (-0.1446)
- rmse: 0.02926 -> 0.03145 (+0.00219)
- Decision: partial keep for signal quality research (big precision/F1 gain), but reject as production candidate due side-accuracy and RMSE degradation.
- Next mutation (for next run, execute first): keep classifier_pos_weight=7.0 and add side-aware post-filter (require directional dominance margin between up/down halves at inference) to improve side accuracy without giving back too much precision.
## 2026-04-05 21:33:26Z
- la=16: best=hgb_d6_lr006_w | f1=0.0924, precision=0.0485, recall=1.0000, side_acc=0.26115485564304464, rmse=0.03480, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1588, precision=0.0863, recall=1.0000, side_acc=0.34100481347773764, rmse=0.03806, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2340, precision=0.1340, recall=0.9209, side_acc=0.5644275454161386, rmse=0.02999, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T213416Z.json



## 2026-04-05 21:38:32Z
- Continuity: picked up from 2026-04-05 21:00:41Z. Last best config remained baseline `hgb_d6_lr006_w` from `experiments/results/hourly/run_20260405T202818Z.json`; last failed hypothesis was that `classifier_pos_weight=7.0` hurdle variant could be promoted despite side/RMSE degradation.
- Executed unfinished next mutation first: added side-aware post-filter with directional dominance margin (|up_max-down_max| >= 0.0015) on hurdle inference.
- Baseline cycle artifact: experiments/results/hourly/run_20260405T213416Z.json
- Structured compare artifact: experiments/results/exp05_tcn_structured_loss_compare_20260405T213832Z.json
- Hurdle artifact: experiments/results/exp06_tcn_hurdle_map_20260405T213627Z.json
- Structured-loss gate check @ look_ahead=64: structured f1=0.0355, precision=0.0185 vs baseline f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation applied: side_margin_postfilter=0.0015 (gate_threshold=0.62, pos_weight=7.0, reg_weight_scale=12.0).
- Metrics delta vs previous hurdle run (exp06_tcn_hurdle_map_20260405T210041Z.json):
- f1: 0.2651 -> 0.2641 (-0.0010)
- precision: 0.1546 -> 0.1534 (-0.0012)
- side_acc: 0.4086 -> 0.4024 (-0.0063)
- rmse: 0.03145 -> 0.03236 (+0.00090)
- Delta vs single-stage baseline in same run:
- f1: 0.2378 -> 0.2641 (+0.0263)
- precision: 0.1368 -> 0.1534 (+0.0166)
- side_acc: 0.5411 -> 0.4024 (-0.1387)
- rmse: 0.02924 -> 0.03236 (+0.00312)
- side filter keep-rate within gated rows: 1.000
- Decision: reject side-margin=0.0015 as ineffective (keep-rate=1.000 means virtually no filtering; side_acc/RMSE still degrade materially).
- Next mutation (for next run, execute first): keep gate_threshold=0.62 and pos_weight=7.0, increase side_margin only to 0.0045 to force actual directional filtering and test side_acc recovery without collapsing precision/F1.
## 2026-04-05 22:40:40Z
- la=16: best=hgb_d6_lr006_w | f1=0.0924, precision=0.0485, recall=0.9858, side_acc=0.2577903682719547, rmse=0.03398, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1522, precision=0.0824, recall=1.0000, side_acc=0.32752817210595636, rmse=0.03862, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2255, precision=0.1285, recall=0.9209, side_acc=0.5236231478892927, rmse=0.02979, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T224132Z.json


## 2026-04-05 22:45:52Z
- Continuity: picked up from 2026-04-05 22:40:40Z baseline heartbeat (`experiments/results/hourly/run_20260405T224132Z.json`) and latest hurdle state (`experiments/results/exp06_hurdle_mutation_state.json`, side_margin=0.0045, keep-rate=0.994). Last best config remained `hgb_d6_lr006_w`; last failed hypothesis was that small side-margin post-filter would recover side accuracy without hurting precision/F1.
- Executed unfinished next mutation first: `side_margin_postfilter` increased to 0.0100 (from 0.0045) because keep-rate was still ~1.0.
- Baseline cycle artifact: experiments/results/hourly/run_20260405T224132Z.json
- Structured compare artifact: experiments/results/exp05_tcn_structured_loss_compare_20260405T224407Z.json
- Hurdle artifacts: experiments/results/exp06_tcn_hurdle_map_20260405T224223Z.json (first mutation), experiments/results/exp06_tcn_hurdle_map_20260405T224329Z.json (extra iteration)
- Structured-loss gate check @ look_ahead=64: structured f1=0.0330, precision=0.0172 vs baseline f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation #1 applied: side_margin_postfilter=0.0100 (gate_threshold=0.62, pos_weight=7.0, reg_weight_scale=12.0).
- vs prior hurdle (20260405T214752Z): f1 0.2515 -> 0.2527 (+0.0012), precision 0.1452 -> 0.1467 (+0.0015), side_acc 0.4050 -> 0.3990 (-0.0060), rmse 0.03200 -> 0.03227 (+0.00027), keep-rate=0.978.
- Decision: reject (filter still barely engaged and side_acc/RMSE remained materially worse than single-stage baseline).
- Extra in-session mutation #2 (one-factor): raise gate_threshold to 0.66 while keeping side_margin=0.01 fixed (to force fewer noisy gated rows).
- vs mutation #1: f1 0.2527 -> 0.2478 (-0.0049), precision 0.1467 -> 0.1435 (-0.0032), side_acc 0.3990 -> 0.3931 (-0.0059), rmse 0.03227 -> 0.03177 (-0.00050), keep-rate 0.978 -> 0.987.
- vs single-stage baseline in same run: f1 0.2266 -> 0.2478 (+0.0211), precision 0.1295 -> 0.1435 (+0.0141), side_acc 0.5645 -> 0.3931 (-0.1714), rmse 0.02926 -> 0.03177 (+0.00251).
- Decision: reject as practical candidate (long-horizon side accuracy remains far below baseline and RMSE remains inflated despite better sparse F1/precision).
- Baseline reference (this run, look_ahead=64): f1=0.2255, precision=0.1285, side_acc=0.5236, rmse=0.02979, zero_rmse=0.01770.
- Next mutation (for next run, execute first): keep gate_threshold=0.66 and side_margin=0.01 fixed; mutate only `pos_weight` from 7.0 -> 6.0 to test whether lower classifier aggressiveness can recover side accuracy/RMSE while preserving precision gains.
## 2026-04-05 23:47:57Z
- la=16: best=hgb_d6_lr006_w | f1=0.0946, precision=0.0497, recall=1.0000, side_acc=0.24111257406188283, rmse=0.03391, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1633, precision=0.0890, recall=0.9964, side_acc=0.33448275862068966, rmse=0.03795, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2330, precision=0.1333, recall=0.9237, side_acc=0.5516232409084576, rmse=0.02942, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260405T234847Z.json

## 2026-04-05 23:52:20Z
- Continuity: picked up from latest run artifact `experiments/results/hourly/run_20260405T234847Z.json` and prior hurdle artifact `experiments/results/exp06_tcn_hurdle_map_20260405T224329Z.json`. Last best config remained `hgb_d6_lr006_w`; last failed hypothesis was that reducing hurdle classifier aggressiveness would recover side accuracy/RMSE without sacrificing sparse-event gains.
- Executed unfinished next mutation first: reduced `pos_weight` 7.0 -> 6.0 with `gate_threshold=0.66` and `side_margin=0.01` fixed, as queued in the previous journal entry.
- Baseline cycle artifact: experiments/results/hourly/run_20260405T234847Z.json
- Structured compare artifact: experiments/results/exp05_tcn_structured_loss_compare_20260405T235220Z.json
- Hurdle artifact: experiments/results/exp06_tcn_hurdle_map_20260405T234933Z.json
- Structured-loss gate check @ look_ahead=64: structured f1=0.0366, precision=0.0191 vs baseline f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation applied: `classifier_pos_weight=6.0` (gate_threshold=0.66, side_margin=0.01, reg_weight_scale=12.0).
- vs prior hurdle (exp06_tcn_hurdle_map_20260405T224329Z.json): f1 0.2478 -> 0.2521 (+0.0043), precision 0.1435 -> 0.1463 (+0.0028), side_acc 0.3931 -> 0.3959 (+0.0028), rmse 0.03177 -> 0.03173 (-0.00004), keep-rate=0.983.
- vs single-stage baseline in same run: f1 0.2420 -> 0.2521 (+0.0101), precision 0.1398 -> 0.1463 (+0.0065), side_acc 0.5874 -> 0.3959 (-0.1915), rmse 0.02922 -> 0.03173 (+0.00251).
- Decision: reject as practical candidate (still fails long-horizon directional-quality and dense-error constraints despite higher sparse-event F1/precision).
- Baseline reference (this run, look_ahead=64): f1=0.2330, precision=0.1333, side_acc=0.5516, rmse=0.02942, zero_rmse=0.01770.
- Next mutation (for next run, execute first): Keep gate_threshold=0.66, pos_weight=6.0, reg_weight_scale=12.0; increase side_margin 0.0100 -> 0.0150 only (single-factor) because side_keep_rate is still 0.983 (filter effectively not engaging) and side_acc remains too low.
## 2026-04-06 00:53:40Z
- la=16: best=hgb_d6_lr006_w | f1=0.0921, precision=0.0483, recall=1.0000, side_acc=0.22380803011292347, rmse=0.03388, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1525, precision=0.0825, recall=1.0000, side_acc=0.3244712990936556, rmse=0.03811, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2274, precision=0.1297, recall=0.9209, side_acc=0.5831017231795442, rmse=0.02916, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260406T005431Z.json



## 2026-04-06 00:58:46Z
- Continuity: picked up from latest journal handoff (2026-04-05 23:52:20Z). Last best config remained `hgb_d6_lr006_w` from `experiments/results/hourly/run_20260405T234847Z.json`; last failed hypothesis was that lowering hurdle aggressiveness (`pos_weight=6.0`) would recover side quality enough for practical use.
- Executed unfinished next mutation first: increased `side_margin` from 0.0100 -> 0.0150 with `gate_threshold=0.66`, `pos_weight=6.0`, `reg_weight_scale=12.0` fixed.
- Baseline cycle artifact: `experiments/results/hourly/run_20260406T005431Z.json`
- Structured compare artifact: `experiments/results/exp05_tcn_structured_loss_compare_20260406T005553Z.json`
- Hurdle artifact: `experiments/results/exp06_tcn_hurdle_map_20260406T005628Z.json`
- Structured-loss gate check @ look_ahead=64: structured f1=0.0366, precision=0.0191 vs baseline f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation applied: `side_margin_postfilter=0.0150`.
  - vs prior hurdle (`exp06_tcn_hurdle_map_20260405T234933Z.json`): f1 0.2521 -> 0.2698 (+0.0177), precision 0.1463 -> 0.1587 (+0.0124), side_acc 0.3959 -> 0.4162 (+0.0203), rmse 0.03173 -> 0.03052 (-0.00122), keep-rate 0.983 -> 0.975.
  - vs single-stage baseline in same run: f1 0.2240 -> 0.2698 (+0.0458), precision 0.1273 -> 0.1587 (+0.0315), side_acc 0.5604 -> 0.4162 (-0.1441), rmse 0.02978 -> 0.03052 (+0.00073).
- Decision: reject as practical candidate (precision/F1 improved strongly, but long-horizon side accuracy is still far below baseline and below practical gate).
- Next mutation (for next run, execute first): keep `side_margin=0.015` and `pos_weight=6.0`; increase only `gate_threshold` 0.66 -> 0.70 to further cut noisy triggers and test if side_acc can recover without giving up too much precision.
## 2026-04-06 02:01:00Z
- la=16: best=hgb_d6_lr006_w | f1=0.0861, precision=0.0451, recall=0.9150, side_acc=0.23272238103245335, rmse=0.03401, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1531, precision=0.0830, recall=0.9929, side_acc=0.3248700816629547, rmse=0.03781, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2360, precision=0.1356, recall=0.9081, side_acc=0.6060649611957459, rmse=0.02901, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260406T020150Z.json

## 2026-04-14 11:42:51Z
- Continuity: starting from exp16 strict holdout result where `l1_evt005_pw2` preserved geometry / RMSE but stayed too conservative to trade.
- Hypothesis: conditioning train/test on violent recent windows may concentrate learnable market-reaction structure and improve event quality without RMSE blow-up.
- Mutation applied: event-window filter on newest BTC files only (`07:00`, `08:00`, `09:00` train; `10:00` test), keeping top 35% windows by observable intrawindow move/range/vol/book-activity score.
- Event-filter artifact: experiments/results/exp17_event_filtered_h64_tcn_20260414T114251Z.json
- Window counts: train 51/144 selected, test 17/48 selected, older-file fallback not needed.
- Best route: `l1_evt005_pw2` with precision=0.2957, f1=0.3285, rmse=1.14068 vs zero_rmse=1.14359, fixed pnl=-0.02344.
- Visual verdict: not noise; partial timing alignment, but prediction still underfires / collapses vertically, matching near-flat PnL.
- Decision: keep event-window conditioning in the palette, but reject current mapper+trigger stack as tradable because better event metrics did not convert to profits.
- Next mutation: keep event-window conditioning fixed and mutate only trigger extraction (local-max / side-aware map trigger or threshold sweep) instead of changing the mapper again.

## 2026-04-14 12:06:16Z
- Continuity: used exp17 as the seed result and expanded to a 25-run event-filtered suite on the same honest recent BTC holdout.
- Leakage check: event-window ranking still used only present-window observables (abs return/range/realized vol/book activity), not future target intensity or future pnl.
- Mutation applied: 25-variant sweep over event loss, pos_weight, width, epochs, decay, active-pixel weighting, and trade-threshold calibration; train files fixed to `07:00,08:00,09:00`, test file fixed to `10:00`.
- Data scale note: this was a screening run, not a huge training job. Script capped each file at `max_windows=48`; train windows `144 -> 51` after filtering, test windows `48 -> 17`; rolling window size `256`; approximate train timesteps `13056`, test timesteps `4352`.
- Runtime note: experiment start `12:06:16Z`, first per-variant artifact about `12:07:03Z`, summary artifact about `12:08:02Z`, total wall-clock about `106s`.
- Cache correction: the repo already had shaped-array cache in `shaper.py`/`cache_manager.py`; the slowness came from custom experiment scripts forcing cache off. Future scientists should not repeat that mistake.
- Suite artifact: experiments/results/exp18_event_filtered_suite25_20260414T120616Z.json
- Best route by route-score: `l1_evt005_pw2_h64` with precision=0.2861, f1=0.3734, rmse=1.14125 vs zero_rmse≈1.14359, fixed pnl=-0.02344.
- Most interesting positive-pnl routes: `regonly_activew3` pnl=6.4375, `regonly_wd1e3` pnl=0.9375, `regonly_huber_thr010` pnl=0.9375.
- Visual verdict on top route: better structural alignment than weaker runs, but still underfiring rather than harvesting enough opportunities.
- Decision: the event-filtered regime looks learnable; mapper sweep is good enough for now; tradability blocker has shifted to trigger extraction / execution conversion.
- Next mutation: freeze dataset + mapper family for one step and run trigger-only follow-ups (local-max / side-aware extraction and threshold sweeps around `l1_evt005_pw2_h64`, `regonly_wd1e3`, `regonly_huber_thr010`; confirm `regonly_activew3` across neighboring thresholds/regimes before trusting its pnl).

## 2026-04-06 02:04:07Z
- Continuity: picked up from latest journal handoff (2026-04-06 00:58:46Z) and artifacts `experiments/results/hourly/run_20260406T020150Z.json` + `experiments/results/exp06_tcn_hurdle_map_20260406T020235Z.json`. Last best config remained `hgb_d6_lr006_w`; last failed hypothesis was that stronger gate threshold with capped side-margin would recover long-horizon side quality.
- Executed unfinished next mutation first: `gate_threshold` raised to 0.70 with `pos_weight=6.0`, `side_margin=0.015`, `reg_weight_scale=12.0` fixed.
- Baseline cycle artifact: `experiments/results/hourly/run_20260406T020150Z.json`
- Structured compare artifact: `experiments/results/exp05_tcn_structured_loss_compare_20260406T020407Z.json`
- Hurdle artifact: `experiments/results/exp06_tcn_hurdle_map_20260406T020235Z.json`
- Structured-loss gate check @ look_ahead=64: structured f1=0.0346, precision=0.0180 vs baseline f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation applied: `gate_threshold=0.70`.
  - vs prior hurdle (`exp06_tcn_hurdle_map_20260406T005628Z.json`): f1 0.2698 -> 0.2540 (-0.0158), precision 0.1587 -> 0.1489 (-0.0098), side_acc 0.4162 -> 0.3901 (-0.0261), rmse 0.03052 -> 0.03138 (+0.00086), keep-rate=0.969.
  - vs single-stage baseline in same run: f1 0.2383 -> 0.2540 (+0.0157), precision 0.1373 -> 0.1489 (+0.0116), side_acc 0.5782 -> 0.3901 (-0.1881), rmse 0.02953 -> 0.03138 (+0.00185).
- Decision: reject as practical candidate (side accuracy remains far below practical gate and RMSE is still above single-stage baseline despite sparse-event gains).
- Baseline reference (this run, look_ahead=64): f1=0.2360, precision=0.1356, side_acc=0.6061, rmse=0.02901, zero_rmse=0.01770.
- Next mutation (for next run, execute first): keep gate_threshold=0.70 and side_margin=0.015 fixed; lower classifier_pos_weight 6.0 -> 5.0 (single-factor) to reduce over-triggering and attempt side_acc/RMSE recovery.
## 2026-04-12 18:35:08Z
- la=16: best=hgb_d6_lr006_w | f1=0.0929, precision=0.0487, recall=1.0000, side_acc=0.25779939617577996, rmse=0.03430, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1677, precision=0.0915, recall=1.0000, side_acc=0.3195813741847414, rmse=0.03816, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2269, precision=0.1294, recall=0.9209, side_acc=0.5760764225391112, rmse=0.02952, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260412T183559Z.json



## 2026-04-12 18:37:20Z
- Continuity: picked up from latest journal handoff (2026-04-06 02:04:07Z) and artifacts `experiments/results/hourly/run_20260412T183559Z.json` + `experiments/results/exp06_tcn_hurdle_map_20260412T183659Z.json`. Last best config remained `hgb_d6_lr006_w`; last failed hypothesis was that stricter gate threshold at `0.70` would recover side quality while preserving sparse-event gains.
- Executed unfinished next mutation first: lowered `classifier_pos_weight` from `6.0 -> 5.0` with `gate_threshold=0.70`, `side_margin=0.015`, `reg_weight_scale=12.0` fixed.
- Baseline cycle artifact: `experiments/results/hourly/run_20260412T183559Z.json`
- Structured compare artifact: `experiments/results/exp05_tcn_structured_loss_compare_20260412T183644Z.json`
- Hurdle artifact: `experiments/results/exp06_tcn_hurdle_map_20260412T183659Z.json`
- Structured-loss gate check @ look_ahead=64: structured f1=0.0356, precision=0.0185 vs baseline-TCN f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation applied: `classifier_pos_weight=5.0`.
  - vs prior hurdle (`exp06_tcn_hurdle_map_20260406T020235Z.json`): f1 0.2540 -> 0.2356 (-0.0184), precision 0.1489 -> 0.1371 (-0.0118), side_acc 0.3901 -> 0.3852 (-0.0049), rmse 0.03138 -> 0.03202 (+0.00064), keep-rate 0.9688 -> 0.9775.
  - vs single-stage baseline in same run: f1 0.2322 -> 0.2356 (+0.0034), precision 0.1331 -> 0.1371 (+0.0040), side_acc 0.5587 -> 0.3852 (-0.1734), rmse 0.02932 -> 0.03202 (+0.00269).
- Decision: reject as practical candidate (lowering `pos_weight` weakened sparse-event quality relative to prior hurdle run while leaving side accuracy and dense error far below practical requirements).
- Next mutation (for next run, execute first): keep `gate_threshold=0.70`, `pos_weight=5.0`, `side_margin=0.015` fixed; lower only `reg_weight_scale` `12.0 -> 8.0` to soften conditional-map overshoot and test whether RMSE/side quality can recover without reopening the event gate.
## 2026-04-12 19:41:31Z
- la=16: best=hgb_d6_lr006_w | f1=0.0950, precision=0.0499, recall=0.9858, side_acc=0.25488873904248144, rmse=0.03399, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1574, precision=0.0854, recall=0.9964, side_acc=0.33993399339933994, rmse=0.03804, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2311, precision=0.1321, recall=0.9209, side_acc=0.5845231296402056, rmse=0.02910, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260412T194221Z.json

## 2026-04-12 19:43:04Z
- Continuity: picked up from latest journal handoff (2026-04-12 18:37:20Z) and artifacts `experiments/results/hourly/run_20260412T194221Z.json` + `experiments/results/exp06_tcn_hurdle_map_20260412T183659Z.json`. Last best config remained `hgb_d6_lr006_w`; last failed hypothesis was that lowering `classifier_pos_weight` to `5.0` would recover side quality / RMSE without giving back the sparse-event edge.
- Executed unfinished next mutation first: lowered `reg_weight_scale` from `12.0 -> 8.0` with `gate_threshold=0.70`, `pos_weight=5.0`, `side_margin=0.015` fixed, because the gate threshold was already capped and the previous run still showed low precision plus large side/RMSE damage.
- Baseline cycle artifact: `experiments/results/hourly/run_20260412T194221Z.json`
- Structured compare artifact: `experiments/results/exp05_tcn_structured_loss_compare_20260412T194333Z.json`
- Hurdle artifact: `experiments/results/exp06_tcn_hurdle_map_20260412T194304Z.json`
- Structured-loss gate check @ look_ahead=64: structured f1=0.0338, precision=0.0176 vs baseline-TCN f1=0.0472, precision=0.0252 -> does NOT beat baseline (fallback required).
- Hurdle mutation applied: `reg_weight_scale=8.0`.
  - vs prior hurdle (`exp06_tcn_hurdle_map_20260412T183659Z.json`): f1 0.2356 -> 0.2277 (-0.0079), precision 0.1371 -> 0.1325 (-0.0046), side_acc 0.3852 -> 0.3769 (-0.0083), rmse 0.03202 -> 0.03547 (+0.00345), keep-rate 0.9775 -> 0.9797.
  - vs single-stage baseline in same run: f1 0.2297 -> 0.2277 (-0.0019), precision 0.1312 -> 0.1325 (+0.0013), side_acc 0.5679 -> 0.3769 (-0.1910), rmse 0.02920 -> 0.03547 (+0.00627).
- Decision: reject as practical candidate (softening conditional-map weighting did not recover dense error or directional quality; it slightly hurt sparse-event quality too).
- Baseline reference (this run, look_ahead=64): f1=0.2311, precision=0.1321, side_acc=0.5845, rmse=0.02910, zero_rmse=0.01770.
- Next mutation (for next run, execute first): keep `gate_threshold=0.70`, `pos_weight=5.0`, `side_margin=0.015` fixed; lower only `reg_weight_scale` `8.0 -> 6.0` to complete the overshoot-softening sweep before changing the event gate, then re-check whether RMSE/side quality recover at all.

## 2026-04-13 22:11:05Z
- la=16: best=hgb_d6_lr006_w | f1=0.0952, precision=0.0500, recall=1.0000, side_acc=0.2502052208175997, rmse=0.03370, zero_rmse=0.01521
- la=32: best=hgb_d6_lr006_w | f1=0.1591, precision=0.0864, recall=1.0000, side_acc=0.32460257380772145, rmse=0.03809, zero_rmse=0.01722
- la=64: best=hgb_d6_lr006_w | f1=0.2236, precision=0.1275, recall=0.9081, side_acc=0.5678131991051454, rmse=0.02970, zero_rmse=0.01770
- Artifact: experiments/results/hourly/run_20260413T221201Z.json

