# Finding and testing tradable order book patterns

Use the **causal 2D TCN as the primary model**. The research question is whether an image of the book and recent trades up to time `t` can forecast which price levels become reachable after `t`, and whether the forecast improves the results of a trading policy. Use simple rules as controls to measure what the TCN adds. The existing map scores and short-route PnL are insufficient to promote a strategy.

The [repository and data findings](repository-findings.md) cover six inspected hours, an inventory of 8,464 parquet files and a causality check of the active model on two recorded BTC images.

## How the image model uses time

`ArrayShaper` builds one image row from the book and trades available at a tick, with the price bins anchored to a running EMA. `OrderBook2DTCN` consumes the rolling sequence of those rows. Each temporal convolution pads on the past side, and the output heads preserve the time axis. For an image ending at `t`, `prediction[-1]` is the forecast available at `t`. Each earlier row `prediction[i]` is a forecast available at its own time, using only image rows through `i`.

Train against future crossings and score them after their horizon completes. This is the intended use of future data. If a hypothesis needs two more ticks of confirmation after a trigger, timestamp the decision at the second confirming tick and apply order latency from there. The same rule applies to an explicitly permitted right-context model. Attribute the decision and its measured returns to the end of the allowed context. A direction or exit chosen using the realized future path is useful as a separate oracle diagnostic.

For one decision per timestamp, run the fixed historical-context model on rolling images and emit the last row, after warmup. Batch those images for throughput. An efficient causal convolution evaluated on one long chunk may see more history than the existing 1,024-row model receives at a rolling endpoint. Test equivalence with the same available history before replacing rolling inference with chunk inference.

## 1. Make the research clock and execution economics explicit

Build timestamped one-second image rows and quote records from raw parquet. Maintain the three books in message-time order and retain both message timestamp and sequence. Forward-fill known book state across genuinely idle intervals, set trade flow to zero there, and preserve message age and validity flags. Stop or exclude trading across disconnects, unresolved integrity failures and missing files. Carrying an old quote through an unexplained outage must not create an executable quote. Keep the legacy occupied-bucket clock available for an evaluation of the existing configuration with its original input geometry.

Use snapshots as resets, zero sizes as removals, and exclude trade snapshots from fresh traded volume. Check sequence continuity across the interleaved recorded channels: the sampled files share sequence progression across channels, so gaps within one channel alone do not demonstrate loss. Account for deliberately omitted subscription messages before calling a hole a feed failure.

For future capture, retain local monotonic receive time, exchange event times, trade IDs, connection/session identity and heartbeat records. Preserve native raw fields through normalization. Existing data cannot retrospectively supply fields that were discarded.

Record the intended market, order size, maker/taker fee tier and execution instrument in every serious result. Begin with spot long/flat, or explicitly cost an inventory sell-and-rebuy policy. A simulated naked short in BTC-USD does not establish that the intended account can execute it. Derivatives require their own executable quotes, basis and funding assumptions.

The sampled 60-second moves are often only a few bps. Establish whether the real fee tier leaves a plausible opportunity before investing in larger models. An illustrative 10 bps hurdle in the diagnostic chart is not an assumed Coinbase fee schedule.

## 2. Search a small set of economically interpretable patterns

Treat depletion, absorption and cross-market confirmation as mechanisms to probe in the image forecasts. Annotate them with simple measurements over trailing 1, 5, 15 and 60 seconds. These annotations support conditional analysis and rule baselines. Keep price-bin order and the depth and trade channels intact for the TCN.

| Priority | Hypothesis | Features known at the decision time | Test that could reject it |
|---|---|---|---|
| 1 | Depletion plus confirmed aggression predicts continuation | Falling ask depth, rising bid-side support, positive signed aggressive trade volume, quote-based order-flow imbalance and a recent upward quote move; mirror for downside | Does it add net executable return beyond recent-price momentum, spread and volatility controls? |
| 2 | Replenishment during heavy aggression predicts absorption and reversal | Repeated sell aggression into a stable/replenishing bid, unusually low price response per traded unit, followed by a causal recovery signal; mirror at the ask | Does replenishment distinguish a reversal from a brief pause before continuation? |
| 3 | BTC flow helps time ETH opportunities | Lagged BTC returns and signed flow, current ETH book response and ETH-BTC relative-price residual | Does it improve an ETH-only model after at least a one-second information lag and conservative execution delay? |

For signed aggressive volume use `volume(maker SELL) - volume(maker BUY)`. Check this interpretation against reconstructed quote behavior on development data. Depth reductions alone cannot identify cancellations versus executions with certainty; call them depletion, and use the trade tape as supporting evidence.

First study the map score and side margin on forecasts computed at rolling endpoints. Report realized crossings and executable returns by calibration-set score bucket, horizon and annotated pattern. Then compare the full model to depth-only, trade-only and price-only inputs, retraining each under the same split. Matched controls with similar trailing volatility, spread, momentum and time of day help distinguish a motif from the market regime. Cluster repeated triggers into episodes and report how many separate episodes produce the effect.

Only investigate horizons whose observed opportunity size can cover actual costs. The 1-second table is the first research resolution; revisit 100 ms only if a surviving hypothesis needs it and survives the feed's batching and latency limits. ETH-BTC is initially a supporting feature because its sampled trade activity is much sparser.

## 3. Use targets aligned with a trade

Define the information cutoff at time `t`. Features use only messages available by that cutoff. Execute at `t + latency`, after inference and order-transmission delay, using the executable side of the reconstructed book.

For a fixed-size long, compute:

`net_quote_pnl = q * (exit_sell_vwap - entry_buy_vwap) - entry_fee - exit_fee`

Convert it to bps of entry notional. VWAP must walk the visible ask/bid levels for the intended size. This accounts for spread through execution prices; do not subtract the same spread again. Add an explicit allowance for unobserved latency and market impact.

Keep first-hit maps as the primary training target. For a revised target, anchor each up/down boundary to that row's executable starting quote and express distance in bps. Encode first-hit time and distinguish a completed observation with no hit from censoring at a file break or session end. Check that changing quotes after `t + H` leaves the label at `t` unchanged and that fresh and cached shaping agree on valid rows. Version the target and its cache schema.

Keep the current L1 target for the initial evaluation comparison. For a follow-up label experiment, consider predicting hit probabilities at specified distances and horizons. Calibrate those probabilities on earlier days. Compare the learned map's first-hit order against realized up/down hits and measure net returns at fixed horizons as evaluation outcomes. Use explicit exit rules for achieved PnL.

For route extraction, start with the existing train-calibrated map maxima and side margin. Add per-distance or per-horizon rules only as a separate experiment. Fit normalization, output caps, thresholds and any probability calibration on training or calibration data that precedes evaluation.

## 4. Use chronological development and a sealed final test

Proposed 2026 schedule, subject to checking the full experiment and notebook manifests for prior use:

| Period | Role |
|---|---|
| Apr 14–May 15 | Feature development, baseline fitting and internal chronological tuning |
| May 16–May 31 | Compare the small preregistered hypothesis set and select one policy |
| Jun 1–Jun 14 | Development walk-forward checks: preceding 12 days for fitting, next 2 days for calibration, following day for evaluation |
| Jun 16–Jul 1 | Candidate sealed final period; this audit did not inspect its file contents |

June 15 and July 2 have already been inspected descriptively and are excluded from the proposed sealed period. Earlier experiments heavily reused April; it is development data. The final period is only sealed after confirming it was not previously used to choose models. Once opened for performance evaluation, it cannot serve as a fresh final test again. If it was already used, reserve newly recorded days.

Split by complete dates and timestamps, keeping all markets on the same side of each split. Purge training decisions whose forward labels overlap the next split. For conservative independent evaluation blocks, allow a gap covering the maximum lookback plus label horizon; separately distinguish this from a live walk-forward run, where using known earlier history for features is legitimate. Do not randomly split overlapping rolling examples.

Freeze the model family, feature set, cost assumptions, refit cadence and policy before final evaluation. A walk-forward model may refit on the past using that fixed recipe. Do not choose the winning route separately for each reported test hour.

The 2024–2025 corpus is useful for earlier discovery and regime transfer tests. Treat it as a separate recording era, with its own integrity checks and schema normalization, rather than pooling it blindly with 2026. Older data cannot replace contemporary forward confirmation.

## 5. Simulate the full eligible timeline once

Every valid decision timestamp gets at most one forecast and one policy decision. Maintain cash, inventory, position size and pending execution state continuously across files. Warm up from past observations and mark explicit resets at genuine data breaks. Evaluate all eligible periods, including quiet periods and periods where the strategy elects not to trade.

Regime gates must use trailing observations and thresholds fixed before the test. Do not rank complete test windows and then trade their earlier rows. Preserve original timestamps through batching; overlapping model contexts can supply predictions, but the ledger must count each real trade once.

Run at least the measured fee tier, a worse fee/slippage scenario, and delays such as 0.5, 1 and 2 seconds as sensitivity assumptions. Order-size scenarios should span the intended size and a larger size. The available data permits assumption-based latency stress tests; actual latency requires new capture measurements.

Do not award passive fills because a quote was merely touched. Aggregated L2 cannot recover exact queue priority. Passive research needs conservative fill bounds, adverse-selection analysis and subsequent forward paper validation; it should be a separate execution hypothesis.

## 6. Decide using economic and statistical evidence

Report daily and per-trade net bps, quote-currency PnL at fixed notional, trade count, turnover, exposure, drawdown, tail loss, average win/loss, break-even round-trip fees and sensitivity to size/delay. Include the whole daily equity curve and per-market/session/regime slices. Map F1, RMSE and directional accuracy remain diagnostic metrics.

Compare against no trade, costed buy-and-hold, matched random entries, simple momentum, and a basic depth/flow rule under the same opportunity set and sizing. For a short policy, also compare against constant short exposure over the same selected sessions; otherwise a falling market can explain the apparent edge.

Use day-block bootstrap intervals for mean daily net PnL and incremental performance, retaining within-day dependence. If dependence spans days, use longer blocks. Log every hypothesis, feature set and threshold tried; restrict the search and account for multiple comparisons in the discovery stage. A few favorable hours or a large count of overlapping target pixels are insufficient evidence.

Suggested research promotion gate: positive aggregate final-period net PnL, a 95% block-bootstrap lower bound above zero at realistic costs, positive incremental value over the strongest relevant baseline, survival of the stressed execution scenario, and no dependence on a single exceptional day. If the test is too short or trades too few to resolve those conditions, classify it as inconclusive and collect more days. It is reasonable for the first search to find no tradable edge.

After passing offline checks, run the frozen policy forward in shadow mode for at least two to four weeks, extending the period if trades are sparse. Log decision information, hypothetical orders and contemporaneous executable depth. Verify real latency and, where appropriate, exchange-supported test execution before relying on passive fill estimates or committing capital.

## First experiment

1. Reproduce `regonly_l1_2d_h8` with the existing 1,000 ms cadence, 1,024 input rows, 64-tick horizon, training pool and route rules. Save the fitted weights and calibration for both evaluation arms. Use development days and keep the candidate final dates sealed.
2. Score the same fitted model and frozen route on the old selected-window evaluator and on every eligible rolling endpoint of the same recordings. Include intermediate comparisons with identical friction and fill assumptions to quantify what changes with timestamp selection and context warmup. Then stress the chronological ledger with delayed arrivals, two-sided fees, displayed depth and gap handling. Mark the old evaluation arm as a historical diagnostic.
3. Check prefix invariance through replay, shaping, normalization, score, eligibility and decisions. Changing information after a cutoff must leave all earlier model inputs, forecasts and order submissions identical within floating-point tolerance. Check forecast targets separately for invariance to changes beyond their declared horizon. Validate ledger results on known spread, fee, latency, size and session-boundary cases.
4. Report map scores at eligible decision timestamps, right-edge hit rates by score bucket, net bps and daily PnL. If scores retain discrimination but routes lose after realistic costs, investigate execution or abstention. If discrimination disappears when evaluation moves to chronological rolling endpoints, address the evaluation or training objective before changing the architecture.
5. Run separate follow-ups for corrected trade tape, a fixed wall clock, revised first-hit labels and input-channel ablations. Repeat the winning frozen configuration on the reserved dates and forward collection.

Future experiments must use new numbered scripts, preserve prior experiment artifacts, call the required pipeline guards, cap `n_samples` or stream examples to fit memory, and register every result as required by `experiments/scientist_handoff_protocol.md`. This audit added documentation and descriptive data artifacts only; implementation and strategy experiments remain the next work.
