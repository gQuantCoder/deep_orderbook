# Why the omniscient/reference PnL is not turning into a stable tradable strategy

## Short answer
Because the omniscient PnL is a reference ceiling, not a causally available policy.

What we found is that the model can sometimes learn **where action is happening**, but it still does a poor or unstable job at converting that into a **robust entry/exit policy with the right side, timing, and churn profile**.

So the gap is not just “prediction quality bad.”
It is more specifically:

1. the model often overfires,
2. the trigger layer is fragile,
3. direction can flip by regime,
4. some positive PnL slices collapse on adjacent holdouts,
5. high-turnover routes can look exciting while being fake/fragile.

---

## What we did, in order

### 1. Honest holdout check killed the first flattering story
We moved from same-file-ish / flattering setups to stricter different-parquet walk-forward holdouts.

Key result:
- `exp13_multifile_btc_walkforward`
- test: `2026-04-14T03-00-00.parquet`
- metrics:
  - precision: `0.0696`
  - recall: `0.9733`
  - f1: `0.1299`
  - rmse: `2.1488`
  - zero-baseline rmse: `0.4435`
- PnL:
  - omniscient: `+15.0625`
  - prediction: `-3.6406`

Conclusion:
- the earlier apparent success was mostly bullshit from flattering evaluation
- under honest holdout, the branch over-triggered badly and did not generalize

### 2. We tried cleaner BTC variants, but they still failed the dense-map/generalization test
We ran stricter BTC variants:
- `exp14_multifile_btc_precision`
- `exp15_multifile_btc_regonly`

Results:
- `exp14`: prediction PnL `+29.3047`, but RMSE still blew out badly vs zero baseline
- `exp15`: best of the honest early variants, prediction PnL `+9.4297`, but still unacceptable RMSE / generalization

Conclusion:
- some dashboards and slice-level PnL looked better
- but they were not trustworthy enough out of sample
- good-looking PnL without bounded dense error was rejected

### 3. We added event-window conditioning to focus on interesting moments
Instead of training on everything, we filtered to eventful windows using only **present-time observable** signals:
- abs return bps
- intrawindow range bps
- realized vol
- book std
- book impulse

This was explicitly checked for leakage and kept causal.

Small recent-data screening result:
- `exp18_event_filtered_suite25`
- best route around `l1_evt005_pw2_h64`
- precision: about `0.2861`
- f1: about `0.3734`
- RMSE roughly matched zero baseline
- fixed-slice prediction PnL: about `-0.0234`

Conclusion:
- event filtering helped find signal-rich windows
- it improved metrics
- it still did not make the strategy trade well

### 4. We scaled up to serious historical data
We stopped kidding ourselves with tiny runs and moved onto large historical BTC data from `/media/photoDS216/crypto`.

Serious run:
- `exp21_historical_volatile_suite`
- 8 train parquet files, 1 holdout file
- train windows: `33,989 -> 11,897` after event filter
- test windows: `4,336 -> 1,518`
- train timesteps: `3,045,632`
- runtime: about `2.4h` on CUDA

Best route:
- `l1_evt005_pw2_h64`
- precision: `0.4823`
- recall: `0.8420`
- f1: `0.6133`
- rmse: `6.5177`
- zero-baseline rmse: `6.6834`
- fixed PnL: `-201.0781`

This was a huge finding:
- the mapper was learning something real enough to produce strong event metrics
- but the actual trading policy still lost money

Conclusion:
- the problem was no longer just “the model learns nothing”
- the problem became: **signal extraction / execution logic is broken**

### 5. We pivoted from mapper search to trigger/strategy search
We froze mapper branches and started sweeping strategy logic instead:
- threshold quantiles
- persistence
- side margin
- cooldown
- max hold

This became `exp22_trigger_sweep_hist_cached`.

Best route there:
- strategy: `q95_p2_hold96`
- final PnL: `-2423.7266`
- precision: `0.4485`
- f1: `0.6102`
- rmse_ratio: `0.9805`

Conclusion:
- decent signal metrics and sane RMSE were **not enough**
- long-only trigger extraction was still getting destroyed

### 6. We found and fixed a real infrastructure mistake: cache was already there
At one point we suspected preprocessing waste and thought new cache architecture might be needed.
That was wrong.

What we found:
- the repo already had shaped-array caching in:
  - `deep_orderbook/shaper.py`
  - `deep_orderbook/cache_manager.py`
  - `deep_orderbook/config.py`
- the real bug was that an experiment helper path had forced:
  - `use_cache=False`
  - `save_cache=False`

So the slowdown was partly self-inflicted by the experiment script, not missing infra.

We fixed that and updated the scientist handoff docs/skills so we do not reinvent the wheel again.

### 7. After cache fix, the next holdout finally produced positive long-side PnL
Run:
- `exp23_trigger_sweep_next_holdout`
- test file: `2025-04-09T19-00-33.parquet`

Best route:
- mapper: `regonly_wd1e3`
- strategy: `q95_p2_hold96`
- final PnL: `+907.4922`
- precision: `0.3235`
- f1: `0.4868`
- rmse_ratio: `1.0133`

Also strong:
- mapper: `l1_evt005_pw2_h64`
- same strategy `q95_p2_hold96`
- final PnL: `+545.8828`
- precision: `0.3560`
- f1: `0.5045`
- rmse_ratio: `0.9947`

Conclusion:
- finally a real positive holdout
- but only on one adjacent regime
- not enough to declare victory

### 8. We tested the inversion / short hypothesis instead of assuming long-only truth
Because losing money with decent map metrics can mean the execution side is wrong, we tested explicit short/reverse execution.

#### Same holdout as exp23, but short-side (`exp24_reverse_short_holdout`)
Best route:
- mapper: `l1_evt005_pw2_h64`
- strategy: `q80_p2_hold48`
- final PnL: `+11732.9766`
- precision: `0.3949`
- f1: `0.5420`
- rmse_ratio: `0.9865`

This was the first big sign that the learned map might be directionally inverted or at least regime-sensitive.

But it was not clean:
- nearby short routes could still be very bad
- e.g. `q95_p1_hold48` on the same holdout was `-1257.1094`

Conclusion:
- short-side can massively outperform long-side on the same mapper
- but the strategy is extremely threshold/regime sensitive

#### Next adjacent holdout short-side (`exp27_l1_short_holdout_20`)
Best route:
- strategy: `q80_p3_hold48`
- final PnL: `+1968.4297`
- precision: `0.2456`
- f1: `0.3848`
- rmse_ratio: `1.0345`

#### Same holdout long-side comparison (`exp28_l1_long_holdout_20`)
Best route:
- strategy: `q90_p1_hold48`
- final PnL: `+1312.0000`
- precision: `0.2708`
- f1: `0.4153`
- rmse_ratio: `1.0310`

Conclusion for holdout 20:
- both long and short could be positive
- short was better on raw PnL
- this looked more like **regime-dependent directional asymmetry** than a simple universal inversion rule

#### Next adjacent holdout short-side again (`exp29_l1_short_holdout_21`)
Best route:
- strategy: `q95_p1_hold48`
- final PnL: `+2001.6875`
- precision: `0.2752`
- f1: `0.4195`
- rmse_ratio: `1.0254`

Conclusion:
- short remained interesting on the next holdout too
- but we still saw route sensitivity and no stable single threshold family that obviously dominates everywhere

---

## What we observed overall

### A. Omniscient/reference PnL is not an executable strategy
It is the answer key after the fact.
It says the slice had opportunity.
It does **not** say the model has learned a causally tradable policy.

### B. The model often learns eventfulness better than execution
Repeated pattern:
- precision/F1 can get decent
- RMSE can stay bounded vs zero baseline
- PnL can still be awful

That means:
- “where something happens” is easier than
- “which side to take, exactly when to enter, how long to hold, and how not to churn to death”

### C. Over-triggering was a constant early failure mode
In the first honest holdouts:
- true event rate was around `0.049`
- predicted event rate exploded to `0.686` in `exp13`

That is a massive spam problem.
Even when later routes got calmer, the system remained highly sensitive to threshold/persistence choices.

### D. Good mapper metrics did not guarantee money
The cleanest example is `exp21` / `exp22`:
- mapper metrics looked materially better
- RMSE was not catastrophic
- strategy still lost hard

So the bottleneck moved from map learning to policy extraction.

### E. Direction seems regime-sensitive, maybe partially inverted
Short-side tests showed that a branch that looks bad or mediocre long-only can become excellent on a particular holdout.
But not consistently enough to say “just flip the sign globally.”

Current honest read:
- there may be real directional asymmetry / inversion on some regimes
- but it is not a simple constant transform
- threshold family and holdout regime both matter a lot

### F. Turnover/churn is a major bullshitting vector
Some routes generated absurd trade counts and terrible PnL:
- e.g. `exp25_l1_short_holdout_18` had strong-looking precision/F1 and good RMSE ratio, but PnL was catastrophically negative:
  - `q95_p1_hold48`: `-18139.1719` with `1728` trades
  - `q90_p1_hold48`: `-30872.7578` with `3823` trades

That is exactly why raw map metrics alone are not enough.
A strategy can be “right” often enough on event classification and still be mechanically terrible as a trader.

---

## Best current explanation for “why is the omniscient reference not trading well?”

Because we are still trying to compress this ladder:

1. opportunity exists in hindsight (`omniscient`)
2. model predicts some future-map structure
3. trigger logic turns that map into entries/exits
4. side choice matches regime
5. hold logic avoids churn and late exits
6. behavior survives adjacent holdouts

We have evidence for parts 1 and sometimes 2.
We have partial evidence for 3 on some holdouts.
We do **not** yet have stable evidence for 4–6.

So the reference is useful, but it is not a policy. It is just proof there was something to capture.

---

## Where the evidence currently points

### Things that look real
- event-window conditioning was worth doing
- large historical training was necessary
- cache reuse mattered operationally
- mapper branches like `l1_evt005_pw2_h64` and `regonly_wd1e3` are not random garbage
- short/reverse evaluation was the right thing to test

### Things that are still broken
- stable trigger extraction
- regime-robust direction choice
- threshold robustness across adjacent holdouts
- churn/friction control
- a single branch/route family that wins repeatedly without cheating itself

---

## Bottom line
The omniscient/reference PnL is not “failing” because the idea is stupid.
It is failing because it was only ever a reference ceiling.

What the experiments say so far is:
- there probably is some learnable signal in the future-map setup
- but the conversion from learned map -> stable tradable policy is still fragile as hell
- the biggest open problem is not just prediction quality anymore
- it is direction + trigger calibration + churn control + regime stability

If you want the blunt version:
- the model can sometimes smell the move
- it still often cannot trade it like an adult
