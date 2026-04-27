# Active research core

This repo has a lot of historical experiment surface area. The active research loop is now intentionally narrower.

## Canonical live branch

- Mapper: `regonly_l1_2d_h8`
- Direction: `short`
- Cadence: `1000ms`
- Geometry: `rolling_window_size=1024`, `look_ahead=64`, `window_stride=4`
- Train pool: recent 24-file 2026 slice (`2026-04-16..2026-04-21` at `00/06/12/18`)
- Validation policy: chronological continuation only
- Cost metric: corrected friction accounting = `final_pnl - trade_count * cost`

## Why the active surface was cut

The bottleneck is no longer map generation. It is execution stability under friction.

So the active route family is cut to the four serious contenders:

1. `q95_p1_hold48` — strict fast
2. `q90_p1_hold48` — medium fast
3. `q90_p2_hold96` — persistence hold96
4. `q90_persistence2_fast_exit` — persistence foil with faster exit

Dropped from active continuation:
- `q80_p1_hold48`
- `q80_p2_hold48`
- `q80_p3_hold48`
- `q95_p2_hold96`
- `q80_persistence2_wide_margin`

Reason: these increased route-search noise faster than they increased tradable truth.

## Canonical continuation script

Use:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp63_active_core_recent24_short_continuation.py
```

What it does:
- auto-discovers completed holdouts from prior summary JSONs
- picks the next 2 fresh chronological `2026-04-23` holdouts
- runs only the active-core route family
- writes JSON, PNG, DB row, notes, journal updates

## What is now historical, not canonical

These scripts remain useful as evidence and prior art, but they should not be treated as the default active loop:
- `exp54`..`exp62` fixed-route continuation family
- `exp61` selector branch
- older architecture churn / broad route sweep scripts

## Promotion standard

A branch is only interesting if it survives:
- multiple fresh chronological holdouts
- corrected cost-5 friction
- without route-leadership flailing every hour

Until then, it stays `not promising yet`.
