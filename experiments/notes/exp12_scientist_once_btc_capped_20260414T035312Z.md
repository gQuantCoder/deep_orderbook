# exp12 scientist once btc capped

- timestamp: 2026-04-14 03:53:12Z
- parquet: `/mnt/data/repos/gaelreinaudi/crypto/2026-04-14T02-00-01.parquet`
- symbol: `BTC-USD`
- mutation: prediction amplitude cap at train q99.5
- hypothesis: BTC showed interesting raw structure but catastrophic dense-map error; capping prediction amplitude to a train-derived percentile may preserve useful pattern timing while controlling blow-up.
- precheck image: `experiments/pictures/exp12_scientist_once_btc_capped_precheck_20260414T035312Z.png`
- dashboard image: `experiments/pictures/exp12_scientist_once_btc_capped_dashboard_20260414T035312Z.png`
- result json: `experiments/results/exp12_scientist_once_btc_capped_20260414T035312Z.json`
- observations:
  - Richness gate passed: books_std=2.4195, target_active_ratio=0.122070
  - Applied prediction cap at train q99.5=7.7531
  - TinyTCN quick run metrics: f1=0.1201, precision=0.0639, recall=1.0000, rmse=0.51082
  - Zero baseline rmse=0.84282
  - Final omniscient pnl=55.92188, prediction pnl=48.21875
- decision: not_promising_yet
- next mutation: if dashboard still looks weak, try the same run on the freshest rolled parquet from `/mnt/data/repos/gaelreinaudi/crypto/` after recorder rollover or switch to BTC-USD fresh slice.
