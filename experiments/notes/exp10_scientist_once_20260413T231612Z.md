# exp10 scientist once

- timestamp: 2026-04-13 23:16:12Z
- parquet: `/mnt/data/repos/gaelreinaudi/crypto/2026-04-13T22-51-51.parquet`
- hypothesis: A visually rich recent 100ms/h64 slice may support a quick TinyTCN learning pass with non-random map and trade structure.
- precheck image: `experiments/pictures/exp10_scientist_once_precheck_20260413T231612Z.png`
- dashboard image: `experiments/pictures/exp10_scientist_once_dashboard_20260413T231612Z.png`
- result json: `experiments/results/exp10_scientist_once_20260413T231612Z.json`
- observations:
  - Richness gate passed: books_std=2.7849, target_active_ratio=0.440918
  - TinyTCN quick run metrics: f1=0.5081, precision=0.4768, recall=0.5438, rmse=0.20647
  - Zero baseline rmse=0.20622
  - Final omniscient pnl=12.09033, prediction pnl=0.76025
- decision: not_promising_yet
- next mutation: if dashboard still looks weak, try the same run on the freshest rolled parquet from `/mnt/data/repos/gaelreinaudi/crypto/` after recorder rollover or switch to BTC-USD fresh slice.
