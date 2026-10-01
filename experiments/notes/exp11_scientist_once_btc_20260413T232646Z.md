# exp11 scientist once btc

- timestamp: 2026-04-13 23:26:46Z
- parquet: `/mnt/data/repos/gaelreinaudi/crypto/2026-04-13T22-51-51.parquet`
- symbol: `BTC-USD`
- hypothesis: After the weak ETH run, switching only the symbol to BTC-USD on the same fresh parquet may reveal cleaner map structure and more tradable behavior.
- precheck image: `experiments/pictures/exp11_scientist_once_btc_precheck_20260413T232646Z.png`
- dashboard image: `experiments/pictures/exp11_scientist_once_btc_dashboard_20260413T232646Z.png`
- result json: `experiments/results/exp11_scientist_once_btc_20260413T232646Z.json`
- observations:
  - Richness gate passed: books_std=2.3890, target_active_ratio=0.418457
  - TinyTCN quick run metrics: f1=0.5873, precision=0.4180, recall=0.9872, rmse=1596.30530
  - Zero baseline rmse=6.65414
  - Final omniscient pnl=163.57812, prediction pnl=45.67188
- decision: not_promising_yet
- next mutation: the BTC symbol improved visual structure and raw pnl, but RMSE exploded relative to zero baseline. Next try should keep BTC and change one factor that controls output amplitude / calibration rather than treating this as a win.
