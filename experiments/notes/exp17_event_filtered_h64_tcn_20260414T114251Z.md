# exp16 batch h64 tcn search

- timestamp: 2026-04-14 11:42:51Z
- train files: 2026-04-14T07-00-00.parquet, 2026-04-14T08-00-00.parquet, 2026-04-14T09-00-00.parquet
- test file: `2026-04-14T10-00-00.parquet`
- precheck image: `experiments/pictures/exp17_event_filtered_h64_tcn_holdout_precheck_20260414T114251Z.png`
- summary json: `experiments/results/exp17_event_filtered_h64_tcn_20260414T114251Z.json`
- event filter: top_fraction=0.35
- selected train windows: 51 / 144
- selected test windows: 17 / 48
- older fallback files added: none
- top route: `l1_evt005_pw2` score=1.6740
- top metrics: precision=0.2957, f1=0.3285, rmse=1.14068, fixed pnl=-0.02344
- top image qc: usable=True, reason=ok, gray_std=0.3028

## ranked variants
- 1. `l1_evt005_pw2` score=1.6740 precision=0.2957 f1=0.3285 rmse=1.14068 fixed_pnl=-0.02344 image=ok
- 2. `regonly_huber_thr010` score=1.5543 precision=0.3208 f1=0.2310 rmse=1.13593 fixed_pnl=0.00000 image=ok
- 3. `precision_evt005_pw2_thr010` score=1.0693 precision=0.1472 f1=0.2477 rmse=1.09895 fixed_pnl=0.93750 image=ok
- 4. `regonly_activew3` score=0.9054 precision=0.1252 f1=0.2202 rmse=1.08061 fixed_pnl=-16.21094 image=ok
