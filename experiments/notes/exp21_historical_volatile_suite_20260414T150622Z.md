# exp16 batch h64 tcn search

- timestamp: 2026-04-14 15:06:22Z
- device: `cuda`
- train files: 2025-03-10T14-00-32.parquet, 2025-03-10T17-00-33.parquet, 2025-03-11T00-00-11.parquet, 2025-03-11T14-00-33.parquet, 2025-03-11T15-00-33.parquet, 2025-04-07T14-00-33.parquet, 2025-04-07T15-00-33.parquet, 2025-04-09T17-00-32.parquet
- test file: `2025-04-09T18-00-33.parquet`
- precheck image: `experiments/pictures/exp21_historical_volatile_suite_holdout_precheck_20260414T150622Z.png`
- summary json: `experiments/results/exp21_historical_volatile_suite_20260414T150622Z.json`
- runtime_seconds_total: 8639.414
- train windows after filter: 11897
- test windows after filter: 1518
- train timesteps: 3045632
- test timesteps: 388608
- max_windows_per_file: None
- event filter: top_fraction=0.35
- selected train windows: 11897 / 33989
- selected test windows: 1518 / 4336
- older fallback files added: none
- top route: `l1_evt005_pw2_h64` score=2.7534
- top metrics: precision=0.4823, f1=0.6133, rmse=6.51771, fixed pnl=-201.07812
- top image qc: usable=True, reason=ok, gray_std=0.3061

## ranked variants
- 1. `l1_evt005_pw2_h64` score=2.7534 precision=0.4823 f1=0.6133 rmse=6.51771 fixed_pnl=-201.07812 image=ok
- 2. `precision_evt005_pw2_thr010` score=2.6599 precision=0.4526 f1=0.6111 rmse=6.62238 fixed_pnl=-288.71094 image=ok
- 3. `regonly_wd1e3` score=2.6582 precision=0.4516 f1=0.6116 rmse=6.59858 fixed_pnl=-320.35156 image=ok
- 4. `baseline_evt025_pw8_lr2e3_h96_e6` score=2.6548 precision=0.4509 f1=0.6110 rmse=6.60214 fixed_pnl=-215.34375 image=ok
- 5. `regonly_huber_thr010` score=2.6512 precision=0.4498 f1=0.6108 rmse=6.60157 fixed_pnl=-406.16406 image=ok
- 6. `regonly_activew3` score=2.6304 precision=0.4427 f1=0.6125 rmse=6.71450 fixed_pnl=-210.46875 image=ok
