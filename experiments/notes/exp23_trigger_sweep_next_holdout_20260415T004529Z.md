# exp23_trigger_sweep_next_holdout

- timestamp: 2026-04-15 00:45:29Z
- device: `cuda`
- route_count: 40
- train_timesteps: 3433984
- test_timesteps: 393472
- runtime_seconds_total: 937.090
- best_route: `regonly_wd1e3 / q95_p2_hold96`
- best pnl: 907.49219
- best precision: 0.3235
- best f1: 0.4868
- best rmse_ratio: 1.0133

## top 10 routes
- 1. `regonly_wd1e3 / q95_p2_hold96` pnl=907.49219 precision=0.3235 f1=0.4868 rmse_ratio=1.0133 trades=156
- 2. `regonly_wd1e3 / q80_persistence2_wide_margin` pnl=809.43750 precision=0.3235 f1=0.4868 rmse_ratio=1.0133 trades=163
- 3. `l1_evt005_pw2_h64 / q95_p2_hold96` pnl=545.88281 precision=0.3560 f1=0.5045 rmse_ratio=0.9947 trades=66
- 4. `l1_evt005_pw2_h64 / q80_persistence2_wide_margin` pnl=409.97656 precision=0.3560 f1=0.5045 rmse_ratio=0.9947 trades=66
- 5. `l1_evt005_pw2_h64 / q95_p1_hold48` pnl=341.50781 precision=0.3560 f1=0.5045 rmse_ratio=0.9947 trades=417
- 6. `l1_evt005_pw2_h64 / q90_p2_hold96` pnl=455.93750 precision=0.3560 f1=0.5045 rmse_ratio=0.9947 trades=471
- 7. `l1_evt005_pw2_h64 / q90_p2_hold48` pnl=320.03125 precision=0.3560 f1=0.5045 rmse_ratio=0.9947 trades=471
- 8. `l1_evt005_pw2_h64 / q90_persistence2_fast_exit` pnl=107.60938 precision=0.3560 f1=0.5045 rmse_ratio=0.9947 trades=471
- 9. `precision_evt005_pw2_thr010 / q95_p2_hold96` pnl=-1074.26562 precision=0.3274 f1=0.4892 rmse_ratio=1.0417 trades=50
- 10. `precision_evt005_pw2_thr010 / q80_persistence2_wide_margin` pnl=-2016.42969 precision=0.3274 f1=0.4892 rmse_ratio=1.0417 trades=111
