# exp24_reverse_short_holdout

- timestamp: 2026-04-15 02:55:17Z
- device: `cuda`
- route_count: 40
- train_timesteps: 3433984
- test_timesteps: 393472
- runtime_seconds_total: 125.886
- best_route: `l1_evt005_pw2_h64 / q80_p2_hold48`
- best pnl: 11732.97656
- best precision: 0.3949
- best f1: 0.5420
- best rmse_ratio: 0.9865

## top 10 routes
- 1. `l1_evt005_pw2_h64 / q80_p2_hold48` pnl=11732.97656 precision=0.3949 f1=0.5420 rmse_ratio=0.9865 trades=1621
- 2. `l1_evt005_pw2_h64 / q80_p3_hold48` pnl=9739.49219 precision=0.3949 f1=0.5420 rmse_ratio=0.9865 trades=1027
- 3. `regonly_huber_thr010 / q90_p1_hold48` pnl=4159.16406 precision=0.3615 f1=0.5269 rmse_ratio=1.0387 trades=1060
- 4. `regonly_huber_thr010 / q80_p2_hold48` pnl=5776.96875 precision=0.3615 f1=0.5269 rmse_ratio=1.0387 trades=1808
- 5. `regonly_huber_thr010 / q95_p1_hold48` pnl=1537.57812 precision=0.3615 f1=0.5269 rmse_ratio=1.0387 trades=218
- 6. `regonly_huber_thr010 / q90_p2_hold48` pnl=1641.60938 precision=0.3615 f1=0.5269 rmse_ratio=1.0387 trades=294
- 7. `regonly_huber_thr010 / q90_p2_hold96` pnl=1641.60938 precision=0.3615 f1=0.5269 rmse_ratio=1.0387 trades=294
- 8. `precision_evt005_pw2_thr010 / q95_p2_hold96` pnl=956.81250 precision=0.3636 f1=0.5286 rmse_ratio=1.0301 trades=37
- 9. `regonly_huber_thr010 / q90_persistence2_fast_exit` pnl=1018.67969 precision=0.3615 f1=0.5269 rmse_ratio=1.0387 trades=294
- 10. `precision_evt005_pw2_thr010 / q80_persistence2_wide_margin` pnl=234.31250 precision=0.3636 f1=0.5286 rmse_ratio=1.0301 trades=66
