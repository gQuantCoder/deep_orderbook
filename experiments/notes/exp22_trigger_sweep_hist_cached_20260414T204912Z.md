# exp22_trigger_sweep_hist_cached

- timestamp: 2026-04-14 20:49:12Z
- device: `cuda`
- route_count: 40
- train_timesteps: 3045632
- test_timesteps: 388608
- runtime_seconds_total: 8607.494
- best_route: `regonly_wd1e3 / q95_p2_hold96`
- best pnl: -2423.72656
- best precision: 0.4485
- best f1: 0.6102
- best rmse_ratio: 0.9805

## top 10 routes
- 1. `regonly_wd1e3 / q95_p2_hold96` pnl=-2423.72656 precision=0.4485 f1=0.6102 rmse_ratio=0.9805 trades=237
- 2. `regonly_wd1e3 / q80_persistence2_wide_margin` pnl=-2423.72656 precision=0.4485 f1=0.6102 rmse_ratio=0.9805 trades=237
- 3. `regonly_huber_thr010 / q95_p2_hold96` pnl=-2585.86719 precision=0.4486 f1=0.6120 rmse_ratio=0.9749 trades=336
- 4. `regonly_huber_thr010 / q80_persistence2_wide_margin` pnl=-2585.86719 precision=0.4486 f1=0.6120 rmse_ratio=0.9749 trades=336
- 5. `precision_evt005_pw2_thr010 / q80_persistence2_wide_margin` pnl=-2988.56250 precision=0.4494 f1=0.6121 rmse_ratio=0.9825 trades=296
- 6. `precision_evt005_pw2_thr010 / q95_p2_hold96` pnl=-3009.69531 precision=0.4494 f1=0.6121 rmse_ratio=0.9825 trades=296
- 7. `l1_evt005_pw2_h64 / q95_p2_hold96` pnl=-3422.90625 precision=0.4866 f1=0.6198 rmse_ratio=0.9666 trades=222
- 8. `l1_evt005_pw2_h64 / q80_persistence2_wide_margin` pnl=-3422.90625 precision=0.4866 f1=0.6198 rmse_ratio=0.9666 trades=222
- 9. `regonly_huber_thr010 / q90_p2_hold48` pnl=-4955.10938 precision=0.4486 f1=0.6120 rmse_ratio=0.9749 trades=1111
- 10. `regonly_huber_thr010 / q90_p2_hold96` pnl=-4955.10938 precision=0.4486 f1=0.6120 rmse_ratio=0.9749 trades=1111
