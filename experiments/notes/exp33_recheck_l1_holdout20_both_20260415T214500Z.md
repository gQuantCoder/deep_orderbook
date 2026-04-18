# exp33_recheck_l1_holdout20_both

- timestamp: 2026-04-15 21:45:00Z
- device: `cuda`
- route_count: 20
- train_timesteps: 3827456
- test_timesteps: 383232
- runtime_seconds_total: 49.099
- best route by score: `l1_evt005_pw2_h64 / short / q80_persistence2_wide_margin`
- best score pnl: 149.83594
- best score precision: 0.2724
- best score f1: 0.4162
- best score rmse_ratio: 1.0309
- best route by raw pnl: `l1_evt005_pw2_h64 / long / q90_p1_hold48`
- best raw pnl: 1885.12500
- best raw pnl/trade: 1.71687

## top 10 routes
- 1. `l1_evt005_pw2_h64 / short / q80_persistence2_wide_margin` pnl=149.83594 pnl_per_trade=1.02627 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=146 score=-3.3959
- 2. `l1_evt005_pw2_h64 / short / q95_p2_hold96` pnl=-8.02344 pnl_per_trade=-0.08817 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=91 score=-3.8028
- 3. `l1_evt005_pw2_h64 / long / q90_p1_hold48` pnl=1885.12500 pnl_per_trade=1.71687 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=1098 score=-16.2972
- 4. `l1_evt005_pw2_h64 / short / q90_persistence2_fast_exit` pnl=-941.65625 pnl_per_trade=-3.67834 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=256 score=-30.7271
- 5. `l1_evt005_pw2_h64 / long / q95_p2_hold96` pnl=-1825.21094 pnl_per_trade=-16.90010 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=108 score=-40.9972
- 6. `l1_evt005_pw2_h64 / long / q95_p1_hold48` pnl=-1618.49219 pnl_per_trade=-4.50833 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=359 score=-49.4144
- 7. `l1_evt005_pw2_h64 / long / q80_persistence2_wide_margin` pnl=-2413.43750 pnl_per_trade=-21.94034 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=110 score=-52.8616
- 8. `l1_evt005_pw2_h64 / short / q95_p1_hold48` pnl=-1773.93750 pnl_per_trade=-4.82048 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=368 score=-52.9729
- 9. `l1_evt005_pw2_h64 / long / q90_p2_hold96` pnl=-1769.26562 pnl_per_trade=-4.69301 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=377 score=-53.3303
- 10. `l1_evt005_pw2_h64 / long / q90_p2_hold48` pnl=-1826.57812 pnl_per_trade=-4.84503 precision=0.2724 f1=0.4162 rmse_ratio=1.0309 trades=377 score=-54.4764
