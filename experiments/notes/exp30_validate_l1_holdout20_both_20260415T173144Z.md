# exp30_validate_l1_holdout20_both

- timestamp: 2026-04-15 17:31:44Z
- device: `cuda`
- route_count: 20
- train_timesteps: 3827456
- test_timesteps: 383232
- runtime_seconds_total: 48.240
- best route by score: `l1_evt005_pw2_h64 / short / q95_p2_hold96`
- best score pnl: 516.62500
- best score precision: 0.2697
- best score f1: 0.4139
- best score rmse_ratio: 1.0303
- best route by raw pnl: `l1_evt005_pw2_h64 / long / q80_p1_hold48`
- best raw pnl: 3180.20312
- best raw pnl/trade: 0.85169

## top 10 routes
- 1. `l1_evt005_pw2_h64 / short / q95_p2_hold96` pnl=516.62500 pnl_per_trade=5.01578 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=103 score=6.0833
- 2. `l1_evt005_pw2_h64 / short / q80_persistence2_wide_margin` pnl=429.56250 pnl_per_trade=2.73607 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=157 score=1.6417
- 3. `l1_evt005_pw2_h64 / short / q90_persistence2_fast_exit` pnl=747.09375 pnl_per_trade=2.19089 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=341 score=-1.2090
- 4. `l1_evt005_pw2_h64 / short / q90_p2_hold48` pnl=297.52344 pnl_per_trade=0.87250 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=341 score=-10.2008
- 5. `l1_evt005_pw2_h64 / short / q90_p2_hold96` pnl=289.09375 pnl_per_trade=0.84778 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=341 score=-10.3694
- 6. `l1_evt005_pw2_h64 / short / q95_p1_hold48` pnl=-402.38281 pnl_per_trade=-1.07017 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=376 score=-25.9483
- 7. `l1_evt005_pw2_h64 / long / q80_persistence2_wide_margin` pnl=-1184.31250 pnl_per_trade=-8.90461 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=133 score=-29.4365
- 8. `l1_evt005_pw2_h64 / long / q95_p1_hold48` pnl=-939.94531 pnl_per_trade=-2.77270 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=339 score=-34.8502
- 9. `l1_evt005_pw2_h64 / short / q80_p3_hold48` pnl=-860.98438 pnl_per_trade=-1.72888 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=498 score=-41.2224
- 10. `l1_evt005_pw2_h64 / long / q95_p2_hold96` pnl=-2292.03906 pnl_per_trade=-19.75896 precision=0.2697 f1=0.4139 rmse_ratio=1.0303 trades=116 score=-50.7408

## friction sanity table (top 3 routes)
- 1. `l1_evt005_pw2_h64 / short / q95_p2_hold96` raw_pnl=516.62500 trades=103 pnl_per_trade=5.01578 adj@1=413.62500 adj@2=310.62500 adj@5=1.62500 adj@10=-513.37500
- 2. `l1_evt005_pw2_h64 / short / q80_persistence2_wide_margin` raw_pnl=429.56250 trades=157 pnl_per_trade=2.73607 adj@1=272.56250 adj@2=115.56250 adj@5=-355.43750 adj@10=-1140.43750
- 3. `l1_evt005_pw2_h64 / short / q90_persistence2_fast_exit` raw_pnl=747.09375 trades=341 pnl_per_trade=2.19089 adj@1=406.09375 adj@2=65.09375 adj@5=-957.90625 adj@10=-2662.90625
