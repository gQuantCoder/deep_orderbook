import asyncio
from deep_orderbook.config import FeedConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l
from deep_orderbook.visu import Visualizer
from deep_orderbook.strategy import Strategy
from deep_orderbook.utils import make_handlers, logger

def setup_config() -> tuple[FeedConfig, ShaperConfig]:
    feed_conf = FeedConfig(
        markets=["ETH-USD"],  # "BTC-USD", "ETH-BTC"
        freq=10.0,
    )
    shaper_config = ShaperConfig(
        only_full_arrays=False,
        view_bips=20,
        num_side_lvl=8,
        look_ahead_side_bips=10,
        look_ahead_side_width=4,
        rolling_window_size=256,
    )
    return feed_conf, shaper_config

async def rolling_plot(config: FeedConfig, shaper_config: ShaperConfig) -> None:
    # vis = Visualizer()
    strategy = Strategy(threshold=0.2)
    logger.info("Starting rolling plot")
    
    try:
        async for books_array, level_prox, pxar in iter_shapes_t2l(
            replay_config=config,
            shaper_config=shaper_config,
            live=True,
        ):
            gt_pnl, pos, gt_up_prox, gt_down_prox = strategy.compute_pnl(pxar, level_prox)
            # vis.update(
            #     books_z_data=books_array,
            #     level_reach_z_data=level_prox,
            #     bidask=pxar,
            #     gt_pnl=gt_pnl,
            #     positions=pos,
            # )
    except Exception as e:
        logger.error(f"Error in rolling plot: {str(e)}")
        raise

def main() -> None:
    logger.setLevel('DEBUG')
    line_handler, noline_handler = make_handlers('live.log')
    logger.addHandler(line_handler)
    
    feed_conf, shaper_config = setup_config()
    try:
        asyncio.run(rolling_plot(feed_conf, shaper_config))
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 