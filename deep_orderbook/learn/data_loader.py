# data_loader.py

import multiprocessing
import asyncio
from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.utils import logger

multiprocessing.set_start_method('spawn', force=True)

class DataLoaderWorker:
    """Data loading worker that reads data from files and puts it into a queue."""

    def __init__(
        self, *, data_queue, replay_config: ReplayConfig, shaper_config: ShaperConfig
    ):
        self.data_queue = data_queue
        self.replay_config = replay_config
        self.shaper_config = shaper_config

    def start(self):
        """Starts the data loading worker in a separate process."""
        p = multiprocessing.Process(target=self.run)
        p.start()
        return p

    def run(self):
        """Worker function to load data and put it into the queue."""
        from deep_orderbook.shaper import iter_shapes_t2l

        while True:
            try:
                rand_replay_config = self.replay_config.but_random_file()
                logger.warning(f"Loading data from {rand_replay_config.file_list()}")

                # Create a new event loop for this process
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def load_data():
                    async for books_array, time_levels, pxar in iter_shapes_t2l(
                        replay_config=rand_replay_config,
                        shaper_config=self.shaper_config,
                    ):
                        try:
                            self.data_queue.put(
                                (books_array, time_levels, pxar),
                            )
                        except Exception as e:
                            logger.error(f"Failed to put data in queue: {e}")
                    logger.info(f"load_data completed")

                loop.run_until_complete(load_data())
                logger.info(
                    f"Data loading completed for {rand_replay_config.file_list()}"
                )
            except Exception as e:
                logger.error(f"Exception in data loading worker: {e}")
                print(
                    f"Exception in data loading worker: {e}\n with :{rand_replay_config.file_list()}"
                )
            finally:
                loop.close()
