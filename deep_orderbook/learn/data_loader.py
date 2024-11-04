# data_loader.py

import threading
import asyncio
import queue
from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.utils import logger


class DataLoaderWorker:
    """Data loading worker that reads data from files and puts it into a queue."""

    def __init__(
        self, *, data_queue, replay_config: ReplayConfig, shaper_config: ShaperConfig
    ):
        self.data_queue = data_queue
        self.replay_config = replay_config
        self.shaper_config = shaper_config

    def start(self):
        """Starts the data loading worker in a separate thread."""
        t = threading.Thread(target=self.run)
        # t.daemon = True
        t.start()

    def run(self):
        """Worker function to load data and put it into the queue."""
        from deep_orderbook.shaper import iter_shapes_t2l

        while True:
            try:
                rand_replay_config = self.replay_config.but_random_file()
                logger.warning(f"Loading data from {rand_replay_config.file_list()}")

                # Create a new event loop for this thread
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
                                # block=False,
                            )
                        except queue.Full:
                            pass
                        # Optional sleep interval
                        # await asyncio.sleep(0.1)
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


async def main():
    # Example usage of DataLoaderWorker
    from deep_orderbook.config import ReplayConfig, ShaperConfig

    replay_config = ReplayConfig(date_regexp="2024-0")
    shaper_config = ShaperConfig(only_full_arrays=False)
    print(f"{replay_config.file_list()=}")

    # Create a queue for data
    data_queue = queue.Queue(maxsize=1000)

    # Initialize and start the data loader worker
    data_loader_worker = DataLoaderWorker(
        data_queue=data_queue, replay_config=replay_config, shaper_config=shaper_config
    )
    num_workers = 4
    for _ in range(num_workers):
        data_loader_worker = DataLoaderWorker(
            data_queue=data_queue,
            replay_config=replay_config,
            shaper_config=shaper_config,
        )
        data_loader_worker.start()

    # Consume data from the queue
    try:
        while True:
            data = data_queue.get(timeout=30)
            books_array, time_levels, pxar = data
            if qs := data_queue.qsize():
                logger.info(
                    f"Queue size: {qs}, books: {books_array.shape}, t2l: {time_levels.shape}"
                )
            await asyncio.sleep(0.1)
    except queue.Empty:
        print("Data queue is empty.")


if __name__ == '__main__':
    asyncio.run(main())
