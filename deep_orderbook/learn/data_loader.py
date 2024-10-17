# data_loader.py

import threading
import asyncio
import queue

class DataLoaderWorker:
    """Data loading worker that reads data from files and puts it into a queue."""

    def __init__(self, *, data_queue, replay_config, shaper_config):
        self.data_queue = data_queue
        self.replay_config = replay_config
        self.shaper_config = shaper_config

    def start(self):
        """Starts the data loading worker in a separate thread."""
        t = threading.Thread(target=self.run)
        t.daemon = True
        t.start()

    def run(self):
        """Worker function to load data and put it into the queue."""
        from deep_orderbook.shaper import iter_shapes_t2l

        while True:
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                rand_replay_config = self.replay_config.but_random_file()

                async def load_data():
                    async for books_array, time_levels, pxar in iter_shapes_t2l(
                        replay_config=rand_replay_config,
                        shaper_config=self.shaper_config,
                    ):
                        self.data_queue.put((books_array, time_levels, pxar))
                        # Optional sleep interval
                        # await asyncio.sleep(0.1)

                loop.run_until_complete(load_data())
            except Exception as e:
                print(f"Exception in data loading worker: {e}")
            finally:
                loop.close()


async def main():
    # Example usage of DataLoaderWorker
    from deep_orderbook.config import ReplayConfig, ShaperConfig

    # Create a queue for data
    data_queue = queue.Queue(maxsize=1000)

    # Define shaper configuration
    shaper_config = ShaperConfig(only_full_arrays=True)

    replay_config = ReplayConfig(date_regexp="2024-0")
    print(f"{replay_config.file_list()=}")

    # Initialize and start the data loader worker
    data_loader_worker = DataLoaderWorker(
        data_queue=data_queue, replay_config=replay_config, shaper_config=shaper_config
    )
    data_loader_worker.start()

    # Consume data from the queue
    try:
        while True:
            data = data_queue.get(timeout=30)
            books_array, time_levels, pxar = data
            print(
                f"Queue size: {data_queue.qsize()}, {books_array.shape=}, {time_levels.shape=}"
            )
            await asyncio.sleep(1)
    except queue.Empty:
        print("Data queue is empty.")


if __name__ == '__main__':
    asyncio.run(main())
