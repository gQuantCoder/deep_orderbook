import asyncio
import queue
import random
import threading
from contextlib import aclosing
from pathlib import Path

import numpy as np

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.utils import logger


class DataLoaderWorker:
    """Streams windows from every matched file.

    ``replay_config.randomize`` is honored inside ``iter_shapes_t2l`` (reservoir shuffle).
    Each cycle walks the full ``replay_config.file_list()`` — shuffled per cycle when
    ``randomize`` is True — so cross-file diversity emerges quickly.
    """

    def __init__(
        self,
        *,
        data_queue: queue.Queue,
        replay_config: ReplayConfig,
        shaper_config: ShaperConfig,
    ) -> None:
        self.data_queue = data_queue
        self.replay_config = replay_config
        self.shaper_config = shaper_config
        self._running = True
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"[DataLoader] worker thread did not exit within {timeout}s")

    async def _stream_file(self, file_path: Path) -> None:
        from deep_orderbook.shaper import iter_shapes_t2l

        cfg = self.replay_config.but(one_path=file_path)
        stream = iter_shapes_t2l(replay_config=cfg, shaper_config=self.shaper_config)
        async with aclosing(stream) as guarded:
            async for item in guarded:
                while self._running:
                    try:
                        self.data_queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if not self._running:
                    return
                await asyncio.sleep(0)

    def _file_order(self) -> list:
        files = list(self.replay_config.file_list())
        if self.replay_config.randomize:
            random.shuffle(files)
        return files

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while self._running:
                files = self._file_order()
                if not files:
                    logger.error(
                        f"[DataLoader] no files match {self.replay_config.date_regexp!r} "
                        f"in {self.replay_config.data_dir} — worker exiting"
                    )
                    return
                for f in files:
                    if not self._running:
                        break
                    try:
                        loop.run_until_complete(self._stream_file(f))
                    except Exception as e:  # noqa: BLE001 — worker must survive bad files
                        logger.error(f"[DataLoader] {f}: {e}")
        finally:
            loop.close()


async def main() -> None:
    # Example usage of DataLoaderWorker
    from deep_orderbook.config import ReplayConfig, ShaperConfig

    replay_config = ReplayConfig(date_regexp="2024-0")
    shaper_config = ShaperConfig(only_full_arrays=False)
    print(f"{replay_config.file_list()=}")

    # Create a queue for data
    data_queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray]] = queue.Queue(maxsize=1000)

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
                    f"Queue size: {qs}, books: {books_array.shape}, t2l: {time_levels.shape} pxar: {pxar.shape}"
                )
            await asyncio.sleep(0.1)
    except queue.Empty:
        print("Data queue is empty.")


if __name__ == '__main__':
    asyncio.run(main())
