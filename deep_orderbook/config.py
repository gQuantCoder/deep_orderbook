import datetime
from pathlib import Path
import random
from typing import Self
from pydantic import BaseModel


class BaseConfig(BaseModel):
    # override given settings, returning a new instance modified
    def but(self, **kwargs) -> Self:
        return self.__class__(**{**self.model_dump(), **kwargs})


class FeedConfig(BaseConfig):
    markets: list[str] = ["ETH-USD", "BTC-USD", "ETH-BTC"]
    max_samples: int = -1
    freq: float = 1.0  # per seconds

    def only_first_market(self) -> list[str]:
        return self.markets[:1]


class ReplayConfig(FeedConfig):
    data_dir: Path = Path("data")
    date_regexp: str = "2024-09"
    one_path: Path | None = None
    skip_until_time: datetime.time = datetime.time(0, 0)
    every: str = "1s"

    def file_list(self) -> list[Path]:
        if self.one_path:
            return [self.one_path]
        return sorted(self.data_dir.glob(f"{self.date_regexp}*.parquet"))
    
    def num_files(self) -> int:
        return len(self.file_list())

    def but_random_file(self) -> Self:
        list_of_all_files = self.file_list()
        random_file = random.choice(list_of_all_files)
        return self.but(one_path=random_file)


class ShaperConfig(BaseConfig):
    zoom_frac: float = 0.004
    num_side_lvl: int = 32
    rolling_window_size: int = 256

    only_full_arrays: bool = False

    for_image_display: bool = False

    look_ahead: int = 8
    look_ahead_side_bips: int = 8
    look_ahead_side_width: int = 16


class TrainConfig(BaseConfig):
    device: str = "cuda"  # "cpu" or "cuda"
    epochs: int = 10
    learning_rate: float = 0.001

    num_workers: int = 1  # Number of data loading threads
    data_queue_size: int = 256  # Maximum number of items in the data queue
    num_levels: int = 4

    batch_size: int = 4
    criterion: str = "MSELoss"  # "MSELoss" or "L1Loss"


class Fullconfig(BaseConfig):
    replay: ReplayConfig = ReplayConfig()
    shaper: ShaperConfig = ShaperConfig()
    train: TrainConfig = TrainConfig()
