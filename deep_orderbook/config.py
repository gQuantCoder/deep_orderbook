import datetime
from pathlib import Path
import random
from typing import Self, Optional
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
    every: str = "1000ms"
    randomize: bool = False

    def file_list(self) -> list[Path]:
        if self.one_path:
            return [self.one_path]
        filename_regexp = f"{self.date_regexp}.parquet"
        # print(f"Searching for {filename_regexp} in {self.data_dir}")
        return sorted(self.data_dir.glob(filename_regexp))
    
    def num_files(self) -> int:
        return len(self.file_list())

    def but_random_file(self) -> Self:
        list_of_all_files = self.file_list()
        random_file = random.choice(list_of_all_files)
        return self.but(one_path=random_file)


class ShaperConfig(BaseConfig):
    view_bips: int = 20
    num_side_lvl: int = 8
    rolling_window_size: int = 256
    window_stride: int = 1  # How many steps to slide the window by (default: 32 for 1/8th overlap)

    only_full_arrays: bool = False

    for_image_display: bool = False

    look_ahead: int = 32
    look_ahead_side_bips: int = 10
    look_ahead_side_width: int = 4

    use_cache: bool = True
    save_cache: bool = True

class TrainConfig(BaseConfig):
    device: str = "cuda"  # "cpu" or "cuda"
    epochs: int = 10
    learning_rate: float = 0.001

    num_workers: int = 1  # Number of data loading threads
    data_queue_size: int = 256  # Maximum number of items in the data queue
    num_levels: int = 4

    batch_size: int = 4
    criterion: str = "MSELoss"  # "MSELoss" or "L1Loss"
    
    # Checkpoint settings
    checkpoint_dir: Path = Path("checkpoints")  # Directory to save checkpoints
    save_checkpoint_batches: int = 100  # Save checkpoint every N batches
    save_checkpoint_mins: float = 5.0  # Minimum time (minutes) between checkpoints
    keep_last_n_checkpoints: int = 5  # Number of most recent checkpoints to keep


class CacheConfig(BaseConfig):
    enabled: bool = True
    cache_dir: Path = Path("cache")
    max_age_days: Optional[int] = 7  # Auto-clear cache files older than this


class Fullconfig(BaseConfig):
    replay: ReplayConfig = ReplayConfig()
    shaper: ShaperConfig = ShaperConfig()
    train: TrainConfig = TrainConfig()
    cache: CacheConfig = CacheConfig()
