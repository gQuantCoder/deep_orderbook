import datetime
from pathlib import Path
from typing import Self
from pydantic import BaseModel


class BaseConfig(BaseModel):
    # override given settings, returning a new instance modified
    def but(self, **kwargs) -> Self:
        return self.__class__(**{**self.model_dump(), **kwargs})


class FeedConfig(BaseConfig):
    markets: list[str] = ["ETH-BTC", "BTC-USD", "ETH-USD"]
    max_samples: int = -1

    def only_first_market(self) -> list[str]:
        return self.markets[:1]


class ReplayConfig(FeedConfig):
    data_dir: Path = Path("data")
    date_regexp: str = "2024-08-05"
    skip_until_time: datetime.time = datetime.time(0, 0)


class ShaperConfig(BaseConfig):
    zoom_frac: float = 0.004
    side_bips: int = 100
    side_width: int = 50

    time_accumulate: int = 256
    look_ahead: int = 64


class TrainConfig(BaseConfig):
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    device: str = "cuda"  # "cpu" or "cuda"
    criterion: str = "MSELoss"  # "MSELoss" or "L1Loss"


class Fullconfig(BaseConfig):
    replay: ReplayConfig = ReplayConfig()
    shaper: ShaperConfig = ShaperConfig()
    train: TrainConfig = TrainConfig()
