import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from deep_orderbook.config import ReplayConfig, ShaperConfig, TrainConfig
from deep_orderbook.learn.trainer import Trainer
from deep_orderbook.learn.losses import StructuredT2LLoss


class _IdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keep shape, keep graph
        return x[:, :1] * self.scale


def test_structured_loss_is_zero_for_perfect_match() -> None:
    loss_fn = StructuredT2LLoss(
        base_loss=nn.MSELoss(),
        monotonic_weight=0.0,
        updown_rank_weight=0.0,
    )
    target = torch.rand(2, 1, 12, 8)
    pred = target.clone()

    loss = loss_fn(pred, target)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_structured_loss_penalizes_wrong_up_down_dominance() -> None:
    loss_fn = StructuredT2LLoss(
        base_loss=nn.MSELoss(),
        pointwise_weight=0.0,
        monotonic_weight=0.0,
        updown_rank_weight=1.0,
    )

    # Target has stronger upside than downside.
    target = torch.zeros(1, 1, 1, 8)
    target[..., 4:] = 0.9
    target[..., :4] = 0.1

    aligned = target.clone()
    inverted = torch.zeros_like(target)
    inverted[..., 4:] = 0.1
    inverted[..., :4] = 0.9

    loss_aligned = loss_fn(aligned, target)
    loss_inverted = loss_fn(inverted, target)

    assert loss_inverted > loss_aligned


def test_structured_loss_can_focus_only_on_last_step() -> None:
    loss_fn = StructuredT2LLoss(
        base_loss=nn.MSELoss(),
        updown_rank_weight=0.0,
        monotonic_weight=0.0,
        focus_last_step=True,
    )

    target = torch.zeros(1, 1, 2, 8)
    pred = target.clone()
    pred[:, :, 0, :] = 10.0  # large error on historical row

    loss = loss_fn(pred, target)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_trainer_builds_structured_loss_when_requested() -> None:
    model = _IdentityModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_config = TrainConfig(
        criterion="StructuredT2L",
        num_workers=0,
        batch_size=1,
    )
    replay_config = ReplayConfig()
    shaper_config = ShaperConfig()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        train_config=train_config,
        replay_config=replay_config,
        shaper_config=shaper_config,
    )

    assert isinstance(trainer.criterion, StructuredT2LLoss)
