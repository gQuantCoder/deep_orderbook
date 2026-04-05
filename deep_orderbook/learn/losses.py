import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredT2LLoss(nn.Module):
    """Field-preserving loss for time-to-level prediction maps.

    The target/pred tensor contract is expected as:
    [batch, channels=1, time, levels]

    levels are split in two halves:
    - first half: down-side proximity map
    - second half: up-side proximity map
    """

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        pointwise_weight: float = 1.0,
        updown_rank_weight: float = 0.25,
        monotonic_weight: float = 0.10,
        rank_margin: float = 0.05,
        focus_last_step: bool = False,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss()
        self.pointwise_weight = pointwise_weight
        self.updown_rank_weight = updown_rank_weight
        self.monotonic_weight = monotonic_weight
        self.rank_margin = rank_margin
        self.focus_last_step = focus_last_step

    def _split_sides(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        level_count = x.shape[-1]
        if level_count % 2 != 0:
            raise ValueError(f"Expected even number of levels, got {level_count}")
        side = level_count // 2
        down = x[..., :side]
        up = x[..., side:]
        return down, up

    def _updown_rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_down, pred_up = self._split_sides(pred)
        tgt_down, tgt_up = self._split_sides(target)

        pred_diff = pred_up.max(dim=-1).values - pred_down.max(dim=-1).values
        tgt_diff = tgt_up.max(dim=-1).values - tgt_down.max(dim=-1).values

        sign = torch.sign(tgt_diff)
        informative = sign != 0
        if not informative.any():
            return pred_diff.new_tensor(0.0)

        signed_margin = sign[informative] * pred_diff[informative]
        return F.relu(self.rank_margin - signed_margin).mean()

    def _monotonicity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        pred_down, pred_up = self._split_sides(pred)

        # Convert both sides to near->far order.
        down_near_far = torch.flip(pred_down, dims=[-1])
        up_near_far = pred_up

        # Proximity should not increase with distance from the current price.
        down_violation = F.relu(down_near_far[..., 1:] - down_near_far[..., :-1])
        up_violation = F.relu(up_near_far[..., 1:] - up_near_far[..., :-1])

        return 0.5 * (down_violation.mean() + up_violation.mean())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match: {pred.shape=} {target.shape=}"
            )

        if self.focus_last_step:
            pred = pred[:, :, -1:, :]
            target = target[:, :, -1:, :]

        total = pred.new_tensor(0.0)

        if self.pointwise_weight > 0:
            total = total + self.pointwise_weight * self.base_loss(pred, target)

        if self.updown_rank_weight > 0:
            total = total + self.updown_rank_weight * self._updown_rank_loss(pred, target)

        if self.monotonic_weight > 0:
            total = total + self.monotonic_weight * self._monotonicity_loss(pred)

        return total
