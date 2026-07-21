import math
import torch
import torch.nn as nn

class NormalizedMSELoss(nn.Module):
    def __init__(
        self,
        arena_size: float,
        angle_weight: float = 1.0,
        distance_weight:  float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be one of: 'mean', 'sum', 'none'."
            )
        
        self.register_buffer(
            "distance_scale",
            torch.tensor(
                math.sqrt(0.5) * arena_size,
                dtype=torch.float32,
            )
        )
        self.angle_weight = angle_weight
        self.distance_weight = distance_weight
        self.reduction = reduction
        self.eps = eps

    def _calculate_angle_loss(self, targets, predictions):
        wrapped_angle = self._wrap_angle(targets - predictions) 
        normalized_error = wrapped_angle / torch.pi
        squared_error = normalized_error ** 2

        return self._apply_reduction(squared_error)
    
    @staticmethod
    def _wrap_angle(angle):
        return torch.atan2(
            torch.sin(angle), torch.cos(angle)
        )

    def _calculate_distance_loss(self, targets, predictions):
        normalized_error = (
            targets - predictions
            ) / (self.distance_scale + self.eps)
        squared_error = normalized_error ** 2

        return self._apply_reduction(squared_error) 

    def _apply_reduction(self, error):
        if self.reduction == "mean":
            return torch.mean(error)
        
        if self.reduction == "sum":
            return torch.sum(error)
        
        return error

    def forward(self, targets, predictions):
        angle_loss = self._calculate_angle_loss(
            targets[:, :, 0],
            predictions[:, :, 0]
        )
        distance_loss = self._calculate_distance_loss(
            targets[:, :, 1],
            predictions[:, :, 1]
        )

        return (
            self.angle_weight * angle_loss
            + self.distance_weight * distance_loss
        )
