import numpy as np
import torch


def generate_relative_polar_position_dataset(
        controller,
        num_samples: int = 1024,
        num_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, positions = generate_trajectory_tensors(
        controller=controller,
        num_samples=num_samples,
        num_steps=num_steps
    )

    relative_positions = calculate_relative_polar_position(positions)

    return features, positions, relative_positions


def generate_trajectory_tensors(
    controller,
    num_samples: int = 1024,
    num_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    trajectories = np.empty(
        (num_samples, num_steps, 4),
        dtype=np.float32,
    )

    for sample_id in range(num_samples):
        trajectory = controller.generate(num_steps)

        trajectories[sample_id] = np.asarray(
            [
                (
                    record["theta"],
                    record["speed"],
                    record["x"],
                    record["y"],
                )
                for record in trajectory
            ],
            dtype=np.float32,
        )

    features = torch.from_numpy(trajectories[:, :, :2])
    positions = torch.from_numpy(trajectories[:, :, 2:])

    return features, positions


def calculate_relative_polar_position(
        positions: torch.Tensor,
        ord: int = 2,
        keepdim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm(
        positions, ord=ord, dim=-1, keepdim=keepdim,
    )

    angles = torch.atan2(
        positions[:, :, 1], positions[:, :, 0],
    ).unsqueeze(dim=2)

    relative_positions = torch.cat(
        [angles, distances],
        dim=-1,
    )

    return relative_positions


def generate_cumulative_distance_dataset(
        controller,
        num_samples: int = 1000,
        num_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, positions = generate_trajectory_tensors(
        controller=controller,
        num_samples=num_samples,
        num_steps=num_steps
    )
    
    step_distances = calculate_step_distances(positions)
    cumulative_distances = torch.cumsum(step_distances, dim=1)

    return features, cumulative_distances


def calculate_step_distances(
        positions: torch.Tensor,
        keepdim: bool = True,
        prepend_zero: bool = True,
) -> torch.Tensor:

    displacements = torch.diff(positions, dim=1)

    distances = torch.linalg.vector_norm(
        displacements,
        ord=2,
        dim=-1,
        keepdim=keepdim,
    )

    if prepend_zero:
        zero_distance = torch.zeros_like(distances[:, :1])
        distances = torch.cat([zero_distance, distances], dim=1)

    return distances