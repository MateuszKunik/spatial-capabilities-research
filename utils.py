import numpy as np
import torch


def extract_coordinates(
    simulation_data: list[dict],
) -> np.ndarray:
    return np.asarray(
        [
            [record["x"], record["y"]]
            for record in simulation_data
        ],
        dtype=np.float32,
    )


def torch_to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().float().numpy()

    return np.asarray(array)