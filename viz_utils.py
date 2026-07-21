import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from utils import torch_to_numpy


def plot_square_arena(
    size: float,
    center: tuple[float, float],
    color: str = "black",
    fill_alpha: float = 0.2,
    line_width: float = 1.5,
):
    half_size = size / 2
    x_center, y_center = center

    x_coordinates = [
        x_center - half_size,
        x_center + half_size,
        x_center + half_size,
        x_center - half_size,
        x_center - half_size,
    ]
    y_coordinates = [
        y_center - half_size,
        y_center - half_size,
        y_center + half_size,
        y_center + half_size,
        y_center - half_size,
    ]

    plt.fill(
        x_coordinates,
        y_coordinates,
        color=color,
        alpha=fill_alpha,
    )

    plt.plot(
        x_coordinates,
        y_coordinates,
        color=color,
        linewidth=line_width,
    )


def plot_simulation(
    data,
    arena,
    figsize: tuple[float, float] = (6, 6),
    arena_color: str = "black",
    arena_fill_alpha: float = 0.2,
    point_color: str = "black",
    point_cmap: str | None = None,
    point_size: float = 30,
    point_edge_color: str | None = None,
    trajectory_color: str = "black",
    show_trajectory: bool = True,
    trajectory_line_width: float = 1.0,
    show_colorbar: bool = False,
):
    plt.figure(figsize=figsize)

    plot_square_arena(
        size=arena.size,
        center=arena.center(),
        color=arena_color,
        fill_alpha=arena_fill_alpha,
    )

    x_coordinates = data[:, 0]
    y_coordinates = data[:, 1]
    steps = np.arange(len(data))

    if show_trajectory:
        plt.plot(
            x_coordinates,
            y_coordinates,
            color=trajectory_color,
            linewidth=trajectory_line_width,
            alpha=0.8,
            zorder=2,
        )

    if point_cmap is None:
        scatter = plt.scatter(
            x=x_coordinates,
            y=y_coordinates,
            color=point_color,
            s=point_size,
            edgecolors=point_edge_color,
            zorder=3,
        )
    else:
        scatter = plt.scatter(
            x=x_coordinates,
            y=y_coordinates,
            c=steps,
            cmap=point_cmap,
            s=point_size,
            edgecolors=point_edge_color,
            zorder=3,
        )

        if show_colorbar:
            colorbar = plt.colorbar(scatter)
            colorbar.set_label("Step")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Simulated trajectory in the arena")
    plt.show()


def plot_sample(
    targets: torch.Tensor | np.ndarray,
    predictions: torch.Tensor | np.ndarray,
    sample_id: int,
    arena_size: float,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    angle_color: str = "tab:blue",
    distance_color: str = "tab:orange",
    target_line_style: str = "-",
    prediction_line_style: str = "--",
    line_width: float = 1.8,
    wrap_angles: bool = True,
    title: str | None = None,
    show_title: bool = True,
    show_legend: bool = True,
    show: bool = True,
    eps: float = 0.1,
):
    targets = torch_to_numpy(targets)
    predictions = torch_to_numpy(predictions)

    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(
            "targets must have shape [num_samples, num_steps, 2]."
        )

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions must have the same shape as targets. "
            f"Received targets={targets.shape} and "
            f"predictions={predictions.shape}."
        )

    if not 0 <= sample_id < targets.shape[0]:
        raise IndexError(
            f"sample_id must be between 0 and {targets.shape[0] - 1}."
        )

    if arena_size <= 0:
        raise ValueError("arena_size must be greater than 0.")

    sample_targets = targets[sample_id]
    sample_predictions = predictions[sample_id]

    target_angles = sample_targets[:, 0]
    target_distances = sample_targets[:, 1]

    predicted_angles = sample_predictions[:, 0]
    predicted_distances = sample_predictions[:, 1]

    if wrap_angles:
        target_angles = np.arctan2(
            np.sin(target_angles),
            np.cos(target_angles),
        )
        predicted_angles = np.arctan2(
            np.sin(predicted_angles),
            np.cos(predicted_angles),
        )

    num_steps = sample_targets.shape[0]
    steps = np.arange(1, num_steps + 1)

    if ax is None:
        figure, angle_axis = plt.subplots(figsize=figsize)
    else:
        angle_axis = ax
        figure = angle_axis.figure

    distance_axis = angle_axis.twinx()

    target_angle_line = angle_axis.plot(
        steps,
        target_angles,
        color=angle_color,
        linestyle=target_line_style,
        linewidth=line_width,
        label="Target angle",
    )[0]

    predicted_angle_line = angle_axis.plot(
        steps,
        predicted_angles,
        color=angle_color,
        linestyle=prediction_line_style,
        linewidth=line_width,
        label="Predicted angle",
    )[0]

    target_distance_line = distance_axis.plot(
        steps,
        target_distances,
        color=distance_color,
        linestyle=target_line_style,
        linewidth=line_width,
        label="Target distance",
    )[0]

    predicted_distance_line = distance_axis.plot(
        steps,
        predicted_distances,
        color=distance_color,
        linestyle=prediction_line_style,
        linewidth=line_width,
        label="Predicted distance",
    )[0]

    angle_axis.set_xlabel("Step")
    angle_axis.set_ylabel(
        "Angle [rad]",
        color=angle_color,
    )
    distance_axis.set_ylabel(
        "Distance",
        color=distance_color,
    )

    angle_axis.set_ylim(-(np.pi + eps), np.pi + eps)
    angle_axis.set_yticks(
        [
            -np.pi,
            -np.pi / 2,
            0,
            np.pi / 2,
            np.pi,
        ],
        labels=[
            r"$-\pi$",
            r"$-\frac{\pi}{2}$",
            "0",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
        ],
    )

    distance_axis.set_ylim(
        -eps, math.sqrt(0.5) * arena_size + eps)

    angle_axis.tick_params(
        axis="y",
        labelcolor=angle_color,
    )
    distance_axis.tick_params(
        axis="y",
        labelcolor=distance_color,
    )

    angle_axis.grid(
        alpha=0.25,
        zorder=0,
    )

    if show_title:
        angle_axis.set_title(
            title
            or f"Angle and distance estimation — sample {sample_id}"
        )

    if show_legend:
        angle_axis.legend(
            handles=[
                target_angle_line,
                predicted_angle_line,
                target_distance_line,
                predicted_distance_line,
            ],
            loc="upper center",
            ncol=2,
        )

    if show and ax is None:
        figure.tight_layout()
        plt.show()

    return figure, angle_axis, distance_axis


def plot_random_samples(
    targets: torch.Tensor | np.ndarray,
    predictions: torch.Tensor | np.ndarray,
    arena_size: float,
    nrows: int = 2,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    random_seed: int | None = None,
    replace: bool = False,
    angle_color: str = "tab:blue",
    distance_color: str = "tab:orange",
    target_line_style: str = "-",
    prediction_line_style: str = "--",
    line_width: float = 1.5,
    wrap_angles: bool = True,
    title: str = "Random samples: angle and distance estimation",
    show_legend: bool = True,
    show: bool = True,
):
    targets = torch_to_numpy(targets)
    predictions = torch_to_numpy(predictions)

    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(
            "targets must have shape [num_samples, num_steps, 2]."
        )

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions must have the same shape as targets. "
            f"Received targets={targets.shape} and "
            f"predictions={predictions.shape}."
        )

    if nrows <= 0 or ncols <= 0:
        raise ValueError("nrows and ncols must be greater than 0.")

    num_plots = nrows * ncols
    num_samples = targets.shape[0]

    if not replace and num_plots > num_samples:
        raise ValueError(
            f"Cannot select {num_plots} unique samples from "
            f"{num_samples} available samples. Set replace=True "
            "to allow repeated samples."
        )

    random_generator = np.random.default_rng(random_seed)

    sample_ids = random_generator.choice(
        num_samples,
        size=num_plots,
        replace=replace,
    )

    if figsize is None:
        figsize = (
            6 * ncols,
            4 * nrows,
        )

    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    for angle_axis, sample_id in zip(
        axes.flat,
        sample_ids,
    ):
        plot_sample(
            targets=targets,
            predictions=predictions,
            sample_id=int(sample_id),
            arena_size=arena_size,
            ax=angle_axis,
            angle_color=angle_color,
            distance_color=distance_color,
            target_line_style=target_line_style,
            prediction_line_style=prediction_line_style,
            line_width=line_width,
            wrap_angles=wrap_angles,
            title=f"Sample {sample_id}",
            show_legend=False,
            show=False,
        )

    figure.suptitle(title)

    if show_legend:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=angle_color,
                linestyle=target_line_style,
                linewidth=line_width,
                label="Target angle",
            ),
            Line2D(
                [0],
                [0],
                color=angle_color,
                linestyle=prediction_line_style,
                linewidth=line_width,
                label="Predicted angle",
            ),
            Line2D(
                [0],
                [0],
                color=distance_color,
                linestyle=target_line_style,
                linewidth=line_width,
                label="Target distance",
            ),
            Line2D(
                [0],
                [0],
                color=distance_color,
                linestyle=prediction_line_style,
                linewidth=line_width,
                label="Predicted distance",
            ),
        ]

        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.01),
        )

        figure.tight_layout(
            rect=(0, 0.07, 1, 0.95),
        )
    else:
        figure.tight_layout(
            rect=(0, 0, 1, 0.95),
        )

    if show:
        plt.show()

    return figure, axes, sample_ids


def plot_loss(
    losses: list[float] | np.ndarray | torch.Tensor,
    figsize: tuple[float, float] = (8, 5),
    color: str = "tab:blue",
    line_width: float = 1.5,
    title: str = "Loss during training",
    logarithmic_scale: bool = False,
):
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    else:
        losses = np.asarray(losses)

    epochs = np.arange(1, len(losses) + 1)

    plt.figure(figsize=figsize)

    plt.plot(
        epochs,
        losses,
        color=color,
        linewidth=line_width,
    )

    if logarithmic_scale:
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()