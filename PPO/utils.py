import os
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_learning_curve(x: List[float], scores: List[float], figure_file: str) -> None:
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def plot_smoothly(signal, window_size=10, time=None):
    if time is None:
        time = np.arange(len(signal))

    smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode="valid")

    rolling_std = [np.std(signal[i : i + window_size]) for i in range(len(smoothed))]

    plt.figure(figsize=(8, 5))
    plt.plot(time[: len(smoothed)], smoothed, color="orange", label="Smoothed Signal")
    plt.fill_between(
        time[: len(smoothed)],
        smoothed - rolling_std,
        smoothed + rolling_std,
        color="orange",
        alpha=0.3,
        label="Variance (Rolling STD)",
    )

    plt.xlabel("Time or Steps")
    plt.ylabel("Signal Value")
    plt.title("Smoothed Signal with Rolling Standard Deviation")
    plt.legend()
    plt.grid()
    plt.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    plt.show()


def anneal_learning_rate(
    optimizer: T.optim.Optimizer,
    initial_lr: float,
    current_step: int,
    total_steps: int,
    final_lr: float = 1e-4,
) -> None:
    """Anneal the learning rate linearly."""
    lr = initial_lr + (final_lr - initial_lr) * current_step / float(total_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_unique_log_dir(base_dir: str) -> str:
    """Generates a unique log directory name by incrementing a suffix if needed."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir  # Return as is if it does not exist

    index = 1
    new_dir = f"{base_dir}_{index}"

    while os.path.exists(new_dir):
        index += 1
        new_dir = f"{base_dir}_{index}"

    os.makedirs(new_dir)
    return new_dir

