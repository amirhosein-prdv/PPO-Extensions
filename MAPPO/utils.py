import os
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
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


class Logger:
    def __init__(self, log_dir: str) -> None:
        log_dir = get_unique_log_dir(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def update_global_step(self, step: int) -> None:
        self.global_step = step

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int = None
    ) -> None:
        """Adds a scalar value to the writer."""
        self.writer.add_scalar(tag, scalar_value, global_step or self.global_step)

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: dict, global_step: int = None
    ) -> None:
        """Adds multiple scalar values to the writer."""
        self.writer.add_scalars(
            main_tag, tag_scalar_dict, global_step or self.global_step
        )

    def add_histogram(
        self,
        tag: str,
        values: T.Tensor,
        global_step: int = None,
        bins: str = "tensorflow",
    ) -> None:
        """Adds a histogram to the writer."""
        self.writer.add_histogram(tag, values, global_step or self.global_step, bins)

    def add_text(self, tag: str, text_string: str, global_step: int = None) -> None:
        """Adds text data to the writer."""
        self.writer.add_text(tag, text_string, global_step or self.global_step)

    def add_dict(self, tag: str, dict_value: dict, global_step: int = None) -> None:
        for k, v in dict_value.items():
            self.writer.add_scalar(tag + f"/{k}", v, global_step or self.global_step)

    def close(self) -> None:
        """Closes the writer."""
        self.writer.close()
