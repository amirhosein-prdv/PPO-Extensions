import torch as T
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

from .utils import get_unique_log_dir


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

    def add_image(
        self, tag: str, img_tensor, global_step: int = None, dataformats: str = "CHW"
    ) -> None:
        """Adds an image to the writer."""
        self.writer.add_image(
            tag, img_tensor, global_step or self.global_step, dataformats=dataformats
        )

    def close(self) -> None:
        """Closes the writer."""
        self.writer.close()
