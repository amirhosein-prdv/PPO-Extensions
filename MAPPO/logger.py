import torch as T
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from pettingzoo import ParallelEnv

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

    def close(self) -> None:
        """Closes the writer."""
        self.writer.close()


class EvaluationLogger:
    def __init__(
        self,
        eval_env: ParallelEnv,
        multiAgents,
        logger: Logger,
        eval_episodes=5,
        verbose=0,
    ):
        """
        Evaluate the model on multiple episodes after each policy update.

        :param eval_env: Gym-like environment for evaluation.
        :param eval_episodes: Number of episodes for evaluation.
        :param verbose: Verbosity level (0: no output, 1: info messages).
        """
        self.eval_env = eval_env
        self.multiAgents = multiAgents
        self.logger = logger
        self.eval_episodes = eval_episodes
        self.verbose = verbose

    def evaluate_and_log(self):
        """Run evaluation episodes and step averaged metrics."""
        metrics_buffer = {
            "reward": [],
            "rewards": {},
        }

        self.multiAgents.eval()
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            done = False

            while not done:
                actions, _, _ = self.multiAgents.get_actions(state, deterministic=True)

                next_states, rewards, terminations, truncations, infos = (
                    self.eval_env.step(actions)
                )
                agent = list(infos.keys())[0]
                done = terminations[agent] or truncations[agent]
                state = next_states

                info = infos[agent]  # Assuming all agents share the same info structure

                metrics_buffer["reward"].append(info["reward"])

                for k, v in rewards.items():
                    metrics_buffer["rewards"].setdefault(k, []).append(v)

        # Compute averages and log them
        rew = sum(metrics_buffer["reward"]) / len(metrics_buffer["reward"])
        self.logger.add_scalar("reward (eval)", rew)

        for k, v in metrics_buffer["rewards"].items():
            self.logger.add_scalar(f"reward {k} (eval)", sum(v) / len(v))

        if self.verbose > 0:
            print("Evaluation metrics logged.")


class StepLogger:
    def __init__(
        self,
        logger: Logger,
        step_interval: Optional[int] = None,
        suffix_title="step",
        verbose=0,
    ):
        self.suffix_title = suffix_title
        self.verbose = verbose
        self.logger = logger
        self.step_interval = step_interval
        self.step_counter = 0

        self.initialize_buffer()

    def initialize_buffer(self) -> None:
        self.metrics_buffer = {
            "rewards": {},
            "reward": [],
        }

    def record_log(self) -> None:
        self._log_buffered_metrics()
        self._reset_buffer()
        if self.verbose > 0:
            print("Step metrics logged.")

    def add_info(self, infos) -> None:
        self.step_counter += 1
        agent = list(infos.keys())[0]
        info = infos[agent]  # Assuming all agents share the same info structure
        self._buffer_metrics(info)

        if self.step_interval is not None:
            if self.step_counter % self.step_interval == 0:
                self.record_log()

    def _buffer_metrics(self, info) -> None:

        if "rewards" in info:
            for k, v in info["rewards"].items():
                self.metrics_buffer["rewards"].setdefault(k, []).append(v)

        if "reward" in info:
            self.metrics_buffer["reward"].append(info["reward"])

    def _log_buffered_metrics(self) -> None:

        if "rewards" in self.metrics_buffer:
            for k, v in self.metrics_buffer["rewards"].items():
                self.logger.add_scalar(f"reward {k} (eval)", sum(v) / len(v))

        if "reward" in self.metrics_buffer:
            avg_value = sum(self.metrics_buffer["reward"]) / len(
                self.metrics_buffer["reward"]
            )
            self.logger.add_scalar(
                f"reward ({self.step_interval}-{self.suffix_title})", avg_value
            )

    def _reset_buffer(self) -> None:
        self.initialize_buffer()
