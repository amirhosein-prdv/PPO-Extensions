import numpy as np
from typing import List, Tuple


class RolloutBuffer:
    def __init__(self, batch_size: int, gamma: float, gae_lambda: float) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.next_states: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

        self.batch_size: int = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def get_data(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.logprobs),
            np.array(self.values),
            np.array(self.advantages),
            np.array(self.returns),
        )

    def generate_batches(
        self,
    ) -> List[np.ndarray]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return batches

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_GAE_and_returns(self, last_value, done):
        self.advantages = np.zeros_like(self.values, dtype=np.float32)
        self.returns = np.zeros_like(self.values, dtype=np.float32)

        last_gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(done)
                next_val = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t + 1])
                next_val = self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []
