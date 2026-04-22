import numpy as np
from typing import List, Tuple, Dict


class MultiAgentRolloutBuffer:
    def __init__(
        self,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        possible_agents: List[str],
    ):
        self.states = {agent: [] for agent in possible_agents}
        self.actions = {agent: [] for agent in possible_agents}
        self.logprobs = {agent: [] for agent in possible_agents}
        self.values = {agent: [] for agent in possible_agents}
        self.rewards = {agent: [] for agent in possible_agents}
        self.dones = {agent: [] for agent in possible_agents}

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size: int = batch_size
        self.possible_agents: List[str] = possible_agents

    def generate_batches(self, agent: str) -> List[np.ndarray]:
        n_states = len(self.states[agent])
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return batches

    def get_data(self, agent: str) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        return (
            np.array(self.states[agent]),
            np.array(self.actions[agent]),
            np.array(self.logprobs[agent]),
            np.array(self.values[agent]),
            np.array(self.advantages[agent]),
            np.array(self.returns[agent]),
        )

    def store(
        self,
        state: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        logprobs: Dict[str, np.ndarray],
        values: Dict[str, np.ndarray],
        reward: Dict[str, float],
        done: bool,
    ) -> None:
        for agent in self.possible_agents:
            self.states[agent].append(state[agent])
            self.actions[agent].append(action[agent])
            self.logprobs[agent].append(logprobs[agent])
            self.values[agent].append(values[agent])
            self.rewards[agent].append(reward[agent])
            self.dones[agent].append(done[agent])

    def compute_GAE_and_returns(self, last_value, done):
        self.advantages, self.returns = {}, {}
        for agent in self.possible_agents:
            self.advantages[agent] = np.zeros_like(self.values[agent], dtype=np.float32)
            self.returns[agent] = np.zeros_like(self.values[agent], dtype=np.float32)

            last_gae = 0
            for t in reversed(range(len(self.rewards[agent]))):
                if t == len(self.rewards[agent]) - 1:
                    next_non_terminal = 1.0 - float(done[agent])
                    next_val = last_value[agent]
                else:
                    next_non_terminal = 1.0 - float(self.dones[agent][t + 1])
                    next_val = self.values[agent][t + 1]
                delta = (
                    self.rewards[agent][t]
                    + self.gamma * next_val * next_non_terminal
                    - self.values[agent][t]
                )
                last_gae = (
                    delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                )
                self.advantages[agent][t] = last_gae

            self.returns[agent] = self.advantages[agent] + self.values[agent]

    def clear(self) -> None:
        self.states = {agent: [] for agent in self.possible_agents}
        self.actions = {agent: [] for agent in self.possible_agents}
        self.logprobs = {agent: [] for agent in self.possible_agents}
        self.values = {agent: [] for agent in self.possible_agents}
        self.rewards = {agent: [] for agent in self.possible_agents}
        self.dones = {agent: [] for agent in self.possible_agents}
