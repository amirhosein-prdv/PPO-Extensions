import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import List, Tuple, Dict

from .Networks_simple import ActorCriticNetwork


# -------------------- PPO Agent for Continuous Actions (Inner Loop) --------------------
class PPOAgent:
    """PPO implementation for continuous action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
        },
    ):

        self.policy = ActorCriticNetwork(state_dim, action_dim, policy_kwargs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.device = self.policy.device

    def update(self, trajectories: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Perform PPO updates on collected trajectories.
        """
        self.policy.train()

        # Concatenate all trajectories
        states = torch.cat([t["states"] for t in trajectories])
        actions = torch.cat([t["actions"] for t in trajectories])
        old_logprobs = torch.cat([t["logprobs"] for t in trajectories])
        returns = torch.cat([t["returns"] for t in trajectories])
        advantages = torch.cat([t["advantages"] for t in trajectories])
        old_values = torch.cat([t["values"] for t in trajectories])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)

            # minibatch iteration
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)
                batch_returns = returns[batch_idx].to(self.device)
                batch_old_values = old_values[batch_idx].squeeze().to(self.device)

                # Get current policy distribution and values
                values, new_logprobs, entropy = self.policy.evaluate_action(
                    batch_states, batch_actions
                )

                # PPO clipped objective
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped) - NOW batch_old_values is defined
                values = values.squeeze()
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = F.mse_loss(values, batch_returns)
                value_loss_clipped = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()

        num_updates = self.epochs * (dataset_size // self.batch_size + 1)
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def collect_trajectory(
        self, env: gym.Env, max_steps: int = 200
    ) -> Dict[str, torch.Tensor]:
        """
        Run one episode and collect data for PPO update.
        """
        self.policy.eval()

        state, _ = env.reset()
        states, actions, rewards, dones, logprobs, values = [], [], [], [], [], []

        for _ in range(max_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, value, logprob = self.policy(state_t)

            next_state, reward, terminated, truncated, _ = env.step(
                action.squeeze(0).cpu().numpy()
            )
            done = terminated or truncated

            states.append(state_t)
            actions.append(action.cpu())
            logprobs.append(logprob.cpu())
            values.append(value.squeeze(0).cpu())
            rewards.append(reward)
            dones.append(done)

            state = next_state

            if done:
                break

        # Bootstrap value for the last state
        with torch.no_grad():
            last_value = self.policy.get_value(torch.FloatTensor(state).unsqueeze(0))
            last_value = last_value.squeeze().cpu()

        # Compute advantages and returns
        advantages, returns = self.compute_gae_advantages(
            rewards, dones, values, last_value
        )

        return {
            "states": torch.cat(states),
            "actions": torch.cat(actions),
            "logprobs": torch.cat(logprobs),
            "rewards": rewards,
            "returns": returns,
            "advantages": advantages,
            "values": torch.stack(values),
        }

    def compute_gae_advantages(
        self,
        rewards: List[float],
        dones: List[bool],
        values: List[torch.Tensor],
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation (GAE) for continuous actions."""
        advantages = []
        returns = []
        gae = 0

        # Convert to tensors
        values_tensor = torch.stack(values)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values_tensor[t + 1]

            # TD error
            delta = (
                rewards[t] + self.gamma * next_val * (1 - dones[t]) - values_tensor[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

            # Compute return
            ret = advantages[0] + values_tensor[t]
            returns.insert(0, ret)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return advantages, returns
