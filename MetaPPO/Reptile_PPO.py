import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import List, Tuple, Dict, Optional

from .PPO import PPOAgent
from .Networks import ActorCriticNetwork


# -------------------- Reptile Meta-Learner (PPO in outer Loop) --------------------
class ReptilePPO:
    """
    Meta-learner that uses Reptile to adapt the PPO agent to new tasks.
    The inner loop runs PPO updates on a support set (task-specific).
    The outer loop updates the initial parameters across tasks.
    """

    def __init__(
        self,
        env_fn,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 3e-4,
        meta_lr: float = 1e-3,
        meta_clip_epsilon: float = 0.2,
        meta_vf_coef: float = 0.5,
        meta_ent_coef: float = 0.01,
        inner_steps: int = 5,
        inner_epochs: int = 4,
        inner_batch_size: int = 64,
        outer_batch_size: int = 4,
        max_steps: int = 200,
        traj_per_task: int = 3,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
        },
    ):
        """
        Args:
            env_fn: function that returns a new environment (e.g., lambda: gym.make('CartPole-v1'))
            state_dim: dimension of state space
            action_dim: dimension of action space
            inner_lr: learning rate for inner PPO updates (task adaptation)
            meta_lr: learning rate for outer meta-update
            inner_steps: number of PPO update steps per task (fast adaptation)
            inner_batch_size: batch size for inner PPO updates
            outer_batch_size: number of tasks per meta-batch
            policy_kwargs: the size of pre-shared actor-critic
            traj_per_task: number of trajectories per task
        """
        self.env_fn = env_fn

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_clip_epsilon = meta_clip_epsilon
        self.meta_vf_coef = meta_vf_coef
        self.meta_ent_coef = meta_ent_coef

        self.max_steps = max_steps
        self.inner_steps = inner_steps
        self.inner_epochs = inner_epochs
        self.inner_batch_size = inner_batch_size
        self.outer_batch_size = outer_batch_size
        self.traj_per_task = traj_per_task  # Number of trajectories for inner loop
        self.policy_kwargs = policy_kwargs

        # Base policy (meta-initialization)
        self.base_policy = ActorCriticNetwork(state_dim, action_dim, policy_kwargs)
        self.meta_optimizer = optim.Adam(self.base_policy.parameters(), lr=meta_lr)

        # Store training metrics
        self.meta_losses = []

    def _clone_policy(self) -> ActorCriticNetwork:
        """
        Create a deep copy of the base policy for inner updates.
        """
        clone = ActorCriticNetwork(self.state_dim, self.action_dim, self.policy_kwargs)
        clone.load_state_dict(self.base_policy.state_dict())
        return clone

    def _inner_update(
        self,
        task_env: gym.Env,
        base_policy: nn.Module,
    ) -> Tuple[nn.Module, Dict[str, float], List[Dict]]:
        """
        Perform PPO updates on a specific task.

        Args:
            task_env: Environment for the current task
            policy: Policy to adapt
            query: If True, collect separate query trajectories for meta-loss

        Returns:
            Adapted policy, training stats, and trajectories
        """
        # Create PPO agent with the given policy
        agent = PPOAgent(
            self.state_dim,
            self.action_dim,
            lr=self.inner_lr,
            batch_size=self.inner_batch_size,
            epochs=self.inner_epochs,
            policy_kwargs=self.policy_kwargs,
        )
        agent.policy = base_policy
        agent.optimizer = optim.Adam(base_policy.parameters(), lr=self.inner_lr)

        # Collect support trajectories for inner update
        support_trajectories = []
        for _ in range(self.traj_per_task):
            traj = agent.collect_trajectory(task_env, self.max_steps)
            support_trajectories.append(traj)

        # Perform inner loop updates
        stats = {}
        for step in range(self.inner_steps):
            stats = agent.update(support_trajectories)

        # If query mode, collect fresh trajectories for meta-loss
        query_trajectories = []
        for _ in range(self.traj_per_task):
            traj = agent.collect_trajectory(task_env, self.max_steps)
            query_trajectories.append(traj)

        return agent.policy, stats, query_trajectories

    def compute_meta_loss(
        self, policy: ActorCriticNetwork, trajectories: List[Dict]
    ) -> torch.Tensor:
        """Compute meta-loss on query trajectories using the adapted policy."""
        device = self.base_policy.device
        policy.train()

        # Concatenate all query trajectories
        states = torch.cat([t["states"] for t in trajectories])
        actions = torch.cat([t["actions"] for t in trajectories])
        advantages = torch.cat([t["advantages"] for t in trajectories]).to(device)
        returns = torch.cat([t["returns"] for t in trajectories]).to(device)
        old_logprobs = torch.cat([t["logprobs"] for t in trajectories]).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute policy loss on query set
        values, new_logprobs, entropy = policy.evaluate_action(states, actions)

        # Clipped surrogate objective
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio, 1.0 - self.meta_clip_epsilon, 1.0 + self.meta_clip_epsilon
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # ----- Value Loss -----
        values = values.squeeze()
        value_loss = F.mse_loss(values, returns)

        # ----- Entropy Bonus -----
        entropy_loss = entropy.mean()

        # ----- Total Loss -----
        total_loss = (
            policy_loss
            + self.meta_vf_coef * value_loss
            - self.meta_ent_coef * entropy_loss
        )

        return total_loss

    def meta_train(
        self,
        num_meta_iterations: int = 100,
        eval_interval: int = 10,
        eval_env_fn: Optional[callable] = None,
    ):
        """Main meta-training loop."""
        print("Starting meta-training...")

        for iteration in range(num_meta_iterations):
            # Sample a batch of tasks
            task_envs = [self.env_fn() for _ in range(self.outer_batch_size)]
            all_stats = []

            # Inner loop: adapt to each task and collect query trajectories
            query_trajectories_per_task = []
            adapted_policies = []

            for task_env in task_envs:
                # Clone base policy for this task
                task_policy = self._clone_policy()

                # Perform inner updates and collect query trajectories
                adapted_policy, stats, query_trajs = self._inner_update(
                    task_env, task_policy
                )
                adapted_policies.append(adapted_policy)
                query_trajectories_per_task.append(query_trajs)
                all_stats.append(stats)

            # ----- Reptile outer update (move base policy towards adapted policies) -----
            # Compute average parameter difference across tasks
            avg_param_deltas = [
                torch.zeros_like(p) for p in self.base_policy.parameters()
            ]
            for adapted_policy in adapted_policies:
                for i, (base_param, adapted_param) in enumerate(
                    zip(self.base_policy.parameters(), adapted_policy.parameters())
                ):
                    avg_param_deltas[i] += adapted_param.data - base_param.data

            # Average over tasks
            for i in range(len(avg_param_deltas)):
                avg_param_deltas[i] /= len(adapted_policies)

            # Update base policy parameters (in-place, no gradients needed)
            with torch.no_grad():
                for base_param, delta in zip(
                    self.base_policy.parameters(), avg_param_deltas
                ):
                    base_param += self.meta_lr * delta

            # (Optional) Compute average meta-loss for logging
            meta_loss_val = 0.0
            count = 0
            for adapted_policy, query_trajs in zip(
                adapted_policies, query_trajectories_per_task
            ):
                if query_trajs:
                    adapted_query_loss = self.compute_meta_loss(
                        adapted_policy, query_trajs
                    )
                    meta_loss_val += adapted_query_loss.item()
                    count += 1
            if count > 0:
                meta_loss_val /= count
            self.meta_losses.append(meta_loss_val)

            # ## Outer loop: compute meta-gradient
            # meta_loss_val = 0.0
            # meta_grads = [torch.zeros_like(p) for p in self.base_policy.parameters()]
            # for adapted_policy, query_trajs in zip(
            #     adapted_policies, query_trajectories_per_task
            # ):
            #     if query_trajs:
            #         adapted_query_loss = self.compute_meta_loss(
            #             adapted_policy, query_trajs
            #         )
            #         meta_loss_val += adapted_query_loss.item()

            #         adapted_grads = torch.autograd.grad(
            #             adapted_query_loss, adapted_policy.parameters()
            #         )
            #         for i, g in enumerate(adapted_grads):
            #             meta_grads[i] += g

            # # Update base policy using these gradients
            # with torch.no_grad():
            #     for base_param, grad in zip(self.base_policy.parameters(), meta_grads):
            #         base_param -= self.meta_lr * grad  # or use optimizer update

            # meta_loss_val = meta_loss_val / self.outer_batch_size

            ## Outer loop: compute meta-gradient
            # meta_loss = 0.0
            # for adapted_policy, query_trajs in zip(
            #     adapted_policies, query_trajectories_per_task
            # ):
            #     if query_trajs:
            #         adapted_query_loss = self.compute_meta_loss(
            #             adapted_policy, query_trajs
            #         )
            #         meta_loss += adapted_query_loss

            # meta_loss = meta_loss / self.outer_batch_size

            # Update base policy
            # self.meta_optimizer.zero_grad()
            # meta_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.base_policy.parameters(), 1.0)
            # self.meta_optimizer.step()

            # meta_loss_val = meta_loss.item()
            # self.meta_losses.append(meta_loss_val)

            # Logging
            if iteration % eval_interval == 0:
                avg_policy_loss = np.mean([s["policy_loss"] for s in all_stats])
                avg_value_loss = np.mean([s["value_loss"] for s in all_stats])
                print(
                    f"\nIteration {iteration}: Meta Loss = {meta_loss_val:.4f}, "
                    f"Avg Inner Loss: Policy = {avg_policy_loss:.4f}, Value = {avg_value_loss:.4f}."
                )

                # Evaluation
                if eval_env_fn:
                    self.evaluate(eval_env_fn, num_episodes=3)

            # Clean up
            for env in task_envs:
                env.close()

    def adapt_to_new_task(
        self, task_env: gym.Env, adaptation_steps: int = 5, num_trajectories: int = 5
    ) -> PPOAgent:
        """Quickly adapt the base policy to a new task."""
        agent = PPOAgent(
            self.state_dim,
            self.action_dim,
            lr=self.inner_lr,
            policy_kwargs=self.policy_kwargs,
            batch_size=self.inner_batch_size,
            epochs=self.inner_epochs,
        )
        agent.policy = self._clone_policy()
        agent.optimizer = optim.Adam(agent.policy.parameters(), lr=self.inner_lr)

        # Collect trajectories
        trajectories = []
        for _ in range(num_trajectories):
            traj = agent.collect_trajectory(task_env, self.max_steps)
            trajectories.append(traj)

        # Adapt
        for step in range(adaptation_steps):
            stats = agent.update(trajectories)

        return agent

    def evaluate(self, env_fn, num_episodes: int = 5, adaptation_steps: int = 3):
        """Evaluate the meta-learned policy on new tasks."""
        total_rewards = []

        for episode in range(num_episodes):
            env = env_fn()
            agent = self.adapt_to_new_task(env, adaptation_steps=adaptation_steps)

            # Test the adapted policy
            state, _ = env.reset()
            episode_reward = 0
            for _ in range(self.max_steps):
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, _ = agent.policy.get_action(state_t, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                done = terminated or truncated
                episode_reward += reward
                if done:
                    break

            total_rewards.append(episode_reward)
            env.close()

        avg_reward = np.mean(total_rewards)
        print(
            f"Evaluation: Avg Reward = {avg_reward:.2f} ± {np.std(total_rewards):.2f}"
        )
        return avg_reward
