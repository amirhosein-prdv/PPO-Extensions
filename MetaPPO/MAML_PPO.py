import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Callable, Union

from .Networks import ActorCriticNetwork
from .logger import Logger


# ---------- MAML Algorithm ----------
class MAML:
    def __init__(
        self,
        env_fn,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 1e-2,
        inner_vf_coef: float = 0.5,
        inner_ent_coef: float = 0.001,
        meta_lr: float = 3e-4,
        meta_vf_coef: float = 0.5,
        meta_ent_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        max_grad_norm: float = 1.0,
        normalize_advantage: bool = False,
        normalize_return: bool = False,
        inner_steps: int = 2,  # number of inner updates
        outer_batch_size: int = 10,  # number of tasks per meta-batch
        max_steps: int = 200,
        traj_per_task: int = 10,
        policy_kwargs: dict = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
        },
        second_order: bool = True,  # True -> exact MAML, False -> FOMAML
        logger: Optional[Logger] = None,
    ):
        self.env_fn = env_fn

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.inner_lr = inner_lr
        self.inner_vf_coef = inner_vf_coef
        self.inner_ent_coef = inner_ent_coef
        self.meta_lr = meta_lr
        self.meta_vf_coef = meta_vf_coef
        self.meta_ent_coef = meta_ent_coef

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_range_vf = clip_range_vf
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return
        self.max_steps = max_steps
        self.inner_steps = inner_steps
        self.outer_batch_size = outer_batch_size
        self.traj_per_task = traj_per_task
        self.policy_kwargs = policy_kwargs

        self.second_order = second_order

        # Base policy (meta-initialization)
        self.base_policy = ActorCriticNetwork(state_dim, action_dim, policy_kwargs)
        self.meta_optimizer = optim.SGD(self.base_policy.parameters(), lr=meta_lr)

        self.device = self.base_policy.device
        self.logger = logger

    # ====================================================================
    #  Meta-training step
    # ====================================================================
    def meta_train(
        self,
        num_meta_iterations: int = 100,
        eval_interval: int = 10,
        eval_env_fn: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """Main meta-training loop."""
        print("Starting meta-training...")

        for iteration in range(num_meta_iterations):
            self.logger.update_global_step(iteration) if self.logger else None

            # Sample a batch of tasks
            task_envs = [self.env_fn() for _ in range(self.outer_batch_size)]

            all_inner_stats = []
            all_meta_stats = []
            task_gradients = []

            base_params = OrderedDict(
                {
                    k: v.clone().detach().requires_grad_(True)
                    for k, v in self.base_policy.named_parameters()
                }
            )

            for task_idx, task_env in enumerate(task_envs):

                # ----- Differentiable inner adaptation -----
                adapted_params, inner_stats = self._inner_update(task_env, base_params)
                all_inner_stats.append(inner_stats)

                # ----- Collect QUERY trajectories using the adapted policy -----
                query_trajs = []
                for _ in range(self.traj_per_task):
                    traj = self._collect_trajectory(task_env, adapted_params)
                    query_trajs.append(traj)

                data = self._flatten_trajectories(query_trajs)

                # ----- Compute meta-loss on query set -----
                task_meta_loss, meta_stats = self._vpg_loss(
                    data, adapted_params, looptype="outer"
                )
                # task_meta_loss, meta_stats = self._ppo_loss(
                #     data, adapted_params, looptype="outer"
                # )
                all_meta_stats.append(meta_stats)

                # ----- Compute gradients for this task -----
                if self.second_order:
                    task_grad = torch.autograd.grad(
                        task_meta_loss,
                        base_params.values(),
                    )
                else:
                    task_grad = torch.autograd.grad(
                        task_meta_loss,
                        adapted_params.values(),
                    )
                task_gradients.append(task_grad)

            # ----- Average gradients across tasks and update -----
            avg_gradients = []
            for grad_group in zip(*task_gradients):
                grads = []
                for g in grad_group:
                    if g is None:
                        raise ValueError("Averaging: Gradient is None")
                    grads.append(g)
                avg_grad = torch.stack(grads).mean(dim=0)
                avg_gradients.append(avg_grad)

            # Apply accumulated gradients to base policy
            self.meta_optimizer.zero_grad()
            for param, grad in zip(self.base_policy.parameters(), avg_gradients):
                if grad is not None:
                    param.grad = grad
                else:
                    raise ValueError("Applying: Gradient is None")

            torch.nn.utils.clip_grad_norm_(
                self.base_policy.parameters(), self.max_grad_norm
            )
            self.meta_optimizer.step()

            inner_stats = {
                key: np.mean([s[key] for s in all_inner_stats])
                for key in all_inner_stats[0].keys()
            }
            meta_stats = {
                key: np.mean([s[key] for s in all_meta_stats])
                for key in all_meta_stats[0].keys()
            }

            if self.logger is not None:
                self.logger.add_scalar("Meta Loss/policy", meta_stats["policy_loss"])
                self.logger.add_scalar("Meta Loss/value", meta_stats["value_loss"])
                self.logger.add_scalar("Meta Loss/entropy", meta_stats["entropy"])
                self.logger.add_scalar("Inner Loss/policy", inner_stats["policy_loss"])
                self.logger.add_scalar("Inner Loss/value", inner_stats["value_loss"])
                self.logger.add_scalar("Inner Loss/entropy", inner_stats["entropy"])

            if iteration % eval_interval == 0:

                print(
                    f"\nIteration {iteration}: ",
                    f"\nMeta Loss: Policy Loss = {meta_stats['policy_loss']:.5f}, Value Loss = {meta_stats['value_loss']:.5f}, Entropy = {meta_stats['entropy']:.5f}",
                    # "\n" + " " * 12,
                    # f"Clipfrac = {meta_stats['clipfracs']:.5f}, Explained Var. = {meta_stats['explained_var']:.5f}, Approx KL. = {meta_stats['approx_kl']:.5e}.",
                    f"\nInner Loss: Policy Loss = {inner_stats['policy_loss']:.5f}, Value Loss = {inner_stats['value_loss']:.3f}, Entropy = {inner_stats['entropy']:.3f}.",
                    # "\n" + " " * 12,
                    # f"Clipfrac = {avg_stats['clipfracs']:.5f}, Explained Var. = {avg_stats['explained_var']:.5f}, Approx KL. = {avg_stats['approx_kl']:.5e}.",
                )

                # Evaluation
                if eval_env_fn:
                    self.evaluate(
                        eval_env_fn,
                        num_episodes=self.outer_batch_size,
                        adaptation_steps=self.inner_steps,
                        num_trajectories=self.traj_per_task,
                    )

            # Clean up
            for env in task_envs:
                env.close()

    # ====================================================================
    #  Inner update: fast adaptation
    # ====================================================================
    def _inner_update(self, task_env, base_params):
        """
        Perform differentiable updates on support trajectories.
        Returns updated parameters (dict) that are a function of base_params.
        """
        # Clone base parameters
        fast_params = OrderedDict({k: v.clone() for k, v in base_params.items()})

        all_inner_stats = []
        for _ in range(self.inner_steps):

            # Collect support trajectories using the current policy
            support_trajs = []
            for _ in range(self.traj_per_task):
                traj = self._collect_trajectory(task_env, fast_params)
                support_trajs.append(traj)
            data = self._flatten_trajectories(support_trajs)

            # Calculate loss on support set
            loss, stats = self._vpg_loss(data, fast_params, looptype="inner")

            # Compute gradients w.r.t fast_params
            grads = torch.autograd.grad(
                loss,
                fast_params.values(),
                create_graph=self.second_order,
                allow_unused=True,
                materialize_grads=True,
            )

            # Update fast_params while maintaining the graph with inner learning rate
            # fast_params = {
            #     k: v - self.inner_lr * grad
            #     for (k, v), grad in zip(fast_params.items(), grads)
            # }
            for (k, v), grad in zip(fast_params.items(), grads):
                if grad is not None:
                    fast_params[k] = v - self.inner_lr * grad
                else:
                    raise ValueError("Inner update: Gradient is None")

            all_inner_stats.append(stats)

        avg_stats = {
            key: np.mean([s[key] for s in all_inner_stats])
            for key in all_inner_stats[0].keys()
        }

        # this is still differentiable w.r.t base_policy
        return fast_params, avg_stats

    # ====================================================================
    #  Loss functions
    # ====================================================================
    def _ppo_loss(
        self,
        data: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
        looptype="inner",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss given a batch of data and policy parameters."""

        values, new_logprobs, entropy = self.base_policy.evaluate_action(
            data["states"], data["actions"], params
        )

        # ----- policy loss -----
        ratio = torch.exp(new_logprobs - data["logprobs"])
        logratio = new_logprobs - data["logprobs"]
        surr1 = ratio * data["advantages"]
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * data["advantages"]
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # ----- Value Loss (clipped) -----
        if self.clip_range_vf is not None:
            value_pred_clipped = data["values"] + torch.clamp(
                values - data["values"],
                -self.clip_range_vf,
                self.clip_range_vf,
            )
            value_loss_unclipped = (values.squeeze(-1) - data["returns"]) ** 2
            value_loss_clipped = (value_pred_clipped - data["returns"]) ** 2
            value_loss = (
                0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            )
        else:
            value_loss = F.mse_loss(values.squeeze(-1), data["returns"])

        # ----- Entropy Loss -----
        entropy_loss = entropy.mean()

        # ----- Total Loss -----
        if looptype == "inner":
            total_loss = (
                policy_loss
                + self.inner_vf_coef * value_loss
                - self.inner_ent_coef * entropy_loss
            )
        elif looptype == "outer":
            total_loss = (
                policy_loss
                + self.meta_vf_coef * value_loss
                - self.meta_ent_coef * entropy_loss
            )
        else:
            raise ValueError("Invalid loop type. Must be 'inner' or 'outer'.")

        # calculate approx_kl & clip_frac
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - logratio).cpu().numpy()
            clipfracs = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

        # Calculate explained variance
        y_pred, y_true = (
            values.detach().cpu().numpy(),
            data["returns"].cpu().numpy(),
        )
        explained_var = (
            np.nan
            if np.var(y_true) == 0
            else 1 - np.var(y_true - y_pred) / np.var(y_true)
        )

        stats = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "explained_var": explained_var,
            "clipfracs": clipfracs,
            "approx_kl": approx_kl,
        }

        return total_loss, stats

    def _vpg_loss(
        self,
        data: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
        looptype="inner",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute REINFORCE loss given a batch of data and policy parameters."""

        values, new_logprobs, entropy = self.base_policy.evaluate_action(
            data["states"], data["actions"], params
        )

        # ----- policy loss -----
        policy_loss = -(new_logprobs * data["advantages"]).mean()

        # ----- Value Loss -----
        value_loss = F.mse_loss(values.squeeze(-1), data["returns"])

        # ----- Entropy Loss -----
        entropy_loss = entropy.mean()

        # ----- Total Loss -----
        if looptype == "inner":
            total_loss = (
                policy_loss
                + self.inner_vf_coef * value_loss
                - self.inner_ent_coef * entropy_loss
            )
        elif looptype == "outer":
            total_loss = (
                policy_loss
                + self.meta_vf_coef * value_loss
                - self.meta_ent_coef * entropy_loss
            )
        else:
            raise ValueError("Invalid loop type. Must be 'inner' or 'outer'.")

        # Calculate approx_kl & clip_frac
        ratio = torch.exp(new_logprobs - data["logprobs"])
        logratio = new_logprobs - data["logprobs"]
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - logratio).cpu().numpy()
            clipfracs = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

        # Calculate explained variance
        y_pred, y_true = (
            values.detach().cpu().numpy(),
            data["returns"].cpu().numpy(),
        )
        explained_var = (
            np.nan
            if np.var(y_true) == 0
            else 1 - np.var(y_true - y_pred) / np.var(y_true)
        )

        stats = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "explained_var": explained_var,
            "clipfracs": clipfracs,
            "approx_kl": approx_kl,
        }

        return total_loss, stats

    # ====================================================================
    #  Trajectory collection
    # ====================================================================
    def _collect_trajectory(
        self,
        env: gym.Env,
        policy_params=None,
        max_steps=None,
    ) -> Dict[str, torch.Tensor]:
        """Run one episode and collect data for policy updates."""
        if max_steps is None:
            max_steps = self.max_steps

        state, _ = env.reset()
        states, actions, rewards, dones, logprobs, values = [], [], [], [], [], []

        for _ in range(max_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, value, logprob = self.base_policy(state_t, policy_params)

            next_state, reward, terminated, truncated, _ = env.step(
                action.detach().cpu().numpy().squeeze(0)
            )
            done = terminated or truncated

            states.append(state_t)
            actions.append(action.detach().cpu())
            logprobs.append(logprob.detach().cpu())
            values.append(value.squeeze(0).detach().cpu())
            rewards.append(reward)
            dones.append(done)

            state = next_state

            if done:
                break

        # Bootstrap value for the last state
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            last_value = (
                self.base_policy.get_value(state_t, policy_params).squeeze(0).cpu()
            )

        # Compute advantages and returns
        advantages, returns = self._compute_gae_advantages(
            rewards, dones, values, last_value
        )

        return {
            "states": torch.cat(states),
            "actions": torch.cat(actions),
            "logprobs": torch.cat(logprobs),
            "values": torch.cat(values),
            "returns": returns,
            "advantages": advantages,
            "rewards": torch.tensor(rewards, dtype=torch.float32),
        }

    def _compute_gae_advantages(
        self,
        rewards: List[float],
        dones: List[bool],
        values: List[torch.Tensor],
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation (GAE) for continuous actions."""
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values_tensor = torch.cat(values).detach().cpu()

        advantages = torch.zeros_like(rewards)
        gae = 0.0
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
            advantages[t] = gae

        # Compute return
        returns = advantages + values_tensor

        return advantages, returns

    def _flatten_trajectories(
        self, trajectories: List[Dict[str, torch.Tensor]], batch_idxs=None
    ) -> Dict[str, torch.Tensor]:

        states = torch.cat([t["states"] for t in trajectories]).to(self.device)
        actions = torch.cat([t["actions"] for t in trajectories]).to(self.device)
        advantages = torch.cat([t["advantages"] for t in trajectories]).to(self.device)
        returns = torch.cat([t["returns"] for t in trajectories]).to(self.device)
        old_logprobs = torch.cat([t["logprobs"] for t in trajectories]).to(self.device)
        old_values = torch.cat([t["values"] for t in trajectories]).to(self.device)

        # Normalize advantages and returns
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.normalize_return:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if batch_idxs is not None:
            return {
                "states": states[batch_idxs],
                "actions": actions[batch_idxs],
                "returns": returns[batch_idxs],
                "advantages": advantages[batch_idxs],
                "logprobs": old_logprobs[batch_idxs],
                "values": old_values[batch_idxs],
            }
        else:
            return {
                "states": states,
                "actions": actions,
                "returns": returns,
                "advantages": advantages,
                "logprobs": old_logprobs,
                "values": old_values,
            }

    # ====================================================================
    #  Evaluation
    # ====================================================================
    def evaluate(
        self,
        env_fn: Callable[[], gym.Env],
        num_episodes: int = 5,
        adaptation_steps: int = 3,
        num_trajectories: int = 5,
    ) -> float:

        rewards_before = []
        rewards_after = []
        steps_before = []
        steps_after = []
        all_inner_stats = []

        for _ in range(num_episodes):

            env = env_fn()

            # BEFORE ADAPTATION
            state, _ = env.reset()
            episode_reward_before = 0.0
            episode_step = 0
            for _ in range(self.max_steps):

                with torch.no_grad():
                    action, _ = self.base_policy.get_action(
                        state,
                        deterministic=True,
                    )

                next_state, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).cpu().numpy()
                )

                episode_reward_before += reward
                state = next_state

                episode_step += 1
                if terminated or truncated:
                    break

            rewards_before.append(episode_reward_before)
            steps_before.append(episode_step)

            # ADAPTATION
            adapted_params, stats = self.adapt_to_new_task(
                env,
                adaptation_steps=adaptation_steps,
                num_trajectories=num_trajectories,
            )
            all_inner_stats.append(stats)

            # AFTER ADAPTATION
            state, _ = env.reset()
            episode_reward_after = 0.0
            episode_step = 0
            for _ in range(self.max_steps):

                with torch.no_grad():
                    action, _ = self.base_policy.get_action(
                        state,
                        params=adapted_params,
                        deterministic=True,
                    )

                next_state, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).cpu().numpy()
                )

                episode_reward_after += reward
                state = next_state

                episode_step += 1
                if terminated or truncated:
                    break

            rewards_after.append(episode_reward_after)
            steps_after.append(episode_step)

            env.close()

        rewards_before = np.array(rewards_before)
        rewards_after = np.array(rewards_after)
        steps_before = np.array(steps_before)
        steps_after = np.array(steps_after)

        avg_rewards_before = rewards_before / steps_before
        avg_rewards_after = rewards_after / steps_after

        improvement = rewards_after - rewards_before

        avg_stats = {
            key: np.mean([s[key] for s in all_inner_stats])
            for key in all_inner_stats[0].keys()
        }

        if self.logger is not None:
            self.logger.add_scalar(
                "Evaluation/Average Step Before", steps_before.mean()
            )
            self.logger.add_scalar("Evaluation/Average Step After", steps_after.mean())
            self.logger.add_scalar(
                "Evaluation/Average Reward Before", avg_rewards_before.mean()
            )
            self.logger.add_scalar(
                "Evaluation/Average Reward After", avg_rewards_after.mean()
            )
            self.logger.add_scalar("Evaluation/Reward Before", rewards_before.mean())
            self.logger.add_scalar(
                "Evaluation/Reward Before (Std)", rewards_before.std()
            )
            self.logger.add_scalar("Evaluation/Reward After", rewards_after.mean())
            self.logger.add_scalar("Evaluation/Reward After (Std)", rewards_after.std())
            self.logger.add_scalar("Evaluation/Reward Improvement", improvement.mean())
            self.logger.add_scalar(
                "Evaluation/Reward Improvement (Std)", improvement.std()
            )
            self.logger.add_scalar(
                "Evaluation/Success Rate", (improvement > 0).mean() * 100
            )

        print("Evaluation:", end=" ")
        print(
            f"Before adaptation: "
            f"{rewards_before.mean():.2f} ± {rewards_before.std():.2f}",
            end=" | ",
        )
        print(
            f"After adaptation: "
            f"{rewards_after.mean():.2f} ± {rewards_after.std():.2f}",
            end="\n" + " " * 12,
        )
        print(
            f"Improvement: {improvement.mean():.2f} ± {improvement.std():.2f}",
            end=" | ",
        )
        print(f"Success rate (+): {(improvement > 0).mean() * 100:.1f} %")

    def adapt_to_new_task(
        self,
        task_env: gym.Env,
        adaptation_steps: int = 5,
        num_trajectories: int = 5,
    ):
        """Fast adaptation updates."""

        base_params = OrderedDict(
            {
                k: v.clone().detach().requires_grad_(True)
                for k, v in self.base_policy.named_parameters()
            }
        )
        all_stats = []
        adapted_params = base_params

        for step in range(adaptation_steps):

            # collect fresh data each step ---
            trajectories = []
            for _ in range(num_trajectories):
                traj = self._collect_trajectory(task_env, adapted_params)
                trajectories.append(traj)
            data = self._flatten_trajectories(trajectories)

            loss, stats = self._vpg_loss(data, adapted_params, looptype="inner")
            all_stats.append(stats)

            grads = torch.autograd.grad(
                loss, adapted_params.values(), allow_unused=True
            )

            adapted_params = OrderedDict(
                {
                    k: v - self.inner_lr * g if g is not None else v
                    for (k, v), g in zip(adapted_params.items(), grads)
                }
            )

        stats = {
            key: np.mean([s[key] for s in all_stats]) for key in all_stats[0].keys()
        }

        return adapted_params, stats

    def _clone_policy(self) -> ActorCriticNetwork:
        """Create a deep copy of the base policy."""
        clone = ActorCriticNetwork(self.state_dim, self.action_dim, self.policy_kwargs)
        clone.load_state_dict(self.base_policy.state_dict())
        return clone
