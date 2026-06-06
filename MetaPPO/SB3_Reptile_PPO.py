from copy import deepcopy
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Callable, List, Optional, Union

import torch

from .logger import Logger


class SB3ReptilePPO:
    """
    Reptile meta-learning with SB3 PPO as inner learner.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        inner_lr: float = 3e-4,
        inner_vf_coef: float = 0.5,
        inner_ent_coef: float = 0.01,
        meta_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        inner_steps: int = 5,
        inner_epochs: int = 4,
        inner_batch_size: int = 64,
        outer_batch_size: int = 4,
        max_steps: int = 200,
        traj_per_task: int = 3,
        policy_kwargs: dict = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
        },
        logger: Optional[Logger] = None,
        verbose=False,
    ):
        self.env_fn = env_fn

        self.inner_lr = inner_lr
        self.inner_vf_coef = inner_vf_coef
        self.inner_ent_coef = inner_ent_coef
        self.meta_lr = meta_lr

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_range_vf = clip_range_vf
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        self.max_steps = max_steps
        self.inner_steps = inner_steps
        self.inner_epochs = inner_epochs
        self.inner_batch_size = inner_batch_size
        self.outer_batch_size = outer_batch_size
        self.traj_per_task = traj_per_task  # Number of trajectories for inner loop
        self.policy_kwargs = dict(
            net_arch=policy_kwargs["feature"]
            + [
                dict(pi=policy_kwargs["pi"], vf=policy_kwargs["vf"]),
            ],
            log_std_init=-0.5,
            ortho_init=True,
        )

        self.steps_per_task = self.max_steps * self.traj_per_task
        self.inner_timesteps = self.inner_steps * self.steps_per_task

        self.logger = logger
        self.verbose = verbose

        # base meta-model
        base_env = env_fn()
        self.meta_model = PPO(
            "MlpPolicy",
            base_env,
            learning_rate=self.inner_lr,
            batch_size=self.inner_batch_size,
            n_steps=self.steps_per_task,
            n_epochs=self.inner_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_epsilon,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.inner_ent_coef,
            vf_coef=self.inner_vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=0,
            policy_kwargs=self.policy_kwargs,
            normalize_advantage=self.normalize_advantage,
        )

    # -----------------------------
    # META TRAINING LOOP
    # -----------------------------
    def meta_train(
        self,
        num_meta_iterations: int,
        eval_env_fn: Optional[Callable] = None,
        eval_interval: int = 1,
    ):
        print("Starting Reptile-PPO Meta-Training...")

        for it in range(num_meta_iterations):

            adapted_models = []

            # -------------------------
            # INNER LOOP
            # -------------------------
            task_envs = [self.env_fn() for _ in range(self.outer_batch_size)]

            if self.verbose:
                print(f"\n[Meta Iter {it}] Adapting to {len(task_envs)} tasks...")

            for task_env in task_envs:

                adapted_model = self._inner_adapt(task_env)
                adapted_models.append(adapted_model)

                task_env.close()

            # -------------------------
            # OUTER LOOP
            # -------------------------
            self._reptile_update(adapted_models)

            # -------------------------
            # LOGGING
            # -------------------------
            if self.logger:
                self.logger.update_global_step(it)

            if it % eval_interval == 0 and eval_env_fn is not None:
                score = self.evaluate(eval_env_fn, num_episodes=5)

    # -----------------------------
    # INNER LOOP (SB3 PPO adaptation)
    # -----------------------------
    def _inner_adapt(self, task_env: gym.Env):
        task_model = PPO(
            "MlpPolicy",
            task_env,
            policy_kwargs=self.policy_kwargs,
            learning_rate=self.inner_lr,
            batch_size=self.inner_batch_size,
            n_steps=self.steps_per_task,
            n_epochs=self.inner_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_epsilon,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.inner_ent_coef,
            vf_coef=self.inner_vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=0,
            normalize_advantage=self.normalize_advantage,
        )

        # initialize from meta model
        task_model.set_parameters(self.meta_model.get_parameters())

        # adaptation
        task_model.learn(
            total_timesteps=self.inner_timesteps, progress_bar=self.verbose
        )

        return task_model

    # -----------------------------
    # REPTILE UPDATE
    # -----------------------------
    def _reptile_update(self, adapted_models: List[PPO]):
        meta_params = self.meta_model.get_parameters()

        delta = deepcopy(meta_params)
        for key in delta["policy"]:
            delta["policy"][key] = 0

        for model in adapted_models:
            params = model.get_parameters()
            for key in meta_params["policy"]:
                delta["policy"][key] += (
                    params["policy"][key] - meta_params["policy"][key]
                )

        # average
        for key in delta["policy"]:
            delta["policy"][key] /= len(adapted_models)

        # update meta model
        new_params = deepcopy(meta_params)

        for key in new_params["policy"]:
            new_params["policy"][key] = (
                meta_params["policy"][key] + self.meta_lr * delta["policy"][key]
            )

        self.meta_model.set_parameters(new_params)

    # -----------------------------
    # FAST ADAPTATION (TEST TIME)
    # -----------------------------
    def adapt_to_task(self, env: gym.Env, steps: Optional[int] = None):
        if steps is None:
            steps = self.inner_timesteps
        else:
            steps = steps * self.steps_per_task

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=self.policy_kwargs,
            learning_rate=self.inner_lr,
            batch_size=self.inner_batch_size,
            n_steps=self.steps_per_task,
            n_epochs=self.inner_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_epsilon,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.inner_ent_coef,
            vf_coef=self.inner_vf_coef,
            max_grad_norm=self.max_grad_norm,
            normalize_advantage=self.normalize_advantage,
            verbose=0,
        )

        model.set_parameters(self.meta_model.get_parameters())
        model.learn(total_timesteps=steps, progress_bar=self.verbose)

        return model

    # -----------------------------
    # EVALUATION
    # -----------------------------
    def evaluate(
        self,
        env_fn: Callable[[], gym.Env],
        num_episodes: int = 5,
        adaptation_steps: int = 3,
    ) -> float:

        rewards_before = []
        rewards_after = []
        steps_before = []
        steps_after = []

        for _ in range(num_episodes):

            env = env_fn()

            # -----------------------------
            # BEFORE ADAPTATION
            # -----------------------------
            obs, _ = env.reset()
            episode_reward_before = 0.0

            episode_step = 0
            for _ in range(self.inner_timesteps):

                with torch.no_grad():
                    action, _ = self.meta_model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward_before += reward

                episode_step += 1
                if terminated or truncated:
                    break

            rewards_before.append(episode_reward_before)
            steps_before.append(episode_step)

            # -----------------------------
            # ADAPTATION
            # -----------------------------
            if self.verbose:
                print(f"Adapting to new task with {adaptation_steps} steps...")

            adapted_model = self.adapt_to_task(
                env,
                steps=adaptation_steps,
            )

            # -----------------------------
            # AFTER ADAPTATION
            # -----------------------------
            obs, _ = env.reset()
            episode_reward_after = 0.0
            episode_step = 0
            for _ in range(self.inner_timesteps):

                with torch.no_grad():
                    action, _ = adapted_model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward_after += reward

                episode_step += 1
                if terminated or truncated:
                    break

            rewards_after.append(episode_reward_after)
            steps_after.append(episode_step)

            env.close()

        # -----------------------------
        # METRICS
        # -----------------------------
        rewards_before = np.array(rewards_before)
        rewards_after = np.array(rewards_after)
        steps_before = np.array(steps_before)
        steps_after = np.array(steps_after)

        avg_rewards_before = rewards_before / steps_before
        avg_rewards_after = rewards_after / steps_after

        improvement = rewards_after - rewards_before

        # -----------------------------
        # LOGGING
        # -----------------------------
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
            self.logger.add_scalar("Evaluation/Reward After", rewards_after.mean())
            self.logger.add_scalar("Evaluation/Improvement", improvement.mean())
            self.logger.add_scalar(
                "Evaluation/Success Rate", (improvement > 0).mean() * 100
            )

        # -----------------------------
        # PRINT
        # -----------------------------
        print(
            f"Before: {rewards_before.mean():.2f} ± {rewards_before.std():.2f} | "
            f"After: {rewards_after.mean():.2f} ± {rewards_after.std():.2f} | "
            f"Improvement: {improvement.mean():.2f} ± {improvement.std():.2f} | "
            f"Success rate: {(improvement > 0).mean() * 100:.1f}%"
        )

        return rewards_after.mean()
