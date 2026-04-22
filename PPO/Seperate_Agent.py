import numpy as np
import torch as T
from torch.nn import functional as F
from torch.distributions.normal import Normal
from typing import Optional, Tuple, List, Union

from .RolloutBuffer import RolloutBuffer
from .Networks import ActorNetwork, CriticNetwork
from .utils import anneal_learning_rate
from .logger import Logger


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_advantage: bool = True,
        logger: Optional["Logger"] = None,
        policy_kwargs: dict[str, List[int]] = {
            "pi": [64, 64],
            "vf": [64, 64],
        },
        chkpt_dir: str = "./tmp/PPO-Agent",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage

        self.n_epochs = n_epochs
        self.logger = logger

        # if self.logger is not None:
        #     logger.add_text(
        #         "hyperparameters",
        #         "|param|value|\n|-|-|\n%s"
        #         % (
        #             "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        #         ),
        #     )

        vf_net = policy_kwargs["vf"]
        pi_net = policy_kwargs["pi"]

        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            fc_dims=pi_net,
            actor_lr=lr,
            chkpt_dir=chkpt_dir,
        )
        self.critic = CriticNetwork(
            state_dim=state_dim,
            fc_dims=vf_net,
            critic_lr=lr,
            chkpt_dir=chkpt_dir,
        )
        self.memory = RolloutBuffer(batch_size, gamma, gae_lambda)
        self.device = self.actor.device

    def anneal_lr(self, current_step: int, total_steps: int) -> None:
        """Anneal the learning rate of the actor and critic network."""
        anneal_learning_rate(
            self.actor.optimizer, self.actor.initial_lr, current_step, total_steps
        )
        anneal_learning_rate(
            self.critic.optimizer, self.critic.initial_lr, current_step, total_steps
        )

    def get_value(self, observation: np.ndarray) -> float:
        state = T.as_tensor(observation, dtype=T.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        value = self.critic(state)
        return value

    def get_action(
        self, observation: np.ndarray, action: Optional[T.Tensor] = None
    ) -> Tuple[np.ndarray, float, float]:
        state = T.as_tensor(observation, dtype=T.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        dist: Normal
        dist = self.actor(state)
        value = self.critic(state)

        if action is None:
            action = dist.rsample()
        logprobs = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, value, logprobs, entropy

    def evaluate_action(
        self, observation: np.ndarray, action: np.ndarray
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        state = T.as_tensor(observation, dtype=T.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        values = self.critic(state)
        dist: Normal
        dist = self.actor(state)
        logprobs = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return values, logprobs, entropy

    def learn(self) -> None:
        # Generate Data
        (
            state_arr,
            action_arr,
            old_logprob_arr,
            values_arr,
            advantages_arr,
            returns_arr,
        ) = self.memory.get_data()

        returns = T.tensor(returns_arr, dtype=T.float32).squeeze().to(self.device)
        advantages = T.tensor(advantages_arr, dtype=T.float32).squeeze().to(self.device)
        values = T.tensor(values_arr, dtype=T.float32).squeeze().to(self.device)

        clipfracs = []
        critic_buffer = []
        actor_buffer = []
        entropy_buffer = []
        approx_kl_buffer = []
        self.actor.train()
        self.critic.train()
        for _ in range(self.n_epochs):
            # Generate minibatch data
            batches = self.memory.generate_batches()

            # Training
            for minibatch in batches:
                old_logprob = (
                    T.tensor(old_logprob_arr[minibatch]).squeeze().to(self.device)
                )
                new_value, new_logprob, entropy = self.evaluate_action(
                    state_arr[minibatch], action_arr[minibatch]
                )
                new_value = new_value.flatten()
                prob_ratio = T.exp(new_logprob - old_logprob)
                prob_logratio = new_logprob - old_logprob

                # nomalize advantage
                if self.normalize_advantage:
                    normalized_advantages = (
                        advantages[minibatch] - advantages[minibatch].mean()
                    ) / (advantages[minibatch].std() + 1e-8)
                else:
                    normalized_advantages = advantages[minibatch]

                # calculate actor loss
                weighted_probs = normalized_advantages * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * normalized_advantages
                )
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate Critic loss
                if self.clip_range_vf is None:
                    critic_loss = F.mse_loss(returns[minibatch], new_value)
                else:
                    # clip value loss
                    clipped_value = values[minibatch] + T.clamp(
                        new_value - values[minibatch],
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )

                    critic_loss_clipped = (returns[minibatch] - clipped_value) ** 2
                    critic_loss_unclipped = (returns[minibatch] - new_value) ** 2
                    critic_loss_max = T.max(critic_loss_clipped, critic_loss_unclipped)
                    critic_loss = 0.5 * critic_loss_max.mean()

                # Calculate entropy loss
                entropy_loss = entropy.mean()

                total_loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.ent_coef * entropy_loss
                )

                with T.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = T.mean((prob_ratio - 1) - prob_logratio).cpu().numpy()
                    clipfracs += [
                        ((prob_ratio - 1.0).abs() > self.clip_range)
                        .float()
                        .mean()
                        .item()
                    ]

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

                self.actor.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic.optimizer.step()

        # Calculate explained variance
        y_pred, y_true = (
            values.cpu().numpy(),
            returns.cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.logger is not None:
            # record for plotting purposes
            self.logger.add_scalar(
                "train/learning_rate", self.policy.optimizer.param_groups[0]["lr"]
            )
            self.logger.add_scalar("train/loss", total_loss.item())
            self.logger.add_scalar("train/value_loss", np.mean(critic_buffer))
            self.logger.add_scalar("train/policy_gradient_loss", np.mean(actor_buffer))
            self.logger.add_scalar("train/entropy", np.mean(entropy_buffer))
            self.logger.add_scalar("train/approx_kl", np.mean(approx_kl_buffer))
            self.logger.add_scalar("train/clip_fraction", np.mean(clipfracs))
            self.logger.add_scalar("train/explained_variance", explained_var)

        self.memory.clear()

    def save_models(self) -> None:
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self) -> None:
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
