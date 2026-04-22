import numpy as np
import torch as T
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union

from pettingzoo import ParallelEnv

from .Networks import ActorCriticNetwork
from .utils import anneal_learning_rate
from .logger import Logger
from .RolloutBuffer import MultiAgentRolloutBuffer


class MultiAgent:
    def __init__(
        self,
        env: ParallelEnv,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_advantage: bool = True,
        logger: Optional["Logger"] = None,
        chkpt_dir: str = "./tmp/PPO-MultiAgent",
        policy_kwargs: dict[str, dict[str, List[int]]] | dict[str, List[int]] = {
            "feature": [32],
            "pi": [64, 64],
            "vf": [64, 64],
        },
        # policy_kwargs: dict[str, List[int]] = {
        #     "feature": [32],
        #     "pi": [64, 64],
        #     "vf": [64, 64],
        # },
    ) -> None:

        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage

        self.n_epochs = n_epochs
        self.logger = logger

        self.possible_agents = env.possible_agents
        self.num_agents = env.num_agents

        if not isinstance(next(iter(policy_kwargs.values())), dict):
            policy_kwargs = {agent: policy_kwargs for agent in self.possible_agents}

        self.agents = {
            agent: ActorCriticNetwork(
                state_dim=env.observation_space(agent).shape[0],
                action_dim=env.action_space(agent).shape[0],
                feature_fc_dims=policy_kwargs[agent]["feature"],
                actor_fc_dims=policy_kwargs[agent]["pi"],
                critic_fc_dims=policy_kwargs[agent]["vf"],
                policy_lr=lr,
                chkpt_dir=chkpt_dir,
                model_name=f"ppo-actor-critic_{agent}",
            )
            for agent in self.possible_agents
        }

        self.memory = MultiAgentRolloutBuffer(
            batch_size, gamma, gae_lambda, self.possible_agents
        )
        self.device = self.agents[list(self.agents.keys())[0]].device

    def anneal_lr(self, current_step: int, total_steps: int) -> None:
        """Anneal the learning rate of the policy network."""
        for agent in self.agents.values():
            anneal_learning_rate(
                agent.optimizer,
                agent.initial_lr,
                current_step,
                total_steps,
            )

    def get_actions(
        self, state: dict, deterministic: bool = False
    ) -> Tuple[dict, dict, dict]:
        actions = {}
        logprobs = {}
        values = {}
        for agent_name, agent in self.agents.items():
            act, lp = agent.get_action(state[agent_name], deterministic)
            val = agent.get_value(state[agent_name])
            logprobs[agent_name] = lp.detach().cpu().numpy()
            actions[agent_name] = act.detach().squeeze().cpu().numpy()
            values[agent_name] = val.detach().squeeze(0).cpu().numpy()

        return actions, values, logprobs

    def get_values(self, state: dict) -> dict:
        values = {}
        for agent_name, agent in self.agents.items():
            val = agent.get_value(state[agent_name])
            values[agent_name] = val.detach().squeeze(0).cpu().numpy()
        return values

    def eval(self) -> None:
        for agent in self.agents.values():
            agent.eval()

    def learn(self) -> None:
        for agent in self.possible_agents:
            self._learn(agent)
        self.memory.clear()

    def _learn(self, agent_name: str) -> None:
        # Generate Data
        (
            state_arr,
            action_arr,
            old_logprob_arr,
            values_arr,
            advantages_arr,
            returns_arr,
        ) = self.memory.get_data(agent_name)

        agent = self.agents[agent_name]

        returns = T.tensor(returns_arr, dtype=T.float32).squeeze().to(self.device)
        advantages = T.tensor(advantages_arr, dtype=T.float32).squeeze().to(self.device)
        values = T.tensor(values_arr, dtype=T.float32).squeeze().to(self.device)

        clipfracs = []
        critic_buffer = []
        actor_buffer = []
        entropy_buffer = []
        approx_kl_buffer = []
        agent.train()
        for _ in range(self.n_epochs):
            # Generate minibatch data
            batches = self.memory.generate_batches(agent_name)

            # Training
            for minibatch in batches:
                old_logprob = (
                    T.tensor(old_logprob_arr[minibatch]).squeeze().to(self.device)
                )
                new_value, new_logprob, entropy = agent.evaluate_action(
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

                agent.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                agent.optimizer.step()

                # Store training variable to log
                critic_buffer.append(critic_loss.item())
                actor_buffer.append(actor_loss.item())
                entropy_buffer.append(entropy_loss.item())
                approx_kl_buffer.append(approx_kl.item())

        # Calculate explained variance
        y_pred, y_true = (
            values.cpu().numpy(),
            returns.cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.logger is not None:
            # record for plotting purposes
            name = f"train/agent_{agent_name}"
            self.logger.add_scalar(f"{name}/loss", total_loss.item())
            self.logger.add_scalar(f"{name}/value_loss", np.mean(critic_buffer))
            self.logger.add_scalar(
                f"{name}/policy_gradient_loss", np.mean(actor_buffer)
            )
            self.logger.add_scalar(f"{name}/entropy", np.mean(entropy_buffer))
            self.logger.add_scalar(f"{name}/approx_kl", np.mean(approx_kl_buffer))
            self.logger.add_scalar(f"{name}/clip_fraction", np.mean(clipfracs))
            self.logger.add_scalar(f"{name}/explained_variance", np.mean(explained_var))
            self.logger.add_scalar(
                f"{name}/learning_rate", agent.optimizer.param_groups[0]["lr"]
            )

    def save_models(self) -> None:
        print("... saving models ...")
        for agent in self.agents.values():
            agent.save_checkpoint()

    def load_models(self) -> None:
        print("... loading models ...")
        for agent in self.agents.values():
            agent.load_checkpoint()
