import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Optional, Callable

from .Networks import ActorCriticNetwork


class MAMLTRPO:
    def __init__(
        self,
        env_fn,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 1e-2,
        inner_vf_coef: float = 0.0,
        inner_ent_coef: float = 0.001,
        meta_lr: float = 3e-4,
        meta_vf_coef: float = 0.5,
        meta_ent_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 1.0,
        normalize_advantage: bool = False,
        inner_steps: int = 2,
        outer_batch_size: int = 10,
        max_steps: int = 200,
        traj_per_task: int = 10,
        policy_kwargs: dict = {"feature": [], "pi": [64, 64], "vf": [64, 64]},
        second_order: bool = True,
        delta: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        residual_tol: float = 1e-10,
        backtrack_coeff: float = 0.8,
        backtrack_iters: int = 10,
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
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.max_steps = max_steps
        self.inner_steps = inner_steps
        self.outer_batch_size = outer_batch_size
        self.traj_per_task = traj_per_task
        self.policy_kwargs = policy_kwargs
        self.second_order = second_order
        self.delta = delta
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.residual_tol = residual_tol
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters

        self.base_policy = ActorCriticNetwork(state_dim, action_dim, policy_kwargs)
        self.value_optimizer = optim.Adam(
            self.base_policy.critic.parameters(), lr=meta_lr
        )
        self.device = self.base_policy.device

    # ====================================================================
    #  Meta-training loop
    # ====================================================================
    def meta_train(
        self,
        num_meta_iterations: int = 100,
        eval_interval: int = 10,
        eval_env_fn: Optional[Callable] = None,
    ):
        print("Starting MAML-TRPO meta-training...")

        for iteration in range(num_meta_iterations):
            # Sample a batch of tasks
            task_envs = [self.env_fn() for _ in range(self.outer_batch_size)]

            all_inner_stats = []
            all_query_data = []

            for task_env in task_envs:
                # FIX 1: Check for NaN in base policy before starting
                if self._has_nan_params(self.base_policy):
                    print(
                        f"ERROR: NaN detected in base_policy at iteration {iteration}"
                    )
                    raise ValueError("NaN in base policy parameters")

                base_params = {
                    k: v.clone().detach().requires_grad_(True)
                    for k, v in self.base_policy.named_parameters()
                }

                try:
                    adapted_params, inner_stats = self._inner_update(
                        task_env, base_params
                    )

                    # FIX 2: Check for NaN after inner update
                    if self._has_nan_in_dict(adapted_params):
                        print(f"WARNING: NaN in adapted_params, skipping task")
                        continue

                    all_inner_stats.append(inner_stats)

                    # Collect QUERY trajectories
                    query_trajs = []
                    for _ in range(self.traj_per_task):
                        traj = self._collect_trajectory(task_env, adapted_params)
                        query_trajs.append(traj)

                    data = self._flatten_trajectories(query_trajs)

                    all_query_data.append(
                        {
                            "data": data,
                            "adapted_params": adapted_params,
                            "base_params": base_params,
                        }
                    )
                except Exception as e:
                    print(f"ERROR in task processing: {e}")
                    continue

            if len(all_query_data) == 0:
                print(f"WARNING: No valid tasks at iteration {iteration}, skipping")
                continue

            meta_stats = self._outer_step(all_query_data)

            if iteration % eval_interval == 0:
                avg_inner = {
                    k: np.mean([s[k] for s in all_inner_stats])
                    for k in all_inner_stats[0]
                }
                print(
                    f"\nIteration {iteration}:",
                    f"\nMeta  Loss: policy={meta_stats['policy_loss']:.5f}  "
                    f"value={meta_stats['value_loss']:.5f}  "
                    f"entropy={meta_stats['entropy']:.5f}",
                    f"\nInner Loss: policy={avg_inner['policy_loss']:.5f}  "
                    f"value={avg_inner['value_loss']:.3f}  "
                    f"entropy={avg_inner['entropy']:.3f}",
                )
                if eval_env_fn:
                    self.evaluate(
                        eval_env_fn,
                        num_episodes=self.outer_batch_size,
                        adaptation_steps=self.inner_steps,
                        num_trajectories=self.traj_per_task,
                    )

            for env in task_envs:
                env.close()

    # ====================================================================
    #  TRPO Outer Update
    # ====================================================================
    def _outer_step(self, all_query_data: List[Dict]) -> Dict:
        """
        Correct clone-and-accumulate MAML gradient, then TRPO step.
        """
        num_tasks = len(all_query_data)
        self.base_policy.zero_grad()

        total_policy_loss = 0.0
        total_entropy = 0.0

        for task_data in all_query_data:
            data = task_data["data"]
            adapted_params = task_data["adapted_params"]
            base_params = task_data["base_params"]

            _, new_logprobs, entropy = self.base_policy.evaluate_action(
                data["states"], data["actions"], adapted_params
            )
            policy_loss = -(new_logprobs * data["advantages"]).mean()
            entropy_loss = entropy.mean()
            task_loss = policy_loss - self.meta_ent_coef * entropy_loss

            total_policy_loss += policy_loss.item()
            total_entropy += entropy_loss.item()

            task_grads = torch.autograd.grad(
                task_loss,
                base_params.values(),
                allow_unused=True,
                materialize_grads=True,
            )

            # Accumulate gradients
            for param, g in zip(self.base_policy.parameters(), task_grads):
                if g is not None:
                    # FIX 3: Check for NaN in gradients
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        print(f"WARNING: NaN/Inf in gradient for task, skipping")
                        continue

                    if param.grad is None:
                        param.grad = g.detach().clone() / num_tasks
                    else:
                        param.grad.add_(g.detach() / num_tasks)

        # FIX 4: Check accumulated gradients
        if any(
            p.grad is not None
            and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in self.base_policy.actor.parameters()
        ):
            print("ERROR: NaN/Inf in accumulated gradients, skipping update")
            return {
                "policy_loss": total_policy_loss / num_tasks,
                "value_loss": 0.0,
                "entropy": total_entropy / num_tasks,
            }

        # Flat policy gradient
        policy_grad_flat = torch.cat(
            [
                p.grad.view(-1)
                for p in self.base_policy.actor.parameters()
                if p.grad is not None
            ]
        )

        # TRPO: Fisher-vector product
        query_states = torch.cat([td["data"]["states"] for td in all_query_data])
        query_actions = torch.cat([td["data"]["actions"] for td in all_query_data])

        with torch.no_grad():
            _, ref_logprobs, _ = self.base_policy.evaluate_action(
                query_states, query_actions
            )
        ref_logprobs = ref_logprobs.detach()

        def fvp(v: torch.Tensor) -> torch.Tensor:
            """Fisher-vector product with improved stability"""
            _, new_lp, _ = self.base_policy.evaluate_action(query_states, query_actions)
            kl = (ref_logprobs - new_lp).mean()

            grad_kl = torch.autograd.grad(
                kl,
                self.base_policy.actor.parameters(),
                create_graph=True,
                retain_graph=True,
            )
            grad_kl_flat = torch.cat([g.view(-1) for g in grad_kl])

            # FIX 5: Check for NaN in KL gradient
            if torch.isnan(grad_kl_flat).any() or torch.isinf(grad_kl_flat).any():
                print("WARNING: NaN/Inf in KL gradient")
                return v * 0.0  # Return zero to skip this update

            hvp = torch.autograd.grad(
                (grad_kl_flat * v.detach()).sum(),
                self.base_policy.actor.parameters(),
                retain_graph=False,
            )
            hvp_flat = torch.cat([h.view(-1) for h in hvp])

            # FIX 6: Increased damping for stability
            return hvp_flat + self.cg_damping * v

        # CG → natural gradient direction
        x = self._conjugate_gradient(fvp, policy_grad_flat.detach())

        # FIX 7: Check CG result
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("ERROR: NaN/Inf in CG result, skipping policy update")
            # Still do value update
            self.value_optimizer.zero_grad()
            value_loss = self._recompute_value_loss(all_query_data)
            if not (torch.isnan(value_loss) or torch.isinf(value_loss)):
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.base_policy.critic.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

            return {
                "policy_loss": total_policy_loss / num_tasks,
                "value_loss": value_loss.item() if not torch.isnan(value_loss) else 0.0,
                "entropy": total_entropy / num_tasks,
            }

        # Step size
        xFx = (fvp(x.detach()) * x.detach()).sum()
        alpha = torch.sqrt(2.0 * self.delta / (xFx + 1e-8))
        step = alpha * x.detach()

        # Line search
        old_actor_params = self._get_flat_actor_params()
        old_loss_val = total_policy_loss / num_tasks

        success = self._line_search(
            old_actor_params,
            step,
            query_states,
            query_actions,
            ref_logprobs,
            old_loss_val,
            all_query_data,
        )
        if not success:
            print("  Line search failed — reverting.")
            self._set_flat_actor_params(old_actor_params)

        # Value update
        self.value_optimizer.zero_grad()
        value_loss = self._recompute_value_loss(all_query_data)

        # FIX 8: Check value loss before backward
        if torch.isnan(value_loss) or torch.isinf(value_loss):
            print("WARNING: NaN/Inf in value loss, skipping value update")
            return {
                "policy_loss": total_policy_loss / num_tasks,
                "value_loss": 0.0,
                "entropy": total_entropy / num_tasks,
            }

        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.base_policy.critic.parameters(), self.max_grad_norm
        )
        self.value_optimizer.step()

        return {
            "policy_loss": total_policy_loss / num_tasks,
            "value_loss": value_loss.item(),
            "entropy": total_entropy / num_tasks,
        }

    # ====================================================================
    #  Conjugate Gradient
    # ====================================================================
    def _conjugate_gradient(
        self,
        fvp_fn: Callable,
        g: torch.Tensor,
        max_iters: Optional[int] = None,
    ) -> torch.Tensor:
        if max_iters is None:
            max_iters = self.cg_iters

        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()
        rdotr = r.dot(r)

        for i in range(max_iters):
            # FIX 9: Better NaN handling in CG
            if torch.isnan(rdotr) or torch.isinf(rdotr):
                print(f"  CG: NaN/Inf at iteration {i}, returning zero")
                return torch.zeros_like(g)

            Fp = fvp_fn(p)

            # Check Fp for NaN
            if torch.isnan(Fp).any() or torch.isinf(Fp).any():
                print(f"  CG: NaN/Inf in Fp at iteration {i}, returning current x")
                return x

            pFp = p.dot(Fp)

            # FIX 10: Prevent division by very small numbers
            if abs(pFp) < 1e-10:
                print(f"  CG: pFp too small at iteration {i}, stopping")
                return x

            a = rdotr / (pFp + 1e-8)
            x += a * p
            r -= a * Fp
            rdr_new = r.dot(r)

            if rdr_new < self.residual_tol:
                print(f"  CG converged at iteration {i}")
                break
            p = r + (rdr_new / rdotr) * p
            rdotr = rdr_new

        return x

    # ====================================================================
    #  Line Search
    # ====================================================================
    def _line_search(
        self,
        old_params: torch.Tensor,
        step: torch.Tensor,
        query_states: torch.Tensor,
        query_actions: torch.Tensor,
        ref_logprobs: torch.Tensor,
        old_loss: float,
        all_query_data: List[Dict],
    ) -> bool:
        for i in range(self.backtrack_iters):
            self._set_flat_actor_params(old_params + self.backtrack_coeff**i * step)

            # FIX 11: Check for NaN after parameter update
            if self._has_nan_params(self.base_policy.actor):
                print(f"  Line search step {i}: NaN in params, trying smaller step")
                continue

            with torch.no_grad():
                _, new_lp, _ = self.base_policy.evaluate_action(
                    query_states, query_actions
                )

                # Check for NaN in logprobs
                if torch.isnan(new_lp).any() or torch.isinf(new_lp).any():
                    print(
                        f"  Line search step {i}: NaN in logprobs, trying smaller step"
                    )
                    continue

                kl = (ref_logprobs - new_lp).mean().item()
                new_loss = self._compute_meta_loss_no_grad(all_query_data)

            if kl <= self.delta and new_loss <= old_loss:
                print(f"  Line search OK (step {i}): KL={kl:.5f} loss={new_loss:.5f}")
                return True

        return False

    # ====================================================================
    #  Helpers
    # ====================================================================
    def _has_nan_params(self, module: nn.Module) -> bool:
        """Check if any parameter in the module contains NaN or Inf"""
        for param in module.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False

    def _has_nan_in_dict(self, params_dict: Dict[str, torch.Tensor]) -> bool:
        """Check if any tensor in the dict contains NaN or Inf"""
        for v in params_dict.values():
            if torch.isnan(v).any() or torch.isinf(v).any():
                return True
        return False

    @torch.no_grad()
    def _compute_meta_loss_no_grad(self, all_query_data: List[Dict]) -> float:
        losses = []
        for td in all_query_data:
            try:
                _, lp, ent = self.base_policy.evaluate_action(
                    td["data"]["states"], td["data"]["actions"], td["adapted_params"]
                )
                loss = (
                    -(lp * td["data"]["advantages"]).mean()
                    - self.meta_ent_coef * ent.mean()
                ).item()

                if not (np.isnan(loss) or np.isinf(loss)):
                    losses.append(loss)
            except:
                continue

        return float(np.mean(losses)) if losses else float("inf")

    def _recompute_value_loss(self, all_query_data: List[Dict]) -> torch.Tensor:
        vals, rets = [], []
        for td in all_query_data:
            v, _, _ = self.base_policy.evaluate_action(
                td["data"]["states"], td["data"]["actions"], td["adapted_params"]
            )
            vals.append(v)
            rets.append(td["data"]["returns"])
        return F.mse_loss(torch.cat(vals).squeeze(-1), torch.cat(rets))

    def _get_flat_actor_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.base_policy.actor.parameters()])

    def _set_flat_actor_params(self, flat: torch.Tensor):
        offset = 0
        for p in self.base_policy.actor.parameters():
            n = p.numel()
            p.data.copy_(flat[offset : offset + n].view(p.shape))
            offset += n

    # ====================================================================
    #  Inner update: REINFORCE for fast adaptation
    # ====================================================================
    def _inner_update(self, task_env, base_params):
        """
        Perform differentiable updates on support trajectories.
        FIX 12: Added gradient clipping and stability checks
        """
        fast_params = {k: v.clone() for k, v in base_params.items()}

        all_stats = []
        for step_idx in range(self.inner_steps):

            # Collect support trajectories
            support_trajs = []
            for _ in range(self.traj_per_task):
                traj = self._collect_trajectory(task_env, fast_params)
                support_trajs.append(traj)
            data = self._flatten_trajectories(support_trajs)

            # Calculate loss
            loss, stats = self._reinforce_loss(data, fast_params, looptype="inner")

            # FIX 13: Check loss for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf in inner loss at step {step_idx}")
                break

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                fast_params.values(),
                create_graph=self.second_order,
                allow_unused=True,
                materialize_grads=True,
            )

            # FIX 14: Clip gradients and check for NaN
            clipped_grads = []
            grad_norm = 0.0
            for grad in grads:
                if grad is not None:
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(
                            f"  WARNING: NaN/Inf in inner gradient at step {step_idx}"
                        )
                        return fast_params, {
                            "policy_loss": 0,
                            "value_loss": 0,
                            "entropy": 0,
                        }
                    grad_norm += grad.norm().item() ** 2
                    clipped_grads.append(grad)
                else:
                    clipped_grads.append(None)

            grad_norm = grad_norm**0.5

            # FIX 15: Clip inner gradients
            if grad_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (grad_norm + 1e-8)
                clipped_grads = [
                    g * clip_coef if g is not None else None for g in clipped_grads
                ]

            # Update fast_params
            for (k, v), grad in zip(fast_params.items(), clipped_grads):
                if grad is not None:
                    fast_params[k] = v - self.inner_lr * grad
                else:
                    # Keep parameter unchanged if gradient is None
                    fast_params[k] = v

            all_stats.append(stats)

        avg_stats = {
            key: np.mean([s[key] for s in all_stats]) for key in all_stats[0].keys()
        }

        return fast_params, avg_stats

    # ====================================================================
    #  Loss
    # ====================================================================
    def _reinforce_loss(
        self,
        data: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
        looptype: str = "inner",
    ) -> Tuple[torch.Tensor, Dict]:

        values, new_logprobs, entropy = self.base_policy.evaluate_action(
            data["states"], data["actions"], params
        )
        policy_loss = -(new_logprobs * data["advantages"]).mean()
        value_loss = F.mse_loss(values.squeeze(-1), data["returns"])
        entropy_loss = entropy.mean()

        if looptype == "inner":
            total_loss = (
                policy_loss
                + self.inner_vf_coef * value_loss
                - self.inner_ent_coef * entropy_loss
            )
        else:
            total_loss = (
                policy_loss
                + self.meta_vf_coef * value_loss
                - self.meta_ent_coef * entropy_loss
            )

        logratio = new_logprobs - data["logprobs"]
        approx_kl = ((logratio.exp() - 1) - logratio).mean().item()
        y_pred = values.detach().cpu().numpy()
        y_true = data["returns"].cpu().numpy()
        ev = (
            np.nan
            if np.var(y_true) == 0
            else 1 - np.var(y_true - y_pred) / np.var(y_true)
        )

        stats = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "explained_var": ev,
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

            delta = (
                rewards[t] + self.gamma * next_val * (1 - dones[t]) - values_tensor[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

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

        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    ) -> None:
        rb, ra = [], []
        for _ in range(num_episodes):
            env = env_fn()
            state, _ = env.reset()
            ep = 0.0
            for _ in range(self.max_steps):
                with torch.no_grad():
                    action, _ = self.base_policy.get_action(state, deterministic=True)
                ns, r, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
                ep += r
                state = ns
                if term or trunc:
                    break
            rb.append(ep)

            adapted, _ = self.adapt_to_new_task(env, adaptation_steps, num_trajectories)

            state, _ = env.reset()
            ep = 0.0
            for _ in range(self.max_steps):
                with torch.no_grad():
                    action, _ = adapted.get_action(state, deterministic=True)
                ns, r, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
                ep += r
                state = ns
                if term or trunc:
                    break
            ra.append(ep)
            env.close()

        rb, ra = np.array(rb), np.array(ra)
        imp = ra - rb
        print(
            f"Evaluation: Before {rb.mean():.2f}±{rb.std():.2f} | "
            f"After {ra.mean():.2f}±{ra.std():.2f} | "
            f"Δ {imp.mean():.2f}±{imp.std():.2f} | "
            f"Success {(imp > 0).mean()*100:.1f}%"
        )

    def adapt_to_new_task(
        self,
        task_env: gym.Env,
        adaptation_steps: int = 5,
        num_trajectories: int = 5,
    ) -> Tuple["ActorCriticNetwork", Dict]:
        adapted = self._clone_policy()
        optimizer = optim.Adam(adapted.parameters(), lr=self.inner_lr)
        loss = None
        for _ in range(adaptation_steps):
            trajs = [
                self._collect_trajectory(task_env, dict(adapted.named_parameters()))
                for _ in range(num_trajectories)
            ]
            data = self._flatten_trajectories(trajs)
            v, lp, ent = adapted.evaluate_action(data["states"], data["actions"])
            loss = (
                -(lp * data["advantages"]).mean()
                + self.meta_vf_coef * F.mse_loss(v.squeeze(-1), data["returns"])
                - self.meta_ent_coef * ent.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), self.max_grad_norm)
            optimizer.step()
        return adapted, {"loss": loss.item() if loss else 0.0}

    def _clone_policy(self) -> "ActorCriticNetwork":
        clone = ActorCriticNetwork(self.state_dim, self.action_dim, self.policy_kwargs)
        clone.load_state_dict(self.base_policy.state_dict())
        return clone
