from typing import Optional
import gymnasium as gym
import numpy as np
import torch
from PPO.Agent import Agent
from PPO.utils import get_unique_log_dir, plot_learning_curve
from PPO.logger import Logger, StepLogger, EvaluationLogger


def make_env(gym_id, seed: Optional[int] = None):
    def thunk():
        env = gym.make(gym_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(
        #     env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        # )
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if seed is not None:
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    env_name = "Hopper"
    env = make_env("Hopper-v5")
    env = env()
    # eval_env = make_env("Hopper-v5")
    # eval_env = eval_env()
    max_action = env.action_space.high[0]

    n_epochs = 5
    batch_size = 32
    episode_num = 15000
    training_interval_step = 512
    timeLimit = env.spec.max_episode_steps

    chkpt_dir = f"./results/{env_name}/models/PPO"
    chkpt_dir = get_unique_log_dir(chkpt_dir)

    figure_folder = f"./results/{env_name}/plots/PPO"
    figure_file = get_unique_log_dir(figure_folder) + "/LearningCurve.png"

    logger = Logger(log_dir=f"./results/{env_name}/tb/PPO")

    policy_kwargs = {
        "feature": [],
        "pi": [256] * 2,
        "vf": [256] * 2,
    }

    agent = Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        policy_kwargs=policy_kwargs,
        batch_size=batch_size,
        n_epochs=n_epochs,
        logger=logger,
        chkpt_dir=chkpt_dir,
        ent_coef=0.00229519,
        gae_lambda=0.99,
        gamma=0.999,
        lr=9.8e-5,
        max_grad_norm=0.7,
        clip_coef=0.2,
        vf_coef=0.835671,
    )

    # step_logger = StepLogger(logger, step_interval=1, suffix_title="eps")
    # eval_logger = EvaluationLogger(eval_env, agent, logger, eval_episodes=10)

    best_score = -np.inf
    score_history = []

    learn_steps = timeLimit * episode_num // training_interval_step
    learn_iters = 0
    avg_score = 0

    n_steps = 0
    for eps in range(episode_num):
        state, info = env.reset()
        last_done = False
        score = 0
        t_step = 0
        while not last_done:
            agent.policy.eval()
            with torch.no_grad():
                action, value, logprob = agent.policy(state)
            action = action.detach().cpu().numpy().squeeze()
            logprob = logprob.detach().cpu().numpy()
            value = value.squeeze(0).detach().cpu().numpy()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            n_steps += 1
            t_step += 1
            # step_logger.add_info(info)
            logger.update_global_step(n_steps)
            logger.add_scalar("rollout/step_reward", reward, n_steps)

            # if truncated:
            #     reward += 0.99 * agent.policy.get_value(next_state).detach().item()

            agent.memory.store(
                state,
                action,
                logprob,
                value,
                reward,
                last_done,
            )
            if n_steps % training_interval_step == 0:  # or done: (Think !)
                agent.anneal_lr(current_step=1, total_steps=1000)
                last_value = agent.policy.get_value(next_state).detach().item()
                agent.memory.compute_GAE_and_returns(last_value=last_value, done=done)
                agent.learn()
                # eval_logger.evaluate_and_log()
                learn_iters += 1
            state = next_state
            last_done = done

        # # step_logger.record_log()

        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        logger.add_scalar("rollout/ep_rew", score, n_steps)
        logger.add_scalar("rollout/ep_rew_mean", avg_score, n_steps)
        logger.add_scalar("rollout/ep_len_mean", t_step, n_steps)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode",
            eps + 1,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "global_steps",
            n_steps,
            "episode_step",
            t_step,
        )

    # x = [i + 1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
