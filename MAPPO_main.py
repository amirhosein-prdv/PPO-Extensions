from typing import Optional
import gymnasium as gym
import numpy as np
from MAPPO.MultiAgent import MultiAgent
from MAPPO.utils import plot_learning_curve, get_unique_log_dir
from MAPPO.logger import Logger, StepLogger, EvaluationLogger


def make_env(gym_id, seed: Optional[int] = None):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeRewards(env)
        env = gym.wrappers.TransformRewards(
            env, lambda rewards: np.clip(rewards, -10, 10)
        )
        if seed is not None:
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    env = make_env("BipedalWalker-v3")
    env = env()
    # eval_env = make_env("BipedalWalker-v3")
    # eval_env = eval_env()

    possible_agents = env.possible_agents

    episode_num = 4000
    training_interval_step = 200
    timeLimit = env.spec.max_episode_steps

    chkpt_dir = "./results/models/PPO"
    chkpt_dir = get_unique_log_dir(chkpt_dir)

    # figure_folder = "./results/plots/PPO"
    # figure_file = get_unique_log_dir(figure_folder) + "/LearningCurve.png"

    logger = Logger(log_dir="./results/tb/PPO")

    policy_kwargs = {
        "feature": [64, 32],
        "pi": [256, 256],
        "vf": [256, 128],
    }

    multiAgents = MultiAgent(
        env=env,
        batch_size=32,
        n_epochs=15,
        logger=logger,
        policy_kwargs=policy_kwargs,
        chkpt_dir=chkpt_dir,
    )

    # episode_logger = StepLogger(logger, step_interval=1, suffix_title="eps")
    # eval_logger = EvaluationLogger(eval_env, multiAgents, logger, eval_episodes=10)

    best_score = -np.inf
    score_history = []
    reward_history = []

    learn_steps = timeLimit * episode_num // training_interval_step
    learn_iters = 0
    avg_score = 0

    n_steps = 0
    for eps in range(episode_num):
        states, infos = env.reset()
        last_done = False
        last_dones = {agent: False for agent in possible_agents}
        score = 0
        t_step = 0
        while not last_done:
            multiAgents.eval()
            actions, values, logprobs = multiAgents.get_actions(states)
            next_states, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                agent: terminations[agent] or truncations[agent]
                for agent in possible_agents
            }
            done = any(terminations.values()) or any(truncations.values())
            score += sum([v for v in rewards.values()])

            n_steps += 1
            t_step += 1
            # episode_logger.add_info(infos)
            logger.update_global_step(n_steps)
            logger.add_dict("rollout/step_rewards", rewards, n_steps)

            multiAgents.memory.store(
                states,
                actions,
                logprobs,
                values,
                rewards,
                last_dones,
            )

            if n_steps % training_interval_step == 0:
                multiAgents.anneal_lr(current_step=learn_iters, total_steps=learn_steps)
                last_values = multiAgents.get_values(next_states)
                multiAgents.memory.compute_GAE_and_returns(last_values, dones)
                multiAgents.learn()
                # eval_logger.evaluate_and_log()
                learn_iters += 1
            states = next_states
            last_dones = dones
            last_done = done

        # episode_logger.record_log()

        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        mean_reward = score / t_step
        reward_history.append(mean_reward)
        avg_reward = np.mean(reward_history[-10:])

        logger.add_scalar("rollout/reward", mean_reward, n_steps)
        logger.add_scalar("rollout/avg_reward", avg_reward, n_steps)
        logger.add_scalar("rollout/ep_rew", score, n_steps)
        logger.add_scalar("rollout/ep_rew_mean", avg_score, n_steps)
        logger.add_scalar("rollout/ep_len_mean", t_step, n_steps)

        if avg_score > best_score:
            best_score = avg_score
            multiAgents.save_models()

        print(
            "episode",
            eps,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "global_steps",
            n_steps,
            "episode_step",
            t_step,
        )

    # x = [i + 1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
