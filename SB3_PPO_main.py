import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

env_name = "Pendulum"
env = gym.make(f"{env_name}-v1")
env = gym.wrappers.NormalizeObservation(env)
eval_env = gym.make(f"{env_name}-v1", render_mode="rgb_array")

# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./SB3_results/{env_name}/models/",
    log_path=f"./SB3_results/{env_name}/logs/",
    eval_freq=2000,
    deterministic=True,
    render=True,
    verbose=0,
)

# Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#     save_freq=1000,
#     save_path=f"./SB3_results/{env_name}/models/",
#     name_prefix=f"{env_name}_model",
#     save_replay_buffer=True,
#     save_vecnormalize=True,
# )

callback = CallbackList(
    [
        # checkpoint_callback,
        eval_callback,
    ]
)

model = PPO(
    "MlpPolicy",
    env=env,
    n_steps=1000,
    verbose=0,
    tensorboard_log=f"./SB3_results/{env_name}/tb/",
    ent_coef=0.01,
    gae_lambda=0.95,
    gamma=0.99,
    learning_rate=3e-4,
    max_grad_norm=0.5,
    clip_range=0.2,
    vf_coef=0.5,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64] * 3,
            vf=[64] * 3,
        )
    ),
)

model.learn(
    total_timesteps=4_500_000,
    tb_log_name=f"{env_name}-ppo",
    progress_bar=True,
    callback=callback,
)


# model = PPO.load(
#     f"./SB3_results/"env_name"/models/best_model.zip",
# )

eval_env = Monitor(eval_env)
eval_env = gym.wrappers.RecordVideo(
    eval_env, f"./SB3_results/{env_name}/Video/", lambda x: True
)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward}, Std: {std_reward}")
