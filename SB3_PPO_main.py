import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


env = gym.make("Hopper-v5")
env = gym.wrappers.NormalizeObservation(env)
eval_env = gym.make("Hopper-v5", render_mode="rgb_array")

# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./SB3_results/Hopper/models/",
    log_path="./SB3_results/Hopper/logs/",
    eval_freq=500,
    deterministic=True,
    render=True,
    verbose=0,
)

# Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#     save_freq=1000,
#     save_path="./SB3_results/Hopper/models/",
#     name_prefix="hopper_model",
#     save_replay_buffer=True,
#     save_vecnormalize=True,
# )

callback = CallbackList(
    [
        # checkpoint_callback,
        eval_callback,
    ]
)

policy_kwargs = dict(net_arch=[256] * 2)
model = PPO(
    "MlpPolicy",
    env=env,
    n_steps=1024,
    verbose=0,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./SB3_results/Hopper/tb/",
    ent_coef=0.00229519,
    gae_lambda=0.99,
    gamma=0.999,
    learning_rate=9.8e-5,
    max_grad_norm=0.7,
    clip_range=0.2,
    vf_coef=0.835671,
)

model.learn(
    total_timesteps=4_500_000,
    tb_log_name="Hopper-ppo",
    progress_bar=True,
    callback=callback,
)


# model = PPO.load(
#     "./SB3_results/Hopper/models/best_model.zip",
# )

eval_env = Monitor(eval_env)
eval_env = gym.wrappers.RecordVideo(
    eval_env, "./SB3_results/Hopper/Video_4/", lambda x: True
)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward}, Std: {std_reward}")
