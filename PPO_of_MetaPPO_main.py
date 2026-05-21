# main.py
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import torch

from MetaPPO.PPO import PPOAgent


def train_pendulum(
    env_name: str = "Pendulum-v1",
    num_episodes: int = 500,
    steps_per_episode: int = 200,
    update_frequency: int = 5,
    eval_every: int = 10,
):
    """
    Train PPO agent on Pendulum-v1 environment.
    """
    # Create environment
    env = gym.make(env_name, render_mode=None)  # No rendering during training
    eval_env = gym.make(env_name, render_mode=None)  # For evaluation

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        epochs=5,
        batch_size=64,
        max_grad_norm=0.5,
        policy_kwargs={
            "feature": [],
            "pi": [64] * 3,
            "vf": [64] * 3,
        },
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    running_reward = deque(maxlen=100)

    print("\nStarting training...")
    print("-" * 50)

    for episode in tqdm(range(1, num_episodes + 1), desc="Training Progress"):
        # Collect trajectories for update_frequency episodes
        trajectories = []
        episode_reward = 0
        episode_length = 0

        for _ in range(update_frequency):
            trajectory = agent.collect_trajectory(env, max_steps=steps_per_episode)
            trajectories.append(trajectory)

            # Calculate episode statistics
            episode_reward += sum(trajectory["rewards"]).item()
            episode_length += len(trajectory["rewards"])

        # Update agent with collected trajectories
        update_stats = agent.update(trajectories)

        # Store metrics
        avg_reward = episode_reward / update_frequency
        avg_length = episode_length / update_frequency

        episode_rewards.append(avg_reward)
        episode_lengths.append(avg_length)
        episode_losses.append(update_stats["loss"])
        running_reward.append(avg_reward)

        # Print progress
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Running Reward (100 ep): {np.mean(running_reward):.2f}")
            print(f"  Loss: {update_stats['loss']:.4f}")
            print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {update_stats['value_loss']:.4f}")
            print(f"  Entropy: {update_stats['entropy']:.4f}")

        # Render evaluation
        if episode % eval_every == 0:
            evaluate_agent(agent, eval_env, num_episodes=1, render=False)

    # Plot training results
    plot_training_results(episode_rewards, episode_lengths, episode_losses)

    env.close()
    eval_env.close()

    return agent, episode_rewards


def evaluate_agent(
    agent: PPOAgent,
    env: gym.Env,
    num_episodes: int = 5,
    render: bool = False,
    max_steps: int = 200,
):
    """
    Evaluate the trained agent.
    """
    agent.policy.eval()
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if render:
                env.render()

            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _ = agent.policy.get_action(state_t, deterministic=True)

            next_state, reward, terminated, truncated, _ = env.step(
                action.squeeze(0).cpu().numpy()
            )
            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return episode_rewards


def plot_training_results(rewards, lengths, losses):
    """
    Plot training metrics.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot rewards
    axes[0].plot(rewards, alpha=0.6, label="Episode Reward")
    # Add moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(
            range(window - 1, len(rewards)),
            moving_avg,
            "r-",
            label=f"Moving Avg ({window} episodes)",
        )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot episode lengths
    axes[1].plot(lengths, alpha=0.6)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Lengths")
    axes[1].grid(True, alpha=0.3)

    # Plot losses
    axes[2].plot(losses, alpha=0.6, label="Total Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Training Loss")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()


def test_trained_model(model_path: str, num_episodes: int = 5):
    """
    Test a previously trained model.
    """
    env = gym.make("Pendulum-v1", render_mode="human")

    # Create agent with same architecture
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_kwargs={
            "feature": [],
            "pi": [64] * 3,
            "vf": [64] * 3,
        },
    )

    # Load trained weights
    agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy.eval()

    print(f"\nTesting model: {model_path}")
    rewards = evaluate_agent(agent, env, num_episodes=num_episodes, render=True)
    print(f"Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    # Train the agent
    trained_agent, rewards = train_pendulum(
        env_name="Pendulum-v1",
        num_episodes=1500,  # Number of training episodes
        steps_per_episode=200,  # Max steps per episode
        update_frequency=5,  # Update policy every N episodes
        eval_every=10,  # evaluation every N episodes
    )

    # Optional: Test the trained model
    # test_trained_model("ppo_pendulum_final.pth", num_episodes=3)
