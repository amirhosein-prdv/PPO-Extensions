import random
import numpy as np
import metaworld
import gymnasium as gym

from MetaPPO.MAML_PPO import MAMLPPO as Meta

# from MetaPPO.MAML_TRPO import MAMLTRPO as Meta
# from MetaPPO.Reptile_PPO import ReptilePPO as Meta


def create_pendulum_env():
    """Create Pendulum environment with varying parameters."""
    env = gym.make("Pendulum-v1")

    # Modify environment parameters for task variation
    env.unwrapped.g = np.random.uniform(5, 15)

    # env.unwrapped.max_speed = np.random.uniform(6, 10)
    # env.unwrapped.max_torque = np.random.uniform(1.5, 2.5)

    return env


def create_meta_world_env():
    """
    Samples from MetaWorld ML10 TEST tasks (no leakage).
    """
    ml10 = metaworld.ML10()

    env_name = random.choice(list(ml10.test_classes.keys()))
    env = ml10.test_classes[env_name]()

    task = random.choice([t for t in ml10.test_tasks if t.env_name == env_name])
    env.set_task(task)

    return env


def create_reach_env(eval: bool = False):
    """
    Create MetaWorld ML1 reach-v3 environment.
    ML1 = single task (reach), different goals.
    """

    ml1 = metaworld.ML1("reach-v3")

    if not eval:
        env = ml1.train_classes["reach-v3"]()
        tasks = ml1.train_tasks
    else:
        env = ml1.test_classes["reach-v3"]()
        tasks = ml1.test_tasks

    task = random.choice(tasks)
    env.set_task(task)

    return env


def env_factory():
    # return create_pendulum_env()
    # return create_meta_world_env()
    return create_reach_env()


def eval_env_factory():
    # return create_pendulum_env()
    # return create_meta_world_env()
    return create_reach_env(eval=True)


if __name__ == "__main__":

    test_env = env_factory()
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    test_env.close()

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # ----------------------------
    # MAML MODEL
    # ----------------------------
    maml = Meta(
        env_fn=env_factory,
        state_dim=state_dim,
        action_dim=action_dim,
        inner_lr=1e-4,
        inner_vf_coef=0.0,
        inner_ent_coef=1e-5,
        meta_lr=1e-3,
        meta_vf_coef=0.5,
        meta_ent_coef=1e-5,
        inner_steps=1,
        outer_batch_size=20,
        traj_per_task=10,
        max_steps=500,
        policy_kwargs={
            "feature": [],
            "pi": [128, 128],
            "vf": [128, 128],
        },
        second_order=True,
    )

    # ----------------------------
    # TRAINING
    # ----------------------------
    print("\n=== Training MAML ===")
    maml.meta_train(
        num_meta_iterations=1000, eval_interval=1, eval_env_fn=eval_env_factory
    )
