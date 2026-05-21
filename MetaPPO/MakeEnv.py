import random
import numpy as np
import metaworld
import gymnasium as gym


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
