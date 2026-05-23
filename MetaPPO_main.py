from MetaPPO.MakeEnv import *

from MetaPPO.MAML_PPO import MAMLPPO
from MetaPPO.MAML_TRPO import MAMLTRPO
from MetaPPO.Reptile_PPO import ReptilePPO

from MetaPPO.logger import Logger
from MetaPPO.utils import get_unique_log_dir


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

    env_name = "Reach"
    alg_name = "ReptilePPO"
    chkpt_dir = f"./results/Meta/{alg_name}/{env_name}/models"
    chkpt_dir = get_unique_log_dir(chkpt_dir)

    logger = Logger(log_dir=f"./results/Meta/{alg_name}/{env_name}/tb")

    # ----------------------------
    # MAML MODEL
    # ----------------------------
    meta_trpo = MAMLTRPO(
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
        outer_batch_size=10,
        traj_per_task=10,
        max_steps=500,
        policy_kwargs={
            "feature": [],
            "pi": [128, 128],
            "vf": [128, 128],
        },
        second_order=True,
        logger=logger,
    )

    meta_ppo = MAMLPPO(
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
        logger=logger,
    )

    meta_reptile = ReptilePPO(
        env_fn=env_factory,
        state_dim=state_dim,
        action_dim=action_dim,
        inner_lr=1e-4,
        inner_vf_coef=0.5,
        inner_ent_coef=1e-5,
        meta_lr=1e-3,
        meta_vf_coef=0.5,
        meta_ent_coef=1e-5,
        inner_steps=1,
        inner_epochs=4,
        inner_batch_size=256,
        outer_batch_size=10,
        traj_per_task=10,
        max_steps=500,
        policy_kwargs={
            "feature": [],
            "pi": [128, 128],
            "vf": [128, 128],
        },
        logger=logger,
    )

    # ----------------------------
    # TRAINING
    # ----------------------------
    print("\n=== Training MAML ===")
    meta_reptile.meta_train(
        num_meta_iterations=1000, eval_interval=1, eval_env_fn=eval_env_factory
    )
