from MetaPPO.MakeEnv import *

from MetaPPO.MAML_PPO import MAML
from MetaPPO.MAML_TRPO import MAMLTRPO
from MetaPPO.Reptile_PPO import ReptilePPO
from MetaPPO.SB3_Reptile_PPO import SB3ReptilePPO

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
    alg_name = "MetaPPO"
    # chkpt_dir = f"./results/Meta/{alg_name}/{env_name}/models"
    # chkpt_dir = get_unique_log_dir(chkpt_dir)

    logger = Logger(log_dir=f"./Results/Meta/{alg_name}/{env_name}/tb")

    # ----------------------------
    # MAML MODEL
    # ----------------------------
    match alg_name:
        case "MetaTRPO":
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
        case "MetaPPO":
            meta_ppo = MAML(
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
                traj_per_task=5,
                max_steps=500,
                policy_kwargs={
                    "feature": [],
                    "pi": [128, 128],
                    "vf": [128, 128],
                },
                second_order=True,
                logger=logger,
            )
        case "ReptilePPO":
            meta_reptile = ReptilePPO(
                env_fn=env_factory,
                state_dim=state_dim,
                action_dim=action_dim,
                inner_lr=1e-4,
                inner_vf_coef=0.5,
                inner_ent_coef=1e-5,
                meta_lr=1e-3,
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
        case "SB3ReptilePPO":
            sb3_meta_reptile = SB3ReptilePPO(
                env_fn=env_factory,
                inner_lr=3e-4,
                meta_lr=1e-4,
                inner_vf_coef=0.5,
                inner_ent_coef=0.0,
                inner_epochs=4,
                inner_batch_size=64,
                outer_batch_size=4,
                traj_per_task=4,
                policy_kwargs={
                    "feature": [],
                    "pi": [256, 256],
                    "vf": [256, 256],
                },
                normalize_advantage=True,
                logger=logger,
                verbose=True,
            )

    # ----------------------------
    # TRAINING
    # ----------------------------
    print(f"\n=== Training Meta-Learning with {alg_name} algorithm. ===")

    match alg_name:
        case "MetaTRPO":
            meta_trpo.meta_train(
                num_meta_iterations=1000, eval_interval=1, eval_env_fn=eval_env_factory
            )
        case "MetaPPO":
            meta_ppo.meta_train(
                num_meta_iterations=1000, eval_interval=1, eval_env_fn=eval_env_factory
            )
        case "ReptilePPO":
            meta_reptile.meta_train(
                num_meta_iterations=1000, eval_interval=1, eval_env_fn=eval_env_factory
            )
        case "SB3ReptilePPO":
            sb3_meta_reptile.meta_train(
                num_meta_iterations=1000, eval_interval=1, eval_env_fn=eval_env_factory
            )
