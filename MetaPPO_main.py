from MetaPPO.MakeEnv import *
from MetaPPO.MAML_PPO import MAMLPPO as Meta

# from MetaPPO.MAML_TRPO import MAMLTRPO as Meta
# from MetaPPO.Reptile_PPO import ReptilePPO as Meta


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
