from dexrl.utils.configclass import configclass
import os
import time


@configclass
class DexrlCfg:
    # global configs
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(ROOT_PATH, "../outputs")
    cuda: bool = True
    device: str = "cuda" if cuda else "cpu"
    # isaaclab_path: str = global_config["simulation"]["isaaclab_path"]
    # dual_arm_lab_path: str = global_config["simulation"]["dual_arm_lab_path"]

    # experiment configs
    exp_group: str = "example_group"
    exp_name: str = f"example_run/{time.strftime('%Y%m%d_%H%M%S')}"
    enable_wandb: bool = False
    wandb_project_name: str = "erl"  # the wandb's project name
    wandb_entity: str = None  # the entity (team) of wandb's project
    seed: int = 1
    headless: bool = False

    record_video: bool = True
    video_path: str = "videos"
    video_episode_pre: int = 100

    log_period: int = 100

    # env configs
    env_id: str = "Hopper-v4"
    num_envs: int = 1
    exp_path: str = f"{output_path}/{exp_name}"
    disable_fabric: bool = False

    # agent configs
    algo_name: str = "SAC"
    torch_deterministic: bool = True

    # Algorithm specific arguments
    total_timesteps: int = 4000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 10
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.1
    autotune: bool = False

    eval_period: int = 900
    eval_episodes: int = 5
    output_dir: str = "output" # 代码所在的目录

    # demo configs
    demo_path: str = None # 保存demo的目录
    # demo_path: str = os.path.join(ROOT_PATH, "algos/demo_data.pkl")
    pretrain_steps: int = 0


    enable_vision: bool = True

if __name__ == "__main__":
    cfg = DexrlCfg()
    print(cfg)