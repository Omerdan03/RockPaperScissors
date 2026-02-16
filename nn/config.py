from dataclasses import dataclass


@dataclass
class Config:
    # Board
    board_size: int = 6

    # Network
    input_channels: int = 15
    num_res_blocks: int = 6
    hidden_channels: int = 128
    num_actions: int = 144  # 6 * 6 * 4

    # Self-play
    episodes_per_iter: int = 128
    heuristic_ratio: float = 0.5  # fraction of episodes played vs heuristic AI
    max_turns: int = 200
    temp_start: float = 1.0
    temp_end: float = 0.3
    temp_anneal_iters: int = 300

    # PPO
    lr: float = 1e-3
    ppo_epochs: int = 4
    mini_batch_size: int = 512
    clip_eps: float = 0.2
    entropy_coeff: float = 0.02
    value_coeff: float = 0.5
    gamma: float = 0.99
    max_grad_norm: float = 0.5

    # Schedule
    total_iterations: int = 1000
    eval_every: int = 25
    checkpoint_every: int = 50

    # Eval
    eval_games: int = 100
    eval_temperature: float = 0.1

    # Paths
    checkpoint_dir: str = "trained_models"
