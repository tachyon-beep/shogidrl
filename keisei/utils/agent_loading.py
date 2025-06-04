"""
agent_loading.py: Utilities for loading PPO agents and initializing opponents.
"""

import os
from typing import Any, Optional

from keisei.config_schema import AppConfig, EnvConfig, ParallelConfig, TrainingConfig
from keisei.utils.opponents import (
    BaseOpponent,
    SimpleHeuristicOpponent,
    SimpleRandomOpponent,
)
from keisei.utils.unified_logger import log_error_to_stderr, log_info_to_stderr


def load_evaluation_agent(
    checkpoint_path: str,
    device_str: str,
    policy_mapper,
    input_channels: int,
    input_features: Optional[str] = "core46",
) -> Any:
    import torch

    from keisei.config_schema import (
        AppConfig,
        DemoConfig,
        EnvConfig,
        EvaluationConfig,
        LoggingConfig,
        TrainingConfig,
        WandBConfig,
    )
    from keisei.core.neural_network import ActorCritic
    from keisei.core.ppo_agent import PPOAgent

    if not os.path.isfile(checkpoint_path):
        log_error_to_stderr(
            "AgentLoading", f"Checkpoint file {checkpoint_path} not found"
        )
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
    # Use dummy configs for required fields
    config = AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device=device_str,
            input_channels=input_channels,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=500,
        ),
        training=TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=1,
            learning_rate=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            input_features=input_features or "core46",
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=1,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            enable_periodic_evaluation=False,
            max_moves_per_game=500,
            log_file_path_eval="/tmp/eval.log",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="/tmp/eval.log", model_dir="/tmp/", run_name="eval-run"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="eval",
            entity=None,
            run_name_prefix="eval-run",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )

    # Create temporary model for loading
    device = torch.device(device_str)
    temp_model = ActorCritic(input_channels, policy_mapper.get_total_actions()).to(
        device
    )

    # Create agent with model (dependency injection)
    agent = PPOAgent(
        model=temp_model, config=config, device=device, name="EvaluationAgent"
    )

    agent.load_model(checkpoint_path)
    agent.model.eval()
    log_info_to_stderr(
        "AgentLoading",
        f"Loaded agent from {checkpoint_path} on device {device_str} for evaluation",
    )
    return agent


def initialize_opponent(
    opponent_type: str,
    opponent_path: Optional[str],
    device_str: str,
    policy_mapper,
    input_channels: int,
) -> Any:
    from keisei.core.ppo_agent import PPOAgent

    if opponent_type == "random":
        from keisei.utils.opponents import SimpleRandomOpponent

        return SimpleRandomOpponent()
    elif opponent_type == "heuristic":
        from keisei.utils.opponents import SimpleHeuristicOpponent

        return SimpleHeuristicOpponent()
    elif opponent_type == "ppo":
        if not opponent_path:
            raise ValueError("Opponent path must be provided for PPO opponent type.")
        return load_evaluation_agent(
            opponent_path, device_str, policy_mapper, input_channels
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


__all__ = [
    "load_evaluation_agent",
    "initialize_opponent",
]
