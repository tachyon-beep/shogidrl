"""
agent_loading.py: Utilities for loading PPO agents and initializing opponents.
"""

import os
from typing import Any, Optional

from keisei.utils.opponents import (
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
    import torch  # pylint: disable=import-outside-toplevel

    from keisei.config_schema import (  # pylint: disable=import-outside-toplevel
        AppConfig,
        DemoConfig,
        DisplayConfig,
        EnvConfig,
        EvaluationConfig,
        LoggingConfig,
        ParallelConfig,
        TrainingConfig,
        WandBConfig,
    )
    from keisei.core.neural_network import (
        ActorCritic,
    )  # pylint: disable=import-outside-toplevel
    from keisei.core.ppo_agent import (
        PPOAgent,
    )  # pylint: disable=import-outside-toplevel

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
            minibatch_size=2,  # Updated from 1 to 2
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
            enable_value_clipping=False,  # Added
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            enable_periodic_evaluation=False,  # Moved up
            evaluation_interval_timesteps=50000,
            strategy="single_opponent",  # Added
            num_games=1,
            max_concurrent_games=4,  # Added
            timeout_per_game=None,  # Added
            opponent_type="random",
            max_moves_per_game=500,
            randomize_positions=True,  # Added
            random_seed=None,  # Added
            save_games=True,  # Added
            save_path=None,  # Added
            log_file_path_eval="/tmp/eval.log",
            log_level="INFO",  # Added
            wandb_log_eval=False,
            update_elo=True,  # Added
            elo_registry_path="elo_ratings.json",  # Added
            agent_id=None,  # Added
            opponent_id=None,  # Added
            previous_model_pool_size=5,  # Added
            enable_in_memory_evaluation=True,  # Added
            model_weight_cache_size=5,  # Added
            enable_parallel_execution=True,  # Added
            process_restart_threshold=100,  # Added
            temp_agent_device="cpu",  # Added
            clear_cache_after_evaluation=True,  # Added
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
        display=DisplayConfig(  # Added with default values
            enable_board_display=True,
            enable_trend_visualization=True,
            enable_elo_ratings=True,
            enable_enhanced_layout=True,
            display_moves=False,
            turn_tick=0.5,
            board_unicode_pieces=True,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=True,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=True,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=500,
            show_moves_trend=True,
            show_completion_rate=True,
            show_enhanced_win_rates=True,
            show_turns_trend=True,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=True,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
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
    if opponent_type == "random":
        return SimpleRandomOpponent()
    if opponent_type == "heuristic":
        return SimpleHeuristicOpponent()
    if opponent_type == "ppo":
        if not opponent_path:
            raise ValueError("Opponent path must be provided for PPO opponent type.")
        return load_evaluation_agent(
            opponent_path, device_str, policy_mapper, input_channels
        )

    raise ValueError(f"Unknown opponent type: {opponent_type}")


__all__ = [
    "load_evaluation_agent",
    "initialize_opponent",
]
