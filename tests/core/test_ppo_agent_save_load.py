"""
Unit tests for PPOAgent model saving and loading.
"""

import os

import torch

from keisei.config_schema import (
    AppConfig,
    DisplayConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper


def _create_test_model(config):
    """Helper function to create ActorCritic model for PPOAgent testing."""
    mapper = PolicyOutputMapper()
    return ActorCritic(config.env.input_channels, mapper.get_total_actions())


def test_model_save_and_load(tmp_path):
    """Test saving and loading of the PPO agent's model."""
    # Setup dimensions and policy mapper
    policy_output_mapper = PolicyOutputMapper()
    config = AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=4,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=policy_output_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=512,
        ),
        training=TrainingConfig(
            total_timesteps=500_000,
            steps_per_epoch=2048,
            ppo_epochs=10,
            minibatch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            input_features="core46",
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
            enable_value_clipping=False,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            strategy="single_opponent",
            max_concurrent_games=1,
            timeout_per_game=300.0,
            randomize_positions=False,
            random_seed=None,
            save_games=True,
            save_path=None,
            log_level="INFO",
            update_elo=True,
            enable_in_memory_evaluation=True,
            model_weight_cache_size=5,
            enable_parallel_execution=True,
            process_restart_threshold=100,
            temp_agent_device="cpu",
            clear_cache_after_evaluation=True,
            num_games=20,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            enable_periodic_evaluation=False,
            max_moves_per_game=512,
            log_file_path_eval="/tmp/eval.log",
            wandb_log_eval=False,
            elo_registry_path=None,
            agent_id=None,
            opponent_id=None,
            previous_model_pool_size=5,
        ),
        logging=LoggingConfig(
            log_file="logs/training_log.txt", model_dir="models/", run_name="test_run"
        ),
        wandb=WandBConfig(
            enabled=True,
            project="keisei-shogi",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        display=DisplayConfig(
            enable_board_display=False,
            enable_trend_visualization=False,
            enable_elo_ratings=False,
            enable_enhanced_layout=False,
            display_moves=False,
            turn_tick=0.5,
            board_unicode_pieces=False,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=False,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=False,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=0,
            show_moves_trend=False,
            show_completion_rate=False,
            show_enhanced_win_rates=False,
            show_turns_trend=False,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=False,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
    )

    device = config.env.device

    # Create model for dependency injection
    model = _create_test_model(config)
    agent = PPOAgent(model=model, config=config, device=torch.device(device))
    # Corrected to use agent.model instead of agent.policy
    original_model_state_dict = {
        k: v.cpu() for k, v in agent.model.state_dict().items()
    }

    model_path = tmp_path / "test_model.pth"
    # Provide default values for the new arguments
    agent.save_model(model_path, global_timestep=0, total_episodes_completed=0)

    assert os.path.exists(model_path)

    # Create a new agent and load the model
    new_model = _create_test_model(config)
    new_agent = PPOAgent(model=new_model, config=config, device=torch.device(device))
    new_agent.load_model(model_path)
    # Corrected to use new_agent.model
    loaded_model_state_dict = {
        k: v.cpu() for k, v in new_agent.model.state_dict().items()
    }

    # Compare model parameters
    for key in original_model_state_dict:
        assert torch.equal(
            original_model_state_dict[key], loaded_model_state_dict[key]
        ), f"Model parameter mismatch for key: {key}"

    # Test loading into an agent with a different network instance but same architecture
    third_model = _create_test_model(config)
    third_agent = PPOAgent(
        model=third_model, config=config, device=torch.device(device)
    )
    # Modify some parameters to test that loading restores original values
    # Use a general approach that works with any model structure
    for param in third_agent.model.parameters():
        param.data.fill_(0.12345)
        break  # Just modify the first parameter we find

    third_agent.load_model(model_path)
    # Corrected to use third_agent.model
    third_loaded_model_state_dict = {
        k: v.cpu() for k, v in third_agent.model.state_dict().items()
    }
    for key in original_model_state_dict:
        assert torch.equal(
            original_model_state_dict[key], third_loaded_model_state_dict[key]
        ), f"Model parameter mismatch for key: {key} after loading into a third agent"

    # Clean up
    os.remove(model_path)


def test_agent_checkpoint_save_and_load(tmp_path):
    """Test comprehensive model saving and loading including metadata."""
    # Create a minimal config for model testing
    policy_output_mapper = PolicyOutputMapper()
    config = AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=policy_output_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=512,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=64,
            ppo_epochs=2,
            minibatch_size=32,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=500,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=1,
            opponent_type="random",
            evaluation_interval_timesteps=500,
            enable_periodic_evaluation=False,
            max_moves_per_game=100,
            log_file_path_eval="/tmp/eval.log",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir=str(tmp_path), run_name="checkpoint_test"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        display=DisplayConfig(
            enable_board_display=False,
            enable_trend_visualization=False,
            enable_elo_ratings=False,
            enable_enhanced_layout=False,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=False,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=False,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=False,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=0,
            show_moves_trend=False,
            show_completion_rate=False,
            show_enhanced_win_rates=False,
            show_turns_trend=False,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=False,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
    )

    # Create and initialize agent
    model = _create_test_model(config)
    agent = PPOAgent(model, config, torch.device("cpu"))

    # Save original model and optimizer state
    original_model_state = {k: v.clone() for k, v in agent.model.state_dict().items()}
    original_optimizer_state = agent.optimizer.state_dict()

    # Save model with metadata using save_model
    model_path = tmp_path / "test_model_with_metadata.pth"
    test_timestep = 100
    test_episodes = 10
    agent.save_model(
        str(model_path),
        global_timestep=test_timestep,
        total_episodes_completed=test_episodes,
    )

    # Verify model file exists
    assert model_path.exists()

    # Create new agent and load model
    new_model = _create_test_model(config)
    new_agent = PPOAgent(new_model, config, torch.device("cpu"))

    # Modify new agent's parameters to ensure loading actually changes them
    for param in new_agent.model.parameters():
        param.data.fill_(0.99999)
        break

    # Load model - save_model/load_model only handles model weights, not metadata
    new_agent.load_model(str(model_path))

    # Verify model parameters are restored
    for key in original_model_state:
        assert torch.allclose(
            new_agent.model.state_dict()[key], original_model_state[key], atol=1e-6
        ), f"Model parameter mismatch for key: {key}"

    # Note: PPOAgent.save_model/load_model includes optimizer and scheduler state,
    # so we can also verify optimizer state is preserved
    loaded_optimizer_state = new_agent.optimizer.state_dict()
    assert len(loaded_optimizer_state["param_groups"]) == len(
        original_optimizer_state["param_groups"]
    )
    assert (
        loaded_optimizer_state["param_groups"][0]["lr"]
        == original_optimizer_state["param_groups"][0]["lr"]
    )


def test_agent_model_metadata_saving(tmp_path):
    """Test that model metadata is correctly saved with save_model."""
    policy_output_mapper = PolicyOutputMapper()
    config = AppConfig(
        # ...existing minimal config...
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=policy_output_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=512,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=64,
            ppo_epochs=2,
            minibatch_size=32,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=500,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=1,
            opponent_type="random",
            evaluation_interval_timesteps=500,
            enable_periodic_evaluation=False,
            max_moves_per_game=100,
            log_file_path_eval="/tmp/eval.log",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir=str(tmp_path), run_name="metadata_test"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        display=DisplayConfig(
            enable_board_display=False,
            enable_trend_visualization=False,
            enable_elo_ratings=False,
            enable_enhanced_layout=False,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=False,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=False,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=False,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=0,
            show_moves_trend=False,
            show_completion_rate=False,
            show_enhanced_win_rates=False,
            show_turns_trend=False,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=False,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
    )

    model = _create_test_model(config)
    agent = PPOAgent(model, config, torch.device("cpu"))

    # Test different metadata values
    test_timestep = 12345
    test_episodes = 67

    model_path = tmp_path / "metadata_model.pth"
    agent.save_model(
        str(model_path),
        global_timestep=test_timestep,
        total_episodes_completed=test_episodes,
    )

    # Verify that the model file contains the metadata
    # Note: PPOAgent.save_model includes metadata in the saved dictionary
    loaded_data = torch.load(str(model_path))
    assert loaded_data["global_timestep"] == test_timestep
    assert loaded_data["total_episodes_completed"] == test_episodes

    # Verify that model state is also present
    assert "model_state_dict" in loaded_data
    assert "optimizer_state_dict" in loaded_data


def test_checkpoint_with_lr_scheduler(tmp_path):
    """Test checkpoint saving and loading with learning rate scheduler state."""
    policy_output_mapper = PolicyOutputMapper()
    config = AppConfig(
        # ...existing minimal config with scheduler...
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=policy_output_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=512,
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=64,
            ppo_epochs=2,
            minibatch_size=32,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=500,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type="linear",
            lr_schedule_kwargs={"final_lr_fraction": 0.1},
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=1,
            opponent_type="random",
            evaluation_interval_timesteps=500,
            enable_periodic_evaluation=False,
            max_moves_per_game=100,
            log_file_path_eval="/tmp/eval.log",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir=str(tmp_path), run_name="scheduler_test"
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        display=DisplayConfig(
            enable_board_display=False,
            enable_trend_visualization=False,
            enable_elo_ratings=False,
            enable_enhanced_layout=False,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=False,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=False,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=False,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=0,
            show_moves_trend=False,
            show_completion_rate=False,
            show_enhanced_win_rates=False,
            show_turns_trend=False,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=False,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
        ),
    )

    model = _create_test_model(config)
    agent = PPOAgent(model, config, torch.device("cpu"))

    # Step the scheduler a few times to change its state
    if agent.scheduler:
        for _ in range(5):
            agent.optimizer.step()
            agent.scheduler.step()

    # Save original learning rate
    original_lr = agent.optimizer.param_groups[0]["lr"]

    # Save model using save_model (PPOAgent doesn't have save_checkpoint)
    model_path = tmp_path / "scheduler_model.pth"
    agent.save_model(str(model_path), global_timestep=100, total_episodes_completed=10)

    # Create new agent and load model
    new_model = _create_test_model(config)
    new_agent = PPOAgent(new_model, config, torch.device("cpu"))
    new_agent.load_model(str(model_path))

    # Verify learning rate is restored
    restored_lr = new_agent.optimizer.param_groups[0]["lr"]
    assert (
        abs(restored_lr - original_lr) < 1e-8
    ), f"Learning rate not restored correctly: {restored_lr} vs {original_lr}"
