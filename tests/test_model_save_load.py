"""
Unit tests for PPOAgent model saving and loading.
"""

import os

import torch

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
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
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=20,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
            enable_periodic_evaluation=False,
            max_moves_per_game=512,
            log_file_path_eval="/tmp/eval.log",
            wandb_log_eval=False,
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
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
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
