"""
test_trainer_session_integration_fixed.py: Integration tests for Trainer and SessionManager.

Tests that verify the SessionManager is properly integrated into the Trainer
and that session management functionality works correctly end-to-end.
"""

from unittest.mock import Mock, mock_open, patch

import pytest

from keisei.config_schema import (
    AppConfig,
    EnvConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.training.trainer import Trainer


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.run_name = kwargs.get("run_name")
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=AppConfig)

    # Environment config
    env_config = Mock(spec=EnvConfig)
    env_config.seed = 42
    env_config.device = "cpu"
    env_config.num_actions_total = 13527
    env_config.input_channels = 46
    config.env = env_config

    # Training config
    training_config = Mock(spec=TrainingConfig)
    training_config.total_timesteps = 1000
    training_config.steps_per_epoch = 64
    training_config.model_type = "resnet"
    training_config.input_features = "core46"
    training_config.tower_depth = 5
    training_config.tower_width = 128
    training_config.se_ratio = 0.25
    training_config.mixed_precision = False
    training_config.checkpoint_interval_timesteps = 1000
    training_config.evaluation_interval_timesteps = 1000
    training_config.gamma = 0.99
    training_config.lambda_gae = 0.95
    # Add missing required attributes for PPOAgent
    training_config.learning_rate = 3e-4
    training_config.clip_epsilon = 0.2
    training_config.value_loss_coeff = 0.5
    training_config.entropy_coef = 0.01
    training_config.ppo_epochs = 10
    training_config.minibatch_size = 64
    training_config.gradient_clip_max_norm = 0.5
    training_config.weight_decay = 0.0
    training_config.render_every_steps = 1
    training_config.refresh_per_second = 4
    training_config.enable_spinner = True
    training_config.ddp = False
    config.training = training_config

    # Logging config
    logging_config = Mock(spec=LoggingConfig)
    logging_config.run_name = None
    logging_config.log_file = "logs/training_log.txt"
    logging_config.model_dir = "models/"
    config.logging = logging_config

    # WandB config
    wandb_config = Mock(spec=WandBConfig)
    wandb_config.run_name_prefix = "test"
    wandb_config.enabled = False
    wandb_config.project = "test-project"
    wandb_config.entity = None
    wandb_config.watch_model = False
    wandb_config.watch_log_freq = 1000
    wandb_config.watch_log_type = "all"
    config.wandb = wandb_config

    # Demo config
    demo_config = Mock()
    demo_config.enable_demo_mode = False
    demo_config.demo_mode_delay = 0.5
    config.demo = demo_config

    return config


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs()


class TestTrainerSessionIntegration:
    """Test SessionManager integration in Trainer."""

    def test_trainer_initialization_with_session_manager(self, mock_config, mock_args):
        """Test that Trainer properly initializes SessionManager."""
        with (
            patch("keisei.training.utils.setup_seeding"),
            patch("keisei.training.utils.serialize_config") as mock_serialize,
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
            patch("keisei.training.utils.setup_wandb") as mock_setup_wandb,
            patch("builtins.open", mock_open()),
            patch("keisei.shogi.ShogiGame") as mock_shogi_game,
            patch("keisei.shogi.features.FEATURE_SPECS") as mock_feature_specs,
            patch("keisei.utils.PolicyOutputMapper") as mock_policy_mapper,
            patch("keisei.core.ppo_agent.PPOAgent") as mock_ppo_agent,
            patch(
                "keisei.core.experience_buffer.ExperienceBuffer"
            ) as mock_experience_buffer,
            patch("keisei.training.models.model_factory") as mock_model_factory,
        ):

            # Setup mocks
            mock_setup_dirs.return_value = {
                "run_artifact_dir": "/tmp/test_run",
                "model_dir": "/tmp/test_run/models",
                "log_file_path": "/tmp/test_run/training.log",
                "eval_log_file_path": "/tmp/test_run/eval.log",
            }
            mock_setup_wandb.return_value = True
            mock_serialize.return_value = '{"test": "config"}'

            # Mock feature specs
            mock_feature_spec = Mock()
            mock_feature_spec.num_planes = 46
            mock_feature_specs.__getitem__.return_value = mock_feature_spec

            # Mock game components
            mock_game_instance = Mock()
            mock_game_instance.reset.return_value = Mock()
            mock_shogi_game.return_value = mock_game_instance

            # Mock policy mapper
            mock_policy_instance = Mock()
            mock_policy_instance.get_total_actions.return_value = 4096
            mock_policy_mapper.return_value = mock_policy_instance

            # Mock model
            mock_model = Mock()
            mock_model_factory.return_value = mock_model

            # Mock agent
            mock_agent_instance = Mock()
            mock_agent_instance.name = "TestAgent"
            mock_ppo_agent.return_value = mock_agent_instance

            # Mock experience buffer
            mock_buffer_instance = Mock()
            mock_experience_buffer.return_value = mock_buffer_instance

            # Create trainer
            trainer = Trainer(mock_config, mock_args)

            # Verify SessionManager was created and configured
            assert hasattr(trainer, "session_manager")
            assert trainer.session_manager is not None

            # Verify session properties are accessible through trainer
            assert trainer.run_name == trainer.session_manager.run_name
            assert trainer.run_artifact_dir == "/tmp/test_run"
            assert trainer.model_dir == "/tmp/test_run/models"
            assert trainer.log_file_path == "/tmp/test_run/training.log"
            assert trainer.is_train_wandb_active is True

            # Verify session setup methods were called
            mock_setup_dirs.assert_called_once()
            mock_setup_wandb.assert_called_once()
            mock_serialize.assert_called_once()

    def test_trainer_run_name_precedence(self, mock_config):
        """Test that run name precedence works correctly."""
        with (
            patch("keisei.training.utils.setup_seeding"),
            patch("keisei.training.utils.serialize_config"),
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
            patch("keisei.training.utils.setup_wandb") as mock_setup_wandb,
            patch("builtins.open", mock_open()),
            patch("keisei.shogi.ShogiGame"),
            patch("keisei.shogi.features.FEATURE_SPECS") as mock_feature_specs,
            patch("keisei.utils.PolicyOutputMapper"),
            patch("keisei.core.ppo_agent.PPOAgent"),
            patch("keisei.core.experience_buffer.ExperienceBuffer"),
            patch("keisei.training.models.model_factory"),
        ):

            # Setup mocks
            mock_setup_dirs.return_value = {
                "run_artifact_dir": "/tmp/test_run",
                "model_dir": "/tmp/test_run/models",
                "log_file_path": "/tmp/test_run/training.log",
                "eval_log_file_path": "/tmp/test_run/eval.log",
            }
            mock_setup_wandb.return_value = False

            mock_feature_spec = Mock()
            mock_feature_spec.num_planes = 46
            mock_feature_specs.__getitem__.return_value = mock_feature_spec

            # Test CLI args precedence
            args_with_name = MockArgs(run_name="cli_run_name")
            trainer = Trainer(mock_config, args_with_name)
            assert trainer.run_name == "cli_run_name"

    def test_session_manager_finalization(self, mock_config, mock_args):
        """Test that session finalization works correctly."""
        with (
            patch("keisei.training.utils.setup_seeding"),
            patch("keisei.training.utils.serialize_config"),
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
            patch("keisei.training.utils.setup_wandb") as mock_setup_wandb,
            patch("builtins.open", mock_open()),
            patch("keisei.shogi.ShogiGame"),
            patch("keisei.shogi.features.FEATURE_SPECS") as mock_feature_specs,
            patch("keisei.utils.PolicyOutputMapper"),
            patch("keisei.core.ppo_agent.PPOAgent"),
            patch("keisei.core.experience_buffer.ExperienceBuffer"),
            patch("keisei.training.models.model_factory"),
        ):

            # Setup mocks
            mock_setup_dirs.return_value = {
                "run_artifact_dir": "/tmp/test_run",
                "model_dir": "/tmp/test_run/models",
                "log_file_path": "/tmp/test_run/training.log",
                "eval_log_file_path": "/tmp/test_run/eval.log",
            }
            mock_setup_wandb.return_value = True

            mock_feature_spec = Mock()
            mock_feature_spec.num_planes = 46
            mock_feature_specs.__getitem__.return_value = mock_feature_spec

            trainer = Trainer(mock_config, mock_args)

            # Test session finalization
            with patch("wandb.run"), patch("wandb.finish") as mock_wandb_finish:
                trainer.session_manager.finalize_session()
                mock_wandb_finish.assert_called_once()

    def test_session_manager_error_handling(self, mock_config, mock_args):
        """Test that SessionManager error handling works correctly."""
        with (
            patch("keisei.training.utils.setup_seeding"),
            patch("keisei.training.utils.serialize_config"),
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
            patch("keisei.training.utils.setup_wandb"),
            patch("keisei.training.models.model_factory"),
        ):

            # Setup directory failure
            mock_setup_dirs.side_effect = OSError("Permission denied")

            # Should raise RuntimeError from SessionManager
            with pytest.raises(RuntimeError, match="Failed to setup directories"):
                Trainer(mock_config, mock_args)
