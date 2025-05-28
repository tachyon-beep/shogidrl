"""
test_session_manager_integration.py: Integration tests for SessionManager with Trainer.

These tests verify that SessionManager properly integrates wi    @patch('keisei.training.trainer.ShogiGame')
    @patch('keisei.training.trainer.PPOAgent')
    @patch('keisei.training.trainer.model_factory')
    @patch('keisei.shogi.features.FEATURE_SPECS')
    @patch('keisei.training.utils.setup_directories')
    @patch('keisei.training.utils.setup_wandb')
    @patch('keisei.training.utils.serialize_config')
    @patch('keisei.training.utils.setup_seeding')
    def test_trainer_uses_session_manager(self, mock_setup_seeding, mock_serialize,
                                         mock_setup_wandb, mock_setup_dirs,
                                         mock_feature_specs, mock_model_factory,ner class
and maintains backward compatibility while providing the expected functionality.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from keisei.config_schema import (
    AppConfig,
    EnvConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.training.session_manager import SessionManager
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
    env_config.device = "cuda"
    env_config.num_actions_total = 8192
    env_config.input_channels = 119
    config.env = env_config

    # Training config
    training_config = Mock(spec=TrainingConfig)
    training_config.total_timesteps = 1000000
    training_config.steps_per_epoch = 2048
    training_config.model_type = "resnet"
    training_config.input_features = "core46"
    training_config.mixed_precision = False
    training_config.tower_depth = 9
    training_config.tower_width = 256
    training_config.se_ratio = 0.25
    training_config.ppo_epochs = 10
    training_config.minibatch_size = 64
    training_config.learning_rate = 3e-4
    training_config.gamma = 0.99
    training_config.lambda_gae = 0.95
    training_config.clip_epsilon = 0.2
    training_config.value_loss_coeff = 0.5
    training_config.entropy_coef = 0.01
    training_config.checkpoint_interval_timesteps = 10000
    training_config.evaluation_interval_timesteps = 50000
    config.training = training_config

    # Logging config
    logging_config = Mock(spec=LoggingConfig)
    logging_config.run_name = None
    config.logging = logging_config

    # WandB config
    wandb_config = Mock(spec=WandBConfig)
    wandb_config.run_name_prefix = "keisei"
    wandb_config.enabled = False
    config.wandb = wandb_config

    # Demo config
    demo_config = Mock()
    demo_config.enable_demo_mode = False
    demo_config.demo_mode_delay = 0.1
    config.demo = demo_config

    return config


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs(resume=None)


class TestSessionManagerTrainerIntegration:
    """Test SessionManager integration with Trainer class."""

    @patch("keisei.training.trainer.ShogiGame")
    @patch("keisei.training.trainer.PPOAgent")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.utils.setup_directories")
    @patch("keisei.training.utils.setup_wandb")
    @patch("keisei.training.utils.serialize_config")
    @patch("keisei.training.utils.setup_seeding")
    def test_trainer_uses_session_manager(
        self,
        mock_setup_seeding,
        mock_serialize,
        mock_setup_wandb,
        mock_setup_dirs,
        mock_feature_specs,
        mock_model_factory,
        mock_ppo_agent,
        mock_shogi_game,
        mock_config,
        mock_args,
    ):
        """Test that Trainer properly uses SessionManager for session management."""
        # Setup mocks
        mock_setup_dirs.return_value = {
            "run_artifact_dir": "/tmp/test_run",
            "model_dir": "/tmp/test_run/models",
            "log_file_path": "/tmp/test_run/training.log",
            "eval_log_file_path": "/tmp/test_run/eval.log",
        }
        mock_setup_wandb.return_value = False
        mock_serialize.return_value = '{"test": "config"}'

        # Mock feature spec
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 119
        mock_feature_specs.__getitem__.return_value = mock_feature_spec

        # Mock model factory
        mock_model = Mock()
        mock_model_factory.return_value = mock_model

        # Mock agent
        mock_agent_instance = Mock()
        mock_agent_instance.name = "TestAgent"
        mock_ppo_agent.return_value = mock_agent_instance

        # Mock game
        mock_game_instance = Mock()
        mock_game_instance.reset.return_value = Mock()
        mock_shogi_game.return_value = mock_game_instance

        with (
            patch("builtins.open"),
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
            patch("glob.glob", return_value=[]),
            patch("os.makedirs"),
            patch("os.path.exists", return_value=True),
        ):
            trainer = Trainer(mock_config, mock_args)

        # Verify SessionManager is created and used
        assert hasattr(trainer, "session_manager")
        assert isinstance(trainer.session_manager, SessionManager)

        # Verify session properties are accessible through trainer
        assert trainer.run_name == trainer.session_manager.run_name
        assert trainer.run_artifact_dir == trainer.session_manager.run_artifact_dir
        assert trainer.model_dir == trainer.session_manager.model_dir
        assert trainer.log_file_path == trainer.session_manager.log_file_path
        assert trainer.is_train_wandb_active == trainer.session_manager.is_wandb_active

        # Verify setup methods were called
        mock_setup_dirs.assert_called_once()
        mock_setup_wandb.assert_called_once()
        mock_serialize.assert_called_once()
        mock_setup_seeding.assert_called_once()

    @patch("keisei.training.trainer.ShogiGame")
    @patch("keisei.training.trainer.PPOAgent")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.utils.setup_directories")
    @patch("keisei.training.utils.setup_wandb")
    @patch("keisei.training.utils.serialize_config")
    @patch("keisei.training.utils.setup_seeding")
    def test_trainer_session_info_logging(
        self,
        mock_setup_seeding,
        mock_serialize,
        mock_setup_wandb,
        mock_setup_dirs,
        mock_feature_specs,
        mock_model_factory,
        mock_ppo_agent,
        mock_shogi_game,
        mock_config,
        mock_args,
    ):
        """Test that Trainer uses SessionManager for session info logging."""
        # Setup mocks
        mock_setup_dirs.return_value = {
            "run_artifact_dir": "/tmp/test_run",
            "model_dir": "/tmp/test_run/models",
            "log_file_path": "/tmp/test_run/training.log",
            "eval_log_file_path": "/tmp/test_run/eval.log",
        }
        mock_setup_wandb.return_value = False
        mock_serialize.return_value = '{"test": "config"}'

        # Mock feature spec
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 119
        mock_feature_specs.__getitem__.return_value = mock_feature_spec

        # Mock model factory
        mock_model = Mock()
        mock_model_factory.return_value = mock_model

        # Mock agent
        mock_agent_instance = Mock()
        mock_agent_instance.name = "TestAgent"
        mock_agent_instance.model = "MockModel"
        mock_ppo_agent.return_value = mock_agent_instance

        # Mock game
        mock_game_instance = Mock()
        mock_game_instance.reset.return_Value = Mock()
        mock_shogi_game.return_value = mock_game_instance

        with (
            patch("builtins.open"),
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
            patch("glob.glob", return_value=[]),
            patch("os.makedirs"),
            patch("os.path.exists", return_value=True),
        ):
            trainer = Trainer(mock_config, mock_args)

        # Test session info logging
        logged_messages = []

        def mock_log_both(msg, **kwargs):
            logged_messages.append(msg)

        # Mock the session manager's log_session_info method
        with patch.object(
            trainer.session_manager, "log_session_info"
        ) as mock_log_session:
            trainer._log_run_info(mock_log_both)

            # Verify SessionManager's log_session_info was called
            mock_log_session.assert_called_once()

            # Verify it was called with correct parameters
            call_args = mock_log_session.call_args
            assert call_args[1]["agent_info"]["type"] == "Mock"
            assert call_args[1]["agent_info"]["name"] == "TestAgent"
            assert call_args[1]["global_timestep"] == 0
            assert call_args[1]["total_episodes_completed"] == 0

    def test_session_manager_standalone_functionality(self, mock_config, mock_args):
        """Test that SessionManager works independently for basic operations."""
        session_manager = SessionManager(
            mock_config, mock_args, run_name="test_session"
        )

        # Test basic properties
        assert session_manager.run_name == "test_session"

        # Test properties raise errors before setup
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = session_manager.run_artifact_dir

        with pytest.raises(RuntimeError, match="WandB not yet initialized"):
            _ = session_manager.is_wandb_active

    @patch("keisei.training.utils.setup_directories")
    @patch("keisei.training.utils.setup_wandb")
    @patch("keisei.training.utils.serialize_config")
    @patch("builtins.open")
    @patch("os.path.join")
    def test_session_manager_full_workflow(
        self,
        mock_join,
        mock_open,
        mock_serialize,
        mock_setup_wandb,
        mock_setup_dirs,
        mock_config,
        mock_args,
    ):
        """Test SessionManager complete workflow."""
        # Setup mocks
        mock_setup_dirs.return_value = {
            "run_artifact_dir": "/tmp/test_session",
            "model_dir": "/tmp/test_session/models",
            "log_file_path": "/tmp/test_session/training.log",
            "eval_log_file_path": "/tmp/test_session/eval.log",
        }
        mock_setup_wandb.return_value = True
        mock_serialize.return_value = '{"session": "config"}'
        mock_join.return_value = "/tmp/test_session/effective_config.json"

        session_manager = SessionManager(
            mock_config, mock_args, run_name="test_session"
        )

        # Execute full workflow
        dirs = session_manager.setup_directories()
        wandb_active = session_manager.setup_wandb()
        session_manager.save_effective_config()

        # Verify workflow results
        assert dirs["run_artifact_dir"] == "/tmp/test_session"
        assert wandb_active is True
        assert session_manager.is_wandb_active is True
        assert session_manager.run_artifact_dir == "/tmp/test_session"

        # Test session summary
        summary = session_manager.get_session_summary()
        assert summary["run_name"] == "test_session"
        assert summary["is_wandb_active"] is True
        assert summary["seed"] == 42

    def test_session_manager_backward_compatibility(self, mock_config, mock_args):
        """Test that SessionManager maintains backward compatibility."""
        # Test run name precedence (same as before)
        explicit_name = "explicit_run"
        session_manager = SessionManager(mock_config, mock_args, run_name=explicit_name)
        assert session_manager.run_name == explicit_name

        # Test with args run name
        args_name = "args_run"
        args_with_name = MockArgs(run_name=args_name)
        session_manager2 = SessionManager(mock_config, args_with_name)
        assert session_manager2.run_name == args_name

        # Test with config run name
        config_name = "config_run"
        mock_config.logging.run_name = config_name
        session_manager3 = SessionManager(mock_config, mock_args)
        assert session_manager3.run_name == config_name


class TestSessionManagerErrorHandling:
    """Test SessionManager error handling scenarios."""

    def test_directory_setup_failure(self, mock_config, mock_args):
        """Test handling of directory setup failures."""
        session_manager = SessionManager(mock_config, mock_args, run_name="test")

        with patch(
            "keisei.training.utils.setup_directories",
            side_effect=OSError("Permission denied"),
        ):
            with pytest.raises(RuntimeError, match="Failed to setup directories"):
                session_manager.setup_directories()

    def test_wandb_setup_failure(self, mock_config, mock_args):
        """Test handling of WandB setup failures."""
        session_manager = SessionManager(mock_config, mock_args, run_name="test")
        session_manager._run_artifact_dir = "/tmp/test"

        with patch(
            "keisei.training.utils.setup_wandb", side_effect=Exception("WandB error")
        ):
            with patch("sys.stderr"):
                result = session_manager.setup_wandb()
                assert result is False
                assert session_manager.is_wandb_active is False

    def test_config_saving_failure(self, mock_config, mock_args):
        """Test handling of config saving failures."""
        session_manager = SessionManager(mock_config, mock_args, run_name="test")
        session_manager._run_artifact_dir = "/tmp/test"

        with patch(
            "keisei.training.utils.serialize_config",
            side_effect=TypeError("Serialization error"),
        ):
            with pytest.raises(RuntimeError, match="Failed to save effective config"):
                session_manager.save_effective_config()
