"""
test_session_management.py: Comprehensive tests for SessionManager and its integration with Trainer.

This file covers the standalone unit tests for the SessionManager class, including
initialization, lifecycle management (directory, WandB setup, config saving),
logging, and error handling.

It also includes integration tests to verify that the Trainer class correctly
utilizes the SessionManager for all session-related tasks.
"""
from unittest.mock import Mock, patch, mock_open, MagicMock

import pytest

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.training.session_manager import SessionManager
from keisei.training.trainer import Trainer


# Mock Fixtures
class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.run_name = kwargs.get("run_name")
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_app_config():
    """Create a mock application configuration for testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            num_actions_total=13527,
            input_channels=46,
            seed=42,
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
        ),
        evaluation=EvaluationConfig(
            num_games=20,
            opponent_type="random",
            evaluation_interval_timesteps=50000,
        ),
        logging=LoggingConfig(
            log_file="test.log",
            model_dir="/tmp/test_models",
            run_name=None,
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(
            enable_demo_mode=False,
            demo_mode_delay=0.5,
        ),
    )


@pytest.fixture
def mock_cli_args():
    """Create mock command-line arguments."""
    return MockArgs(resume=None)


# Unit Tests for SessionManager
class TestSessionManagerInitialization:
    """Test SessionManager initialization logic and run name precedence."""

    def test_init_with_explicit_run_name(self, mock_app_config, mock_cli_args):
        """Test initialization with an explicit run_name parameter."""
        explicit_name = "explicit_test_run"
        manager = SessionManager(mock_app_config, mock_cli_args, run_name=explicit_name)
        assert manager.run_name == explicit_name

    def test_init_with_args_run_name(self, mock_app_config):
        """Test initialization with run_name from CLI args."""
        args_name = "args_test_run"
        args = MockArgs(run_name=args_name)
        manager = SessionManager(mock_app_config, args)
        assert manager.run_name == args_name

    def test_init_with_config_run_name(self, mock_app_config, mock_cli_args):
        """Test initialization with run_name from the config file."""
        config_name = "config_test_run"
        mock_app_config.logging.run_name = config_name
        manager = SessionManager(mock_app_config, mock_cli_args)
        assert manager.run_name == config_name

    @patch("keisei.training.session_manager.generate_run_name")
    def test_init_with_auto_generated_name(self, mock_generate, mock_app_config, mock_cli_args):
        """Test initialization with an auto-generated run_name."""
        generated_name = "auto_generated_run"
        mock_generate.return_value = generated_name
        manager = SessionManager(mock_app_config, mock_cli_args)
        assert manager.run_name == generated_name
        mock_generate.assert_called_once_with(mock_app_config, None)

    def test_init_run_name_precedence(self, mock_app_config):
        """Test the precedence order for determining the run_name."""
        explicit_name = "explicit_name"
        args_name = "args_name"
        config_name = "config_name"
        args_with_name = MockArgs(run_name=args_name)
        mock_app_config.logging.run_name = config_name

        # Explicit name should have the highest priority
        manager1 = SessionManager(mock_app_config, args_with_name, run_name=explicit_name)
        assert manager1.run_name == explicit_name

        # Args name should have priority over config name
        manager2 = SessionManager(mock_app_config, args_with_name)
        assert manager2.run_name == args_name


class TestSessionManagerLifecycle:
    """Test the complete lifecycle of SessionManager: setup, config saving, and finalization."""

    @pytest.fixture
    def setup_mocks(self):
        """A single fixture to set up all necessary mocks."""
        with patch("keisei.training.utils.setup_directories") as mock_setup_dirs, \
             patch("keisei.training.utils.setup_wandb") as mock_setup_wandb, \
             patch("keisei.training.utils.serialize_config") as mock_serialize, \
             patch("keisei.training.utils.setup_seeding") as mock_setup_seeding, \
             patch("builtins.open", new_callable=mock_open) as mock_file, \
             patch("os.path.join") as mock_join:

            mock_setup_dirs.return_value = {
                "run_artifact_dir": "/tmp/test_run",
                "model_dir": "/tmp/test_run/models",
                "log_file_path": "/tmp/test_run/training.log",
                "eval_log_file_path": "/tmp/test_run/eval.log",
            }
            mock_setup_wandb.return_value = True
            mock_serialize.return_value = '{"test": "config"}'
            mock_join.return_value = "/tmp/test_run/effective_config.json"

            yield {
                "setup_dirs": mock_setup_dirs,
                "setup_wandb": mock_setup_wandb,
                "serialize": mock_serialize,
                "setup_seeding": mock_setup_seeding,
                "file": mock_file,
            }

    def test_full_workflow(self, setup_mocks, mock_app_config, mock_cli_args):
        """Test the complete session setup workflow."""
        manager = SessionManager(mock_app_config, mock_cli_args, run_name="test_workflow")

        # 1. Setup Seeding
        manager.setup_seeding()
        setup_mocks["setup_seeding"].assert_called_once_with(mock_app_config)

        # 2. Setup Directories
        dirs = manager.setup_directories()
        assert dirs["run_artifact_dir"] == "/tmp/test_run"
        assert manager.run_artifact_dir == "/tmp/test_run"
        setup_mocks["setup_dirs"].assert_called_once_with(mock_app_config, "test_workflow")

        # 3. Setup WandB
        wandb_active = manager.setup_wandb()
        assert wandb_active is True
        assert manager.is_wandb_active is True
        setup_mocks["setup_wandb"].assert_called_once_with(mock_app_config, "test_workflow", "/tmp/test_run")

        # 4. Save Effective Config
        manager.save_effective_config()
        setup_mocks["serialize"].assert_called_once_with(mock_app_config)
        setup_mocks["file"].assert_called_once_with("/tmp/test_run/effective_config.json", "w", encoding="utf-8")
        setup_mocks["file"]().write.assert_called_once_with('{"test": "config"}')

    @patch("wandb.finish")
    def test_finalize_session(self, _mock_wandb_finish, mock_app_config, mock_cli_args):
        """Test session finalization with and without an active WandB run."""
        # Case 1: WandB is active
        manager1 = SessionManager(mock_app_config, mock_cli_args)
        manager1._is_wandb_active = True  # pylint: disable=protected-access
        with patch("wandb.run", MagicMock()):  # Ensure wandb.run is not None
            manager1.finalize_session()
        _mock_wandb_finish.assert_called_once()

        # Case 2: WandB is not active
        _mock_wandb_finish.reset_mock()
        manager2 = SessionManager(mock_app_config, mock_cli_args)
        manager2._is_wandb_active = False  # pylint: disable=protected-access
        manager2.finalize_session()
        _mock_wandb_finish.assert_not_called()


class TestSessionManagerErrorHandling:
    """Test SessionManager error handling and property access before setup."""

    def test_properties_raise_error_before_setup(self, mock_app_config, mock_cli_args):
        """Test that properties raise RuntimeError if accessed before setup."""
        manager = SessionManager(mock_app_config, mock_cli_args)

        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = manager.run_artifact_dir
        with pytest.raises(RuntimeError, match="Directories not yet set up"):
            _ = manager.model_dir
        with pytest.raises(RuntimeError, match="WandB not yet initialized"):
            _ = manager.is_wandb_active

    @patch("keisei.training.utils.setup_directories", side_effect=OSError("Permission denied"))
    def test_directory_setup_failure(self, _mock_setup_dirs, mock_app_config, mock_cli_args):
        """Test handling of directory setup failures."""
        manager = SessionManager(mock_app_config, mock_cli_args)
        with pytest.raises(RuntimeError, match="Failed to setup directories"):
            manager.setup_directories()

    @patch("keisei.training.utils.setup_wandb", side_effect=Exception("WandB API error"))
    def test_wandb_setup_failure(self, _mock_setup_wandb, mock_app_config, mock_cli_args):
        """Test that WandB setup failure is handled gracefully."""
        manager = SessionManager(mock_app_config, mock_cli_args)
        manager._run_artifact_dir = "/tmp/test"  # pylint: disable=protected-access

        with patch("sys.stderr"):  # Suppress error message print
            result = manager.setup_wandb()
            assert result is False
            assert manager.is_wandb_active is False

    @patch("keisei.training.utils.serialize_config", side_effect=TypeError("Serialization error"))
    def test_config_saving_failure(self, _mock_serialize, mock_app_config, mock_cli_args):
        """Test handling of config saving failures."""
        manager = SessionManager(mock_app_config, mock_cli_args)
        manager._run_artifact_dir = "/tmp/test"  # pylint: disable=protected-access
        with pytest.raises(RuntimeError, match="Failed to save effective config"):
            manager.save_effective_config()


class TestSessionManagerLoggingAndSummary:
    """Test session info logging and summary generation."""

    def test_get_session_summary(self, mock_app_config, mock_cli_args):
        """Test generation of the session summary."""
        manager = SessionManager(mock_app_config, mock_cli_args, run_name="summary_test")

        # Before setup
        summary1 = manager.get_session_summary()
        assert summary1["run_name"] == "summary_test"
        assert summary1["run_artifact_dir"] is None
        assert summary1["is_wandb_active"] is None
        assert summary1["seed"] == 42

        # After setup
        manager._run_artifact_dir = "/tmp/summary"      # pylint: disable=protected-access
        manager._model_dir = "/tmp/summary/models"      # pylint: disable=protected-access
        manager._log_file_path = "/tmp/summary/training.log"  # pylint: disable=protected-access
        manager._is_wandb_active = True                 # pylint: disable=protected-access

        summary2 = manager.get_session_summary()
        assert summary2["run_artifact_dir"] == "/tmp/summary"
        assert summary2["is_wandb_active"] is True

    def test_log_session_info(self, mock_app_config, mock_cli_args):
        """Test the content of session info logging."""
        manager = SessionManager(mock_app_config, mock_cli_args, run_name="log_test")
        manager._run_artifact_dir = "/tmp/log_test_dir"  # pylint: disable=protected-access
        manager._is_wandb_active = False               # pylint: disable=protected-access

        logged_messages = []
        def mock_logger(msg):
            logged_messages.append(msg)

        # Test logging for a new run
        manager.log_session_info(
            mock_logger,
            agent_info={"type": "PPO", "name": "TestAgent"},
            global_timestep=0
        )

        assert any("Keisei Training Run: log_test" in msg for msg in logged_messages)
        assert any("Run directory: /tmp/log_test_dir" in msg for msg in logged_messages)
        assert any("Starting fresh training." in msg for msg in logged_messages)
        assert any("Agent: PPO (TestAgent)" in msg for msg in logged_messages)

        # Test logging for a resumed run
        logged_messages.clear()
        manager.log_session_info(
            mock_logger,
            resumed_from_checkpoint="/path/to/checkpoint.pth",
            global_timestep=50000,
            total_episodes_completed=100
        )

        assert any("Resumed training from checkpoint" in msg for msg in logged_messages)
        assert any(
            "Resuming from timestep 50000, 100 episodes completed" in msg for msg in logged_messages
        )


# Integration Tests for Trainer
class TestTrainerIntegration:
    """Test the integration of SessionManager within the Trainer class."""

    @pytest.fixture
    def setup_trainer_mocks(self):
        """A fixture to set up all necessary mocks for Trainer initialization."""
        with patch("keisei.training.trainer.ShogiGame"), \
             patch("keisei.training.trainer.PPOAgent") as mock_ppo_agent, \
             patch("keisei.training.trainer.model_factory"), \
             patch("keisei.shogi.features.FEATURE_SPECS"), \
             patch("keisei.training.utils.setup_directories") as mock_setup_dirs, \
             patch("keisei.training.utils.setup_wandb") as mock_setup_wandb, \
             patch("keisei.training.utils.serialize_config") as mock_serialize, \
             patch("keisei.training.utils.setup_seeding") as mock_setup_seeding, \
             patch("builtins.open"), \
             patch("os.path.join", side_effect=lambda *args: "/".join(args)), \
             patch("glob.glob", return_value=[]), \
             patch("os.path.exists", return_value=True):

            # Common return values for mocks
            mock_setup_dirs.return_value = {
                "run_artifact_dir": "/tmp/trainer_run",
                "model_dir": "/tmp/trainer_run/models",
                "log_file_path": "/tmp/trainer_run/training.log",
                "eval_log_file_path": "/tmp/trainer_run/eval.log",
            }
            mock_setup_wandb.return_value = False
            mock_ppo_agent.return_value = Mock(name="TestAgent", model="MockModel")

            yield {
                "setup_dirs": mock_setup_dirs,
                "setup_wandb": mock_setup_wandb,
                "serialize": mock_serialize,
                "setup_seeding": mock_setup_seeding,
            }

    def test_trainer_initializes_and_uses_session_manager(self, setup_trainer_mocks, mock_app_config, mock_cli_args):
        """Test that Trainer correctly initializes and delegates to SessionManager."""
        trainer = Trainer(mock_app_config, mock_cli_args)

        # Verify SessionManager is created and used
        assert hasattr(trainer, "session_manager")
        assert isinstance(trainer.session_manager, SessionManager)

        # Verify setup methods from SessionManager were called during Trainer init
        setup_trainer_mocks["setup_seeding"].assert_called_once()
        setup_trainer_mocks["setup_dirs"].assert_called_once()
        setup_trainer_mocks["setup_wandb"].assert_called_once()
        setup_trainer_mocks["serialize"].assert_called_once()

        # Verify session properties are correctly accessed from the manager
        assert trainer.run_name == trainer.session_manager.run_name
        assert trainer.run_artifact_dir == "/tmp/trainer_run"
        assert trainer.model_dir == "/tmp/trainer_run/models"
        assert trainer.log_file_path == "/tmp/trainer_run/training.log"
        assert trainer.is_train_wandb_active is False

    def test_trainer_delegates_info_logging_to_session_manager(self, _, mock_app_config, mock_cli_args):
        """Test that Trainer uses SessionManager for logging run information."""
        trainer = Trainer(mock_app_config, mock_cli_args)

        def mock_log_both(_msg, **_kwargs):
            pass  # Dummy logger

        # Patch the session manager's method to spy on the call
        with patch.object(trainer.session_manager, "log_session_info") as mock_log_session:
            trainer._log_run_info(mock_log_both)  # pylint: disable=protected-access

            # Verify SessionManager's log_session_info was called once
            mock_log_session.assert_called_once()

            # Verify it was called with the correct information from the trainer
            call_args = mock_log_session.call_args
            assert call_args[1]["agent_info"]["name"] == "TestAgent"
            assert call_args[1]["global_timestep"] == 0
            assert call_args[1]["total_episodes_completed"] == 0
