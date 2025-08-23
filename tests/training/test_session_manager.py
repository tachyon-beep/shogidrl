"""
test_session_management.py: Comprehensive tests for SessionManager and its integration with Trainer.

This file covers the standalone unit tests for the SessionManager class, including
initialization, lifecycle management (directory, WandB setup, config saving),
logging, and error handling.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

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
        display=DisplayConfig(
            display_moves=False,
            turn_tick=0.5,
        ),
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
    )


@pytest.fixture
def mock_cli_args():
    """Create mock command-line arguments."""
    return MockArgs(resume=None)


@pytest.fixture
def temp_base_dir():
    """Create a temporary directory for testing directory operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Enhanced Unit Tests for SessionManager
class TestSessionManagerDirectoryOperations:
    """Test SessionManager directory creation and path handling robustness."""

    def test_directory_setup_creates_missing_directories(
        self, mock_app_config, mock_cli_args, temp_base_dir
    ):
        """Test that directory setup correctly creates all necessary directories."""
        run_artifact_path = os.path.join(temp_base_dir, "run_artifacts", "test_run")
        models_path = os.path.join(run_artifact_path, "models")

        def mock_setup_directories_side_effect(config, run_name):
            """Mock side effect that creates directories like the real function."""
            os.makedirs(run_artifact_path, exist_ok=True)
            os.makedirs(models_path, exist_ok=True)
            return {
                "run_artifact_dir": run_artifact_path,
                "model_dir": models_path,
                "log_file_path": os.path.join(run_artifact_path, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_path, "rich_periodic_eval_log.txt"
                ),
            }

        with patch(
            "keisei.training.utils.setup_directories",
            side_effect=mock_setup_directories_side_effect,
        ):
            manager = SessionManager(
                mock_app_config, mock_cli_args, run_name="test_run"
            )
            manager.setup_directories()

            # Verify directories were created
            assert os.path.exists(run_artifact_path)
            assert os.path.exists(models_path)

            # Verify SessionManager properties work
            assert manager.run_artifact_dir == run_artifact_path
            assert manager.model_dir == models_path

    def test_save_effective_config_with_missing_directory(
        self, mock_app_config, mock_cli_args, temp_base_dir
    ):
        """Test that save_effective_config handles missing run_artifact_dir correctly."""
        run_artifact_path = os.path.join(temp_base_dir, "nonexistent", "test_run")

        def mock_setup_directories_side_effect(config, run_name):
            """Mock that doesn't create the directory initially."""
            return {
                "run_artifact_dir": run_artifact_path,
                "model_dir": os.path.join(run_artifact_path, "models"),
                "log_file_path": os.path.join(run_artifact_path, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_path, "rich_periodic_eval_log.txt"
                ),
            }

        def mock_serialize_config(config):
            """Mock config serialization."""
            return {"mocked": "config"}

        with (
            patch(
                "keisei.training.utils.setup_directories",
                side_effect=mock_setup_directories_side_effect,
            ),
            patch(
                "keisei.training.utils.serialize_config",
                side_effect=mock_serialize_config,
            ),
            patch("builtins.open", mock_open()) as mock_file,
        ):

            manager = SessionManager(
                mock_app_config, mock_cli_args, run_name="test_run"
            )
            manager.setup_directories()

            # Ensure the directory doesn't exist initially
            assert not os.path.exists(run_artifact_path)

            # This should create the directory and save the config
            manager.save_effective_config()

            # Verify directory was created
            assert os.path.exists(run_artifact_path)

            # Verify file was opened for writing
            expected_config_path = os.path.join(
                run_artifact_path, "effective_config.json"
            )
            mock_file.assert_called_with(expected_config_path, "w", encoding="utf-8")

    def test_wandb_enabled_directory_consistency(
        self, mock_app_config, mock_cli_args, temp_base_dir
    ):
        """Test that W&B enabled/disabled scenarios maintain directory consistency."""
        # Test with W&B enabled
        mock_app_config.wandb.enabled = True
        run_artifact_path = os.path.join(temp_base_dir, "wandb_test_run")

        def mock_setup_directories_wandb(config, run_name):
            os.makedirs(run_artifact_path, exist_ok=True)
            return {
                "run_artifact_dir": run_artifact_path,
                "model_dir": os.path.join(run_artifact_path, "models"),
                "log_file_path": os.path.join(run_artifact_path, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_path, "rich_periodic_eval_log.txt"
                ),
            }

        def mock_setup_wandb_success(config, run_name, run_artifact_dir):
            return True

        with (
            patch(
                "keisei.training.utils.setup_directories",
                side_effect=mock_setup_directories_wandb,
            ),
            patch(
                "keisei.training.utils.setup_wandb",
                side_effect=mock_setup_wandb_success,
            ),
        ):
            manager = SessionManager(
                mock_app_config, mock_cli_args, run_name="wandb_test"
            )
            manager.setup_directories()
            wandb_result = manager.setup_wandb()

            assert wandb_result is True
            assert manager.is_wandb_active is True
            assert os.path.exists(run_artifact_path)

    def test_multiple_session_managers_different_directories(
        self, mock_app_config, mock_cli_args, temp_base_dir
    ):
        """Test that multiple SessionManager instances can coexist with different directories."""
        run1_path = os.path.join(temp_base_dir, "run1")
        run2_path = os.path.join(temp_base_dir, "run2")

        def mock_setup_run1(config, run_name):
            os.makedirs(run1_path, exist_ok=True)
            return {
                "run_artifact_dir": run1_path,
                "model_dir": os.path.join(run1_path, "models"),
                "log_file_path": os.path.join(run1_path, "training.log"),
                "eval_log_file_path": os.path.join(
                    run1_path, "rich_periodic_eval_log.txt"
                ),
            }

        def mock_setup_run2(config, run_name):
            os.makedirs(run2_path, exist_ok=True)
            return {
                "run_artifact_dir": run2_path,
                "model_dir": os.path.join(run2_path, "models"),
                "log_file_path": os.path.join(run2_path, "training.log"),
                "eval_log_file_path": os.path.join(
                    run2_path, "rich_periodic_eval_log.txt"
                ),
            }

        with patch(
            "keisei.training.utils.setup_directories", side_effect=mock_setup_run1
        ):
            manager1 = SessionManager(mock_app_config, mock_cli_args, run_name="run1")
            manager1.setup_directories()

        with patch(
            "keisei.training.utils.setup_directories", side_effect=mock_setup_run2
        ):
            manager2 = SessionManager(mock_app_config, mock_cli_args, run_name="run2")
            manager2.setup_directories()

        # Verify both managers have different, correct paths
        assert manager1.run_artifact_dir == run1_path
        assert manager2.run_artifact_dir == run2_path
        assert os.path.exists(run1_path)
        assert os.path.exists(run2_path)


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
    def test_init_with_auto_generated_name(
        self, mock_generate, mock_app_config, mock_cli_args
    ):
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
        manager1 = SessionManager(
            mock_app_config, args_with_name, run_name=explicit_name
        )
        assert manager1.run_name == explicit_name

        # Args name should have priority over config name
        manager2 = SessionManager(mock_app_config, args_with_name)
        assert manager2.run_name == args_name


class TestSessionManagerLifecycle:
    """Test the complete lifecycle of SessionManager: setup, config saving, and finalization."""

    @pytest.fixture
    def setup_mocks(self):
        """A single fixture to set up all necessary mocks."""
        with (
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
            patch("keisei.training.utils.setup_wandb") as mock_setup_wandb,
            patch("keisei.training.utils.serialize_config") as mock_serialize,
            patch("keisei.training.utils.setup_seeding") as mock_setup_seeding,
            patch("builtins.open", new_callable=mock_open) as mock_file,
            patch("os.path.join") as mock_join,
        ):

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
        manager = SessionManager(
            mock_app_config, mock_cli_args, run_name="test_workflow"
        )

        # 1. Setup Seeding
        manager.setup_seeding()
        setup_mocks["setup_seeding"].assert_called_once_with(mock_app_config)

        # 2. Setup Directories
        dirs = manager.setup_directories()
        assert dirs["run_artifact_dir"] == "/tmp/test_run"
        assert manager.run_artifact_dir == "/tmp/test_run"
        setup_mocks["setup_dirs"].assert_called_once_with(
            mock_app_config, "test_workflow"
        )

        # 3. Setup WandB
        wandb_active = manager.setup_wandb()
        assert wandb_active is True
        assert manager.is_wandb_active is True
        setup_mocks["setup_wandb"].assert_called_once_with(
            mock_app_config, "test_workflow", "/tmp/test_run"
        )

        # 4. Save Effective Config
        manager.save_effective_config()
        setup_mocks["serialize"].assert_called_once_with(mock_app_config)
        setup_mocks["file"].assert_called_once_with(
            "/tmp/test_run/effective_config.json", "w", encoding="utf-8"
        )
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

    @patch(
        "keisei.training.utils.setup_directories",
        side_effect=OSError("Permission denied"),
    )
    def test_directory_setup_failure(
        self, _mock_setup_dirs, mock_app_config, mock_cli_args
    ):
        """Test handling of directory setup failures."""
        manager = SessionManager(mock_app_config, mock_cli_args)
        with pytest.raises(RuntimeError, match="Failed to setup directories"):
            manager.setup_directories()

    @patch(
        "keisei.training.utils.setup_wandb", side_effect=Exception("WandB API error")
    )
    def test_wandb_setup_failure(
        self, _mock_setup_wandb, mock_app_config, mock_cli_args
    ):
        """Test that WandB setup failure is handled gracefully."""
        manager = SessionManager(mock_app_config, mock_cli_args)
        manager._run_artifact_dir = "/tmp/test"  # pylint: disable=protected-access

        with patch("sys.stderr"):  # Suppress error message print
            result = manager.setup_wandb()
            assert result is False
            assert manager.is_wandb_active is False

    @patch(
        "keisei.training.utils.serialize_config",
        side_effect=TypeError("Serialization error"),
    )
    def test_config_saving_failure(
        self, _mock_serialize, mock_app_config, mock_cli_args
    ):
        """Test handling of config saving failures."""
        manager = SessionManager(mock_app_config, mock_cli_args)
        manager._run_artifact_dir = "/tmp/test"  # pylint: disable=protected-access
        with pytest.raises(RuntimeError, match="Failed to save effective config"):
            manager.save_effective_config()


class TestSessionManagerLoggingAndSummary:
    """Test session info logging and summary generation."""

    def test_get_session_summary(self, mock_app_config, mock_cli_args):
        """Test generation of the session summary."""
        manager = SessionManager(
            mock_app_config, mock_cli_args, run_name="summary_test"
        )

        # Before setup
        summary1 = manager.get_session_summary()
        assert summary1["run_name"] == "summary_test"
        assert summary1["run_artifact_dir"] is None
        assert summary1["is_wandb_active"] is None
        assert summary1["seed"] == 42

        # After setup
        manager._run_artifact_dir = "/tmp/summary"  # pylint: disable=protected-access
        manager._model_dir = "/tmp/summary/models"  # pylint: disable=protected-access
        manager._log_file_path = (
            "/tmp/summary/training.log"  # pylint: disable=protected-access
        )
        manager._is_wandb_active = True  # pylint: disable=protected-access

        summary2 = manager.get_session_summary()
        assert summary2["run_artifact_dir"] == "/tmp/summary"
        assert summary2["is_wandb_active"] is True

    def test_log_session_info(self, mock_app_config, mock_cli_args):
        """Test the content of session info logging."""
        manager = SessionManager(mock_app_config, mock_cli_args, run_name="log_test")
        manager._run_artifact_dir = (
            "/tmp/log_test_dir"  # pylint: disable=protected-access
        )
        manager._is_wandb_active = False  # pylint: disable=protected-access

        logged_messages = []

        def mock_logger(msg):
            logged_messages.append(msg)

        # Test logging for a new run
        manager.log_session_info(
            mock_logger,
            agent_info={"type": "PPO", "name": "TestAgent"},
            global_timestep=0,
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
            total_episodes_completed=100,
        )

        assert any("Resumed training from checkpoint" in msg for msg in logged_messages)
        assert any(
            "Resuming from timestep 50000, 100 episodes completed" in msg
            for msg in logged_messages
        )


# Integration Tests for Trainer
class TestTrainerIntegration:
    """Test the integration of SessionManager within the Trainer class."""

    @pytest.fixture
    def setup_trainer_mocks(self, mock_app_config):
        """A fixture to set up all necessary mocks for Trainer initialization."""
        # Use autospec=True to ensure mocks have the same signature as the real methods.
        # Patch where modules are LOOKED UP, not where they are defined.
        with (
            patch(
                "keisei.training.env_manager.ShogiGame", autospec=True
            ) as mock_shogi_game_class,
            patch(
                "keisei.training.env_manager.PolicyOutputMapper", autospec=True
            ) as mock_policy_mapper_class,
            patch(
                "keisei.training.models.model_factory", autospec=True
            ) as mock_model_factory,
            patch(
                "keisei.training.setup_manager.PPOAgent", autospec=True
            ) as mock_ppo_agent_class,
            patch(
                "keisei.shogi.features.FEATURE_SPECS", new_callable=dict
            ) as mock_feature_specs,
            patch(
                "keisei.training.session_manager.SessionManager.setup_directories",
                autospec=True,
            ) as mock_setup_dirs,
            patch(
                "keisei.training.session_manager.SessionManager.setup_wandb",
                autospec=True,
            ) as mock_setup_wandb,
            patch(
                "keisei.training.session_manager.SessionManager.save_effective_config",
                autospec=True,
            ) as mock_save_config,
            patch(
                "keisei.training.session_manager.SessionManager.setup_seeding",
                autospec=True,
            ) as mock_setup_seeding,
            patch(
                "keisei.training.session_manager.generate_run_name",
                return_value="mocked_auto_run_name",
            ) as mock_gen_name,
            patch("builtins.open", new_callable=mock_open),
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
            patch("glob.glob", return_value=[]),
            patch("os.path.exists", return_value=True),
            patch("keisei.evaluation.performance_manager.ResourceMonitor", autospec=True) as mock_resource_monitor,
        ):

            # --- Mock SessionManager side effects ---
            def mock_setup_directories_impl(session_manager_self):
                dirs = {
                    "run_artifact_dir": "/tmp/trainer_run",
                    "model_dir": "/tmp/trainer_run/models",
                    "log_file_path": "/tmp/trainer_run/training.log",
                    "eval_log_file_path": "/tmp/trainer_run/eval.log",
                }
                session_manager_self._run_artifact_dir = dirs["run_artifact_dir"]
                session_manager_self._model_dir = dirs["model_dir"]
                session_manager_self._log_file_path = dirs["log_file_path"]
                session_manager_self._eval_log_file_path = dirs["eval_log_file_path"]
                return dirs

            def mock_setup_wandb_impl(session_manager_self):
                session_manager_self._is_wandb_active = False
                return False

            mock_setup_dirs.side_effect = mock_setup_directories_impl
            mock_setup_wandb.side_effect = mock_setup_wandb_impl

            # --- Mock other dependencies ---
            mock_game_instance = mock_shogi_game_class.return_value
            mock_game_instance.get_observation.return_value = Mock(
                name="MockObservation"
            )
            mock_game_instance.reset.return_value = (
                mock_game_instance.get_observation.return_value
            )

            mock_policy_mapper_class.return_value.get_total_actions.return_value = (
                mock_app_config.env.num_actions_total
            )

            mock_model_instance = mock_model_factory.return_value
            mock_model_instance.to.return_value = mock_model_instance
            mock_feature_specs["core46"] = Mock(
                num_planes=mock_app_config.env.input_channels
            )

            # FIX: Configure the PPOAgent mock instance with a 'name' attribute
            mock_agent_instance = mock_ppo_agent_class.return_value
            mock_agent_instance.model = mock_model_instance
            mock_agent_instance.name = "TestAgent"

            # Yield all necessary mocks for assertions in the tests.
            yield {
                "mock_setup_dirs": mock_setup_dirs,
                "mock_setup_wandb": mock_setup_wandb,
                "mock_save_config": mock_save_config,
                "mock_setup_seeding": mock_setup_seeding,
                "mock_ppo_agent_class": mock_ppo_agent_class,
                "mock_generate_run_name": mock_gen_name,
            }

    def test_trainer_initializes_and_uses_session_manager(
        self, setup_trainer_mocks, mock_app_config, mock_cli_args
    ):
        """Test that Trainer correctly initializes and delegates to SessionManager."""
        trainer = Trainer(mock_app_config, mock_cli_args)

        # Verify SessionManager is created and used
        assert hasattr(trainer, "session_manager")
        assert isinstance(trainer.session_manager, SessionManager)

        # Assert calls on the mock objects from the fixture
        setup_trainer_mocks["mock_setup_dirs"].assert_called_once()
        setup_trainer_mocks["mock_setup_wandb"].assert_called_once()
        setup_trainer_mocks["mock_save_config"].assert_called_once()
        setup_trainer_mocks["mock_setup_seeding"].assert_called_once()
        setup_trainer_mocks["mock_generate_run_name"].assert_called_once()

        # Verify session properties are correctly accessed from the manager
        assert trainer.run_name == "mocked_auto_run_name"
        assert trainer.run_artifact_dir == "/tmp/trainer_run"
        assert trainer.model_dir == "/tmp/trainer_run/models"
        assert trainer.log_file_path == "/tmp/trainer_run/training.log"
        assert trainer.is_train_wandb_active is False

    def test_trainer_delegates_info_logging_to_session_manager(
        self, setup_trainer_mocks, mock_app_config, mock_cli_args
    ):
        """Test that Trainer uses SessionManager for logging run information."""
        trainer = Trainer(mock_app_config, mock_cli_args)

        # Create a mock logger function
        mock_log_both = Mock()

        # Spy on the SessionManager's log_session_info method
        with patch.object(
            trainer.session_manager, "log_session_info"
        ) as mock_log_session:
            trainer._log_run_info(mock_log_both)  # pylint: disable=protected-access

            # Verify SessionManager's method was called once
            mock_log_session.assert_called_once()

            # Verify it was called with the correct info from the trainer
            call_kwargs = mock_log_session.call_args.kwargs
            # FIX: The agent instance mock now has the 'name' attribute configured
            assert call_kwargs["agent_info"]["name"] == "TestAgent"
            assert call_kwargs["global_timestep"] == 0
            assert call_kwargs["total_episodes_completed"] == 0
