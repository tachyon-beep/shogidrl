"""
test_trainer_session_integration.py: Integration tests for Trainer and SessionManager.

Tests that verify the SessionManager is properly integrated into the Trainer
and that session management functionality works correctly end-to-end.
"""

import tempfile
from unittest.mock import Mock, mock_open, patch

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
    """Create a real configuration for testing."""
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
def mock_args():
    """Create mock command-line arguments."""
    return MockArgs(
        run_name=None,
        resume=None,
        input_features=None,
        model=None,
        tower_depth=None,
        tower_width=None,
        se_ratio=None,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTrainerSessionIntegration:
    """Test integration between Trainer and SessionManager."""

    @patch("torch.device")
    @patch("keisei.training.trainer.Console")
    @patch("keisei.training.trainer.TrainingLogger")
    @patch("keisei.training.trainer.ShogiGame")
    @patch("keisei.training.trainer.PolicyOutputMapper")
    @patch("keisei.training.trainer.PPOAgent")
    @patch("keisei.training.trainer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.trainer.display.TrainingDisplay")
    @patch("keisei.training.trainer.callbacks.CheckpointCallback")
    @patch("keisei.training.trainer.callbacks.EvaluationCallback")
    def test_trainer_initialization_with_session_manager(
        self,
        mock_eval_callback,
        mock_checkpoint_callback,
        mock_display,
        mock_feature_specs,
        mock_model_factory,
        mock_experience_buffer,
        mock_ppo_agent,
        mock_policy_mapper,
        mock_shogi_game,
        mock_training_logger,
        mock_console,
        mock_torch_device,
        mock_config,
        mock_args,
        temp_dir,
    ):
        """Test that Trainer properly initializes with SessionManager."""
        # Setup feature specs mock
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        # Setup model factory mock
        mock_model = Mock()
        mock_model_factory.return_value = mock_model

        # Setup device mock - return string that PyTorch can handle
        mock_torch_device.return_value = "cpu"

        # Setup game mock
        game_instance = Mock()
        game_instance.seed = Mock()
        mock_shogi_game.return_value = game_instance

        # Setup policy mapper mock
        policy_mapper_instance = Mock()
        policy_mapper_instance.get_total_actions.return_value = 4096
        mock_policy_mapper.return_value = policy_mapper_instance

        # Setup agent mock
        agent_instance = Mock()
        agent_instance.name = "test_agent"
        agent_instance.model = mock_model
        mock_ppo_agent.return_value = agent_instance

        # Setup buffer mock
        buffer_instance = Mock()
        mock_experience_buffer.return_value = buffer_instance

        # Setup logger mock
        logger_instance = Mock()
        mock_training_logger.return_value = logger_instance

        # Setup console mock
        console_instance = Mock()
        mock_console.return_value = console_instance

        # Use temp directory for session manager
        with (
            patch(
                "keisei.training.trainer.SessionManager"
            ) as mock_session_manager_class,
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
            patch("glob.glob", return_value=[]),
            patch("os.makedirs"),
            patch("os.path.exists", return_value=True),
        ):

            # Create mock session manager instance
            mock_session_manager = Mock()
            mock_session_manager.run_name = "test_run_12345"
            mock_session_manager.run_artifact_dir = f"{temp_dir}/artifacts"
            mock_session_manager.model_dir = f"{temp_dir}/models"
            mock_session_manager.log_file_path = f"{temp_dir}/train.log"
            mock_session_manager.eval_log_file_path = f"{temp_dir}/eval.log"
            mock_session_manager.is_wandb_active = False

            # Mock setup methods
            mock_session_manager.setup_directories = Mock()
            mock_session_manager.setup_wandb = Mock()
            mock_session_manager.save_effective_config = Mock()
            mock_session_manager.setup_seeding = Mock()
            mock_session_manager.log_session_info = Mock()
            mock_session_manager.finalize_session = Mock()

            mock_session_manager_class.return_value = mock_session_manager

            # Create Trainer instance
            trainer = Trainer(mock_config, mock_args)

            # Verify SessionManager was created and initialized
            assert hasattr(trainer, "session_manager")
            assert trainer.session_manager is not None

            # Verify session setup methods were called
            mock_session_manager.setup_directories.assert_called_once()
            mock_session_manager.setup_wandb.assert_called_once()
            mock_session_manager.save_effective_config.assert_called_once()
            mock_session_manager.setup_seeding.assert_called_once()

            # Verify session properties were set on trainer
            assert hasattr(trainer, "run_name")
            assert hasattr(trainer, "model_dir")
            assert hasattr(trainer, "log_file_path")

            # Verify other trainer components were initialized
            assert hasattr(trainer, "agent")
            assert hasattr(trainer, "experience_buffer")
            assert hasattr(trainer, "game")
            assert hasattr(trainer, "policy_output_mapper")

    @patch("torch.device")
    @patch("keisei.training.trainer.Console")
    @patch("keisei.training.trainer.TrainingLogger")
    @patch("keisei.training.trainer.ShogiGame")
    @patch("keisei.training.trainer.PolicyOutputMapper")
    @patch("keisei.training.trainer.PPOAgent")
    @patch("keisei.training.trainer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.trainer.display.TrainingDisplay")
    @patch("keisei.training.trainer.callbacks.CheckpointCallback")
    @patch("keisei.training.trainer.callbacks.EvaluationCallback")
    def test_trainer_session_properties_delegation(
        self,
        mock_eval_callback,
        mock_checkpoint_callback,
        mock_display,
        mock_feature_specs,
        mock_model_factory,
        mock_experience_buffer,
        mock_ppo_agent,
        mock_policy_mapper,
        mock_shogi_game,
        mock_training_logger,
        mock_console,
        mock_torch_device,
        mock_config,
        mock_args,
        temp_dir,
    ):
        """Test that Trainer properly delegates session properties to SessionManager."""
        # Setup mocks (similar to previous test)
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model_factory.return_value = mock_model

        # Setup device mock - return string that PyTorch can handle
        mock_torch_device.return_value = "cpu"

        game_instance = Mock()
        game_instance.seed = Mock()
        mock_shogi_game.return_value = game_instance

        policy_mapper_instance = Mock()
        policy_mapper_instance.get_total_actions.return_value = 4096
        mock_policy_mapper.return_value = policy_mapper_instance

        agent_instance = Mock()
        agent_instance.name = "test_agent"
        agent_instance.model = mock_model
        mock_ppo_agent.return_value = agent_instance

        buffer_instance = Mock()
        mock_experience_buffer.return_value = buffer_instance

        logger_instance = Mock()
        mock_training_logger.return_value = logger_instance

        console_instance = Mock()
        mock_console.return_value = console_instance

        # Mock session manager with specific property values
        with (
            patch(
                "keisei.training.trainer.SessionManager"
            ) as mock_session_manager_class,
            patch("builtins.open", mock_open()),
        ):
            mock_session_manager = Mock()
            mock_session_manager.run_name = "test_run_session"
            mock_session_manager.run_artifact_dir = f"{temp_dir}/artifacts"
            mock_session_manager.model_dir = f"{temp_dir}/models"
            mock_session_manager.log_file_path = f"{temp_dir}/train.log"
            mock_session_manager.eval_log_file_path = f"{temp_dir}/eval.log"
            mock_session_manager.is_wandb_active = False

            # Mock setup methods
            mock_session_manager.setup_directories = Mock()
            mock_session_manager.setup_wandb = Mock()
            mock_session_manager.save_effective_config = Mock()
            mock_session_manager.setup_seeding = Mock()

            mock_session_manager_class.return_value = mock_session_manager

            # Create Trainer instance
            trainer = Trainer(mock_config, mock_args)

            # Verify properties were copied from session manager
            assert trainer.run_name == "test_run_session"
            assert trainer.run_artifact_dir == f"{temp_dir}/artifacts"
            assert trainer.model_dir == f"{temp_dir}/models"
            assert trainer.log_file_path == f"{temp_dir}/train.log"
            assert trainer.eval_log_file_path == f"{temp_dir}/eval.log"
            assert trainer.is_train_wandb_active is False

    def test_session_manager_method_integration(self, mock_config, mock_args, temp_dir):
        """Test that session manager methods are properly integrated."""
        with (
            patch(
                "keisei.training.trainer.SessionManager"
            ) as mock_session_manager_class,
            patch("builtins.open", mock_open()),
        ):
            mock_session_manager = Mock()
            mock_session_manager.log_session_info = Mock()
            mock_session_manager.finalize_session = Mock()
            mock_session_manager.setup_directories = Mock()
            mock_session_manager.setup_wandb = Mock()
            mock_session_manager.save_effective_config = Mock()
            mock_session_manager.setup_seeding = Mock()

            # Add required properties
            mock_session_manager.run_name = "test_session"
            mock_session_manager.run_artifact_dir = f"{temp_dir}/artifacts"
            mock_session_manager.model_dir = f"{temp_dir}/models"
            mock_session_manager.log_file_path = f"{temp_dir}/training.log"
            mock_session_manager.eval_log_file_path = f"{temp_dir}/eval.log"
            mock_session_manager.is_wandb_active = False

            mock_session_manager_class.return_value = mock_session_manager

            # Mock all trainer dependencies to avoid initialization issues
            with (
                patch.multiple(
                    "keisei.training.trainer",
                    Console=Mock(),
                    TrainingLogger=Mock(),
                    ShogiGame=Mock(),
                    PolicyOutputMapper=Mock(),
                    PPOAgent=Mock(),
                    ExperienceBuffer=Mock(),
                    display=Mock(),
                    callbacks=Mock(),
                ),
                patch("keisei.training.models.model_factory"),
                patch(
                    "keisei.training.utils.find_latest_checkpoint", return_value=None
                ),
                patch("torch.device", return_value="cpu"),
                patch(
                    "keisei.shogi.features.FEATURE_SPECS",
                    {"core46": Mock(num_planes=46)},
                ),
            ):

                trainer = Trainer(mock_config, mock_args)

                # Test session info logging delegation
                mock_log_both = Mock()
                trainer._log_run_info(mock_log_both)

                # Verify session manager's log_session_info was called with correct arguments
                mock_session_manager.log_session_info.assert_called_once()
                call_args = mock_session_manager.log_session_info.call_args
                assert call_args is not None
                # Verify keyword arguments were passed
                assert "logger_func" in call_args.kwargs
                assert "agent_info" in call_args.kwargs
                assert "global_timestep" in call_args.kwargs
                assert "total_episodes_completed" in call_args.kwargs

                # Test session finalization (would be called at end of training)
                # This would typically be called in training loop completion
                trainer.session_manager.finalize_session()
                mock_session_manager.finalize_session.assert_called_once()
