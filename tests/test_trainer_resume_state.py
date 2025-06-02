"""
test_trainer_resume_state.py: Comprehensive tests for Trainer training state restoration during checkpoint resumption.

Tests verify that Trainer properly restores:
- global_timestep
- total_episodes_completed
- black_wins, white_wins, draws
When resuming from checkpoints through ModelManager integration.
"""

import tempfile
from unittest.mock import Mock, patch

import pytest
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
from keisei.core.ppo_agent import PPOAgent
from keisei.training.model_manager import ModelManager
from keisei.training.trainer import Trainer


def _create_mock_model_with_parameters():
    """Helper to create a properly mocked model with parameters for optimizer."""
    mock_model = Mock()
    mock_model.to.return_value = mock_model
    
    # Create a proper mock parameter that behaves like a PyTorch tensor
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    mock_model.parameters.return_value = [mock_param]
    
    return mock_model


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.resume = kwargs.get("resume")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return AppConfig(
        env=EnvConfig(
            device="cpu", num_actions_total=13527, input_channels=46, seed=42
        ),
        training=TrainingConfig(
            total_timesteps=1000,
            steps_per_epoch=64,
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
            tower_depth=5,
            tower_width=128,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=1000,
            evaluation_interval_timesteps=1000,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(
            num_games=20, opponent_type="random", evaluation_interval_timesteps=1000
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir="/tmp/test_models", run_name=None
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
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
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
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.integration
class TestTrainerResumeState:
    """Test Trainer training state restoration during checkpoint resumption."""

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.core.ppo_agent.PPOAgent")
    def test_trainer_restore_state_from_checkpoint_data(
        self,
        mock_ppo_agent_class,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_session_manager_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that Trainer correctly restores training state variables from checkpoint data."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = _create_mock_model_with_parameters()
        mock_model_factory.return_value = mock_model

        # Mock PPOAgent instance
        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance

        # Mock model and optimizer with proper load_state_dict methods
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock ModelManager instance
        mock_model_manager_instance = mock_model_manager_class.return_value
        mock_model_manager_instance.create_model.return_value = mock_model
        mock_model_manager_instance.handle_checkpoint_resume.return_value = True
        mock_model_manager_instance.checkpoint_data = None  # Will be set later
        mock_model_manager_instance.resumed_from_checkpoint = None

        # Mock EnvManager instance
        mock_env_manager_instance = mock_env_manager_class.return_value
        mock_game = Mock()
        mock_policy_mapper = Mock()
        mock_env_manager_instance.setup_environment.return_value = (
            mock_game,
            mock_policy_mapper,
        )
        mock_env_manager_instance.action_space_size = 13527
        mock_env_manager_instance.obs_space_shape = (46, 9, 9)

        # Create args with resume path
        args = MockArgs(resume="/path/to/checkpoint.pth")

        # Mock checkpoint data with specific values
        checkpoint_data = {
            "global_timestep": 2500,
            "total_episodes_completed": 150,
            "black_wins": 60,
            "white_wins": 55,
            "draws": 35,
            "model_state_dict": {},
            "optimizer_state_dict": {},
        }

        # Mock load_model to return our checkpoint data
        mock_agent_instance.load_model.return_value = checkpoint_data

        # Set up ModelManager to return checkpoint data when handle_checkpoint_resume is called
        def mock_handle_checkpoint_resume(agent, model_dir, resume_path_override=None):
            mock_model_manager_instance.checkpoint_data = checkpoint_data
            mock_model_manager_instance.resumed_from_checkpoint = (
                "/path/to/checkpoint.pth"
            )
            return True

        mock_model_manager_instance.handle_checkpoint_resume.side_effect = (
            mock_handle_checkpoint_resume
        )

        # Mock torch.load to return checkpoint data and os.path.exists
        with (
            patch("torch.load", return_value=checkpoint_data),
            patch("os.path.exists", return_value=True),
        ):
            # Create trainer - checkpoint resume happens during __init__
            trainer = Trainer(mock_config, args)

        # Debug prints
        print("DEBUG: After trainer creation:")
        print(f"DEBUG: trainer.global_timestep = {trainer.global_timestep}")
        print(
            f"DEBUG: trainer.model_manager.checkpoint_data = {getattr(trainer.model_manager, 'checkpoint_data', 'NOT SET')}"
        )
        print(
            f"DEBUG: trainer.model_manager.resumed_from_checkpoint = {getattr(trainer.model_manager, 'resumed_from_checkpoint', 'NOT SET')}"
        )
        print(
            f"DEBUG: mock_agent_instance.load_model.called = {mock_agent_instance.load_model.called}"
        )
        if mock_agent_instance.load_model.called:
            print(
                f"DEBUG: mock_agent_instance.load_model.call_args = {mock_agent_instance.load_model.call_args}"
            )
            print(
                f"DEBUG: mock_agent_instance.load_model.return_value = {mock_agent_instance.load_model.return_value}"
            )

        # Verify state variables are correctly restored from checkpoint data
        assert trainer.global_timestep == 2500
        assert trainer.total_episodes_completed == 150
        assert trainer.black_wins == 60
        assert trainer.white_wins == 55
        assert trainer.draws == 35
        assert trainer.resumed_from_checkpoint == "/path/to/checkpoint.pth"

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.core.ppo_agent.PPOAgent")
    def test_trainer_restore_state_with_missing_checkpoint_fields(
        self,
        mock_ppo_agent_class,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_session_manager_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that Trainer handles missing fields in checkpoint data gracefully."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = _create_mock_model_with_parameters()
        mock_model_factory.return_value = mock_model

        # Mock PPOAgent instance
        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance

        # Mock model and optimizer with proper load_state_dict methods
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock ModelManager instance
        mock_model_manager_instance = mock_model_manager_class.return_value
        mock_model_manager_instance.create_model.return_value = mock_model
        mock_model_manager_instance.handle_checkpoint_resume.return_value = True
        mock_model_manager_instance.checkpoint_data = None  # Will be set later
        mock_model_manager_instance.resumed_from_checkpoint = None

        # Mock EnvManager instance
        mock_env_manager_instance = mock_env_manager_class.return_value
        mock_game = Mock()
        mock_policy_mapper = Mock()
        mock_env_manager_instance.setup_environment.return_value = (
            mock_game,
            mock_policy_mapper,
        )
        mock_env_manager_instance.action_space_size = 13527
        mock_env_manager_instance.obs_space_shape = (46, 9, 9)

        # Create trainer
        args = MockArgs(resume="/path/to/incomplete_checkpoint.pth")

        # Mock incomplete checkpoint data (missing some fields)
        incomplete_checkpoint_data = {
            "global_timestep": 1000,
            # Missing: total_episodes_completed, black_wins, white_wins, draws
            "model_state_dict": {},
            "optimizer_state_dict": {},
        }

        # Mock load_model to return our incomplete checkpoint data
        mock_agent_instance.load_model.return_value = incomplete_checkpoint_data

        # Set up ModelManager to return checkpoint data when handle_checkpoint_resume is called
        def mock_handle_checkpoint_resume(agent, model_dir, resume_path_override=None):
            mock_model_manager_instance.checkpoint_data = incomplete_checkpoint_data
            mock_model_manager_instance.resumed_from_checkpoint = (
                "/path/to/incomplete_checkpoint.pth"
            )
            return True

        mock_model_manager_instance.handle_checkpoint_resume.side_effect = (
            mock_handle_checkpoint_resume
        )

        # Mock torch.load to return incomplete checkpoint data and os.path.exists
        with (
            patch("torch.load", return_value=incomplete_checkpoint_data),
            patch("os.path.exists", return_value=True),
        ):
            # Create trainer - checkpoint resume happens during __init__
            trainer = Trainer(mock_config, args)

        # Verify that missing fields default to 0 (using dict.get() with default)
        assert trainer.global_timestep == 1000  # Present field
        assert (
            trainer.total_episodes_completed == 0
        )  # Missing field, should default to 0
        assert trainer.black_wins == 0  # Missing field, should default to 0
        assert trainer.white_wins == 0  # Missing field, should default to 0
        assert trainer.draws == 0  # Missing field, should default to 0

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.core.ppo_agent.PPOAgent")
    def test_trainer_no_checkpoint_resume_preserves_initial_state(
        self,
        mock_ppo_agent_class,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_session_manager_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that Trainer preserves initial state when no checkpoint is loaded."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = _create_mock_model_with_parameters()
        mock_model_factory.return_value = mock_model

        # Mock PPOAgent instance
        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance

        # Mock model and optimizer with proper load_state_dict methods
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock ModelManager instance
        mock_model_manager_instance = mock_model_manager_class.return_value
        mock_model_manager_instance.create_model.return_value = mock_model
        mock_model_manager_instance.handle_checkpoint_resume.return_value = (
            False  # No checkpoint
        )
        mock_model_manager_instance.checkpoint_data = None
        mock_model_manager_instance.resumed_from_checkpoint = None

        # Mock EnvManager instance
        mock_env_manager_instance = mock_env_manager_class.return_value
        mock_game = Mock()
        mock_policy_mapper = Mock()
        mock_env_manager_instance.setup_environment.return_value = (
            mock_game,
            mock_policy_mapper,
        )
        mock_env_manager_instance.action_space_size = 13527
        mock_env_manager_instance.obs_space_shape = (46, 9, 9)

        # Create trainer with no resume
        args = MockArgs()  # No resume specified
        trainer = Trainer(mock_config, args)

        # Mock ModelManager to return no checkpoint data
        trainer.model_manager.checkpoint_data = None
        trainer.model_manager.resumed_from_checkpoint = None

        # Call the checkpoint resume method
        trainer._handle_checkpoint_resume()

        # Verify state variables remain at initial values (0)
        assert trainer.global_timestep == 0
        assert trainer.total_episodes_completed == 0
        assert trainer.black_wins == 0
        assert trainer.white_wins == 0
        assert trainer.draws == 0
        assert trainer.resumed_from_checkpoint is None

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.core.ppo_agent.PPOAgent")
    def test_trainer_error_handling_agent_not_initialized(
        self,
        mock_ppo_agent_class,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_session_manager_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test that Trainer raises error when agent is not initialized before checkpoint resume."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = _create_mock_model_with_parameters()
        mock_model_factory.return_value = mock_model

        # Mock PPOAgent instance
        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance

        # Mock model and optimizer with proper load_state_dict methods
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock ModelManager instance
        mock_model_manager_instance = mock_model_manager_class.return_value
        mock_model_manager_instance.create_model.return_value = mock_model
        mock_model_manager_instance.handle_checkpoint_resume.return_value = True
        mock_model_manager_instance.checkpoint_data = None
        mock_model_manager_instance.resumed_from_checkpoint = None

        # Mock EnvManager instance
        mock_env_manager_instance = mock_env_manager_class.return_value
        mock_game = Mock()
        mock_policy_mapper = Mock()
        mock_env_manager_instance.setup_environment.return_value = (
            mock_game,
            mock_policy_mapper,
        )
        mock_env_manager_instance.action_space_size = 13527
        mock_env_manager_instance.obs_space_shape = (46, 9, 9)

        # Create trainer
        args = MockArgs(resume="/path/to/checkpoint.pth")
        trainer = Trainer(mock_config, args)

        # Simulate agent not being initialized
        trainer.agent = None

        # Should raise RuntimeError when agent is None
        with pytest.raises(
            RuntimeError, match="Agent not initialized before _handle_checkpoint_resume"
        ):
            trainer._handle_checkpoint_resume()

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.core.ppo_agent.PPOAgent")
    def test_trainer_model_manager_integration_flow(
        self,
        mock_ppo_agent_class,  # PPOAgent (last patch)
        mock_model_factory,  # model_factory
        _mock_experience_buffer,  # ExperienceBuffer
        _mock_policy_mapper,  # PolicyOutputMapper
        mock_feature_specs,  # FEATURE_SPECS
        _mock_shogi_game,  # ShogiGame
        mock_session_manager_class,  # SessionManager
        mock_training_loop_manager_class,  # TrainingLoopManager
        mock_model_manager_class,  # ModelManager
        mock_env_manager_class,  # EnvManager (first patch)
        mock_config,  # Added missing fixture
        temp_dir,
    ):
        """Test complete flow of ModelManager checkpoint resume integration with Trainer."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = _create_mock_model_with_parameters()
        mock_model_factory.return_value = mock_model

        # Mock PPOAgent instance
        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance

        # Mock model and optimizer with proper load_state_dict methods
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock ModelManager instance
        mock_model_manager_instance = mock_model_manager_class.return_value
        mock_model_manager_instance.create_model.return_value = mock_model
        mock_model_manager_instance.handle_checkpoint_resume.return_value = True
        mock_model_manager_instance.checkpoint_data = None  # Will be set later
        mock_model_manager_instance.resumed_from_checkpoint = None

        # Mock EnvManager instance
        mock_env_manager_instance = mock_env_manager_class.return_value
        mock_game = Mock()
        mock_policy_mapper = Mock()
        mock_env_manager_instance.setup_environment.return_value = (
            mock_game,
            mock_policy_mapper,
        )
        mock_env_manager_instance.action_space_size = 13527
        mock_env_manager_instance.obs_space_shape = (46, 9, 9)

        # Create trainer
        checkpoint_path = "/path/to/test_checkpoint.pth"
        args = MockArgs(resume=checkpoint_path)
        # Ensure device is a string, not a MagicMock
        mock_config.env.device = "cpu"
        trainer = Trainer(mock_config, args)

        # Mock the ModelManager.handle_checkpoint_resume method
        with patch.object(
            trainer.model_manager, "handle_checkpoint_resume"
        ) as mock_handle_resume:
            # Setup return value for handle_checkpoint_resume
            mock_handle_resume.return_value = True

            # Mock checkpoint data to be set by ModelManager
            test_checkpoint_data = {
                "global_timestep": 3000,
                "total_episodes_completed": 200,
                "black_wins": 80,
                "white_wins": 70,
                "draws": 50,
                "model_state_dict": {},
                "optimizer_state_dict": {},
            }
            trainer.model_manager.checkpoint_data = test_checkpoint_data
            trainer.model_manager.resumed_from_checkpoint = checkpoint_path

            # Call the checkpoint resume method
            trainer._handle_checkpoint_resume()

            # Verify ModelManager.handle_checkpoint_resume was called correctly
            mock_handle_resume.assert_called_once_with(
                agent=trainer.agent,
                model_dir=trainer.model_dir,
                resume_path_override=checkpoint_path,
            )

            # Verify Trainer state was updated from ModelManager data
            assert trainer.global_timestep == 3000
            assert trainer.total_episodes_completed == 200
            assert trainer.black_wins == 80
            assert trainer.white_wins == 70
            assert trainer.draws == 50
            assert trainer.resumed_from_checkpoint == checkpoint_path

    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.shogi.ShogiGame")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.utils.PolicyOutputMapper")
    @patch("keisei.core.experience_buffer.ExperienceBuffer")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.core.ppo_agent.PPOAgent")
    def test_trainer_end_to_end_resume_state_verification(
        self,
        mock_ppo_agent_class,
        mock_model_factory,
        _mock_experience_buffer,
        _mock_policy_mapper,
        mock_feature_specs,
        _mock_shogi_game,
        mock_session_manager_class,
        mock_training_loop_manager_class,
        mock_model_manager_class,
        mock_env_manager_class,
        mock_config,
        temp_dir,
    ):
        """Test end-to-end flow of state restoration and verification that restored state is properly used."""
        # Setup mocks
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = _create_mock_model_with_parameters()
        mock_model_factory.return_value = mock_model

        # Mock PPOAgent instance
        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance

        # Mock model and optimizer with proper load_state_dict methods
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = f"{temp_dir}/artifacts"
        mock_session_instance.model_dir = f"{temp_dir}/models"
        mock_session_instance.log_file_path = f"{temp_dir}/train.log"
        mock_session_instance.eval_log_file_path = f"{temp_dir}/eval.log"
        mock_session_instance.is_wandb_active = False

        # Mock ModelManager instance
        mock_model_manager_instance = mock_model_manager_class.return_value
        mock_model_manager_instance.create_model.return_value = mock_model
        mock_model_manager_instance.handle_checkpoint_resume.return_value = True
        mock_model_manager_instance.checkpoint_data = None  # Will be set later
        mock_model_manager_instance.resumed_from_checkpoint = None

        # Mock EnvManager instance
        mock_env_manager_instance = mock_env_manager_class.return_value
        mock_game = Mock()
        mock_policy_mapper = Mock()
        mock_env_manager_instance.setup_environment.return_value = (
            mock_game,
            mock_policy_mapper,
        )
        mock_env_manager_instance.action_space_size = 13527
        mock_env_manager_instance.obs_space_shape = (46, 9, 9)

        mock_training_loop_instance = mock_training_loop_manager_class.return_value

        # Create trainer with resume
        checkpoint_path = "/path/to/end_to_end_checkpoint.pth"
        args = MockArgs(resume=checkpoint_path)

        # Mock checkpoint data with specific values
        resumed_checkpoint_data = {
            "global_timestep": 5000,
            "total_episodes_completed": 350,
            "black_wins": 140,
            "white_wins": 120,
            "draws": 90,
            "model_state_dict": {},
            "optimizer_state_dict": {},
        }

        # Mock load_model to return our resumed checkpoint data
        mock_agent_instance.load_model.return_value = resumed_checkpoint_data

        # Set up ModelManager to return checkpoint data when handle_checkpoint_resume is called
        def mock_handle_checkpoint_resume(agent, model_dir, resume_path_override=None):
            mock_model_manager_instance.checkpoint_data = resumed_checkpoint_data
            mock_model_manager_instance.resumed_from_checkpoint = checkpoint_path
            return True

        mock_model_manager_instance.handle_checkpoint_resume.side_effect = (
            mock_handle_checkpoint_resume
        )

        # Mock torch.load to return resumed checkpoint data and os.path.exists
        with (
            patch("torch.load", return_value=resumed_checkpoint_data),
            patch("os.path.exists", return_value=True),
        ):
            # Create trainer - checkpoint resume happens during __init__
            trainer = Trainer(mock_config, args)

        # Verify state is restored from checkpoint
        assert trainer.global_timestep == 5000
        assert trainer.total_episodes_completed == 350
        assert trainer.black_wins == 140
        assert trainer.white_wins == 120
        assert trainer.draws == 90

        # Mock TrainingLoopManager to verify it receives correct trainer state
        mock_training_loop_instance.trainer = trainer

        # Verify the TrainingLoopManager can access restored state
        assert mock_training_loop_instance.trainer.global_timestep == 5000
        assert mock_training_loop_instance.trainer.total_episodes_completed == 350
        assert mock_training_loop_instance.trainer.black_wins == 140
        assert mock_training_loop_instance.trainer.white_wins == 120
        assert mock_training_loop_instance.trainer.draws == 90

        # Verify resumed_from_checkpoint is properly set
        assert trainer.resumed_from_checkpoint == checkpoint_path
