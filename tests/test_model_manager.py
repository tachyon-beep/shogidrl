"""
test_model_manager.py: Comprehensive unit tests for ModelManager class.

Tests cover model configuration, mixed precision setup, checkpoint handling,
WandB artifact creation, and model saving functionality.
"""

import os
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
    TrainingConfig,
    WandBConfig,
)
from keisei.training.model_manager import ModelManager


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self, **kwargs):
        self.resume = kwargs.get("resume", None)
        self.input_features = kwargs.get("input_features", None)
        self.model = kwargs.get("model", None)
        self.tower_depth = kwargs.get("tower_depth", None)
        self.tower_width = kwargs.get("tower_width", None)
        self.se_ratio = kwargs.get("se_ratio", None)


@pytest.fixture
def mock_config():
    """Create a mock AppConfig for testing."""
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
    return MockArgs()


@pytest.fixture
def device():
    """Create a test device."""
    return torch.device("cpu")


@pytest.fixture
def logger_func():
    """Create a mock logger function."""
    return Mock()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestModelManagerInitialization:
    """Test ModelManager initialization and configuration."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_initialization_success(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test successful ModelManager initialization."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_feature_specs.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Explicitly create model after manager initialization
        created_model = manager.create_model()

        # Verify initialization
        assert manager.config == mock_config
        assert manager.args == mock_args
        assert manager.device == device
        assert manager.logger_func == logger_func
        assert manager.input_features == "core46"
        assert manager.model_type == "resnet"
        assert manager.tower_depth == 9
        assert manager.tower_width == 256
        assert abs(manager.se_ratio - 0.25) < 1e-6
        assert manager.obs_shape == (46, 9, 9)
        assert manager.use_mixed_precision is False
        assert manager.scaler is None
        assert (
            manager.model == mock_model
        )  # model_factory was mocked to return mock_model
        assert (
            created_model == mock_model
        )  # create_model should return the created model

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_initialization_with_args_override(
        self, mock_model_factory, mock_feature_specs, mock_config, device, logger_func
    ):  # pylint: disable=too-many-positional-arguments
        """Test initialization with command-line argument overrides."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 60
        mock_feature_specs.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create args with overrides
        args = MockArgs(
            input_features="extended",
            model="cnn",
            tower_depth=12,
            tower_width=512,
            se_ratio=0.5,
        )

        # Create ModelManager
        manager = ModelManager(mock_config, args, device, logger_func)

        # Verify overrides applied
        assert manager.input_features == "extended"
        assert manager.model_type == "cnn"
        assert manager.tower_depth == 12
        assert manager.tower_width == 512
        assert abs(manager.se_ratio - 0.5) < 1e-6
        assert manager.obs_shape == (60, 9, 9)

    @patch("keisei.training.model_manager.GradScaler")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_mixed_precision_cuda_enabled(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_grad_scaler,
        mock_config,
        mock_args,
        logger_func,
    ):
        """Test mixed precision setup with CUDA enabled."""
        # Enable mixed precision in config
        mock_config.training.mixed_precision = True

        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_feature_specs["core46"] = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_scaler = Mock()
        mock_grad_scaler.return_value = mock_scaler

        # Create CUDA device
        cuda_device = torch.device("cuda")

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, cuda_device, logger_func)

        # Verify mixed precision enabled
        assert manager.use_mixed_precision is True
        assert manager.scaler == mock_scaler
        mock_grad_scaler.assert_called_once()
        logger_func.assert_any_call("Mixed precision training enabled (CUDA).")

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_mixed_precision_cpu_warning(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test mixed precision warning when CUDA not available."""
        # Enable mixed precision in config but use CPU device
        mock_config.training.mixed_precision = True

        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_feature_specs["core46"] = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Verify mixed precision disabled with warning
        assert manager.use_mixed_precision is False
        assert manager.scaler is None
        logger_func.assert_any_call(
            "Mixed precision training requested but CUDA is not available/selected. "
            "Proceeding without mixed precision."
        )


class TestModelManagerCheckpointHandling:
    """Test checkpoint loading and resuming functionality."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.training.model_manager.utils.find_latest_checkpoint")
    def test_handle_checkpoint_resume_latest_found(
        self,
        mock_find_checkpoint,
        mock_model_factory,
        mock_feature_specs,
        mock_config,
        device,
        logger_func,
        temp_dir,
    ):
        """Test resuming from latest checkpoint when found."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_feature_specs["core46"] = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
        mock_find_checkpoint.return_value = checkpoint_path

        # Create args with resume="latest"
        args = MockArgs(resume="latest")

        # Create ModelManager
        manager = ModelManager(mock_config, args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        # Test checkpoint resume
        result = manager.handle_checkpoint_resume(mock_agent, temp_dir)

        # Verify checkpoint loaded
        assert result is True
        assert manager.resumed_from_checkpoint == checkpoint_path
        mock_agent.load_model.assert_called_once_with(checkpoint_path)
        logger_func.assert_any_call(  # Updated log message
            f"Resumed from latest checkpoint: {checkpoint_path}"
        )

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.training.model_manager.utils.find_latest_checkpoint")
    def test_handle_checkpoint_resume_not_found(
        self,
        mock_find_checkpoint,
        mock_model_factory,
        mock_features,
        mock_config,
        device,
        logger_func,
        temp_dir,
    ):
        """Test resuming when no checkpoint found."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_find_checkpoint.return_value = None

        # Create args with resume="latest"
        args = MockArgs(resume="latest")

        # Create ModelManager
        manager = ModelManager(mock_config, args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        # Test checkpoint resume
        result = manager.handle_checkpoint_resume(mock_agent, temp_dir)

        # Verify no checkpoint loaded
        assert result is False
        assert manager.resumed_from_checkpoint is None
        mock_agent.load_model.assert_not_called()

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("os.path.exists")  # Add patch for os.path.exists
    def test_handle_checkpoint_resume_explicit_path(
        self,
        mock_os_path_exists,  # Add mock for os.path.exists
        mock_model_factory,
        mock_features,
        mock_config,
        device,
        logger_func,
        temp_dir,
    ):
        """Test resuming from explicit checkpoint path."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        checkpoint_path = "/path/to/specific/checkpoint.pth"
        mock_os_path_exists.return_value = (
            True  # Ensure os.path.exists returns True for the mock path
        )

        # Create args with explicit resume path
        args = MockArgs(resume=checkpoint_path)

        # Create ModelManager
        manager = ModelManager(mock_config, args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        # Test checkpoint resume
        result = manager.handle_checkpoint_resume(mock_agent, temp_dir)

        # Verify checkpoint loaded
        assert result is True
        assert manager.resumed_from_checkpoint == checkpoint_path
        mock_agent.load_model.assert_called_once_with(checkpoint_path)
        logger_func.assert_any_call(  # Updated log message
            f"Resumed from specified checkpoint: {checkpoint_path}"
        )


class TestModelManagerArtifacts:
    """Test WandB artifact creation functionality."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.training.model_manager.wandb")
    def test_create_model_artifact_success(
        self,
        mock_wandb,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test successful model artifact creation."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create a test model file
        model_path = os.path.join(temp_dir, "test_model.pth")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("test model content")

        # Setup WandB mocks
        mock_wandb.run = Mock()
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Test artifact creation
        result = manager.create_model_artifact(
            model_path=model_path,
            artifact_name="test-model",
            run_name="test_run",
            is_wandb_active=True,
            description="Test model",
            metadata={"test": "value"},
            aliases=["latest"],
        )

        # Verify artifact created
        assert result is True
        mock_wandb.Artifact.assert_called_once()
        mock_artifact.add_file.assert_called_once_with(model_path)
        mock_wandb.log_artifact.assert_called_once_with(
            mock_artifact, aliases=["latest"]
        )

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_create_model_artifact_wandb_inactive(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test artifact creation when WandB is inactive."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Test artifact creation with WandB inactive
        result = manager.create_model_artifact(
            model_path="/path/to/model.pth",
            artifact_name="test-model",
            run_name="test_run",
            is_wandb_active=False,
        )

        # Verify artifact not created
        assert result is False

    @patch("keisei.training.model_manager.wandb")
    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_create_model_artifact_file_missing(
        self,
        mock_model_factory,
        mock_features,
        mock_wandb,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):  # pylint: disable=too-many-positional-arguments
        """Test artifact creation when model file is missing."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Setup wandb mock
        mock_wandb.run = Mock()

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Test artifact creation with missing file
        result = manager.create_model_artifact(
            model_path="/nonexistent/model.pth",
            artifact_name="test-model",
            run_name="test_run",
            is_wandb_active=True,
        )

        # Verify artifact not created
        assert result is False
        logger_func.assert_any_call(
            "Warning: Model file /nonexistent/model.pth does not exist, skipping artifact creation."
        )


class TestModelManagerSaving:
    """Test model and checkpoint saving functionality."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_save_final_model_success(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test successful final model saving."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        game_stats = {"black_wins": 10, "white_wins": 5, "draws": 2}

        # Test final model saving
        with patch.object(
            manager, "create_model_artifact", return_value=True
        ) as mock_artifact:
            success, model_path = manager.save_final_model(
                agent=mock_agent,
                model_dir=temp_dir,
                global_timestep=1000,
                total_episodes_completed=17,
                game_stats=game_stats,
                run_name="test_run",
                is_wandb_active=True,
            )

        # Verify model saved
        assert success is True
        assert model_path == os.path.join(temp_dir, "final_model.pth")
        mock_agent.save_model.assert_called_once_with(model_path, 1000, 17)
        mock_artifact.assert_called_once()

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_save_final_checkpoint_success(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test successful final checkpoint saving."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        game_stats = {"black_wins": 10, "white_wins": 5, "draws": 2}

        # Test final checkpoint saving
        with patch.object(
            manager, "create_model_artifact", return_value=True
        ) as mock_artifact:
            success, checkpoint_path = manager.save_final_checkpoint(
                agent=mock_agent,
                model_dir=temp_dir,
                global_timestep=1000,
                total_episodes_completed=17,
                game_stats=game_stats,
                run_name="test_run",
                is_wandb_active=True,
            )

        # Verify checkpoint saved
        assert success is True
        expected_path = os.path.join(temp_dir, "checkpoint_ts1000.pth")
        assert checkpoint_path == expected_path
        mock_agent.save_model.assert_called_once_with(
            expected_path, 1000, 17, stats_to_save=game_stats
        )
        mock_artifact.assert_called_once()

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_save_final_checkpoint_zero_timestep(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test final checkpoint saving with zero timestep."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.FEATURE_SPECS = {"core46": mock_feature_spec}

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Create mock agent
        mock_agent = Mock()

        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        # Test final checkpoint saving with zero timestep
        success, checkpoint_path = manager.save_final_checkpoint(
            agent=mock_agent,
            model_dir=temp_dir,
            global_timestep=0,
            total_episodes_completed=0,
            game_stats=game_stats,
            run_name="test_run",
            is_wandb_active=True,
        )

        # Verify checkpoint not saved
        assert success is False
        assert checkpoint_path is None
        mock_agent.save_model.assert_not_called()


class TestModelManagerUtilities:
    """Test utility methods and information retrieval."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_get_model_info(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):  # pylint: disable=too-many-positional-arguments
        """Test model information retrieval."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Get model info
        info = manager.get_model_info()

        # Verify information
        expected_info = {
            "model_type": "resnet",
            "input_features": "core46",
            "tower_depth": 9,
            "tower_width": 256,
            "se_ratio": 0.25,
            "obs_shape": (46, 9, 9),
            "num_planes": 46,
            "use_mixed_precision": False,
            "device": "cpu",
        }
        assert info == expected_info

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    # PPOAgent patch is removed as we instantiate it directly.
    def test_model_creation_and_agent_instantiation(  # Renamed test
        self,
        # mock_ppo_agent, # Removed
        mock_model_factory,
        mock_features,  # This is the patch object for 'keisei.shogi.features.FEATURE_SPECS'
        mock_config,
        mock_args,
        device,
        logger_func,
    ):  # pylint: disable=too-many-positional-arguments
        """Test model creation via ModelManager and subsequent PPOAgent instantiation."""
        from keisei.core.ppo_agent import PPOAgent  # Import for direct instantiation

        # Setup mocks for model creation
        mock_feature_spec_instance = Mock(name="MockFeatureSpecInstance")
        mock_feature_spec_instance.num_planes = 46
        # mock_features is the patch object for 'keisei.shogi.features.FEATURE_SPECS'
        # We need to mock its __getitem__ if it's a dict-like access.
        mock_features.__getitem__.return_value = mock_feature_spec_instance

        # This is the model that model_factory will return
        mock_model_from_factory = Mock(name="MockModelFromFactory")
        mock_model_from_factory.to.return_value = mock_model_from_factory
        mock_model_factory.return_value = mock_model_from_factory

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # 1. Create model using ModelManager
        # This call will set manager.model internally and return the model
        returned_model = manager.create_model()
        assert returned_model == mock_model_from_factory
        assert manager.model == mock_model_from_factory  # Verify internal state

        # 2. Instantiate PPOAgent (as Trainer would do)
        # We instantiate PPOAgent directly.
        agent = PPOAgent(
            config=mock_config,
            device=device,
        )

        # Ensure returned_model is not None before assigning it to agent.model
        # manager.create_model() is type-hinted to return ActorCriticProtocol,
        # and raises an error if it can't. So, returned_model should not be None here.
        assert (
            returned_model is not None
        ), "manager.create_model() should have returned a model or raised an error."
        agent.model = returned_model  # Now Pylance should be satisfied

        # Verify agent created and model assigned
        # agent.model was just assigned returned_model, which was asserted to be mock_model_from_factory
        assert agent.model == mock_model_from_factory
        assert agent.device == device
        assert agent.config == mock_config


class TestModelManagerEnhancedCheckpointHandling:
    """Enhanced tests for checkpoint loading scenarios and edge cases."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("keisei.training.model_manager.utils.find_latest_checkpoint")
    def test_load_checkpoint_multiple_available(
        self,
        mock_find_checkpoint,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test checkpoint loading when multiple checkpoints exist."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Mock find_latest_checkpoint to return the latest checkpoint
        latest_checkpoint_path = os.path.join(mock_config.logging.model_dir, "checkpoint_2000.pth")
        mock_find_checkpoint.return_value = latest_checkpoint_path

        # Mock checkpoint data
        checkpoint_data = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "global_timestep": 2000,
            "total_episodes_completed": 50,
            "black_wins": 25,
            "white_wins": 20,
            "draws": 5,
        }

        # Create args with resume="latest"
        args_with_resume = MockArgs(resume="latest")
        manager = ModelManager(mock_config, args_with_resume, device, logger_func)
        manager.create_model()

        # Mock an agent for testing
        mock_agent = Mock()
        mock_agent.load_model.return_value = checkpoint_data

        # Test loading latest checkpoint
        result = manager.handle_checkpoint_resume(mock_agent, mock_config.logging.model_dir)

        assert result is True
        assert manager.checkpoint_data is not None
        assert manager.checkpoint_data["global_timestep"] == 2000
        mock_agent.load_model.assert_called_with(latest_checkpoint_path)

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("os.path.exists")
    def test_load_checkpoint_specific_not_found(
        self,
        mock_exists,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test loading a specific checkpoint that doesn't exist."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Mock specific checkpoint doesn't exist
        mock_exists.return_value = False

        manager = ModelManager(mock_config, mock_args, device, logger_func)
        manager.create_model()

        mock_agent = Mock()

        # Test loading specific non-existent checkpoint - set args to use specific path
        args_with_resume = MockArgs(resume="checkpoint_9999.pth")
        manager_with_resume = ModelManager(mock_config, args_with_resume, device, logger_func)
        manager_with_resume.create_model()

        result = manager_with_resume.handle_checkpoint_resume(mock_agent, "/some/model/dir")

        assert result is False
        logger_func.assert_any_call("Specified resume checkpoint not found: checkpoint_9999.pth")

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("os.path.exists")
    @patch("torch.load")
    def test_load_checkpoint_corrupted_data(
        self,
        mock_torch_load,
        mock_exists,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test handling of corrupted checkpoint data."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_exists.return_value = True
        # Mock corrupted checkpoint (missing required keys)
        mock_torch_load.return_value = {"incomplete": "data"}

        manager = ModelManager(mock_config, mock_args, device, logger_func)
        manager.create_model()

        mock_agent = Mock()
        mock_agent.load_model.return_value = {"incomplete": "data"}

        # Test loading corrupted checkpoint
        args_with_resume = MockArgs(resume="latest")
        manager_with_resume = ModelManager(mock_config, args_with_resume, device, logger_func)
        manager_with_resume.create_model()

        # Mock the utils.find_latest_checkpoint to return a valid path
        with patch("keisei.training.model_manager.utils.find_latest_checkpoint", return_value="/path/to/corrupt.pth"):
            result = manager_with_resume.handle_checkpoint_resume(mock_agent, "/some/model/dir")

        # Should handle gracefully - either succeed or fail, but not crash
        assert isinstance(result, bool)  # Should return a boolean

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    def test_save_checkpoint_directory_creation(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test that save_checkpoint creates directories if they don't exist."""
        # Setup mocks
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        # Use a non-existent subdirectory
        nonexistent_model_dir = os.path.join(temp_dir, "models", "subdir")
        mock_config.logging.model_dir = nonexistent_model_dir

        manager = ModelManager(mock_config, mock_args, device, logger_func)
        manager.create_model()

        mock_agent = Mock()
        mock_agent.model = manager.model
        mock_agent.optimizer = Mock()

        # Test checkpoint saving - should create directory
        stats = {
            "black_wins": 10,
            "white_wins": 8,
            "draws": 7,
        }
        manager.save_checkpoint(
            agent=mock_agent,
            model_dir=nonexistent_model_dir,
            timestep=1000,
            episode_count=25,
            stats=stats,
            run_name="test_run",
            is_wandb_active=False
        )

        # Verify directory was created
        assert os.path.exists(nonexistent_model_dir)

        # Verify save was attempted via agent.save_model
        mock_agent.save_model.assert_called_once_with(
            os.path.join(nonexistent_model_dir, "checkpoint_ts1000.pth"),
            1000,
            25,
            stats_to_save=stats,
        )


class TestModelManagerWandBArtifactEnhancements:
    """Enhanced tests for W&B artifact creation with edge cases."""

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("wandb.Artifact")
    @patch("wandb.log_artifact")
    @patch("wandb.run")
    def test_create_model_artifact_with_metadata(
        self,
        mock_wandb_run,
        mock_log_artifact,
        mock_artifact_class,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test model artifact creation with comprehensive metadata."""
        # Setup mocks
        mock_wandb_run.return_value = True  # Mock that wandb.run is active
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        # Create test model file
        model_path = os.path.join(temp_dir, "test_model.pth")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("dummy model data")

        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Test artifact creation with metadata
        result = manager.create_model_artifact(
            model_path=model_path,
            artifact_name="enhanced-test-model",
            run_name="test_run_with_metadata",
            is_wandb_active=True,
            metadata={"epochs": 100, "accuracy": 0.95},
        )

        # Verify artifact creation was attempted (mocked W&B environment)
        assert result is True
        mock_artifact_class.assert_called_once()
        mock_log_artifact.assert_called_once()
        mock_artifact_class.assert_called_once()
        mock_artifact.add_file.assert_called_once_with(model_path)
        mock_log_artifact.assert_called_once_with(mock_artifact, aliases=None)

    @patch("keisei.shogi.features.FEATURE_SPECS")
    @patch("keisei.training.models.model_factory")
    @patch("wandb.Artifact")
    @patch("wandb.log_artifact", side_effect=RuntimeError("W&B API Error"))
    @patch("wandb.run")
    def test_create_model_artifact_wandb_failure_handling(
        self,
        mock_wandb_run,
        mock_log_artifact,
        mock_artifact_class,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test robust handling of W&B API failures during artifact creation."""
        # Setup mocks
        mock_wandb_run.return_value = True  # Mock that wandb.run is active
        mock_feature_spec = Mock()
        mock_feature_spec.num_planes = 46
        mock_features.__getitem__.return_value = mock_feature_spec

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        # Create test model file
        model_path = os.path.join(temp_dir, "test_model.pth")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("dummy model data")

        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Test artifact creation with W&B failure
        result = manager.create_model_artifact(
            model_path=model_path,
            artifact_name="failing-test-model",
            run_name="test_run_fail",
            is_wandb_active=True,
        )

        # Should handle failure gracefully
        assert result is False
        logger_func.assert_any_call("Error creating W&B artifact for %s: W&B API Error" % model_path)

    def test_create_model_artifact_wandb_inactive(
        self,
        mock_config,
        mock_args,
        device,
        logger_func,
        temp_dir,
    ):
        """Test that artifact creation is skipped when W&B is inactive."""
        with patch("keisei.shogi.features.FEATURE_SPECS") as mock_features, \
             patch("keisei.training.models.model_factory") as mock_model_factory:
            
            # Setup mocks
            mock_feature_spec = Mock()
            mock_feature_spec.num_planes = 46
            mock_features.__getitem__.return_value = mock_feature_spec

            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model_factory.return_value = mock_model
            
            # Create test model file
            model_path = os.path.join(temp_dir, "test_model.pth")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("dummy model data")
            
            manager = ModelManager(mock_config, mock_args, device, logger_func)
            
            # Test with W&B inactive
            result = manager.create_model_artifact(
                model_path=model_path,
                artifact_name="inactive-wandb-test",
                run_name="test_run",
                is_wandb_active=False  # W&B not active
            )
            
            # Should skip artifact creation
            assert result is False
            # The method returns False immediately when W&B is inactive,
            # without logging a specific message
