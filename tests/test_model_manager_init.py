"""
Unit tests for ModelManager initialization and basic utilities.
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
            num_actions_total=1, 
            input_channels=1, 
            seed=42,
            max_moves_per_game=500
        ),
        training=TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=1,
            learning_rate=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=1,
            enable_spinner=False,
            input_features="core",
            tower_depth=1,
            tower_width=1,
            se_ratio=0.1,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=1,
            evaluation_interval_timesteps=1,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            enable_periodic_evaluation=True,
            num_games=1, 
            opponent_type="random", 
            evaluation_interval_timesteps=1,
            max_moves_per_game=500,
            log_file_path_eval="test_eval.log",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="test.log", model_dir=tempfile.gettempdir(), run_name=None
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1,
            watch_log_type="all",
            log_model_artifact=False,
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
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
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


class TestModelManagerInitialization:
    """Test ModelManager initialization and configuration."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
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
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Verify initialization
        assert manager.config == mock_config
        assert manager.args == mock_args
        assert manager.device == device
        assert manager.logger_func == logger_func
        assert manager.model is None  # Model not created yet
        assert manager.resumed_from_checkpoint is None
        assert manager.checkpoint_data is None

        # Verify feature setup
        assert manager.input_features == mock_config.training.input_features
        assert manager.obs_shape == (46, 9, 9)

        # Verify mixed precision setup for CPU
        assert not manager.use_mixed_precision
        assert manager.scaler is None

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_initialization_with_args_override(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_config,
        device,
        logger_func,
    ):
        """Test ModelManager initialization with args override."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=20)
        mock_model_factory.return_value = Mock()

        # Create args with overrides
        args_with_overrides = MockArgs(
            input_features="custom",
            model="cnn",
            tower_depth=8,
            tower_width=128,
            se_ratio=0.1,
        )

        # Create ModelManager
        manager = ModelManager(mock_config, args_with_overrides, device, logger_func)

        # Verify args override config values
        assert manager.input_features == "custom"
        assert manager.model_type == "cnn"
        assert manager.tower_depth == 8
        assert manager.tower_width == 128
        assert abs(manager.se_ratio - 0.1) < 1e-9  # Use approximate comparison for float
        assert manager.obs_shape == (20, 9, 9)

    @patch("keisei.training.model_manager.GradScaler")
    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_mixed_precision_cuda_enabled(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_grad_scaler,
        mock_config,
        mock_args,
        logger_func,
    ):
        """Test mixed precision setup when CUDA is available."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()
        mock_scaler_instance = Mock()
        mock_grad_scaler.return_value = mock_scaler_instance

        # Enable mixed precision in config
        mock_config.training.mixed_precision = True

        # Use CUDA device
        cuda_device = torch.device("cuda")

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, cuda_device, logger_func)

        # Verify mixed precision is enabled
        assert manager.use_mixed_precision is True
        assert manager.scaler == mock_scaler_instance
        mock_grad_scaler.assert_called_once()

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_mixed_precision_cpu_warning(
        self,
        mock_model_factory,
        mock_feature_specs,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test mixed precision warning when requested on CPU."""
        # Setup mocks
        mock_feature_specs.__getitem__.return_value = Mock(num_planes=46)
        mock_model_factory.return_value = Mock()
        mock_logger = Mock()

        # Enable mixed precision in config (but device is CPU)
        mock_config.training.mixed_precision = True

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, mock_logger)

        # Verify mixed precision is disabled with warning
        assert manager.use_mixed_precision is False
        assert manager.scaler is None

        # Verify warning was logged
        mock_logger.assert_called()
        warning_call = mock_logger.call_args[0][0]
        assert (
            "Mixed precision training requested but CUDA is not available"
            in warning_call
        )


class TestModelManagerUtilities:
    """Test utility methods and information retrieval."""

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_get_model_info(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test get_model_info method returns correct information."""
        # Setup mocks
        mock_feature_spec = Mock(num_planes=46)
        mock_features.__getitem__.return_value = mock_feature_spec
        mock_model_factory.return_value = Mock()

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Get model info
        info = manager.get_model_info()

        # Verify returned information
        expected_info = {
            "model_type": mock_config.training.model_type,
            "input_features": mock_config.training.input_features,
            "tower_depth": mock_config.training.tower_depth,
            "tower_width": mock_config.training.tower_width,
            "se_ratio": mock_config.training.se_ratio,
            "obs_shape": (46, 9, 9),
            "num_planes": 46,
            "use_mixed_precision": False,  # CPU device
            "device": "cpu",
        }

        assert info == expected_info

    @patch("keisei.training.model_manager.features.FEATURE_SPECS")
    @patch("keisei.training.model_manager.model_factory")
    def test_model_creation_and_agent_instantiation(
        self,
        mock_model_factory,
        mock_features,
        mock_config,
        mock_args,
        device,
        logger_func,
    ):
        """Test model creation through ModelManager."""
        # Setup mocks
        mock_feature_spec = Mock(num_planes=46)
        mock_features.__getitem__.return_value = mock_feature_spec

        # Create a mock model that responds to .to()
        mock_model = Mock()
        mock_model.to.return_value = mock_model  # .to() returns itself
        mock_model_factory.return_value = mock_model

        # Create ModelManager
        manager = ModelManager(mock_config, mock_args, device, logger_func)

        # Create model
        created_model = manager.create_model()

        # Verify model factory was called with correct parameters
        mock_model_factory.assert_called_once_with(
            model_type=mock_config.training.model_type,
            obs_shape=(46, 9, 9),
            num_actions=mock_config.env.num_actions_total,
            tower_depth=mock_config.training.tower_depth,
            tower_width=mock_config.training.tower_width,
            se_ratio=0.1,  # se_ratio is 0.1 in mock config
        )

        # Verify model was moved to device
        mock_model.to.assert_called_once_with(device)

        # Verify model is stored and returned
        assert manager.model == mock_model
        assert created_model == mock_model
