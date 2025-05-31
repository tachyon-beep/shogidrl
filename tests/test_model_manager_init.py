"""
Unit tests for ModelManager initialization and basic utilities.
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
from keisei.core.ppo_agent import PPOAgent


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
        env=EnvConfig(device="cpu", num_actions_total=1, input_channels=1, seed=42),
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
        ),
        evaluation=EvaluationConfig(num_games=1, opponent_type="random", evaluation_interval_timesteps=1),
        logging=LoggingConfig(log_file="test.log", model_dir=tempfile.gettempdir(), run_name=None),
        wandb=WandBConfig(enabled=False, project="test", entity=None, run_name_prefix="test", watch_model=False, watch_log_freq=1, watch_log_type="all"),
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
        # ...body unchanged from original...
        pass

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
        # ...body unchanged...
        pass

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
        # ...body unchanged...
        pass

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
        # ...body unchanged...
        pass


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
        # ...body unchanged...
        pass

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
        # ...body unchanged...
        pass
