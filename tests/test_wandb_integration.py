"""
test_wandb_integration.py: Tests for Weights & Biases integration in Keisei.

This module tests the W&B artifacts functionality, sweep parameter handling,
and W&B logging integration in the Trainer class.
"""

from typing import Any, Dict
from unittest.mock import Mock, patch

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
from keisei.training.train_wandb_sweep import apply_wandb_sweep_config
from keisei.training.utils import setup_wandb


class DummyArgs:
    """Mock args object for testing."""
    def __init__(self, **kwargs):
        self.run_name = "test_run"
        self.resume = None
        self.__dict__.update(kwargs)


def make_test_config(**overrides) -> AppConfig:
    """Create a test configuration with W&B settings."""
    training_data: Dict[str, Any] = {
        "total_timesteps": 1000,
        "steps_per_epoch": 64,
        "ppo_epochs": 2,
        "minibatch_size": 32,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_epsilon": 0.2,
        "value_loss_coeff": 0.5,
        "entropy_coef": 0.01,
        "render_every_steps": 1,
        "refresh_per_second": 4,
        "enable_spinner": False,
        "input_features": "core46",
        "tower_depth": 5,
        "tower_width": 64,
        "se_ratio": 0.0,
        "model_type": "resnet",
        "mixed_precision": False,
        "ddp": False,
        "gradient_clip_max_norm": 0.5,
        "lambda_gae": 0.95,
        "checkpoint_interval_timesteps": 500,
        "evaluation_interval_timesteps": 1000,
    }
    training_data.update({k: v for k, v in overrides.items() if k in training_data})
    training = TrainingConfig(**training_data)

    env_data: Dict[str, Any] = {
        "device": "cpu",
        "input_channels": 46,
        "num_actions_total": 4158,
        "seed": 42,
    }
    env_data.update({k: v for k, v in overrides.items() if k in env_data})
    env = EnvConfig(**env_data)

    evaluation = EvaluationConfig(
        num_games=1,
        opponent_type="random",
        evaluation_interval_timesteps=1000
    )

    logging = LoggingConfig(
        log_file="test_training.log",
        model_dir="test_models",
        run_name="test_run"
    )

    # W&B config with test-friendly defaults
    wandb_enabled = overrides.get("wandb_enabled", False)
    wandb = WandBConfig(
        enabled=wandb_enabled,
        project="keisei-test",
        entity=None,
        run_name_prefix="test",
        watch_model=False,
        watch_log_freq=1000,
        watch_log_type="all"
    )

    demo = DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0)

    return AppConfig(
        training=training,
        env=env,
        evaluation=evaluation,
        logging=logging,
        wandb=wandb,
        demo=demo,
    )


class TestWandBArtifacts:
    """Test W&B artifacts functionality."""

    def test_create_model_artifact_wandb_disabled(self, tmp_path):
        """Test artifact creation when W&B is disabled."""
        config = make_test_config(wandb_enabled=False)
        args = DummyArgs()

        with patch('keisei.training.utils.setup_wandb', return_value=False):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = False

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Mock log function
            log_mock = Mock()

            # Should return False when W&B is disabled
            result = trainer._create_model_artifact(  # pylint: disable=protected-access
                model_path=str(model_path),
                artifact_name="test-model",
                description="Test model",
                metadata={"test": True},
                aliases=["latest"],
                log_both=log_mock
            )

            assert result is False
            log_mock.assert_not_called()

    @patch('wandb.run')
    @patch('wandb.Artifact')
    @patch('wandb.log_artifact')
    def test_create_model_artifact_success(self, mock_log_artifact, mock_artifact_class, mock_wandb_run, tmp_path):
        """Test successful artifact creation when W&B is enabled."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        # Mock W&B objects
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact
        mock_wandb_run.return_value = True

        with patch('keisei.training.utils.setup_wandb', return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_123"

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Mock log function
            log_mock = Mock()

            # Test artifact creation
            result = trainer._create_model_artifact(  # pylint: disable=protected-access
                model_path=str(model_path),
                artifact_name="test-model",
                artifact_type="model",
                description="Test model for unit testing",
                metadata={"timesteps": 1000, "test": True},
                aliases=["latest", "test"],
                log_both=log_mock
            )

            # Verify result
            assert result is True

            # Verify artifact was created with correct parameters
            mock_artifact_class.assert_called_once_with(
                name="test_run_123-test-model",
                type="model",
                description="Test model for unit testing",
                metadata={"timesteps": 1000, "test": True}
            )

            # Verify file was added to artifact
            mock_artifact.add_file.assert_called_once_with(str(model_path))

            # Verify artifact was logged with aliases
            mock_log_artifact.assert_called_once_with(mock_artifact, aliases=["latest", "test"])

            # Verify logging message
            log_mock.assert_called_once()
            log_call_args = log_mock.call_args[0][0]
            assert "test_run_123-test-model" in log_call_args
            assert "created and uploaded" in log_call_args
            assert "latest" in log_call_args
            assert "test" in log_call_args

    def test_create_model_artifact_missing_file(self, tmp_path):
        """Test artifact creation with missing model file."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with patch('keisei.training.utils.setup_wandb', return_value=True), \
             patch('wandb.run', return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True

            # Use non-existent file path
            missing_path = tmp_path / "missing_model.pth"

            # Mock log function
            log_mock = Mock()

            # Should return False for missing file
            result = trainer._create_model_artifact(  # pylint: disable=protected-access
                model_path=str(missing_path),
                artifact_name="test-model",
                log_both=log_mock
            )

            assert result is False
            log_mock.assert_called_once()
            log_call_args = log_mock.call_args[0][0]
            assert "does not exist" in log_call_args

    @patch('wandb.run')
    @patch('wandb.Artifact')
    @patch('wandb.log_artifact')
    def test_create_model_artifact_wandb_error(self, mock_log_artifact, mock_artifact_class, mock_wandb_run, tmp_path):
        """Test artifact creation when W&B throws an error."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        # Mock W&B to throw an error
        mock_log_artifact.side_effect = RuntimeError("W&B API error")
        mock_artifact_class.return_value = Mock()
        mock_wandb_run.return_value = True

        with patch('keisei.training.utils.setup_wandb', return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_error"

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Mock log function
            log_mock = Mock()

            # Should return False when W&B throws error
            result = trainer._create_model_artifact(  # pylint: disable=protected-access
                model_path=str(model_path),
                artifact_name="test-model",
                log_both=log_mock
            )

            assert result is False

            # Verify error was logged
            log_mock.assert_called_once()
            log_call_args = log_mock.call_args
            assert "Error creating W&B artifact" in log_call_args[0][0]
            assert log_call_args[1]["log_level"] == "error"

    def test_create_model_artifact_default_parameters(self, tmp_path):
        """Test artifact creation with default parameters."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with patch('wandb.run', return_value=True), \
             patch('wandb.Artifact') as mock_artifact_class, \
             patch('wandb.log_artifact'), \
             patch('keisei.training.utils.setup_wandb', return_value=True):

            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_defaults"

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Test with minimal parameters
            result = trainer._create_model_artifact(  # pylint: disable=protected-access
                model_path=str(model_path),
                artifact_name="minimal-model"
            )

            assert result is True

            # Verify defaults were used
            mock_artifact_class.assert_called_once_with(
                name="test_run_defaults-minimal-model",
                type="model",  # default
                description="Model checkpoint from run test_run_defaults",  # default
                metadata={}  # default
            )


class TestWandBSweepIntegration:
    """Test W&B sweep parameter handling."""

    def test_sweep_config_mapping(self):
        """Test that sweep configuration parameters are mapped correctly."""
        # Create a Mock that properly supports dict() conversion
        mock_config = Mock()
        mock_config.learning_rate = 1e-3
        mock_config.gamma = 0.98
        mock_config.ppo_epochs = 8
        mock_config.tower_depth = 12

        # Mock the dict() conversion by implementing keys() method
        def mock_keys():
            return ['learning_rate', 'gamma', 'ppo_epochs', 'tower_depth']

        mock_config.keys = mock_keys

        # Mock dict() function behavior for wandb.config
        def mock_dict_conversion(obj):
            if obj is mock_config:
                return {
                    'learning_rate': 1e-3,
                    'gamma': 0.98,
                    'ppo_epochs': 8,
                    'tower_depth': 12
                }
            return {}

        with patch('wandb.run', return_value=True), \
             patch('wandb.config', mock_config), \
             patch('builtins.dict', side_effect=mock_dict_conversion):

            overrides = apply_wandb_sweep_config()

            # Verify sweep parameters are mapped to config paths
            assert overrides['training.learning_rate'] == pytest.approx(1e-3)
            assert overrides['training.gamma'] == pytest.approx(0.98)
            assert overrides['training.ppo_epochs'] == 8
            assert overrides['training.tower_depth'] == 12
            assert overrides['wandb.enabled'] is True

    def test_sweep_config_no_wandb_run(self):
        """Test sweep config when no W&B run is active."""
        with patch('wandb.run', None):
            overrides = apply_wandb_sweep_config()
            assert not overrides

    def test_sweep_config_partial_parameters(self):
        """Test sweep config with only some parameters present."""
        # Mock config with only some parameters
        mock_config = Mock()
        mock_config.learning_rate = 5e-4

        # Mock the dict() conversion with only learning_rate
        def mock_keys():
            return ['learning_rate']

        mock_config.keys = mock_keys

        def mock_dict_conversion(obj):
            if obj is mock_config:
                return {'learning_rate': 5e-4}
            return {}

        # Mock hasattr to return False for missing parameters
        def mock_hasattr(_obj, name):
            return name == 'learning_rate'

        with patch('wandb.run', return_value=True), \
             patch('wandb.config', mock_config), \
             patch('builtins.dict', side_effect=mock_dict_conversion), \
             patch('builtins.hasattr', side_effect=mock_hasattr):

            overrides = apply_wandb_sweep_config()

            # Only learning_rate should be included
            assert overrides['training.learning_rate'] == pytest.approx(5e-4)
            assert overrides['wandb.enabled'] is True
            assert len([k for k in overrides if k.startswith('training.')]) == 1


class TestWandBUtilities:
    """Test W&B utility functions."""

    def test_setup_wandb_disabled(self):
        """Test W&B setup when disabled in config."""
        config = make_test_config(wandb_enabled=False)

        result = setup_wandb(config, "test_run", "/tmp/test")

        assert result is False

    @patch('wandb.init')
    def test_setup_wandb_success(self, mock_wandb_init):
        """Test successful W&B setup."""
        config = make_test_config(wandb_enabled=True)

        result = setup_wandb(config, "test_run", "/tmp/test")

        assert result is True
        mock_wandb_init.assert_called_once()

        # Verify init was called with correct parameters
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs['project'] == "keisei-test"
        assert call_kwargs['name'] == "test_run"
        assert call_kwargs['mode'] == "online"
        assert call_kwargs['id'] == "test_run"

    @patch('wandb.init')
    def test_setup_wandb_init_error(self, mock_wandb_init):
        """Test W&B setup when init throws an error."""
        config = make_test_config(wandb_enabled=True)
        mock_wandb_init.side_effect = OSError("Network error")

        result = setup_wandb(config, "test_run", "/tmp/test")

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
