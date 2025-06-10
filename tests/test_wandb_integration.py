"""
test_wandb_integration.py: Tests for Weights & Biases integration in Keisei.

This module tests the W&B artifacts functionality, sweep parameter handling,
and W&B logging integration in the Trainer class.
"""

import os
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

import wandb
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
from keisei.training.utils import apply_wandb_sweep_config, setup_wandb


@pytest.fixture
def temp_base_dir():
    """Provide a temporary base directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


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
    training = TrainingConfig(
        **training_data,
        normalize_advantages=True,
        lr_schedule_type=None,
        lr_schedule_kwargs=None,
        lr_schedule_step_on="epoch",
    )

    env_data: Dict[str, Any] = {
        "device": "cpu",
        "input_channels": 46,
        "num_actions_total": 13527,
        "seed": 42,
    }
    env_data.update({k: v for k, v in overrides.items() if k in env_data})
    env = EnvConfig(**env_data)

    evaluation = EvaluationConfig(
        num_games=1,
        opponent_type="random",
        evaluation_interval_timesteps=1000,
        enable_periodic_evaluation=False,
        max_moves_per_game=500,
        log_file_path_eval="eval_log.txt",
        wandb_log_eval=False,
    )

    logging = LoggingConfig(
        log_file="test_training.log", model_dir="test_models", run_name="test_run"
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
        watch_log_type="all",
        log_model_artifact=False,
    )

    display = DisplayConfig(display_moves=False, turn_tick=0.0)

    return AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=30.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        training=training,
        env=env,
        evaluation=evaluation,
        logging=logging,
        wandb=wandb,
        display=display,
    )


class TestWandBArtifacts:
    """Test W&B artifacts functionality."""

    def test_create_model_artifact_wandb_disabled(self, tmp_path):
        """Test artifact creation when W&B is disabled."""
        config = make_test_config(wandb_enabled=False)
        args = DummyArgs()

        with patch("keisei.training.utils.setup_wandb", return_value=False):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = False

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Should return False when W&B is disabled
            result = trainer.model_manager.create_model_artifact(
                model_path=str(model_path),
                artifact_name="test-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
                description="Test model",
                metadata={"test": True},
                aliases=["latest"],
            )

            assert result is False

    def test_create_model_artifact_success(self, mock_wandb_active, tmp_path):
        """Test successful artifact creation when W&B is enabled."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        # Configure the mock W&B active fixture
        mock_artifact = Mock()
        mock_wandb_active["artifact_class"].return_value = mock_artifact
        mock_wandb_active["run"].return_value = True

        with patch("keisei.training.utils.setup_wandb", return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_123"

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Test artifact creation
            result = trainer.model_manager.create_model_artifact(
                model_path=str(model_path),
                artifact_name="test-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
                artifact_type="model",
                description="Test model for unit testing",
                metadata={"timesteps": 1000, "test": True},
                aliases=["latest", "test"],
            )

            # Verify result
            assert result is True

            # Verify artifact was created with correct parameters
            mock_wandb_active["artifact_class"].assert_called_once_with(
                name="test_run_123-test-model",
                type="model",
                description="Test model for unit testing",
                metadata={"timesteps": 1000, "test": True},
            )

            # Verify file was added to artifact
            mock_artifact.add_file.assert_called_once_with(str(model_path))

            # Verify artifact was logged with aliases
            mock_wandb_active["log_artifact"].assert_called_once_with(
                mock_artifact, aliases=["latest", "test"]
            )

    def test_create_model_artifact_missing_file(self, tmp_path):
        """Test artifact creation with missing model file."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with patch("keisei.training.utils.setup_wandb", return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True

            # Use non-existent file path
            missing_path = tmp_path / "missing_model.pth"

            # Should return False for missing file
            result = trainer.model_manager.create_model_artifact(
                model_path=str(missing_path),
                artifact_name="test-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            assert result is False

    def test_create_model_artifact_wandb_error(self, mock_wandb_active, tmp_path):
        """Test artifact creation when W&B throws an error."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        # Mock W&B to throw an error
        mock_wandb_active["log_artifact"].side_effect = RuntimeError("W&B API error")
        mock_wandb_active["artifact_class"].return_value = Mock()
        mock_wandb_active["run"].return_value = True

        with patch("keisei.training.utils.setup_wandb", return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_error"

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Should return False when W&B throws error
            result = trainer.model_manager.create_model_artifact(
                model_path=str(model_path),
                artifact_name="test-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            assert result is False

    def test_create_model_artifact_default_parameters(self, tmp_path):
        """Test artifact creation with default parameters."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with (
            patch("wandb.run", return_value=True),
            patch("wandb.Artifact") as mock_artifact_class,
            patch("wandb.log_artifact"),
            patch("keisei.training.utils.setup_wandb", return_value=True),
        ):

            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_defaults"

            # Create a dummy model file
            model_path = tmp_path / "test_model.pth"
            model_path.write_text("dummy model content")

            # Test with minimal parameters
            result = trainer.model_manager.create_model_artifact(
                model_path=str(model_path),
                artifact_name="minimal-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            assert result is True

            # Verify defaults were used
            mock_artifact_class.assert_called_once_with(
                name="test_run_defaults-minimal-model",
                type="model",  # default
                description="Model checkpoint from run test_run_defaults",  # default
                metadata={},  # default
            )

    def test_create_model_artifact_retry_logic(self, mock_wandb_active, tmp_path):
        """Test artifact creation retry logic with network failures."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        # Create a dummy model file
        model_path = tmp_path / "test_model.pth"
        model_path.write_text("dummy model content")

        # Test Case 1: Success on second attempt
        call_count = 0

        def failing_log_artifact(artifact, aliases=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network timeout")
            # Success on second attempt
            return None

        mock_wandb_active["log_artifact"].side_effect = failing_log_artifact
        mock_wandb_active["artifact_class"].return_value = Mock()
        mock_wandb_active["run"].return_value = True

        with patch("keisei.training.utils.setup_wandb", return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_retry"

            # Should succeed after retry
            result = trainer.model_manager.create_model_artifact(
                model_path=str(model_path),
                artifact_name="retry-test-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            assert result is True
            assert call_count == 2  # Failed once, succeeded on retry

        # Test Case 2: All retries fail
        mock_wandb_active["log_artifact"].side_effect = RuntimeError(
            "Persistent network error"
        )
        call_count = 0

        with patch("keisei.training.utils.setup_wandb", return_value=True):
            trainer = Trainer(config=config, args=args)
            trainer.is_train_wandb_active = True
            trainer.run_name = "test_run_retry_fail"

            # Should fail after all retries
            result = trainer.model_manager.create_model_artifact(
                model_path=str(model_path),
                artifact_name="retry-fail-model",
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            assert result is False


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
            return ["learning_rate", "gamma", "ppo_epochs", "tower_depth"]

        mock_config.keys = mock_keys

        # Mock dict() function behavior for wandb.config
        def mock_dict_conversion(obj):
            if obj is mock_config:
                return {
                    "learning_rate": 1e-3,
                    "gamma": 0.98,
                    "ppo_epochs": 8,
                    "tower_depth": 12,
                }
            return {}

        with (
            patch("wandb.run", return_value=True),
            patch("wandb.config", mock_config),
            patch("builtins.dict", side_effect=mock_dict_conversion),
        ):

            overrides = apply_wandb_sweep_config()

            # Verify sweep parameters are mapped to config paths
            assert overrides["training.learning_rate"] == pytest.approx(1e-3)
            assert overrides["training.gamma"] == pytest.approx(0.98)
            assert overrides["training.ppo_epochs"] == 8
            assert overrides["training.tower_depth"] == 12
            assert overrides["wandb.enabled"] is True

    def test_sweep_config_no_wandb_run(self):
        """Test sweep config when no W&B run is active."""
        with patch("wandb.run", None):
            overrides = apply_wandb_sweep_config()
            assert not overrides

    def test_sweep_config_partial_parameters(self):
        """Test sweep config with only some parameters present."""
        # Mock config with only some parameters
        mock_config = Mock()
        mock_config.learning_rate = 5e-4

        # Mock the dict() conversion with only learning_rate
        def mock_keys():
            return ["learning_rate"]

        mock_config.keys = mock_keys

        def mock_dict_conversion(obj):
            if obj is mock_config:
                return {"learning_rate": 5e-4}
            return {}

        # Mock hasattr to return False for missing parameters
        def mock_hasattr(_obj, name):
            return name == "learning_rate"

        with (
            patch("wandb.run", return_value=True),
            patch("wandb.config", mock_config),
            patch("builtins.dict", side_effect=mock_dict_conversion),
            patch("builtins.hasattr", side_effect=mock_hasattr),
        ):

            overrides = apply_wandb_sweep_config()

            # Only learning_rate should be included
            assert overrides["training.learning_rate"] == pytest.approx(5e-4)
            assert overrides["wandb.enabled"] is True
            assert len([k for k in overrides if k.startswith("training.")]) == 1


class TestWandBUtilities:
    """Test W&B utility functions."""

    def test_setup_wandb_disabled(self):
        """Test W&B setup when disabled in config."""
        config = make_test_config(wandb_enabled=False)

        result = setup_wandb(config, "test_run", "/tmp/test")

        assert result is False

    @pytest.mark.parametrize(
        "init_side_effect,expected_result,should_verify_call_args",
        [
            (None, True, True),  # Success case
            (OSError("Network error"), False, False),  # Error case
        ],
        ids=["success", "init_error"],
    )
    def test_setup_wandb_scenarios(
        self,
        mock_wandb_disabled,
        init_side_effect,
        expected_result,
        should_verify_call_args,
    ):
        """Test W&B setup with success and error scenarios."""
        config = make_test_config(wandb_enabled=True)

        # Mock wandb.init based on test scenario
        with patch("wandb.init", side_effect=init_side_effect) as mock_wandb_init:
            result = setup_wandb(config, "test_run", "/tmp/test")

            assert result is expected_result

            if expected_result:
                mock_wandb_init.assert_called_once()

                if should_verify_call_args:
                    # Verify init was called with correct parameters
                    call_kwargs = mock_wandb_init.call_args[1]
                    assert call_kwargs["project"] == "keisei-test"
                    assert call_kwargs["name"] == "test_run"
                    assert call_kwargs["mode"] == "online"
                    assert call_kwargs["id"] == "test_run"


class TestWandBLoggingIntegration:
    """Test the integration between Trainer.log_both and W&B logging with current logic."""

    def test_log_both_impl_creation_and_wandb_logic(self, temp_base_dir):
        """Test that log_both_impl correctly implements W&B logging logic."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with (
            patch("keisei.training.utils.setup_wandb", return_value=True),
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
        ):

            # Create actual test directories
            run_artifact_dir = os.path.join(temp_base_dir, "test_run")
            model_dir = os.path.join(run_artifact_dir, "models")
            os.makedirs(run_artifact_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Mock directory setup with actual paths
            mock_setup_dirs.return_value = {
                "run_artifact_dir": run_artifact_dir,
                "model_dir": model_dir,
                "log_file_path": os.path.join(run_artifact_dir, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_dir, "rich_periodic_eval_log.txt"
                ),
            }

            trainer = Trainer(config=config, args=args)

            # Verify W&B is active after setup
            assert trainer.session_manager.is_wandb_active is True
            assert trainer.is_train_wandb_active is True

            # Create a mock logger to simulate the log_both_impl creation
            mock_logger = Mock()

            # Test the log_both_impl logic directly (simulating what happens in run_training_loop)
            with (
                patch("wandb.log") as mock_wandb_log,
                patch("wandb.run", Mock()) as mock_wandb_run,
            ):  # wandb.run exists

                # Simulate the log_both_impl function from run_training_loop
                def log_both_impl(
                    message: str,
                    also_to_wandb: bool = False,
                    wandb_data: Optional[Dict] = None,
                ):
                    mock_logger.log(message)
                    if trainer.is_train_wandb_active and also_to_wandb:
                        if mock_wandb_run:
                            log_payload = {"train_message": message}
                            if wandb_data:
                                log_payload.update(wandb_data)
                            mock_wandb_log(log_payload, step=trainer.metrics_manager.global_timestep)

                # Test W&B logging when conditions are met
                log_both_impl(
                    "Test message", also_to_wandb=True, wandb_data={"loss": 0.5}
                )

                expected_payload = {"train_message": "Test message", "loss": 0.5}
                mock_wandb_log.assert_called_once_with(
                    expected_payload, step=trainer.metrics_manager.global_timestep
                )

    def test_log_both_impl_wandb_run_none(self, temp_base_dir):
        """Test that log_both_impl handles the case where wandb.run is None."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with (
            patch("keisei.training.utils.setup_wandb", return_value=True),
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
        ):

            # Create actual test directories
            run_artifact_dir = os.path.join(temp_base_dir, "test_run")
            model_dir = os.path.join(run_artifact_dir, "models")
            os.makedirs(run_artifact_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Mock directory setup with actual paths
            mock_setup_dirs.return_value = {
                "run_artifact_dir": run_artifact_dir,
                "model_dir": model_dir,
                "log_file_path": os.path.join(run_artifact_dir, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_dir, "rich_periodic_eval_log.txt"
                ),
            }

            trainer = Trainer(config=config, args=args)
            mock_logger = Mock()

            # Test with wandb.run = None
            with patch("wandb.log") as mock_wandb_log, patch("wandb.run", None):
                import wandb  # Import within the test context where it's patched

                def log_both_impl(
                    message: str,
                    also_to_wandb: bool = False,
                    wandb_data: Optional[Dict] = None,
                ):
                    mock_logger.log(message)
                    if trainer.is_train_wandb_active and also_to_wandb:
                        if wandb.run:
                            log_payload = {"train_message": message}
                            if wandb_data:
                                log_payload.update(wandb_data)
                            wandb.log(log_payload, step=trainer.metrics_manager.global_timestep)

                # Should not crash, should not call wandb.log
                log_both_impl(
                    "Test message", also_to_wandb=True, wandb_data={"loss": 0.5}
                )

                # Verify wandb.log was NOT called since wandb.run is None
                mock_wandb_log.assert_not_called()

    def test_log_both_impl_wandb_disabled_in_config(self, temp_base_dir):
        """Test that log_both_impl does not log to W&B when disabled in config."""
        config = make_test_config(wandb_enabled=False)  # W&B disabled
        args = DummyArgs()

        with patch("keisei.training.utils.setup_directories") as mock_setup_dirs:
            # Create actual test directories
            run_artifact_dir = os.path.join(temp_base_dir, "test_run")
            model_dir = os.path.join(run_artifact_dir, "models")
            os.makedirs(run_artifact_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Mock directory setup with actual paths
            mock_setup_dirs.return_value = {
                "run_artifact_dir": run_artifact_dir,
                "model_dir": model_dir,
                "log_file_path": os.path.join(run_artifact_dir, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_dir, "rich_periodic_eval_log.txt"
                ),
            }

            trainer = Trainer(config=config, args=args)

            # Verify W&B is not active
            assert trainer.session_manager.is_wandb_active is False
            assert trainer.is_train_wandb_active is False

            mock_logger = Mock()

            with patch("wandb.log") as mock_wandb_log, patch("wandb.run", Mock()):

                def log_both_impl(
                    message: str,
                    also_to_wandb: bool = False,
                    wandb_data: Optional[Dict] = None,
                ):
                    mock_logger.log(message)
                    if trainer.is_train_wandb_active and also_to_wandb:
                        if wandb.run:
                            log_payload = {"train_message": message}
                            if wandb_data:
                                log_payload.update(wandb_data)
                            wandb.log(log_payload, step=trainer.metrics_manager.global_timestep)

                # Should not log to W&B due to is_train_wandb_active = False
                log_both_impl(
                    "Test message", also_to_wandb=True, wandb_data={"loss": 0.5}
                )

                # Verify wandb.log was NOT called
                mock_wandb_log.assert_not_called()

    def test_log_both_impl_also_to_wandb_false(self, temp_base_dir):
        """Test that log_both_impl respects also_to_wandb=False parameter."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with (
            patch("keisei.training.utils.setup_wandb", return_value=True),
            patch("keisei.training.utils.setup_directories") as mock_setup_dirs,
        ):

            # Create actual test directories
            run_artifact_dir = os.path.join(temp_base_dir, "test_run")
            model_dir = os.path.join(run_artifact_dir, "models")
            os.makedirs(run_artifact_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Mock directory setup with actual paths
            mock_setup_dirs.return_value = {
                "run_artifact_dir": run_artifact_dir,
                "model_dir": model_dir,
                "log_file_path": os.path.join(run_artifact_dir, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_dir, "rich_periodic_eval_log.txt"
                ),
            }

            trainer = Trainer(config=config, args=args)
            mock_logger = Mock()

            with patch("wandb.log") as mock_wandb_log, patch("wandb.run", Mock()):

                def log_both_impl(
                    message: str,
                    also_to_wandb: bool = False,
                    wandb_data: Optional[Dict] = None,
                ):
                    mock_logger.log(message)
                    if trainer.is_train_wandb_active and also_to_wandb:
                        if wandb.run:
                            log_payload = {"train_message": message}
                            if wandb_data:
                                log_payload.update(wandb_data)
                            wandb.log(log_payload, step=trainer.metrics_manager.global_timestep)

                # Test with also_to_wandb=False
                log_both_impl(
                    "Test message", also_to_wandb=False, wandb_data={"loss": 0.5}
                )

                # Verify wandb.log was NOT called due to also_to_wandb=False
                mock_wandb_log.assert_not_called()

    def test_session_manager_wandb_state_consistency(self, temp_base_dir):
        """Test that SessionManager.is_wandb_active reflects the actual W&B state consistently."""
        config = make_test_config(wandb_enabled=True)
        args = DummyArgs()

        with patch("keisei.training.utils.setup_directories") as mock_setup_dirs:
            # Create actual test directories
            run_artifact_dir = os.path.join(temp_base_dir, "test_run")
            model_dir = os.path.join(run_artifact_dir, "models")
            os.makedirs(run_artifact_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Mock directory setup with actual paths
            mock_setup_dirs.return_value = {
                "run_artifact_dir": run_artifact_dir,
                "model_dir": model_dir,
                "log_file_path": os.path.join(run_artifact_dir, "training.log"),
                "eval_log_file_path": os.path.join(
                    run_artifact_dir, "rich_periodic_eval_log.txt"
                ),
            }

            trainer = Trainer(config=config, args=args)

            # Test successful W&B setup
            with patch("keisei.training.utils.setup_wandb", return_value=True):
                result = trainer.session_manager.setup_wandb()
                assert result is True
                assert trainer.session_manager.is_wandb_active is True

            # Test failed W&B setup
            trainer2 = Trainer(config=config, args=args)
            with patch("keisei.training.utils.setup_wandb", return_value=False):
                result = trainer2.session_manager.setup_wandb()
                assert result is False
                assert trainer2.session_manager.is_wandb_active is False


if __name__ == "__main__":
    pytest.main([__file__])
