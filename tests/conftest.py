"""
conftest.py: Shared fixtures for all tests in the DRL Shogi Client project.
"""

import multiprocessing as mp
import sys  # Add this import
from unittest.mock import MagicMock, Mock, patch

import pytest

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
from keisei.utils import PolicyOutputMapper

# Try to set the start method as early as possible for pytest runs
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
        print(
            "Successfully set multiprocessing start method to 'spawn' in conftest.py",
            file=sys.stderr,
        )
except RuntimeError as e:
    print(
        f"Could not set multiprocessing start method in conftest.py: {e}",
        file=sys.stderr,
    )
except AttributeError:
    # Fallback for older Python versions that might not have get_start_method with allow_none
    # or if get_start_method itself fails when no method has been set yet.
    try:
        # Attempt to set it directly if get_start_method is problematic or indicates no method set.
        # This path is more speculative and depends on Python version specifics.
        mp.set_start_method("spawn", force=True)  # Still use force=True
        print(
            "Successfully set multiprocessing start method to 'spawn' in conftest.py (fallback/direct set)",
            file=sys.stderr,
        )
    except RuntimeError as e_inner:
        print(
            f"Could not set multiprocessing start method in conftest.py (fallback/direct set): {e_inner}",
            file=sys.stderr,
        )


# Place all test scaffolding here, not in individual test files.


# W&B Fixtures for consistent mocking across tests
@pytest.fixture
def mock_wandb_disabled():
    """Fixture that completely disables W&B for tests by mocking all common functions."""
    mock_run = MagicMock()
    mock_run.return_value = None  # wandb.run returns None when disabled

    with (
        patch("wandb.init") as mock_init,
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
        patch("wandb.watch") as mock_watch,
        patch("wandb.run", mock_run),  # Use a mock object instead of None
        patch("wandb.Artifact") as mock_artifact,
        patch("wandb.log_artifact") as mock_log_artifact,
    ):
        # Configure return values for common usage patterns
        mock_init.return_value = None
        mock_log.return_value = None
        mock_finish.return_value = None
        mock_watch.return_value = None
        mock_artifact.return_value = Mock()
        mock_log_artifact.return_value = None

        yield {
            "init": mock_init,
            "log": mock_log,
            "finish": mock_finish,
            "watch": mock_watch,
            "run": mock_run,  # Add run to the dictionary
            "artifact": mock_artifact,
            "log_artifact": mock_log_artifact,
        }


@pytest.fixture
def mock_wandb_active():
    """Fixture that mocks W&B as active with proper mock objects."""
    mock_run = MagicMock()
    mock_artifact = Mock()

    with (
        patch("wandb.init") as mock_init,
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
        patch("wandb.watch") as mock_watch,
        patch("wandb.run", mock_run),  # Set wandb.run to active mock
        patch("wandb.Artifact", return_value=mock_artifact) as mock_artifact_class,
        patch("wandb.log_artifact") as mock_log_artifact,
    ):
        # Configure return values for active W&B session
        mock_init.return_value = mock_run
        mock_log.return_value = None
        mock_finish.return_value = None
        mock_watch.return_value = None
        mock_log_artifact.return_value = None

        yield {
            "init": mock_init,
            "log": mock_log,
            "finish": mock_finish,
            "watch": mock_watch,
            "run": mock_run,
            "artifact_class": mock_artifact_class,
            "artifact": mock_artifact,
            "log_artifact": mock_log_artifact,
        }


@pytest.fixture
def mock_setup_wandb():
    """Fixture for mocking the setup_wandb utility function."""
    with patch("keisei.training.utils.setup_wandb") as mock_setup:
        mock_setup.return_value = False  # Default to disabled
        yield mock_setup


# Add more fixtures as the codebase grows (e.g., mock agents, sample moves, etc.)

# =============================================================================
# Configuration Fixtures - Eliminates Config Duplication Anti-Pattern
# =============================================================================


@pytest.fixture
def policy_mapper():
    """Reusable PolicyOutputMapper fixture."""
    return PolicyOutputMapper()


@pytest.fixture
def minimal_env_config():
    """Minimal environment configuration for unit tests."""
    return EnvConfig(
        device="cpu",
        input_channels=46,
        num_actions_total=13527,
        seed=42,
        max_moves_per_game=200,  # Reasonable limit for test scenarios
    )


@pytest.fixture
def minimal_training_config():
    """Minimal training configuration for unit tests."""
    return TrainingConfig(
        total_timesteps=1000,
        steps_per_epoch=32,
        ppo_epochs=1,
        minibatch_size=2,
        learning_rate=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        value_loss_coeff=0.5,
        entropy_coef=0.01,
        render_every_steps=1,
        refresh_per_second=4,
        enable_spinner=False,  # Disable for tests
        input_features="core46",
        tower_depth=5,  # Smaller for faster tests
        tower_width=128,  # Smaller for faster tests
        se_ratio=0.25,
        model_type="resnet",
        mixed_precision=False,
        ddp=False,
        gradient_clip_max_norm=0.5,
        lambda_gae=0.95,
        checkpoint_interval_timesteps=1000,
        evaluation_interval_timesteps=1000,
        weight_decay=0.0,
    )


@pytest.fixture
def test_evaluation_config():
    """Evaluation configuration optimized for testing."""
    return EvaluationConfig(
        num_games=1,  # Minimal for fast tests
        opponent_type="random",
        evaluation_interval_timesteps=1000,
    )


@pytest.fixture
def test_logging_config(tmp_path):
    """Logging configuration using temporary paths."""
    return LoggingConfig(
        log_file=str(tmp_path / "test.log"),
        model_dir=str(tmp_path / "models"),
        run_name="test_run",
    )


@pytest.fixture
def disabled_wandb_config():
    """WandB configuration disabled for testing."""
    return WandBConfig(
        enabled=False,
        project="test-project",
        entity=None,
        run_name_prefix="test",
        watch_model=False,
        watch_log_freq=1000,
        watch_log_type="all",
    )


@pytest.fixture
def test_demo_config():
    """Demo configuration for testing."""
    return DemoConfig(
        enable_demo_mode=False,
        demo_mode_delay=0.0,  # No delays in tests
    )


@pytest.fixture
def disabled_parallel_config():
    """Parallel configuration disabled for unit tests."""
    return ParallelConfig(
        enabled=False,
        num_workers=1,  # Single worker for tests
        batch_size=2,  # Small batch for tests
        sync_interval=100,
        compression_enabled=False,  # Disabled for simplicity
        timeout_seconds=5.0,  # Shorter timeout
        max_queue_size=100,  # Smaller queue
        worker_seed_offset=1000,
    )


@pytest.fixture
def minimal_app_config(
    minimal_env_config,
    minimal_training_config,
    test_evaluation_config,
    test_logging_config,
    disabled_wandb_config,
    test_demo_config,
    disabled_parallel_config,
):
    """Complete minimal AppConfig for unit tests."""
    return AppConfig(
        env=minimal_env_config,
        training=minimal_training_config,
        evaluation=test_evaluation_config,
        logging=test_logging_config,
        wandb=disabled_wandb_config,
        demo=test_demo_config,
        parallel=disabled_parallel_config,
    )


@pytest.fixture
def fast_training_config(minimal_training_config):
    """Training config optimized for very fast test execution."""
    config = minimal_training_config.model_copy()
    config.total_timesteps = 100
    config.steps_per_epoch = 8
    config.ppo_epochs = 1
    config.minibatch_size = 2
    config.tower_depth = 2  # Very small network
    config.tower_width = 64
    config.checkpoint_interval_timesteps = 100
    config.evaluation_interval_timesteps = 100
    return config


@pytest.fixture
def fast_app_config(
    minimal_env_config,
    fast_training_config,
    test_evaluation_config,
    test_logging_config,
    disabled_wandb_config,
    test_demo_config,
    disabled_parallel_config,
):
    """Complete AppConfig optimized for very fast test execution."""
    return AppConfig(
        env=minimal_env_config,
        training=fast_training_config,
        evaluation=test_evaluation_config,
        logging=test_logging_config,
        wandb=disabled_wandb_config,
        demo=test_demo_config,
        parallel=disabled_parallel_config,
    )


@pytest.fixture
def integration_test_config(policy_mapper, tmp_path):
    """Configuration suitable for integration tests with proper action mapping."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=200,  # Reasonable limit for test scenarios
        ),
        training=TrainingConfig(
            total_timesteps=200,  # Small for integration tests
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=4,
            learning_rate=1e-3,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=3,  # Small network for speed
            tower_width=64,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=200,
            evaluation_interval_timesteps=200,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(
            num_games=2,
            opponent_type="random",
            evaluation_interval_timesteps=200,
        ),
        logging=LoggingConfig(
            log_file=str(tmp_path / "integration_test.log"),
            model_dir=str(tmp_path / "integration_models"),
            run_name="integration_test",
        ),
        wandb=WandBConfig(
            enabled=False,
            project="integration-test",
            entity=None,
            run_name_prefix="integration",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(
            enable_demo_mode=False,
            demo_mode_delay=0.0,
        ),
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=4,
            sync_interval=50,
            compression_enabled=False,
            timeout_seconds=5.0,
            max_queue_size=50,
            worker_seed_offset=1000,
        ),
    )
