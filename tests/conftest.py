"""
conftest.py: Shared fixtures for all tests in the DRL Shogi Client project.
"""

import multiprocessing as mp
import sys  # Add this import
from unittest.mock import MagicMock, Mock, patch

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
from keisei.constants import (
    CORE_OBSERVATION_CHANNELS,
    SHOGI_BOARD_SIZE,
    TEST_BUFFER_SIZE,
)
from keisei.utils import PolicyOutputMapper

# =============================================================================
# Default Constants for Tests - Used by PPO Agent Tests
# =============================================================================


class _EnvDefaults:
    """Default environment configuration values for tests."""

    seed = 42
    num_actions_total = 13527  # Default from schema


class _TrainDefaults:
    """Default training configuration values for tests."""

    learning_rate = 1e-3
    gamma = 0.99
    clip_epsilon = 0.2
    value_loss_coeff = 0.5
    entropy_coef = 0.01
    render_every_steps = 1
    refresh_per_second = 4
    se_ratio = 0.25
    gradient_clip_max_norm = 0.5
    lambda_gae = 0.95


# Create instances to be imported by test files
ENV_DEFAULTS = _EnvDefaults()
TRAIN_DEFAULTS = _TrainDefaults()

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
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=13527,  # Default from schema
        seed=42,
        max_moves_per_game=500,  # Default from schema
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
        gamma=0.99,  # Default from schema
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
        lambda_gae=0.95,  # Default from schema
        checkpoint_interval_timesteps=1000,
        evaluation_interval_timesteps=1000,
        weight_decay=0.0,
        normalize_advantages=True,
        enable_value_clipping=False,  # Required parameter
        lr_schedule_type=None,
        lr_schedule_kwargs=None,
        lr_schedule_step_on="epoch",
    )


@pytest.fixture
def test_evaluation_config():
    """Evaluation configuration optimized for testing."""
    return EvaluationConfig(
        num_games=1,  # Minimal for fast tests
        opponent_type="random",
        evaluation_interval_timesteps=1000,
        enable_periodic_evaluation=False,
        max_moves_per_game=200,
        log_file_path_eval="eval_log.txt",
        wandb_log_eval=False,
        elo_registry_path="test_elo_ratings.json",
        agent_id="test_agent",
        opponent_id="test_opponent",
        previous_model_pool_size=2,  # Small for tests
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
        log_model_artifact=False,
    )


@pytest.fixture
def test_display_config():
    """Display configuration for testing."""
    return DisplayConfig(
        enable_board_display=False,
        enable_trend_visualization=False,
        enable_elo_ratings=False,
        enable_enhanced_layout=False,
        display_moves=False,
        turn_tick=0.0,  # No delays in tests
        board_unicode_pieces=False,
        board_highlight_last_move=False,
        sparkline_width=15,
        trend_history_length=100,
        elo_initial_rating=1500.0,
        elo_k_factor=32.0,
        dashboard_height_ratio=2,
        progress_bar_height=4,
        show_text_moves=False,
        move_list_length=10,
        moves_latest_top=True,
        moves_flash_ms=0,  # No flashing in tests
        show_moves_trend=False,
        show_completion_rate=False,
        show_enhanced_win_rates=False,
        show_turns_trend=False,
        metrics_window_size=100,
        trend_smoothing_factor=0.1,
        metrics_panel_height=6,
        enable_trendlines=False,
        log_layer_keyword_filters=["test_layer"],  # Minimal for tests
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
    test_display_config,
    disabled_parallel_config,
):
    """Complete minimal AppConfig for unit tests."""
    return AppConfig(
        env=minimal_env_config,
        training=minimal_training_config,
        evaluation=test_evaluation_config,
        logging=test_logging_config,
        wandb=disabled_wandb_config,
        display=test_display_config,
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
    test_display_config,
    disabled_parallel_config,
):
    """Complete AppConfig optimized for very fast test execution."""
    return AppConfig(
        env=minimal_env_config,
        training=fast_training_config,
        evaluation=test_evaluation_config,
        logging=test_logging_config,
        wandb=disabled_wandb_config,
        display=test_display_config,
        parallel=disabled_parallel_config,
    )


@pytest.fixture
def integration_test_config(policy_mapper, tmp_path):
    """Configuration suitable for integration tests with proper action mapping."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=CORE_OBSERVATION_CHANNELS,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
            max_moves_per_game=500,  # Default from schema
        ),
        training=TrainingConfig(
            total_timesteps=200,  # Small for integration tests
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=4,
            learning_rate=1e-3,
            gamma=0.99,  # Default from schema
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
            lambda_gae=0.95,  # Default from schema
            checkpoint_interval_timesteps=200,
            evaluation_interval_timesteps=200,
            weight_decay=0.0,
            normalize_advantages=True,
            enable_value_clipping=False,  # Required parameter
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=2,
            opponent_type="random",
            evaluation_interval_timesteps=200,
            enable_periodic_evaluation=False,  # Added missing parameter
            max_moves_per_game=200,  # Added missing parameter
            log_file_path_eval=str(
                tmp_path / "integration_eval.log"
            ),  # Added missing parameter
            wandb_log_eval=False,  # Added missing parameter
            elo_registry_path=str(tmp_path / "integration_elo.json"),
            agent_id="integration_agent",
            opponent_id="integration_opponent",
            previous_model_pool_size=3,
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
            log_model_artifact=False,
        ),
        display=DisplayConfig(
            enable_board_display=False,
            enable_trend_visualization=False,
            enable_elo_ratings=False,
            enable_enhanced_layout=False,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=False,
            board_highlight_last_move=False,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=False,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=0,
            show_moves_trend=False,
            show_completion_rate=False,
            show_enhanced_win_rates=False,
            show_turns_trend=False,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=False,
            log_layer_keyword_filters=["integration_test_layer"],
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


# =============================================================================
# PPOAgent Testing Fixtures - Eliminates Setup Duplication
# =============================================================================


@pytest.fixture
def ppo_test_model():
    """Create a test ActorCritic model for PPOAgent testing."""
    from keisei.core.neural_network import ActorCritic

    mapper = PolicyOutputMapper()
    return ActorCritic(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=mapper.get_total_actions(),
    )


@pytest.fixture
def ppo_agent_basic(minimal_app_config, ppo_test_model):
    """Basic PPOAgent with minimal configuration for unit tests."""
    import torch

    from keisei.core.ppo_agent import PPOAgent

    return PPOAgent(
        model=ppo_test_model,
        config=minimal_app_config,
        device=torch.device("cpu"),
        name="TestPPOAgent",
    )


@pytest.fixture
def ppo_agent_fast(fast_app_config, ppo_test_model):
    """PPOAgent optimized for fast test execution."""
    import torch

    from keisei.core.ppo_agent import PPOAgent

    return PPOAgent(
        model=ppo_test_model,
        config=fast_app_config,
        device=torch.device("cpu"),
        name="FastTestPPOAgent",
    )


@pytest.fixture
def populated_experience_buffer():
    """Pre-populated experience buffer for PPO learning tests."""
    import torch

    from keisei.core.experience_buffer import ExperienceBuffer

    buffer = ExperienceBuffer(
        buffer_size=TEST_BUFFER_SIZE,
        gamma=0.99,  # Default from schema
        lambda_gae=0.95,  # Default from schema
        device="cpu",
    )

    # Create consistent dummy data
    dummy_obs_tensor = torch.randn(
        CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
    )
    dummy_legal_mask = torch.ones(
        13527, dtype=torch.bool, device="cpu"  # Default from schema
    )

    # Add varied experiences
    rewards = [1.0, -0.5, 2.0, 0.0]
    values = [0.8, 0.2, 1.5, 0.1]

    for i in range(4):
        buffer.add(
            obs=dummy_obs_tensor,
            action=i % 1000,  # Valid action index
            reward=rewards[i],
            log_prob=0.1 * (i + 1),
            value=values[i],
            done=(i == 3),
            legal_mask=dummy_legal_mask,
        )

    buffer.compute_advantages_and_returns(0.0)
    return buffer


@pytest.fixture
def dummy_observation():
    """Standard dummy observation tensor for PPO tests."""
    import numpy as np
    import torch

    rng = np.random.default_rng(42)
    obs_np = rng.random(
        (CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    ).astype(np.float32)
    return torch.from_numpy(obs_np).to(torch.device("cpu"))


@pytest.fixture
def dummy_legal_mask():
    """Standard dummy legal mask for PPO tests."""
    import torch

    mask = torch.ones(13527, dtype=torch.bool, device="cpu")  # Default from schema
    mask[0] = False  # Make first action illegal for testing
    return mask


def create_test_experience_data(buffer_size: int, device: str = "cpu"):
    """Helper function to generate consistent dummy experience data."""
    import torch

    dummy_obs = torch.randn(
        CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device=device
    )
    dummy_mask = torch.ones(
        13527, dtype=torch.bool, device=device  # Default from schema
    )

    # Generate varied but consistent data
    experiences = []
    for i in range(buffer_size):
        experiences.append(
            {
                "obs": dummy_obs,
                "action": i % 1000,
                "reward": float(i - buffer_size // 2),  # Mix of positive/negative
                "log_prob": 0.1 * (i + 1),
                "value": 0.5 * i,
                "done": (i == buffer_size - 1),
                "legal_mask": dummy_mask,
            }
        )

    return experiences


def assert_valid_ppo_metrics(metrics: dict):
    """Validate PPO training metrics structure and content."""
    import numpy as np

    required_metrics = [
        "ppo/policy_loss",
        "ppo/value_loss",
        "ppo/entropy",
        "ppo/kl_divergence_approx",
        "ppo/learning_rate",
    ]

    for metric in required_metrics:
        assert metric in metrics, f"Missing required metric: {metric}"
        assert isinstance(
            metrics[metric], (int, float)
        ), f"Metric {metric} should be numeric"
        assert not np.isnan(metrics[metric]), f"Metric {metric} is NaN"
        assert not np.isinf(metrics[metric]), f"Metric {metric} is infinite"
