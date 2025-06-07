"""
Shared fixtures and utilities for evaluation tests.

This module contains common test fixtures, constants, and mock classes
used across all evaluation test modules.
"""

from unittest.mock import MagicMock, Mock

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
from keisei.utils import PolicyOutputMapper

# Constants used across evaluation tests
INPUT_CHANNELS = 46


# Mock PPO Agent class for testing
class MockPPOAgent:
    """Mock PPO Agent for testing purposes."""

    def __init__(self, *args, **kwargs):
        self.device = "cpu"
        self.name = kwargs.get("name", "MockAgent")

    def select_action(self, observation, legal_mask=None):
        """Mock action selection - returns first legal action."""
        if legal_mask is not None:
            legal_indices = legal_mask.nonzero(as_tuple=True)[0]
            if len(legal_indices) > 0:
                return legal_indices[0].item()
        return 0  # Fallback action

    def get_action_and_value(self, observation, legal_mask=None):
        """Mock get_action_and_value method."""
        import torch

        action = self.select_action(observation, legal_mask)
        return action, torch.tensor(0.0), torch.tensor(0.0)

    def load_model(self, checkpoint_path):
        """Mock model loading."""
        return {}


def make_test_config():
    """Create a minimal test configuration for evaluation tests."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=INPUT_CHANNELS,
            num_actions_total=13527,
            seed=42,
            max_moves_per_game=200,
        ),
        training=TrainingConfig(
            total_timesteps=100,
            steps_per_epoch=8,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1e-3,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=False,
            input_features="core46",
            tower_depth=2,
            tower_width=64,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=100,
            weight_decay=0.0,
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            num_games=1,
            opponent_type="random",
            evaluation_interval_timesteps=100,
            enable_periodic_evaluation=False,
            max_moves_per_game=200,
            log_file_path_eval="eval_log.txt",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="test.log",
            model_dir="test_models",
            run_name="test_run",
        ),
        wandb=WandBConfig(
            enabled=False,
            project="test-project",
            entity=None,
            run_name_prefix="test",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False,
        ),
        display=DisplayConfig(
            display_moves=False,
            turn_tick=0.0,
        ),
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=2,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=5.0,
            max_queue_size=100,
            worker_seed_offset=1000,
        ),
    )


@pytest.fixture
def policy_mapper():
    """Fixture providing PolicyOutputMapper instance."""
    return PolicyOutputMapper()


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return make_test_config()


@pytest.fixture
def shogi_game_initial():
    """Fixture providing a fresh ShogiGame instance for testing."""
    from keisei.shogi.shogi_game import ShogiGame

    return ShogiGame()


@pytest.fixture
def eval_logger_setup(tmp_path):
    """Fixture providing evaluation logger setup for testing."""
    from keisei.utils.utils import EvaluationLogger

    log_file = tmp_path / "test_eval.log"
    logger = EvaluationLogger(str(log_file), also_stdout=False)

    # Return a context manager that properly opens the logger
    class LoggerContext:
        def __init__(self, logger, log_file_path):
            self.logger = logger
            self.log_file_path = log_file_path

        def __enter__(self):
            return self.logger.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.logger.__exit__(exc_type, exc_val, exc_tb)

    return LoggerContext(logger, str(log_file)), str(log_file)
