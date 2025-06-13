"""
Shared fixtures and utilities for evaluation tests.

This module contains common test fixtures, constants, and mock classes
used across all evaluation test modules.
"""

# CONSOLIDATION COMPLETE: All shared fixtures, factories, and config templates are now centralized here (or in factories.py).
# Remove any duplicate fixture logic from individual test files. This is the canonical location for shared test infrastructure.
# Last consolidation: 2025-06-13
#
# Next: Update EVAL_REMEDIATION_PLAN.md to reflect this consolidation.

import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Generator, Any
from unittest.mock import MagicMock, Mock

import pytest
import torch

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
            enable_board_display=True,
            enable_trend_visualization=True,
            enable_elo_ratings=True,
            enable_enhanced_layout=True,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=True,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=True,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=True,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=500,
            show_moves_trend=True,
            show_completion_rate=True,
            show_enhanced_win_rates=True,
            show_turns_trend=True,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=True,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"],
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


# Phase 1 Foundation Fixes: Test Isolation and Async Testing Standards

@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """Provides isolated temporary directory for each test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_isolation():
    """Ensures each test starts with clean state."""
    # Clear any existing PyTorch caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset any global state
    torch.manual_seed(42)
    
    yield
    
    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def thread_isolation():
    """Ensures thread safety and isolation for concurrent tests."""
    # Store original thread count
    original_thread_count = threading.active_count()
    
    yield
    
    # Verify no threads were leaked
    final_thread_count = threading.active_count()
    assert final_thread_count <= original_thread_count + 1, \
        f"Thread leak detected: {final_thread_count} vs {original_thread_count}"


@pytest.fixture
def async_test_timeout():
    """Standard timeout for async tests to prevent hanging."""
    return 10.0  # 10 seconds timeout


@pytest.fixture 
def event_loop():
    """Standard event loop fixture for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_agent_factory():
    """Factory for creating consistent mock agents across tests."""
    def create_mock_agent(name: str = "MockAgent", device: str = "cpu"):
        agent = MockPPOAgent(name=name)
        agent.device = device
        return agent
    return create_mock_agent


@pytest.fixture
def performance_monitor():
    """Monitor test performance to ensure they meet Phase 1 requirements."""
    import time
    start_time = time.perf_counter()
    
    yield
    
    execution_time = time.perf_counter() - start_time
    # Phase 1 requirement: individual tests should complete within 5 seconds
    assert execution_time < 5.0, \
        f"Test took {execution_time:.3f}s, should be under 5s per Phase 1 requirements"


@pytest.fixture
def memory_monitor():
    """Monitor memory usage to detect leaks during testing."""
    import psutil
    import gc
    
    # Force garbage collection before monitoring
    gc.collect()
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    # Force garbage collection after test
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Phase 1 requirement: individual tests should not leak significant memory
    assert memory_increase < 100, \
        f"Memory increased by {memory_increase:.1f} MB, possible memory leak"


class AsyncTestHelper:
    """Helper class for standardized async testing patterns."""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pytest.fail(f"Async operation timed out after {timeout}s")
    
    @staticmethod
    async def simulate_concurrent_execution(tasks, max_concurrent: int = 4):
        """Simulate concurrent execution with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[limited_task(task) for task in tasks])


@pytest.fixture
def async_helper():
    """Provides async testing helper utilities."""
    return AsyncTestHelper()


# Error injection utilities for fault tolerance testing

class ErrorInjector:
    """Utility for injecting controlled errors in tests."""
    
    def __init__(self, failure_rate: float = 0.3, seed: int = 42):
        import random
        self.failure_rate = failure_rate
        self.random = random.Random(seed)
    
    def should_fail(self) -> bool:
        """Determine if operation should fail based on failure rate."""
        return self.random.random() < self.failure_rate
    
    def get_random_error(self):
        """Get a random error type for testing."""
        error_types = [TimeoutError, RuntimeError, ValueError]
        return self.random.choice(error_types)


@pytest.fixture
def error_injector():
    """Provides error injection utilities for fault tolerance testing."""
    return ErrorInjector()


# Cleanup utilities to ensure test isolation

@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically clean up test environment after each test."""
    yield
    
    # Clean up any remaining temporary files
    import tempfile
    import os
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith("pytest_"):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except PermissionError:
                pass  # Ignore cleanup errors
