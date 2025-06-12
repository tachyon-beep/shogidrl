"""
Shared fixtures for display tests.
"""

from unittest.mock import Mock

import pytest

from tests.display.test_utilities import (
    DisplayMockLibrary,
    ErrorScenarioFactory,
    PerformanceTestHelper,
    TestDataFactory,
)


@pytest.fixture
def unicode_test_data():
    """Fixture providing Unicode test data for various edge cases."""
    return {
        "japanese_pieces": "王飛角金銀桂香歩",
        "chinese_numbers": "一二三四五六七八九",
        "special_symbols": "★☆◎◇◆※",
        "mixed_content": "7g7f★王手",
        "complex_unicode": "🏆🎯⚡🔥💫",
        "mathematical": "∞±≤≥∈∉∩∪",
        "arrows": "→←↑↓↗↘↙↖",
    }


@pytest.fixture
def extreme_dimensions():
    """Fixture providing extreme terminal dimensions for testing."""
    return [
        (1, 1),  # Minimum dimensions
        (5, 3),  # Very small
        (10, 5),  # Small
        (40, 15),  # Below normal
        (80, 24),  # Standard terminal
        (120, 30),  # Large terminal
        (200, 50),  # Very large
        (500, 100),  # Extreme width
        (80, 200),  # Extreme height
        (1000, 1000),  # Maximum test case
    ]


@pytest.fixture
def large_metrics_dataset():
    """Fixture providing large metrics dataset for performance testing."""
    return TestDataFactory.create_metrics_history(length=1000, include_outliers=True)


@pytest.fixture
def training_session_in_progress():
    """Fixture providing a realistic mid-session state."""
    return TestDataFactory.create_training_session(duration_minutes=15, move_count=75)


@pytest.fixture
def display_with_error_state():
    """Fixture for testing error recovery scenarios."""
    return {
        "corrupted_game_state": ErrorScenarioFactory.create_corrupted_game_state(),
        "malformed_config": ErrorScenarioFactory.create_malformed_config(),
        "extreme_metrics": ErrorScenarioFactory.create_extreme_metrics(),
    }


@pytest.fixture
def performance_helper():
    """Fixture providing performance testing helper."""
    return PerformanceTestHelper()


@pytest.fixture
def mock_library():
    """Fixture providing mock library."""
    return DisplayMockLibrary()


@pytest.fixture
def mock_config():
    """Fixture providing a mock configuration object."""
    mock_config = Mock()
    mock_config.display = Mock()
    mock_config.display.refresh_rate = 10.0
    mock_config.display.enable_animations = True
    mock_config.display.show_debug_info = False
    mock_config.display.max_board_width = 80
    mock_config.display.max_board_height = 24
    mock_config.display.metrics_window_size = 100  # Return integer, not Mock
    mock_config.display.sparkline_width = 15
    mock_config.display.trend_history_length = 100
    mock_config.display.refresh_per_second = 4.0  # Add refresh rate

    # Add training configuration
    mock_config.training = Mock()
    mock_config.training.total_timesteps = 10000
    mock_config.training.enable_spinner = True
    mock_config.training.refresh_per_second = 4.0  # Add training refresh rate

    return mock_config


@pytest.fixture
def mock_trainer():
    """Fixture providing a mock trainer object with proper step_manager configuration."""
    mock_trainer = Mock()
    mock_trainer.metrics_manager = Mock()
    mock_trainer.metrics_manager.get_hot_squares = Mock(return_value=["1a", "2b", "3c"])
    mock_trainer.metrics_manager.global_timestep = 1000
    mock_trainer.metrics_manager.total_episodes_completed = 50
    mock_trainer.metrics_manager.black_wins = 30
    mock_trainer.metrics_manager.white_wins = 20
    mock_trainer.metrics_manager.draws = 5
    mock_trainer.metrics_manager.sente_opening_history = [
        "Standard Opening",
        "Rapid Attack",
    ]
    mock_trainer.metrics_manager.gote_opening_history = [
        "Defense Formation",
        "Counter Attack",
    ]

    # Configure step_manager with proper string values
    mock_trainer.step_manager = Mock()
    mock_trainer.step_manager.move_log = ["7g7f", "8c8d", "6g6f"]
    mock_trainer.step_manager.sente_best_capture = "Rook on 2h"
    mock_trainer.step_manager.gote_best_capture = "Bishop on 7g"
    mock_trainer.step_manager.sente_capture_count = 2
    mock_trainer.step_manager.gote_capture_count = 1
    mock_trainer.step_manager.sente_drop_count = 0
    mock_trainer.step_manager.gote_drop_count = 1
    mock_trainer.step_manager.sente_promo_count = 1
    mock_trainer.step_manager.gote_promo_count = 0

    return mock_trainer


@pytest.fixture
def mock_console():
    """Fixture providing a mock console object with proper encoding."""
    mock_console = Mock()
    mock_console.size.width = 120
    mock_console.size.height = 30
    mock_console.print = Mock()
    mock_console.clear = Mock()
    mock_console.update_screen = Mock()
    mock_console.get_time = Mock(return_value=0.0)
    mock_console.encoding = "utf-8"  # Ensure encoding returns string, not Mock
    mock_console.log = Mock()
    return mock_console
