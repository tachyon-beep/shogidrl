"""
Test utilities for display tests including data factories and mock helpers.

This module provides reusable components to improve test quality, consistency,
and maintainability across the display test suite.
"""

import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame

try:
    import psutil  # pylint: disable=unused-import

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# Create a simple TrainingMetrics class for testing
class TrainingMetrics:
    """Simple training metrics for testing."""

    def __init__(self):
        self.accuracy = 0.0
        self.speed = 0.0
        self.elo_rating = 1200


class MetricsHistory:
    """Simple metrics history for testing - matches the real MetricsHistory interface."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.win_rates_history: List[Dict[str, float]] = []
        self.learning_rates: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.kl_divergences: List[float] = []
        self.entropies: List[float] = []
        self.clip_fractions: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_rewards: List[float] = []

    def _trim(self, values: List[Any]) -> None:
        while len(values) > self.max_history:
            values.pop(0)

    def add_episode_data(self, win_rates: Dict[str, float]) -> None:
        self.win_rates_history.append(win_rates)
        self._trim(self.win_rates_history)

    def add_ppo_data(self, metrics: Dict[str, float]) -> None:
        if "ppo/learning_rate" in metrics:
            self.learning_rates.append(metrics["ppo/learning_rate"])
            self._trim(self.learning_rates)
        if "ppo/policy_loss" in metrics:
            self.policy_losses.append(metrics["ppo/policy_loss"])
            self._trim(self.policy_losses)
        if "ppo/value_loss" in metrics:
            self.value_losses.append(metrics["ppo/value_loss"])
            self._trim(self.value_losses)
        if "ppo/kl_divergence_approx" in metrics:
            self.kl_divergences.append(metrics["ppo/kl_divergence_approx"])
            self._trim(self.kl_divergences)
        if "ppo/entropy" in metrics:
            self.entropies.append(metrics["ppo/entropy"])
            self._trim(self.entropies)
        if "ppo/clip_fraction" in metrics:
            self.clip_fractions.append(metrics["ppo/clip_fraction"])
            self._trim(self.clip_fractions)

    def add_metrics(self, metrics: TrainingMetrics):
        """Legacy compatibility method."""
        # This method is intentionally empty for compatibility


class TestDataFactory:
    """Factory for creating realistic test data across display tests."""

    @staticmethod
    def create_game_state(
        move_count: int = 10,
        difficulty: str = "normal",  # pylint: disable=unused-argument
        include_captures: bool = True,  # pylint: disable=unused-argument
        board_size: Tuple[int, int] = (9, 9),  # pylint: disable=unused-argument
    ) -> ShogiGame:
        """Create realistic game states for testing."""
        game = ShogiGame()
        game.reset()

        # Make some random moves
        for _ in range(min(move_count, 20)):  # Limit to reasonable number
            legal_moves = game.get_legal_moves()
            if legal_moves and not game.game_over:
                move = random.choice(legal_moves)
                try:
                    game.make_move(move)
                except (ValueError, AttributeError):
                    break  # Stop if move fails
            else:
                break

        return game

    @staticmethod
    def create_training_session(
        duration_minutes: int = 30,
        elo_progression: Optional[List[int]] = None,
        move_count: int = 150,
    ) -> Dict[str, Any]:
        """Create complete training sessions with realistic data."""
        if elo_progression is None:
            # Generate realistic ELO progression
            base_elo = 1200
            elo_progression = [base_elo]
            for _ in range(move_count):
                change = random.randint(-20, 25)  # Slight upward bias
                new_elo = max(800, min(2400, elo_progression[-1] + change))
                elo_progression.append(new_elo)

        # Generate realistic timing data
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        move_times = []
        current_time = start_time

        for _ in range(move_count):
            # Realistic thinking times (1-30 seconds, with occasional longer thinks)
            think_time = (
                random.uniform(1, 30)
                if random.random() > 0.1
                else random.uniform(30, 120)
            )
            current_time += timedelta(seconds=think_time)
            move_times.append(think_time)

        return {
            "game_state": TestDataFactory.create_game_state(
                move_count=min(move_count, 30)
            ),
            "elo_progression": elo_progression,
            "move_times": move_times,
            "start_time": start_time,
            "end_time": current_time,
            "total_moves": move_count,
            "average_move_time": sum(move_times) / len(move_times) if move_times else 0,
        }

    @staticmethod
    def create_metrics_history(
        length: int = 100,
        include_outliers: bool = True,
        metric_types: Optional[List[str]] = None,
    ) -> MetricsHistory:
        """Create realistic metrics history for testing."""
        if metric_types is None:
            metric_types = ["accuracy", "speed", "elo_rating", "move_quality"]

        history = MetricsHistory()

        for i in range(length):
            metrics = TrainingMetrics()

            # Generate realistic metric values
            if "accuracy" in metric_types:
                base_accuracy = 0.7 + (i / length) * 0.2  # Improving accuracy
                noise = random.uniform(-0.1, 0.1)
                metrics.accuracy = max(0.0, min(1.0, base_accuracy + noise))

            if "speed" in metric_types:
                # Speed in moves per minute
                base_speed = 15 + random.uniform(-3, 5)
                if include_outliers and random.random() < 0.05:
                    base_speed *= random.uniform(0.3, 2.5)  # Occasional outliers
                metrics.speed = max(1.0, base_speed)

            if "elo_rating" in metric_types:
                base_elo = 1200 + (i * 2) + random.randint(-15, 20)
                metrics.elo_rating = max(800, min(2400, base_elo))

            history.add_metrics(metrics)

        return history

    @staticmethod
    def create_unicode_test_data() -> Dict[str, str]:
        """Create test data with various Unicode characters for testing rendering."""
        return {
            "japanese_pieces": "王飛角金銀桂香歩",
            "chinese_numbers": "一二三四五六七八九",
            "special_symbols": "▲△☗☖⚡✓✗",
            "mixed_content": "Player 1 vs 対戦相手 (Rating: 1200)",
            "long_unicode": "特殊文字テスト" * 10,
            "empty_string": "",
            "whitespace_only": "   \t\n   ",
            "emoji_content": "🎯🏆⭐🎮👑",
        }

    @staticmethod
    def create_extreme_dimensions() -> List[Tuple[int, int]]:
        """Create test cases for extreme terminal dimensions."""
        return [
            (1, 1),  # Minimum
            (10, 5),  # Very small
            (80, 24),  # Standard
            (120, 40),  # Large
            (200, 100),  # Very large
            (1000, 500),  # Extreme
            (80, 1),  # Very narrow
            (10, 100),  # Very tall
        ]

    @staticmethod
    def generate_episode_metrics(count: int) -> List[Dict[str, Any]]:
        """Generate episode metrics for performance testing."""
        episodes = []
        for i in range(count):
            episode = {
                "episode_id": i,
                "reward": random.uniform(-100, 100),
                "length": random.randint(10, 200),
                "win_rate": random.uniform(0.0, 1.0),
                "elo_change": random.randint(-20, 25),
                "move_quality": random.uniform(0.3, 0.95),
                "time_taken": random.uniform(30, 300),
            }
            episodes.append(episode)
        return episodes


class DisplayMockLibrary:
    """Library of reusable mock configurations for display testing."""

    @staticmethod
    def mock_console_with_dimensions(width: int, height: int) -> Mock:
        """Create console mock with specific dimensions and behavior validation."""
        console_mock = Mock(spec=Console)
        console_mock.size.width = width
        console_mock.size.height = height
        console_mock.print = Mock()
        console_mock.clear = Mock()
        console_mock.update_screen = Mock()
        console_mock.get_time = Mock(return_value=0.0)  # Add missing get_time method
        console_mock.encoding = "utf-8"  # Ensure encoding returns string, not Mock

        # Track what gets printed for validation
        console_mock.printed_content = []

        def track_print(*args, **kwargs):
            console_mock.printed_content.append((args, kwargs))

        console_mock.print.side_effect = track_print
        return console_mock

    @staticmethod
    def mock_rich_table_with_validation() -> Mock:
        """Create Rich table mock that validates content and structure."""
        table_mock = Mock(spec=Table)
        table_mock.columns = []
        table_mock.rows = []
        table_mock.add_column = Mock()
        table_mock.add_row = Mock()

        # Track table construction
        def track_add_column(header, **kwargs):
            table_mock.columns.append({"header": header, "kwargs": kwargs})

        def track_add_row(*row_data, **kwargs):
            table_mock.rows.append({"data": row_data, "kwargs": kwargs})

        table_mock.add_column.side_effect = track_add_column
        table_mock.add_row.side_effect = track_add_row

        return table_mock

    @staticmethod
    def mock_rich_panel_with_content_validation() -> Mock:
        """Create Rich panel mock that validates content rendering."""
        panel_mock = Mock(spec=Panel)
        panel_mock.renderable = None
        panel_mock.title = None
        panel_mock.content_history = []

        def track_content(content, **kwargs):
            panel_mock.content_history.append({"content": content, "kwargs": kwargs})
            panel_mock.renderable = content
            return panel_mock

        # Configure the mock to track content when called as a constructor
        panel_mock.side_effect = track_content
        return panel_mock

    @staticmethod
    def create_mock_game_state_with_validation() -> Mock:
        """Create game state mock that validates state consistency."""
        game_state_mock = Mock()
        game_state_mock.get_legal_moves = Mock(return_value=[])
        game_state_mock.current_player = Color.BLACK
        game_state_mock.game_over = False
        game_state_mock.move_count = 0

        return game_state_mock


class PerformanceTestHelper:
    """Helper for performance testing of display components."""

    @staticmethod
    def measure_display_update_time(update_func, *args, **kwargs) -> float:
        """Measure the time taken for a display update operation."""
        start_time = time.perf_counter()
        update_func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time

    @staticmethod
    def stress_test_display_updates(
        component, update_method: str, data_generator, iterations: int = 100
    ) -> Dict[str, float]:
        """Perform stress testing on display components."""
        times = []

        for i in range(iterations):
            test_data = data_generator(i)
            update_func = getattr(component, update_method)

            start_time = time.perf_counter()
            update_func(test_data)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        return {
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
        }

    @staticmethod
    def memory_usage_test(test_func, *args, **kwargs) -> Dict[str, Any]:
        """Basic memory usage tracking for tests."""
        if not HAS_PSUTIL:
            return {
                "result": test_func(*args, **kwargs),
                "memory_before": 0,
                "memory_after": 0,
                "memory_diff_mb": 0.0,
            }

        # psutil is available if we reach here
        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        result = test_func(*args, **kwargs)

        memory_after = process.memory_info().rss
        memory_diff = memory_after - memory_before

        return {
            "result": result,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_diff": memory_diff,
            "memory_diff_mb": memory_diff / 1024 / 1024,
        }


class ErrorScenarioFactory:
    """Factory for creating various error scenarios for testing."""

    @staticmethod
    def create_corrupted_game_state() -> ShogiGame:
        """Create a game state with intentional corruption for error testing."""
        game = ShogiGame()
        game.reset()
        # Force some corrupted state
        game.game_over = True
        # Note: Cannot set current_player to None due to type constraints
        return game

    @staticmethod
    def create_malformed_config() -> Dict[str, Any]:
        """Create malformed configuration data for testing."""
        return {
            "display": {
                "theme": "invalid_theme",
                "width": -10,  # Invalid width
                "height": "not_a_number",  # Wrong type
                "update_interval": None,  # Missing required value
                "colors": {
                    "background": "#invalid_color",
                    "text": "rgb(300, 300, 300)",  # Out of range
                },
            }
        }

    @staticmethod
    def create_extreme_metrics() -> TrainingMetrics:
        """Create metrics with extreme values for boundary testing."""
        metrics = TrainingMetrics()
        metrics.accuracy = 2.5  # Invalid (> 1.0)
        metrics.speed = -10  # Invalid (negative)
        metrics.elo_rating = 999999  # Extreme value
        return metrics


# Test fixtures for enhanced scenarios
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
def performance_helper():
    """Fixture providing performance testing helper."""
    return PerformanceTestHelper()


@pytest.fixture
def mock_library():
    """Fixture providing mock library."""
    return DisplayMockLibrary()


@pytest.fixture
def large_metrics_dataset():
    """Fixture providing large metrics dataset for performance testing."""
    return TestDataFactory.create_metrics_history(length=1000, include_outliers=True)
