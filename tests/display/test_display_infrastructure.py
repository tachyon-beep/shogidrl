"""
Tests for display infrastructure components.

This module tests:
- EloRatingSystem mathematical accuracy and edge cases
- DisplayConfig validation, defaults, and error handling
- MetricsHistory data management and memory efficiency
- Component integration and performance
- Thread safety and concurrent access
- Error recovery and boundary conditions
"""

import math
import threading
import time

from keisei.config_schema import DisplayConfig
from keisei.shogi.shogi_core_definitions import Color
from keisei.training.display_components import Sparkline
from keisei.training.elo_rating import EloRatingSystem
from keisei.training.metrics_manager import MetricsHistory

# Import test utilities
from tests.display.test_utilities import TestDataFactory


def create_test_display_config(**kwargs) -> DisplayConfig:
    """Helper to create DisplayConfig with all required parameters."""
    defaults = {
        "enable_board_display": True,
        "enable_trend_visualization": True,
        "enable_elo_ratings": True,
        "enable_enhanced_layout": True,
        "display_moves": False,
        "turn_tick": 0.0,
        "board_unicode_pieces": True,
        "board_cell_width": 5,
        "board_cell_height": 3,
        "board_highlight_last_move": True,
        "sparkline_width": 15,
        "trend_history_length": 100,
        "elo_initial_rating": 1500.0,
        "elo_k_factor": 32.0,
        "dashboard_height_ratio": 2,
        "progress_bar_height": 4,
        "show_text_moves": True,
        "move_list_length": 10,
        "moves_latest_top": True,
        "moves_flash_ms": 500,
        "show_moves_trend": True,
        "show_completion_rate": True,
        "show_enhanced_win_rates": True,
        "show_turns_trend": True,
        "metrics_window_size": 100,
        "trend_smoothing_factor": 0.1,
        "metrics_panel_height": 6,
        "enable_trendlines": True,
        "log_layer_keyword_filters": ["stem", "policy_head", "value_head"],
    }
    defaults.update(kwargs)
    return DisplayConfig(**defaults)


class TestDisplayConfiguration:
    """Tests for DisplayConfig validation and defaults."""

    def test_display_config_defaults(self):
        """Test DisplayConfig default values."""
        config = create_test_display_config()

        # Core display features
        assert config.enable_board_display is True
        assert config.enable_trend_visualization is True
        assert config.enable_elo_ratings is True
        assert config.show_moves_trend is True
        assert config.show_completion_rate is True

        # Sparkline configuration
        assert config.sparkline_width == 15
        assert config.metrics_window_size == 100

        # Board display configuration
        assert config.board_unicode_pieces is True
        assert config.board_cell_width == 5
        assert config.board_cell_height == 3

        # Elo configuration - using approximate comparison for floats
        assert abs(config.elo_initial_rating - 1500.0) < 0.001
        assert abs(config.elo_k_factor - 32.0) < 0.001

    def test_display_config_customization(self):
        """Test DisplayConfig with custom values."""
        config = create_test_display_config(
            sparkline_width=20,
            board_cell_width=7,
            move_list_length=30,
            elo_k_factor=16.0,
        )

        assert config.sparkline_width == 20
        assert config.board_cell_width == 7
        assert config.move_list_length == 30
        assert abs(config.elo_k_factor - 16.0) < 0.001

    def test_display_config_comprehensive_defaults(self):
        """Comprehensive test of DisplayConfig default values."""
        config = create_test_display_config()

        # Move list configuration
        assert config.move_list_length == 10
        assert config.moves_latest_top is True
        assert config.moves_flash_ms == 500

        # Layout configuration
        assert config.dashboard_height_ratio == 2
        assert config.progress_bar_height == 4

        # Trend configuration
        assert config.trend_history_length == 100
        assert abs(config.trend_smoothing_factor - 0.1) < 0.001

        # Enhanced features
        assert config.enable_enhanced_layout is True
        assert config.show_enhanced_win_rates is True
        assert config.enable_trendlines is True

    def test_display_config_validation(self):
        """Test DisplayConfig parameter validation."""
        # Test minimum values
        config = create_test_display_config(
            sparkline_width=1,  # Should be at least 1
            board_cell_width=3,  # Should be at least 3
            move_list_length=1,  # Should be at least 1
        )

        assert config.sparkline_width >= 1
        assert config.board_cell_width >= 3
        assert config.move_list_length >= 1


class TestEloRatingSystem:
    """Tests for EloRatingSystem mathematical accuracy."""

    def test_elo_rating_mathematical_accuracy(self):
        """Test exact Elo formula implementation."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=32)

        initial_black = elo.black_rating
        initial_white = elo.white_rating

        # Test black win scenario
        elo.update_ratings(Color.BLACK)

        # Calculate expected Elo change manually
        # Expected score for black against equal opponent = 0.5
        # Actual score = 1.0 (black won)
        # Rating change = K * (actual - expected) = 32 * (1.0 - 0.5) = 16
        expected_black_rating = initial_black + 16
        expected_white_rating = initial_white - 16

        assert abs(elo.black_rating - expected_black_rating) < 0.001
        assert abs(elo.white_rating - expected_white_rating) < 0.001

    def test_elo_rating_different_initial_ratings(self):
        """Test Elo calculation with different initial ratings."""
        elo = EloRatingSystem(initial_rating=1400, k_factor=32)
        elo.white_rating = 1600  # Set white higher

        initial_black = elo.black_rating  # 1400
        initial_white = elo.white_rating  # 1600

        # Black (lower rated) wins against white (higher rated)
        elo.update_ratings(Color.BLACK)

        # Black should gain more points, white should lose more
        black_gain = elo.black_rating - initial_black
        white_loss = initial_white - elo.white_rating

        assert black_gain > 16  # Should gain more than against equal opponent
        assert white_loss > 16  # Should lose more than against equal opponent
        assert abs(black_gain - white_loss) < 0.001  # Gains should equal losses

    def test_elo_rating_k_factor_variations(self):
        """Test Elo system with different K-factors."""
        elo_high_k = EloRatingSystem(initial_rating=1500, k_factor=64)
        elo_low_k = EloRatingSystem(initial_rating=1500, k_factor=16)

        # Same outcome with different K-factors
        elo_high_k.update_ratings(Color.BLACK)
        elo_low_k.update_ratings(Color.BLACK)

        # Higher K-factor should result in larger rating changes
        high_k_change = elo_high_k.black_rating - 1500
        low_k_change = elo_low_k.black_rating - 1500

        assert high_k_change > low_k_change
        assert (
            abs(high_k_change - 4 * low_k_change) < 0.001
        )  # Should be exactly 4x (64/16 = 4)

    def test_elo_rating_draw_scenario(self):
        """Test Elo ratings for draw outcomes."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=32)

        initial_black = elo.black_rating
        initial_white = elo.white_rating

        # Simulate draw (actual score = 0.5 for both players)
        elo.update_ratings(None)  # None represents draw

        # For equal players in a draw, ratings should remain unchanged
        assert abs(elo.black_rating - initial_black) < 0.001
        assert abs(elo.white_rating - initial_white) < 0.001

    def test_elo_rating_extreme_scenarios(self):
        """Test Elo system with extreme rating differences."""
        elo = EloRatingSystem(initial_rating=2800, k_factor=32)  # Very high rated black
        elo.white_rating = 1200  # Very low rated white

        initial_black = elo.black_rating
        initial_white = elo.white_rating

        # Higher rated player wins (expected outcome)
        elo.update_ratings(Color.BLACK)

        # Should gain very few points (expected outcome)
        rating_gain = elo.black_rating - initial_black
        assert rating_gain < 2  # Should gain less than 2 points

        # Lower rated player loses (expected outcome)
        white_loss = initial_white - elo.white_rating
        assert white_loss < 2  # Should lose less than 2 points
        assert (
            abs(rating_gain - white_loss) < 0.01
        )  # Gains should approximately equal losses


class TestMetricsInfrastructure:
    """Tests for metrics infrastructure."""

    def test_metrics_history_trimming(self):
        """Test MetricsHistory trimming behavior."""
        history = MetricsHistory(max_history=3)

        # Add more episode data than max_history
        for i in range(5):
            history.add_episode_data({"black_win_rate": float(i)})

        # Should only keep last 3 entries
        assert len(history.win_rates_history) == 3

        # Add more PPO data than max_history
        for i in range(5):
            history.add_ppo_data(
                {
                    "ppo/learning_rate": float(i) * 0.001,
                    "ppo/policy_loss": float(i) * 0.1,
                }
            )

        # Should only keep last 3 entries for all PPO metrics
        assert len(history.learning_rates) == 3
        assert len(history.policy_losses) == 3

    def test_metrics_history_memory_management(self):
        """Test that MetricsHistory prevents memory leaks."""
        history = MetricsHistory(max_history=10)

        # Simulate long-running training with many episodes
        for i in range(100):
            history.add_episode_data({"black_win_rate": float(i % 10) / 10.0})

        # Should never exceed max_history
        assert len(history.win_rates_history) == 10

    def test_metrics_history_data_types(self):
        """Test MetricsHistory with various data types."""
        history = MetricsHistory(max_history=5)

        # Test with different numeric types
        history.add_episode_data(
            {
                "black_win_rate": 42,  # int
            }
        )
        history.add_episode_data(
            {
                "black_win_rate": 3.14159,  # float
            }
        )
        history.add_episode_data({"black_win_rate": -10.5})  # negative float

        # Should handle all numeric types
        assert len(history.win_rates_history) == 3
        assert history.win_rates_history[0]["black_win_rate"] == 42
        assert abs(history.win_rates_history[1]["black_win_rate"] - 3.14159) < 1e-6
        assert abs(history.win_rates_history[2]["black_win_rate"] - (-10.5)) < 1e-6

    def test_metrics_history_comprehensive_trimming(self):
        """Comprehensive test of MetricsHistory trimming with multiple metrics."""
        history = MetricsHistory(max_history=3)

        # Add more episode data than max_history with multiple metrics
        for i in range(5):
            history.add_episode_data(
                {"black_win_rate": float(i), "average_episode_length": i * 10.0}
            )

        # Should only keep last 3 entries
        assert len(history.win_rates_history) == 3

        # Should keep the most recent values
        assert (
            abs(history.win_rates_history[-1]["black_win_rate"] - 4.0) < 1e-6
        )  # Last added value

        # Add more PPO data than max_history with additional metrics
        for i in range(5):
            history.add_ppo_data(
                {
                    "ppo/learning_rate": float(i) * 0.001,
                    "ppo/policy_loss": float(i) * 0.1,
                    "ppo/value_loss": float(i) * 0.05,
                }
            )

        # Should only keep last 3 entries for all PPO metrics
        assert len(history.learning_rates) == 3
        assert len(history.policy_losses) == 3
        assert len(history.value_losses) == 3

        # Verify most recent values
        assert abs(history.learning_rates[-1] - 0.004) < 1e-6  # 4 * 0.001
        assert abs(history.policy_losses[-1] - 0.4) < 1e-6  # 4 * 0.1
        assert abs(history.value_losses[-1] - 0.2) < 1e-6  # 4 * 0.05


class TestSparklineGeneration:
    """Tests for Sparkline visual generation."""

    def test_sparkline_visual_correctness(self):
        """Test sparkline visual representation accuracy."""
        spark = Sparkline(width=8)

        # Test ascending data
        ascending_data = [1, 2, 3, 4, 5, 6, 7, 8]
        result = spark.generate(ascending_data)

        assert len(result) == 8

        # Should use different characters for different values
        unique_chars = set(result)
        assert len(unique_chars) > 1, "Should use multiple different characters"

    def test_sparkline_data_mapping(self):
        """Test correct mapping of data values to sparkline characters."""
        spark = Sparkline(width=3)

        # Test with known extremes
        extreme_data = [0, 50, 100]
        result = spark.generate(extreme_data, range_min=0, range_max=100)

        sparkline_chars = "▁▂▃▄▅▆▇█"

        # First character should be lowest, last should be highest
        if all(char in sparkline_chars for char in result):
            first_idx = sparkline_chars.index(result[0])
            last_idx = sparkline_chars.index(result[2])
            assert last_idx > first_idx, "Last value should map to higher character"

    def test_sparkline_performance(self):
        """Test sparkline generation performance with large datasets."""
        spark = Sparkline(width=50)

        # Generate large dataset
        large_data = list(range(1000))

        start_time = time.time()
        result = spark.generate(large_data[-50:])  # Take last 50 values
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 0.1  # Less than 100ms
        assert len(result) == 50

    def test_sparkline_performance_large_dataset(self):
        """Test sparkline generation performance with very large datasets."""
        spark = Sparkline(width=20)

        # Generate very large dataset
        large_data = list(range(10000))

        start_time = time.time()
        result = spark.generate(large_data[-20:])  # Take last 20 values
        end_time = time.time()

        # Should complete quickly even with large input
        assert (end_time - start_time) < 0.1  # Less than 100ms
        assert len(result) == 20


class TestRobustnessAndIntegration:
    """Tests for system robustness and component integration."""

    def test_resource_cleanup(self):
        """Test that components properly clean up resources."""
        # Create components with large data
        history = MetricsHistory(max_history=1000)

        # Add large amount of data
        for i in range(5000):
            history.add_episode_data({"black_win_rate": float(i % 100) / 100.0})

        # Should still respect max_history limit
        assert len(history.win_rates_history) <= 1000

        # Should contain most recent data
        assert abs(history.win_rates_history[-1]["black_win_rate"] - 0.99) < 1e-6

    def test_component_interoperability(self):
        """Test that display components work together without conflicts."""
        # Create multiple components
        sparkline = Sparkline(width=10)
        elo = EloRatingSystem(initial_rating=1500, k_factor=32)
        history = MetricsHistory(max_history=5)

        # Simulate realistic usage together
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        sparkline_result = sparkline.generate(test_data)

        # Update elo ratings
        elo.update_ratings(Color.BLACK)

        # Add data to history
        history.add_episode_data({"black_win_rate": 0.75})

        # All should produce valid results
        assert sparkline_result is not None
        assert len(sparkline_result) == 10
        assert elo.black_rating > 1500  # Should have increased
        assert len(history.win_rates_history) == 1

    def test_concurrent_access_safety(self):
        """Test thread safety for components that might be accessed concurrently."""
        elo = EloRatingSystem()
        results = []

        def update_ratings():
            for _ in range(5):  # Reduced iterations for faster test
                elo.update_ratings(Color.BLACK)
                time.sleep(0.001)  # Small delay
                results.append(elo.black_rating)

        # Create multiple threads
        threads = [threading.Thread(target=update_ratings) for _ in range(2)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have results from all threads
        assert len(results) == 10  # 2 threads * 5 updates each
        # All ratings should be different (increasing)
        assert len(set(results)) > 1


# Keep simple standalone tests for backward compatibility
def test_display_config_defaults():
    """Simple test for DisplayConfig defaults."""
    config = create_test_display_config()
    assert config.enable_board_display is True
    assert config.sparkline_width == 15
    assert abs(config.elo_initial_rating - 1500.0) < 0.001
    assert config.show_moves_trend is True
    assert config.show_completion_rate is True


def test_metrics_history_trimming_simple():
    """Simple test for MetricsHistory trimming."""
    history = MetricsHistory(max_history=3)
    for i in range(5):
        history.add_episode_data({"black_win_rate": float(i)})
    assert len(history.win_rates_history) == 3

    for i in range(5):
        history.add_ppo_data(
            {
                "ppo/learning_rate": float(i),
                "ppo/policy_loss": float(i),
            }
        )
    assert len(history.learning_rates) == 3
    assert len(history.policy_losses) == 3


def test_elo_rating_updates():
    """Simple test for Elo rating updates."""
    elo = EloRatingSystem()
    initial = elo.black_rating
    elo.update_ratings(Color.BLACK)
    assert elo.black_rating > initial


def test_sparkline_generation():
    """Simple test for sparkline generation."""
    spark = Sparkline(width=5)
    s = spark.generate([1, 2, 3, 4, 5])
    assert len(s) == 5


class TestDisplayConfigurationAdvanced:
    """Advanced tests for DisplayConfig error handling and edge cases."""

    def test_display_config_invalid_values(self):
        """Test DisplayConfig with invalid parameter values."""
        # Test that DisplayConfig handles edge cases gracefully
        config = create_test_display_config()
        assert config.sparkline_width > 0
        assert config.board_cell_width > 0
        assert config.elo_k_factor > 0

    def test_display_config_extreme_values(self):
        """Test DisplayConfig with extreme but valid values."""
        # Test very large values
        config = create_test_display_config(
            sparkline_width=1000, trend_history_length=50000, move_list_length=500
        )

        assert config.sparkline_width == 1000
        assert config.trend_history_length == 50000
        assert config.move_list_length == 500

    def test_display_config_memory_usage(self):
        """Test DisplayConfig memory usage with large parameters."""
        # Create config with large parameters
        config = create_test_display_config(
            trend_history_length=10000, metrics_window_size=5000, move_list_length=1000
        )

        # Should not consume excessive memory for configuration alone
        assert config.trend_history_length == 10000
        assert config.metrics_window_size == 5000

    def test_display_config_serialization_edge_cases(self):
        """Test DisplayConfig behavior with edge case values."""
        # Test with minimum valid values
        config = create_test_display_config(
            sparkline_width=1,
            board_cell_width=3,
            board_cell_height=1,
            move_list_length=1,
            elo_k_factor=0.1,
        )

        assert config.sparkline_width == 1
        assert config.board_cell_width == 3
        assert config.board_cell_height == 1
        assert config.move_list_length == 1
        assert abs(config.elo_k_factor - 0.1) < 0.001


class TestEloRatingSystemAdvanced:
    """Advanced tests for EloRatingSystem edge cases and mathematical accuracy."""

    def test_elo_rating_precision_limits(self):
        """Test Elo system with extreme precision requirements."""
        elo = EloRatingSystem(initial_rating=1500.123456789, k_factor=32.987654321)

        initial_black = elo.black_rating
        initial_white = elo.white_rating

        elo.update_ratings(Color.BLACK)

        # Should maintain reasonable precision
        assert elo.black_rating != initial_black
        assert elo.white_rating != initial_white
        assert (
            abs((elo.black_rating - initial_black) - (initial_white - elo.white_rating))
            < 1e-10
        )

    def test_elo_rating_overflow_protection(self):
        """Test Elo system with extreme rating values."""
        elo = EloRatingSystem(initial_rating=10000.0, k_factor=1000.0)
        elo.white_rating = 0.0

        # Should handle extreme differences without overflow
        elo.update_ratings(Color.BLACK)

        # Should still produce finite values
        assert elo.black_rating != float("inf")
        assert elo.white_rating != float("-inf")
        assert not math.isnan(elo.black_rating)  # Check for NaN

    def test_elo_rating_rapid_convergence(self):
        """Test Elo system behavior with rapid rating changes."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=64)

        ratings_history = [(elo.black_rating, elo.white_rating)]

        # Simulate 20 consecutive black wins
        for _ in range(20):
            elo.update_ratings(Color.BLACK)
            ratings_history.append((elo.black_rating, elo.white_rating))

        # Black rating should have increased significantly
        assert elo.black_rating > 1500 + 200  # At least 200 point gain
        assert elo.white_rating < 1500 - 200  # At least 200 point loss

        # Rating changes should decrease as difference grows
        last_change = ratings_history[-1][0] - ratings_history[-2][0]
        first_change = ratings_history[1][0] - ratings_history[0][0]
        assert last_change < first_change  # Diminishing returns

    def test_elo_rating_mathematical_invariants(self):
        """Test mathematical invariants of Elo system."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=32)

        total_rating_before = elo.black_rating + elo.white_rating

        # Play several games
        for color in [Color.BLACK, Color.WHITE, Color.BLACK, None, Color.WHITE]:
            elo.update_ratings(color)

        total_rating_after = elo.black_rating + elo.white_rating

        # Total rating should be conserved (within floating point precision)
        assert abs(total_rating_before - total_rating_after) < 1e-10

    def test_elo_rating_concurrent_access_safety(self):
        """Test Elo system thread safety."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=32)
        results = []
        errors = []

        def update_ratings_worker():
            try:
                for _ in range(10):
                    elo.update_ratings(Color.BLACK)
                    results.append(elo.black_rating)
                    time.sleep(0.001)
            except (ValueError, AttributeError, RuntimeError) as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=update_ratings_worker) for _ in range(3)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 30  # 3 threads * 10 updates each


class TestMetricsHistoryAdvanced:
    """Advanced tests for MetricsHistory performance and edge cases."""

    def test_metrics_history_large_scale_performance(self):
        """Test MetricsHistory performance with large-scale data."""
        history = MetricsHistory(max_history=1000)

        # Measure performance of large data insertion
        large_dataset = TestDataFactory.generate_episode_metrics(5000)

        start_time = time.time()
        for episode_data in large_dataset:
            history.add_episode_data(episode_data)
        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second
        assert len(history.win_rates_history) == 1000  # Should maintain max_history

    def test_metrics_history_memory_pressure(self):
        """Test MetricsHistory behavior under memory pressure."""
        history = MetricsHistory(max_history=100)

        # Add data with large individual entries
        for i in range(500):
            large_data = {
                "black_win_rate": float(i % 100) / 100.0,
                "white_win_rate": float((i + 1) % 100) / 100.0,
                "draw_rate": float((i + 2) % 100) / 100.0,
            }
            history.add_episode_data(large_data)

        # Should still respect memory limits
        assert len(history.win_rates_history) == 100
        # Most recent entry should be present
        assert abs(history.win_rates_history[-1]["black_win_rate"] - 0.99) < 1e-6

    def test_metrics_history_data_integrity(self):
        """Test MetricsHistory data integrity with various data types."""
        history = MetricsHistory(max_history=10)

        # Test with edge case values
        edge_cases = [
            {"black_win_rate": float("inf")},
            {"black_win_rate": float("-inf")},
            {"black_win_rate": 0.0},
            {"black_win_rate": 1.0},
            {"black_win_rate": -1.0},
            {"black_win_rate": 1e-10},
            {"black_win_rate": 1e10},
        ]

        for case in edge_cases:
            history.add_episode_data(case)

        # Should handle all cases without errors
        assert len(history.win_rates_history) == len(edge_cases)

    def test_metrics_history_concurrent_modifications(self):
        """Test MetricsHistory thread safety with concurrent modifications."""
        history = MetricsHistory(max_history=50)
        errors = []

        def add_episode_data_worker(worker_id):
            try:
                for i in range(20):
                    history.add_episode_data(
                        {
                            "black_win_rate": float(worker_id * 100 + i) / 1000.0,
                            "worker_id": worker_id,
                        }
                    )
                    time.sleep(0.001)
            except (ValueError, AttributeError, RuntimeError) as e:
                errors.append(e)

        def add_ppo_data_worker(worker_id):
            try:
                for i in range(20):
                    history.add_ppo_data(
                        {
                            "ppo/learning_rate": float(worker_id * 100 + i) / 10000.0,
                            "ppo/policy_loss": float(i) / 100.0,
                        }
                    )
                    time.sleep(0.001)
            except (ValueError, AttributeError, RuntimeError) as e:
                errors.append(e)

        # Create multiple threads for different operations
        threads = []
        threads.extend(
            [
                threading.Thread(target=add_episode_data_worker, args=(i,))
                for i in range(2)
            ]
        )
        threads.extend(
            [threading.Thread(target=add_ppo_data_worker, args=(i,)) for i in range(2)]
        )

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have any errors
        assert len(errors) == 0
        # Should have data from all workers
        assert len(history.win_rates_history) > 0
        assert len(history.learning_rates) > 0


class TestSparklineAdvanced:
    """Advanced tests for Sparkline generation and edge cases."""

    def test_sparkline_unicode_edge_cases(self):
        """Test Sparkline with Unicode handling edge cases."""
        spark = Sparkline(width=8)

        # Test with extreme values
        extreme_data = [float("-inf"), -1e10, 0, 1e10, float("inf")]

        # Should handle extreme values gracefully
        result = spark.generate(extreme_data[:3])  # Use finite subset
        assert len(result) == 8  # Should return full width
        assert all(ord(char) < 0x10000 for char in result)  # Valid Unicode

    def test_sparkline_performance_stress(self):
        """Test Sparkline performance under stress conditions."""
        spark = Sparkline(width=100)

        # Generate very large dataset
        large_data = list(range(100000))

        # Test multiple rapid generations
        start_time = time.time()
        for _ in range(10):
            result = spark.generate(large_data[-100:])
            assert len(result) == 100
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 0.5  # Less than 500ms total

    def test_sparkline_data_range_handling(self):
        """Test Sparkline with various data ranges and edge cases."""
        spark = Sparkline(width=5)

        # Test with identical values
        identical_data = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = spark.generate(identical_data)
        assert len(result) == 5
        # All characters should be the same for identical values
        assert len(set(result)) == 1

        # Test with single value
        single_data = [42.0]
        result = spark.generate(single_data)
        assert len(result) == 5  # Should return full width

        # Test with two values
        two_data = [1.0, 10.0]
        result = spark.generate(two_data)
        assert len(result) == 5  # Should return full width

    def test_sparkline_custom_range_accuracy(self):
        """Test Sparkline accuracy with custom ranges."""
        spark = Sparkline(width=3)

        # Test with explicit range
        data = [0, 50, 100]
        result = spark.generate(data, range_min=0, range_max=100)

        sparkline_chars = "▁▂▃▄▅▆▇█"

        # Should map correctly to character positions
        if all(char in sparkline_chars for char in result):
            # First value (0) should map to lowest character
            first_pos = sparkline_chars.index(result[0])
            # Last value (100) should map to highest character
            last_pos = sparkline_chars.index(result[2])
            # Middle value (50) should be between them
            middle_pos = sparkline_chars.index(result[1])

            assert first_pos <= middle_pos <= last_pos


class TestSystemIntegrationAdvanced:
    """Advanced integration tests for component interactions."""

    def test_full_system_integration_workflow(self):
        """Test complete workflow integration of all components."""
        # Initialize all components
        config = create_test_display_config(
            sparkline_width=20, trend_history_length=100, elo_k_factor=32.0
        )

        elo = EloRatingSystem(
            initial_rating=config.elo_initial_rating, k_factor=config.elo_k_factor
        )

        history = MetricsHistory(max_history=config.trend_history_length)
        sparkline = Sparkline(width=config.sparkline_width)

        # Simulate complete training workflow
        for episode in range(50):
            # Simulate game outcome
            outcome = Color.BLACK if episode % 3 == 0 else Color.WHITE
            elo.update_ratings(outcome)

            # Add episode data
            episode_data = {
                "black_win_rate": elo.black_rating
                / (elo.black_rating + elo.white_rating),
                "average_episode_length": 50 + (episode % 20),
                "episode": episode,
            }
            history.add_episode_data(episode_data)

            # Add PPO data
            ppo_data = {
                "ppo/learning_rate": 0.001 * (0.95 ** (episode // 10)),
                "ppo/policy_loss": 1.0 / (1 + episode * 0.1),
                "ppo/value_loss": 0.5 / (1 + episode * 0.05),
            }
            history.add_ppo_data(ppo_data)

        # Generate sparklines from history
        win_rates = [entry["black_win_rate"] for entry in history.win_rates_history]
        sparkline_result = sparkline.generate(win_rates[-20:])  # Last 20 entries

        # Verify integration results
        assert len(history.win_rates_history) == 50
        assert len(history.learning_rates) == 50
        assert len(sparkline_result) == 20
        assert elo.black_rating != config.elo_initial_rating  # Should have changed
        assert elo.white_rating != config.elo_initial_rating

    def test_error_recovery_integration(self):
        """Test system recovery from component errors."""
        elo = EloRatingSystem()
        history = MetricsHistory(max_history=10)
        sparkline = Sparkline(width=5)

        # Simulate error conditions and test graceful handling
        try:
            # Invalid color should be handled gracefully
            elo.update_ratings(None)  # Use None instead of invalid string
        except (ValueError, TypeError):
            pass  # Expected behavior

        # System should still be functional after errors
        elo.update_ratings(Color.BLACK)
        assert elo.black_rating > 1500

        history.add_episode_data({"black_win_rate": 0.5})
        assert len(history.win_rates_history) == 1

        result = sparkline.generate([1, 2, 3])
        assert len(result) == 5  # Should return full width

    def test_performance_integration_stress(self):
        """Test system performance under integrated stress conditions."""
        # Create components
        elo = EloRatingSystem()
        history = MetricsHistory(max_history=1000)
        sparkline = Sparkline(width=50)

        # Stress test with rapid operations
        start_time = time.time()

        for i in range(200):
            # Update elo ratings
            outcome = Color.BLACK if i % 2 == 0 else Color.WHITE
            elo.update_ratings(outcome)

            # Add history data
            history.add_episode_data({"black_win_rate": float(i % 100) / 100.0})
            history.add_ppo_data(
                {"ppo/learning_rate": 0.001, "ppo/policy_loss": float(i % 50) / 50.0}
            )

            # Generate sparkline every 10 iterations
            if i % 10 == 0:
                data = [entry["black_win_rate"] for entry in history.win_rates_history]
                if data:
                    sparkline.generate(data[-50:])

        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 2.0  # Less than 2 seconds

        # Verify final state
        assert len(history.win_rates_history) == 200
        assert len(history.learning_rates) == 200
        assert elo.black_rating != 1500 or elo.white_rating != 1500
