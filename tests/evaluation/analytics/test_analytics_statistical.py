"""
Statistical analysis tests for AdvancedAnalytics components.

This module tests statistical testing functionality including z-tests,
Mann-Whitney U tests, confidence intervals, and statistical recommendations.
"""

from unittest.mock import Mock

import pytest

from keisei.evaluation.analytics.advanced_analytics import (
    AdvancedAnalytics,
    StatisticalTest,
)


class TestAdvancedAnalyticsStatisticalTests:
    """Test statistical analysis functionality."""

    def test_two_proportion_z_test(self, analytics):
        """Test two-proportion z-test implementation."""
        # Use reflection to access private method for testing
        z_test = analytics._two_proportion_z_test(60, 100, 40, 100)

        assert isinstance(z_test, StatisticalTest)
        assert z_test.test_name == "two_proportion_z_test"
        assert isinstance(z_test.p_value, float)
        assert isinstance(z_test.is_significant, bool)
        assert isinstance(z_test.interpretation, str)

    def test_mann_whitney_test_functionality(self, analytics):
        """Test Mann-Whitney U test with actual scipy implementation."""
        test = analytics._mann_whitney_test(
            [1, 2, 3, 4, 5], [3, 4, 5, 6, 7], "test_metric"
        )

        assert test.test_name == "mann_whitney_test_metric"
        assert isinstance(test.p_value, float)
        assert isinstance(test.is_significant, bool)
        assert isinstance(test.statistic, float)
        assert isinstance(test.interpretation, str)

    def test_confidence_interval_calculation(self, analytics):
        """Test confidence interval calculation for win rate difference."""
        ci = analytics._calculate_win_rate_difference_ci(60, 100, 40, 100)

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound

    def test_confidence_interval_edge_cases(self, analytics):
        """Test confidence interval calculation with edge cases."""
        # Zero baseline
        ci = analytics._calculate_win_rate_difference_ci(0, 100, 40, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Zero comparison
        ci = analytics._calculate_win_rate_difference_ci(60, 100, 0, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_performance_recommendation_generation(self, analytics):
        """Test performance recommendation generation."""
        # Test with significant improvement
        tests = [
            StatisticalTest(
                test_name="Test",
                statistic=1.96,
                p_value=0.01,
                is_significant=True,
                confidence_level=0.95,
                interpretation="Significant",
                effect_size=0.6,
            )
        ]

        recommendation = analytics._generate_performance_recommendation(
            win_rate_difference=0.1,  # 10% improvement
            statistical_tests=tests,
            practical_significance=True,
        )

        assert isinstance(recommendation, str)
        assert len(recommendation) > 0
        assert "improvement" in recommendation.lower()

    def test_performance_recommendation_no_significance(self, analytics):
        """Test performance recommendation with no statistical significance."""
        tests = [
            StatisticalTest(
                test_name="Test",
                statistic=0.5,
                p_value=0.8,
                is_significant=False,
                confidence_level=0.95,
                interpretation="Not significant",
            )
        ]

        recommendation = analytics._generate_performance_recommendation(
            win_rate_difference=0.01,  # Small change
            statistical_tests=tests,
            practical_significance=False,
        )

        assert "no statistically significant" in recommendation.lower()

    def test_statistical_significance_detection(self, analytics):
        """Test statistical significance detection in comparisons."""
        # Test with clearly different result sets
        baseline_results = []
        comparison_results = []

        # Baseline: 20% win rate
        for i in range(100):
            game = type("MockGame", (), {})()
            game.is_agent_win = i < 20
            game.moves_count = 50
            game.game_length_seconds = 30.0
            baseline_results.append(game)

        # Comparison: 80% win rate
        for i in range(100):
            game = type("MockGame", (), {})()
            game.is_agent_win = i < 80
            game.moves_count = 50
            game.game_length_seconds = 30.0
            comparison_results.append(game)

        comparison = analytics.compare_performance(
            baseline_results=baseline_results,
            comparison_results=comparison_results,
        )

        # Should detect significant difference
        assert comparison.practical_significance == True
        assert abs(comparison.win_rate_difference - 0.6) < 0.001  # 60% improvement
        assert len(comparison.statistical_tests) > 0

        # At least one test should be significant
        assert any(test.is_significant for test in comparison.statistical_tests)

    def test_confidence_interval_boundary_conditions(self, analytics):
        """Test confidence interval calculation with boundary conditions."""
        # Perfect baseline (100% win rate)
        ci = analytics._calculate_win_rate_difference_ci(100, 100, 80, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Perfect comparison (100% win rate)
        ci = analytics._calculate_win_rate_difference_ci(80, 100, 100, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Small sample sizes
        ci = analytics._calculate_win_rate_difference_ci(3, 5, 2, 5)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_effect_size_calculation(self, analytics):
        """Test effect size calculation in statistical tests."""
        z_test = analytics._two_proportion_z_test(80, 100, 60, 100)

        assert hasattr(z_test, "effect_size")
        assert isinstance(z_test.effect_size, (int, float))
        assert z_test.effect_size >= 0  # Effect size should be non-negative
