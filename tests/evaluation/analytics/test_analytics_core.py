"""Core analytics functionality tests."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from keisei.evaluation.analytics.advanced_analytics import (
    AdvancedAnalytics,
    PerformanceComparison,
    TrendAnalysis,
)


class TestAdvancedAnalyticsCore:
    """Test core AdvancedAnalytics functionality."""

    def test_initialization_default_params(self):
        """Test AdvancedAnalytics initialization with default parameters."""
        analytics = AdvancedAnalytics()

        assert analytics.significance_level == 0.05
        assert analytics.min_practical_difference == 0.05
        assert analytics.trend_window_days == 30

    def test_initialization_custom_params(self):
        """Test AdvancedAnalytics initialization with custom parameters."""
        analytics = AdvancedAnalytics(
            significance_level=0.01,
            min_practical_difference=0.1,
            trend_window_days=60,
        )

        assert analytics.significance_level == 0.01
        assert analytics.min_practical_difference == 0.1
        assert analytics.trend_window_days == 60

    def test_compare_performance_basic(
        self, analytics, sample_game_results, baseline_game_results
    ):
        """Test basic performance comparison functionality."""
        comparison = analytics.compare_performance(
            baseline_results=baseline_game_results,
            comparison_results=sample_game_results,
            baseline_name="Baseline",
            comparison_name="Current",
        )

        assert isinstance(comparison, PerformanceComparison)
        assert comparison.baseline_name == "Baseline"
        assert comparison.comparison_name == "Current"

        # Should show improvement (60% vs 40%)
        assert comparison.win_rate_difference > 0
        assert comparison.win_rate_change_percent > 0

        # Should have statistical tests
        assert isinstance(comparison.statistical_tests, list)
        assert len(comparison.statistical_tests) > 0

        # Should have confidence interval
        assert isinstance(comparison.confidence_interval, tuple)
        assert len(comparison.confidence_interval) == 2

        # Should have recommendation
        assert isinstance(comparison.recommendation, str)
        assert len(comparison.recommendation) > 0

    def test_compare_performance_empty_baseline(self, analytics, sample_game_results):
        """Test performance comparison with empty baseline."""
        comparison = analytics.compare_performance(
            baseline_results=[],
            comparison_results=sample_game_results,
        )

        assert comparison.win_rate_difference == 0.6  # 60% vs 0%
        assert comparison.win_rate_change_percent == float("inf")

    def test_compare_performance_empty_comparison(
        self, analytics, baseline_game_results
    ):
        """Test performance comparison with empty comparison results."""
        comparison = analytics.compare_performance(
            baseline_results=baseline_game_results,
            comparison_results=[],
        )

        assert comparison.win_rate_difference == -0.4  # 0% vs 40%

    def test_analyze_trends_insufficient_data(self, analytics):
        """Test trend analysis with insufficient data."""
        historical_data = [
            (datetime.now() - timedelta(days=1), Mock()),
            (datetime.now(), Mock()),
        ]

        trend = analytics.analyze_trends(historical_data, "win_rate")

        assert isinstance(trend, TrendAnalysis)
        assert trend.metric_name == "win_rate"
        assert trend.trend_direction == "insufficient_data"
        assert trend.trend_strength == 0.0
        assert trend.data_points == 2
        assert trend.time_span_days == 0.0

    def test_analyze_trends_win_rate(self, analytics, historical_evaluation_data):
        """Test trend analysis for win rate metric."""
        trend = analytics.analyze_trends(historical_evaluation_data, "win_rate")

        assert isinstance(trend, TrendAnalysis)
        assert trend.metric_name == "win_rate"
        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
        assert 0 <= trend.trend_strength <= 1
        assert trend.data_points == len(historical_evaluation_data)
        assert trend.time_span_days > 0

        # Should detect increasing trend due to our test data
        assert trend.trend_direction == "increasing"
        assert trend.slope > 0

    def test_analyze_trends_game_length(self, analytics, historical_evaluation_data):
        """Test trend analysis for game length metric."""
        trend = analytics.analyze_trends(historical_evaluation_data, "avg_game_length")

        assert trend.metric_name == "avg_game_length"
        assert trend.trend_direction == "increasing"  # Our test data increases
        assert trend.slope > 0

    def test_analyze_trends_unknown_metric(self, analytics, historical_evaluation_data):
        """Test trend analysis with unknown metric."""
        with patch(
            "keisei.evaluation.analytics.advanced_analytics.logger"
        ) as mock_logger:
            trend = analytics.analyze_trends(
                historical_evaluation_data, "unknown_metric"
            )

            assert trend.metric_name == "unknown_metric"
            # Should still work but with default values
            mock_logger.warning.assert_called()

    def test_analyze_trends_statistical_accuracy(self, analytics, historical_evaluation_data):
        """Test trend analysis statistical accuracy with actual scipy."""
        trend = analytics.analyze_trends(historical_evaluation_data, "win_rate")

        # Should use real scipy linregress
        assert isinstance(trend.slope, float)
        assert isinstance(trend.r_squared, float)
        assert 0 <= trend.r_squared <= 1
        
        # Should detect increasing trend due to our test data
        assert trend.trend_direction == "increasing"
        assert trend.slope > 0

    def test_performance_comparison_edge_cases(self, analytics):
        """Test performance comparison with edge cases."""
        # Test identical performance
        baseline_games = []
        comparison_games = []
        
        for i in range(10):
            # Create identical games
            baseline_game = Mock()
            baseline_game.is_agent_win = i < 5  # 50% win rate
            baseline_game.moves_count = 50
            baseline_game.duration_seconds = 30.0 # CHANGED
            baseline_games.append(baseline_game)
            
            comparison_game = Mock()
            comparison_game.is_agent_win = i < 5  # Same 50% win rate
            comparison_game.moves_count = 50
            comparison_game.duration_seconds = 30.0 # CHANGED
            comparison_games.append(comparison_game)

        comparison = analytics.compare_performance(
            baseline_results=baseline_games,
            comparison_results=comparison_games,
        )

        # Should detect no significant difference
        assert abs(comparison.win_rate_difference) < 0.01  # Very small difference
        assert comparison.practical_significance == False

    def test_trend_analysis_stability(self, analytics):
        """Test trend analysis with stable data."""
        # Create stable performance data
        stable_data = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(10):
            timestamp = base_time + timedelta(days=i * 3)
    
            # Create consistent performance
            games = []
            for j in range(10):
                game = Mock()
                game.is_agent_win = j < 5  # Consistent 50% win rate
                game.moves_count = 50  # Consistent game length
                game.duration_seconds = 30.0 # Changed from game_length_seconds
                # Add other potentially missing attributes for SummaryStats
                game.is_opponent_win = j >= 5
                game.is_draw = False
                games.append(game)
    
            from keisei.evaluation.core.evaluation_result import SummaryStats
            summary_stats = SummaryStats.from_games(games)
            
            from keisei.evaluation.core import EvaluationResult
            result = EvaluationResult(
                context=Mock(),
                games=games,
                summary_stats=summary_stats,
                analytics_data={},
            )
            stable_data.append((timestamp, result))

        trend = analytics.analyze_trends(stable_data, "win_rate")
        
        # Should detect stable trend
        assert trend.trend_direction == "stable"
        assert abs(trend.slope) < 0.01  # Very small slope for stable data

    def test_analytics_parameter_validation(self):
        """Test analytics parameter validation."""
        # Test invalid significance level
        with pytest.raises(ValueError, match="significance_level must be between 0 and 1"): # ADDED match
            AdvancedAnalytics(significance_level=-0.1)
        
        with pytest.raises(ValueError, match="significance_level must be between 0 and 1"): # ADDED match
            AdvancedAnalytics(significance_level=1.1)
        
        # Test invalid practical difference
        with pytest.raises(ValueError, match="min_practical_difference must be non-negative"): # ADDED match
            AdvancedAnalytics(min_practical_difference=-0.1)
        
        # Test invalid trend window
        with pytest.raises(ValueError, match="trend_window_days must be positive"): # ADDED match
            AdvancedAnalytics(trend_window_days=0)

    def test_metric_extraction_robustness(self, analytics):
        """Test robustness of metric extraction from evaluation results."""
        from keisei.evaluation.core import EvaluationResult
        from keisei.evaluation.core.evaluation_result import SummaryStats
        
        # Test with missing attributes
        incomplete_data = []
        base_time = datetime.now() - timedelta(days=10)
        
        for i in range(3):
            timestamp = base_time + timedelta(days=i * 3)
            
            # Create result with minimal data
            result = EvaluationResult(
                context=Mock(),
                games=[],
                summary_stats=SummaryStats(
                    total_games=0,
                    agent_wins=0,
                    opponent_wins=0,
                    draws=0,
                    win_rate=0.0,
                    loss_rate=0.0,
                    draw_rate=0.0,
                    avg_game_length=0.0,
                    total_moves=0,
                    avg_duration_seconds=0.0,
                ),
                analytics_data={},
            )
            incomplete_data.append((timestamp, result))

        # Should handle gracefully - flat line with all zeros should be stable
        trend = analytics.analyze_trends(incomplete_data, "win_rate")
        assert trend.trend_direction == "stable"
        assert abs(trend.slope) < 1e-6  # Very small slope for flat data


class TestPerformanceComparison:
    """Test PerformanceComparison functionality."""

    def test_performance_comparison_creation(self, analytics, sample_game_results, baseline_game_results):
        """Test creation and basic properties of PerformanceComparison."""
        comparison = analytics.compare_performance(
            baseline_results=baseline_game_results,
            comparison_results=sample_game_results,
        )

        # Test basic properties
        assert hasattr(comparison, "baseline_name") # CHANGED from baseline_win_rate
        assert hasattr(comparison, "comparison_name") # CHANGED from comparison_win_rate
        assert hasattr(comparison, "win_rate_difference")
        assert hasattr(comparison, "win_rate_change_percent")
        assert hasattr(comparison, "statistical_tests")
        assert hasattr(comparison, "confidence_interval")
        assert hasattr(comparison, "practical_significance") # CHANGED from is_significant
        assert hasattr(comparison, "recommendation")

    def test_statistical_significance_determination(self, analytics):
        """Test statistical significance determination in comparisons."""
        # Create data that should be significantly different
        high_performance_results = []
        low_performance_results = []
    
        for i in range(50):  # Larger sample for significance
            # High performance: 80% win rate
            game_high = Mock()
            game_high.is_agent_win = i < 40
            game_high.moves_count = 50 # ADDED
            game_high.duration_seconds = 30.0 
            high_performance_results.append(game_high)
    
            # Low performance: 20% win rate
            game_low = Mock()
            game_low.is_agent_win = i < 10
            game_low.moves_count = 50 # ADDED
            game_low.duration_seconds = 30.0 
            low_performance_results.append(game_low)
    
        comparison = analytics.compare_performance(
            baseline_results=low_performance_results,
            comparison_results=high_performance_results,
        )

        # With large effect size and sample, should be significant
        # Check if any statistical test shows significance
        assert any(test.is_significant for test in comparison.statistical_tests) # CHANGED
        assert comparison.win_rate_difference > 0.5  # 60% difference

    def test_recommendation_generation(self, analytics, sample_game_results, baseline_game_results):
        """Test recommendation generation based on comparison results."""
        comparison = analytics.compare_performance(
            baseline_results=baseline_game_results,  # 40% win rate
            comparison_results=sample_game_results,   # 60% win rate
        )

        # Should recommend keeping the improvement
        assert "improvement" in comparison.recommendation.lower() or "better" in comparison.recommendation.lower()
        assert len(comparison.recommendation) > 20  # Should be descriptive
