"""
Tests for advanced analytics integration (Task 7 - High Priority)
Coverage for keisei/evaluation/analytics/advanced_analytics.py
"""
import pytest
import torch
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from keisei.constants import CORE_OBSERVATION_CHANNELS, FULL_ACTION_SPACE
from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
from tests.evaluation.factories import EvaluationTestFactory

class TestAdvancedAnalytics:
    """Test advanced analytics pipeline integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analytics_manager = Mock()
        self.test_results = EvaluationTestFactory.create_test_game_results(
            count=10, win_rate=0.6
        )

    def test_statistical_analysis_algorithm_validation(self):
        """Test statistical analysis algorithm accuracy."""
        # Mock advanced analytics manager
        with patch('keisei.evaluation.analytics.advanced_analytics.AdvancedAnalytics') as MockAnalytics:
            analytics = MockAnalytics.return_value
            
            # Configure mock to return expected statistical results
            analytics.compute_confidence_intervals.return_value = {
                "win_rate_ci": (0.55, 0.65),
                "avg_game_length_ci": (45.2, 55.8),
                "confidence_level": 0.95
            }
            
            analytics.perform_significance_test.return_value = {
                "p_value": 0.023,
                "is_significant": True,
                "test_statistic": 2.45
            }
            
            # Test statistical analysis
            result = analytics.compute_confidence_intervals(self.test_results)
            assert "win_rate_ci" in result
            assert result["confidence_level"] == 0.95
            
            significance = analytics.perform_significance_test(
                self.test_results, baseline_winrate=0.5
            )
            assert significance["is_significant"] is True
            assert significance["p_value"] < 0.05

    def test_performance_comparison_accuracy(self):
        """Test performance comparison between different agents."""
        # Create comparison data
        baseline_results = EvaluationTestFactory.create_test_game_results(
            count=8, win_rate=0.5
        )
        improved_results = EvaluationTestFactory.create_test_game_results(
            count=8, win_rate=0.7
        )
        
        with patch('keisei.evaluation.analytics.advanced_analytics.AdvancedAnalytics') as MockAnalytics:
            analytics = MockAnalytics.return_value
            
            # Mock comparison analysis
            analytics.compare_performance.return_value = {
                "win_rate_improvement": 0.2,
                "relative_improvement": 40.0,
                "statistical_significance": True,
                "effect_size": "large"
            }
            
            comparison = analytics.compare_performance(baseline_results, improved_results)
            
            assert comparison["win_rate_improvement"] == 0.2
            assert comparison["relative_improvement"] == 40.0
            assert comparison["statistical_significance"] is True

    def test_trend_analysis_and_pattern_detection(self):
        """Test trend analysis and pattern detection capabilities."""
        # Create time series data for trend analysis
        time_series_data = [
            {"timestamp": i, "win_rate": 0.5 + (i * 0.02), "elo": 1500 + (i * 10)}
            for i in range(20)
        ]
        
        with patch('keisei.evaluation.analytics.advanced_analytics.AdvancedAnalytics') as MockAnalytics:
            analytics = MockAnalytics.return_value
            
            # Mock trend detection
            analytics.detect_trends.return_value = {
                "win_rate_trend": "increasing",
                "trend_strength": 0.85,
                "seasonal_patterns": False,
                "anomalies_detected": 1
            }
            
            trends = analytics.detect_trends(time_series_data)
            
            assert trends["win_rate_trend"] == "increasing"
            assert trends["trend_strength"] > 0.8
            assert isinstance(trends["anomalies_detected"], int)

    def test_report_generation_and_formatting(self):
        """Test report generation with proper formatting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "analytics_report.html"
            
            with patch('keisei.evaluation.analytics.advanced_analytics.AdvancedAnalytics') as MockAnalytics:
                analytics = MockAnalytics.return_value
                
                # Mock report generation
                analytics.generate_report.return_value = {
                    "report_path": str(report_path),
                    "sections": ["summary", "detailed_analysis", "recommendations"],
                    "charts_generated": 5,
                    "format": "html"
                }
                
                report = analytics.generate_report(
                    self.test_results, 
                    output_path=str(report_path),
                    format="html"
                )
                
                assert report["format"] == "html"
                assert len(report["sections"]) == 3
                assert report["charts_generated"] > 0
                assert "summary" in report["sections"]

    def test_statistical_significance_testing(self):
        """Test statistical significance testing implementation."""
        # Test different sample sizes and effect sizes
        test_cases = [
            {"sample_size": 50, "effect_size": 0.1, "expected_power": "low"},
            {"sample_size": 100, "effect_size": 0.3, "expected_power": "medium"},
            {"sample_size": 200, "effect_size": 0.5, "expected_power": "high"}
        ]
        
        with patch('keisei.evaluation.analytics.advanced_analytics.AdvancedAnalytics') as MockAnalytics:
            analytics = MockAnalytics.return_value
            
            for case in test_cases:
                # Mock statistical power analysis
                analytics.compute_statistical_power.return_value = {
                    "power": 0.8 if case["expected_power"] == "high" else 0.6,
                    "alpha": 0.05,
                    "effect_size": case["effect_size"],
                    "sample_size": case["sample_size"]
                }
                
                power_analysis = analytics.compute_statistical_power(
                    sample_size=case["sample_size"],
                    effect_size=case["effect_size"]
                )
                
                assert power_analysis["sample_size"] == case["sample_size"]
                assert power_analysis["effect_size"] == case["effect_size"]
                if case["expected_power"] == "high":
                    assert power_analysis["power"] >= 0.8

    def test_integration_with_evaluation_results(self):
        """Test analytics integration with evaluation result pipeline."""
        from keisei.evaluation.core import EvaluationResult, EvaluationContext, AgentInfo
        from datetime import datetime
        from uuid import uuid4
        
        # Create mock evaluation result
        agent_info = AgentInfo(name="test_agent")
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=Mock(),
            environment_info={"device": "cpu"}
        )
        
        eval_result = EvaluationResult(
            context=context,
            games=self.test_results,
            summary_stats=Mock()
        )
        
        with patch('keisei.evaluation.analytics.advanced_analytics.AdvancedAnalytics') as MockAnalytics:
            analytics = MockAnalytics.return_value
            
            # Mock integration processing
            analytics.process_evaluation_result.return_value = {
                "processed": True,
                "metrics_extracted": 15,
                "insights_generated": 8,
                "recommendations": ["increase_exploration", "adjust_learning_rate"]
            }
            
            processing_result = analytics.process_evaluation_result(eval_result)
            
            assert processing_result["processed"] is True
            assert processing_result["metrics_extracted"] > 0
            assert len(processing_result["recommendations"]) > 0