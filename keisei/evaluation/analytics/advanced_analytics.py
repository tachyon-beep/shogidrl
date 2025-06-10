"""
Advanced Analytics Pipeline for Keisei Evaluation System.

Provides statistical analysis, trend detection, and automated reporting
capabilities for evaluation results and tournament data.
"""

import json
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional scipy import for advanced statistical tests
try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..core import EvaluationResult, GameResult
from .performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Result of a statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str
    effect_size: Optional[float] = None


@dataclass
class TrendAnalysis:
    """Analysis of performance trends over time."""

    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1, where 1 is strongest trend
    slope: float
    r_squared: float
    data_points: int
    time_span_days: float
    prediction_next_week: Optional[float] = None


@dataclass
class PerformanceComparison:
    """Comparison between two sets of results."""

    baseline_name: str
    comparison_name: str
    win_rate_difference: float
    win_rate_change_percent: float
    statistical_tests: List[StatisticalTest]
    practical_significance: bool
    confidence_interval: Tuple[float, float]
    recommendation: str


class AdvancedAnalytics:
    """
    Advanced analytics pipeline for evaluation results.

    Features:
    - Statistical significance testing
    - Trend analysis over time
    - Performance comparison with confidence intervals
    - Automated insights and recommendations
    - Report generation
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_practical_difference: float = 0.05,  # 5% win rate difference
        trend_window_days: int = 30,
    ):
        self.significance_level = significance_level
        self.min_practical_difference = min_practical_difference
        self.trend_window_days = trend_window_days

    def compare_performance(
        self,
        baseline_results: List[GameResult],
        comparison_results: List[GameResult],
        baseline_name: str = "Baseline",
        comparison_name: str = "Comparison",
    ) -> PerformanceComparison:
        """
        Compare performance between two sets of results with statistical analysis.
        """
        # Calculate basic metrics
        baseline_wins = sum(1 for game in baseline_results if game.is_agent_win)
        baseline_total = len(baseline_results)
        baseline_win_rate = baseline_wins / baseline_total if baseline_total > 0 else 0

        comparison_wins = sum(1 for game in comparison_results if game.is_agent_win)
        comparison_total = len(comparison_results)
        comparison_win_rate = (
            comparison_wins / comparison_total if comparison_total > 0 else 0
        )

        win_rate_difference = comparison_win_rate - baseline_win_rate
        win_rate_change_percent = (
            (win_rate_difference / baseline_win_rate * 100)
            if baseline_win_rate > 0
            else float("inf")
        )

        # Statistical tests
        statistical_tests = []

        # Two-proportion z-test
        if baseline_total > 0 and comparison_total > 0:
            z_test = self._two_proportion_z_test(
                baseline_wins, baseline_total, comparison_wins, comparison_total
            )
            statistical_tests.append(z_test)

        # Mann-Whitney U test (for game lengths or other metrics)
        if len(baseline_results) > 0 and len(comparison_results) > 0:
            baseline_lengths = [game.moves_count for game in baseline_results]
            comparison_lengths = [game.moves_count for game in comparison_results]

            mw_test = self._mann_whitney_test(
                baseline_lengths, comparison_lengths, "game_length"
            )
            statistical_tests.append(mw_test)

        # Confidence interval for win rate difference
        confidence_interval = self._calculate_win_rate_difference_ci(
            baseline_wins, baseline_total, comparison_wins, comparison_total
        )

        # Determine practical significance
        practical_significance = (
            abs(win_rate_difference) >= self.min_practical_difference
        )

        # Generate recommendation
        recommendation = self._generate_performance_recommendation(
            win_rate_difference, statistical_tests, practical_significance
        )

        return PerformanceComparison(
            baseline_name=baseline_name,
            comparison_name=comparison_name,
            win_rate_difference=win_rate_difference,
            win_rate_change_percent=win_rate_change_percent,
            statistical_tests=statistical_tests,
            practical_significance=practical_significance,
            confidence_interval=confidence_interval,
            recommendation=recommendation,
        )

    def analyze_trends(
        self,
        historical_results: List[Tuple[datetime, EvaluationResult]],
        metric: str = "win_rate",
    ) -> TrendAnalysis:
        """
        Analyze performance trends over time.
        """
        if len(historical_results) < 3:
            return TrendAnalysis(
                metric_name=metric,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                slope=0.0,
                r_squared=0.0,
                data_points=len(historical_results),
                time_span_days=0.0,
            )

        # Extract time series data
        timestamps = [timestamp for timestamp, _ in historical_results]
        values = []

        for timestamp, result in historical_results:
            if metric == "win_rate":
                values.append(result.summary_stats.win_rate)
            elif metric == "avg_game_length":
                values.append(result.summary_stats.avg_game_length)
            elif metric == "total_games":
                values.append(result.summary_stats.total_games)
            else:
                logger.warning(f"Unknown metric: {metric}")
                values.append(0.0)

        # Convert timestamps to days since first measurement
        first_timestamp = timestamps[0]
        days = [(ts - first_timestamp).total_seconds() / 86400 for ts in timestamps]

        # Linear regression - using basic implementation if scipy not available
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                days, values
            )
            r_squared = r_value**2
        else:
            # Simple linear regression implementation
            n = len(days)
            sum_x = sum(days)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(days, values))
            sum_x2 = sum(x * x for x in days)

            if n * sum_x2 - sum_x * sum_x == 0:
                slope = 0.0
                intercept = sum_y / n if n > 0 else 0.0
                r_squared = 0.0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n

                # Calculate r-squared
                y_mean = sum_y / n
                ss_tot = sum((y - y_mean) ** 2 for y in values)
                ss_res = sum(
                    (values[i] - (slope * days[i] + intercept)) ** 2 for i in range(n)
                )
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Determine trend direction and strength
        if abs(slope) < 1e-6:
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "increasing"
            trend_strength = min(r_squared, 1.0)
        else:
            trend_direction = "decreasing"
            trend_strength = min(r_squared, 1.0)

        # Time span
        time_span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400

        # Prediction for next week (7 days ahead)
        prediction_next_week = None
        if r_squared > 0.3:  # Only predict if trend is reasonably strong
            next_week_days = days[-1] + 7
            prediction_next_week = slope * next_week_days + intercept

        return TrendAnalysis(
            metric_name=metric,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            data_points=len(historical_results),
            time_span_days=time_span_days,
            prediction_next_week=prediction_next_week,
        )

    def _two_proportion_z_test(
        self, x1: int, n1: int, x2: int, n2: int
    ) -> StatisticalTest:
        """Perform two-proportion z-test."""
        if n1 == 0 or n2 == 0:
            return StatisticalTest(
                test_name="two_proportion_z_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=1 - self.significance_level,
                interpretation="Insufficient data for test",
            )

        p1 = x1 / n1
        p2 = x2 / n2

        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

        # Z-statistic
        if se == 0:
            z_stat = 0.0
        else:
            z_stat = (p2 - p1) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

        is_significant = p_value < self.significance_level

        # Effect size (Cohen's h)
        effect_size = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

        interpretation = self._interpret_z_test(z_stat, p_value, is_significant)

        return StatisticalTest(
            test_name="two_proportion_z_test",
            statistic=z_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=1 - self.significance_level,
            interpretation=interpretation,
            effect_size=abs(effect_size),
        )

    def _mann_whitney_test(
        self, sample1: List[float], sample2: List[float], metric_name: str
    ) -> StatisticalTest:
        """Perform Mann-Whitney U test."""
        if len(sample1) < 3 or len(sample2) < 3:
            return StatisticalTest(
                test_name=f"mann_whitney_{metric_name}",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=1 - self.significance_level,
                interpretation="Insufficient data for Mann-Whitney test",
            )

        try:
            statistic, p_value = scipy_stats.mannwhitneyu(
                sample1, sample2, alternative="two-sided"
            )

            is_significant = p_value < self.significance_level

            interpretation = f"Mann-Whitney U test for {metric_name}: "
            if is_significant:
                interpretation += f"Significant difference detected (p={p_value:.4f})"
            else:
                interpretation += f"No significant difference (p={p_value:.4f})"

            return StatisticalTest(
                test_name=f"mann_whitney_{metric_name}",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=1 - self.significance_level,
                interpretation=interpretation,
            )

        except Exception as e:
            logger.error(f"Mann-Whitney test failed: {e}")
            return StatisticalTest(
                test_name=f"mann_whitney_{metric_name}",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=1 - self.significance_level,
                interpretation=f"Test failed: {str(e)}",
            )

    def _calculate_win_rate_difference_ci(
        self, x1: int, n1: int, x2: int, n2: int, confidence_level: float = None
    ) -> Tuple[float, float]:
        """Calculate confidence interval for win rate difference."""
        if confidence_level is None:
            confidence_level = 1 - self.significance_level

        if n1 == 0 or n2 == 0:
            return (0.0, 0.0)

        p1 = x1 / n1
        p2 = x2 / n2

        # Standard error for difference
        se1 = math.sqrt(p1 * (1 - p1) / n1)
        se2 = math.sqrt(p2 * (1 - p2) / n2)
        se_diff = math.sqrt(se1**2 + se2**2)

        # Critical value
        alpha = 1 - confidence_level
        z_critical = scipy_stats.norm.ppf(1 - alpha / 2)

        # Confidence interval
        diff = p2 - p1
        margin_error = z_critical * se_diff

        return (diff - margin_error, diff + margin_error)

    def _interpret_z_test(
        self, z_stat: float, p_value: float, is_significant: bool
    ) -> str:
        """Generate interpretation for z-test results."""
        interpretation = f"Two-proportion z-test: z={z_stat:.3f}, p={p_value:.4f}. "

        if is_significant:
            if z_stat > 0:
                interpretation += "Significant improvement in win rate."
            else:
                interpretation += "Significant decrease in win rate."
        else:
            interpretation += "No statistically significant difference in win rates."

        return interpretation

    def _generate_performance_recommendation(
        self,
        win_rate_difference: float,
        statistical_tests: List[StatisticalTest],
        practical_significance: bool,
    ) -> str:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Check statistical significance
        significant_tests = [test for test in statistical_tests if test.is_significant]

        if not significant_tests:
            recommendations.append("No statistically significant changes detected.")

        if practical_significance:
            if win_rate_difference > 0:
                recommendations.append(
                    f"Meaningful improvement detected (+{win_rate_difference:.1%}). "
                    "Consider adopting this approach."
                )
            else:
                recommendations.append(
                    f"Meaningful performance decrease detected ({win_rate_difference:.1%}). "
                    "Consider reverting or investigating the cause."
                )
        else:
            recommendations.append(
                "Changes are too small to be practically significant. "
                "Continue monitoring or increase sample size."
            )

        # Add specific test recommendations
        for test in significant_tests:
            if test.effect_size and test.effect_size > 0.5:
                recommendations.append(
                    f"Large effect size detected in {test.test_name} (d={test.effect_size:.2f}). "
                    "This change is likely meaningful."
                )

        return " ".join(recommendations)

    def generate_automated_report(
        self,
        current_results: EvaluationResult,
        baseline_results: Optional[EvaluationResult] = None,
        historical_data: Optional[List[Tuple[datetime, EvaluationResult]]] = None,
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive automated analysis report.
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_type": "comprehensive_evaluation_analysis",
                "keisei_version": "1.0.0",
            },
            "current_performance": {
                "total_games": current_results.summary_stats.total_games,
                "win_rate": current_results.summary_stats.win_rate,
                "wins": current_results.summary_stats.agent_wins,
                "losses": current_results.summary_stats.opponent_wins,
                "draws": current_results.summary_stats.draws,
                "avg_game_length": current_results.summary_stats.avg_game_length,
            },
        }

        # Advanced performance analysis
        analyzer = PerformanceAnalyzer(current_results)
        streaks = analyzer.calculate_win_loss_draw_streaks()
        game_lengths = analyzer.analyze_game_length_distribution()

        report["advanced_metrics"] = {
            "streaks": streaks,
            "game_length_analysis": game_lengths,
        }

        # Comparison analysis if baseline provided
        if baseline_results:
            comparison = self.compare_performance(
                baseline_results.games, current_results.games, "Baseline", "Current"
            )

            report["performance_comparison"] = {
                "win_rate_difference": comparison.win_rate_difference,
                "win_rate_change_percent": comparison.win_rate_change_percent,
                "practical_significance": comparison.practical_significance,
                "confidence_interval": comparison.confidence_interval,
                "statistical_tests": [
                    {
                        "test_name": test.test_name,
                        "p_value": test.p_value,
                        "is_significant": test.is_significant,
                        "interpretation": test.interpretation,
                    }
                    for test in comparison.statistical_tests
                ],
                "recommendation": comparison.recommendation,
            }

        # Trend analysis if historical data provided
        if historical_data:
            win_rate_trend = self.analyze_trends(historical_data, "win_rate")
            game_length_trend = self.analyze_trends(historical_data, "avg_game_length")

            report["trend_analysis"] = {
                "win_rate_trend": {
                    "direction": win_rate_trend.trend_direction,
                    "strength": win_rate_trend.trend_strength,
                    "r_squared": win_rate_trend.r_squared,
                    "prediction_next_week": win_rate_trend.prediction_next_week,
                },
                "game_length_trend": {
                    "direction": game_length_trend.trend_direction,
                    "strength": game_length_trend.trend_strength,
                    "r_squared": game_length_trend.r_squared,
                },
            }

        # Generate insights and recommendations
        insights = self._generate_automated_insights(report)
        report["insights_and_recommendations"] = insights

        # Save report if output file specified
        if output_file:
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Analysis report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")

        return report

    def _generate_automated_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate automated insights from analysis report."""
        insights = []

        current_perf = report["current_performance"]

        # Performance level insights
        win_rate = current_perf["win_rate"]
        if win_rate > 0.7:
            insights.append("🎉 Excellent performance! Win rate above 70%.")
        elif win_rate > 0.55:
            insights.append("✅ Good performance. Win rate shows positive results.")
        elif win_rate > 0.45:
            insights.append("⚖️ Balanced performance. Close to even win/loss ratio.")
        else:
            insights.append("⚠️ Performance below average. Consider model adjustments.")

        # Game length insights
        avg_length = current_perf["avg_game_length"]
        if avg_length > 100:
            insights.append(
                "🔍 Games are running long. Consider investigating efficiency."
            )
        elif avg_length < 20:
            insights.append("⚡ Games are very short. Check for early terminations.")

        # Streak analysis insights
        if "advanced_metrics" in report and "streaks" in report["advanced_metrics"]:
            streaks = report["advanced_metrics"]["streaks"]
            max_win_streak = streaks.get("max_win_streak", 0)
            max_loss_streak = streaks.get("max_loss_streak", 0)

            if max_win_streak > 5:
                insights.append(f"🔥 Impressive win streak of {max_win_streak} games!")
            if max_loss_streak > 5:
                insights.append(
                    f"📉 Concerning loss streak of {max_loss_streak} games."
                )

        # Comparison insights
        if "performance_comparison" in report:
            comp = report["performance_comparison"]
            change_percent = comp.get("win_rate_change_percent", 0)

            if comp.get("practical_significance", False):
                if change_percent > 10:
                    insights.append("📈 Significant improvement detected!")
                elif change_percent < -10:
                    insights.append("📉 Significant performance decline detected.")

        # Trend insights
        if "trend_analysis" in report:
            trend = report["trend_analysis"]["win_rate_trend"]
            direction = trend.get("direction", "unknown")
            strength = trend.get("strength", 0)

            if direction == "increasing" and strength > 0.5:
                insights.append("📊 Strong upward trend in performance!")
            elif direction == "decreasing" and strength > 0.5:
                insights.append("📊 Concerning downward trend detected.")

        if not insights:
            insights.append("📋 Performance appears stable. Continue monitoring.")

        return insights
