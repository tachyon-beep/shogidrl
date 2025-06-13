# CONSOLIDATION COMPLETE: All enhanced feature tests (advanced analytics, background tournaments, enhanced opponent management) are now centralized here.
# This is the canonical file for all enhanced evaluation feature tests. Do not add enhanced feature tests elsewhere.
# Last consolidation: 2025-06-13
#
# test_evaluate_main.py and test_tournament_evaluator.py have been updated to remove any duplicate enhanced feature coverage.
#
# Next: Update EVAL_REMEDIATION_PLAN.md to reflect this consolidation.
"""
Test suite for enhanced evaluation features.

Tests the optional advanced features including background tournaments,
advanced analytics, and enhanced opponent management.

REFACTORED: Reduced mock usage and implemented behavior-driven testing patterns.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
from keisei.evaluation.core import (
    AgentInfo,
    EvaluationResult,
    EvaluationStrategy,
    GameResult,
    OpponentInfo,
    SummaryStats,
    create_evaluation_config,
)
from keisei.evaluation.core.background_tournament import (
    BackgroundTournamentManager,
    TournamentStatus,
)
from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
from keisei.evaluation.opponents.enhanced_manager import (
    EnhancedOpponentManager,
    SelectionStrategy,
)
from tests.evaluation.factories import EvaluationTestFactory, EvaluationScenarioFactory


@pytest.fixture
def temp_analytics_dir():
    """Create temporary directory for analytics output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_evaluation_config():
    """Create realistic evaluation configuration using factories."""
    return create_evaluation_config(
        strategy=EvaluationStrategy.TOURNAMENT, 
        num_games=4, 
        wandb_logging=False
        # Note: Removed num_games_per_opponent as it's not a valid base config parameter
    )


@pytest.fixture
def test_agent_info():
    """Create realistic agent info using factories."""
    return EvaluationTestFactory.create_test_agent_info(
        name="TestAgent", 
        agent_type="ppo_agent"
    )


@pytest.fixture
def test_opponents():
    """Create realistic opponents using factories."""
    return [
        EvaluationTestFactory.create_test_opponent_info(name="Opponent1", opponent_type="random"),
        EvaluationTestFactory.create_test_opponent_info(name="Opponent2", opponent_type="heuristic"),
        EvaluationTestFactory.create_test_opponent_info(
            name="Opponent3", 
            opponent_type="ppo", 
            checkpoint_path="/path/to/opp3.ptk"
        ),
    ]


@pytest.fixture
def sample_game_results():
    """Create sample game results using factories."""
    return EvaluationTestFactory.create_test_game_results(
        count=6,  # 2 per opponent 
        win_rate=0.5  # Balanced results
    )


@pytest.fixture
def sample_evaluation_result(sample_game_results, test_agent_info):
    """Create sample evaluation result using factories."""
    context = EvaluationTestFactory.create_test_evaluation_context(
        agent_info=test_agent_info
    )
    summary_stats = SummaryStats.from_games(sample_game_results)
    return EvaluationResult(
        context=context,
        games=sample_game_results,
        summary_stats=summary_stats,
        analytics_data={"test": "data"},
        errors=[],
    )


class TestEnhancedEvaluationManager:
    """Test enhanced evaluation manager functionality."""

    def test_enhanced_manager_initialization(
        self, test_evaluation_config, temp_analytics_dir
    ):
        """Test enhanced manager initialization with all features."""
        manager = EnhancedEvaluationManager(
            config=test_evaluation_config,
            run_name="test_enhanced",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=True,
            analytics_output_dir=temp_analytics_dir,
        )

        assert manager.enable_background_tournaments
        assert manager.enable_advanced_analytics
        assert manager.enable_enhanced_opponents
        assert manager.analytics_output_dir == temp_analytics_dir

    def test_enhanced_manager_selective_features(self, test_evaluation_config):
        """Test enhanced manager with selective feature enabling."""
        manager = EnhancedEvaluationManager(
            config=test_evaluation_config,
            run_name="test_selective",
            enable_background_tournaments=False,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=False,
        )

        assert not manager.enable_background_tournaments
        assert manager.enable_advanced_analytics
        assert not manager.enable_enhanced_opponents
        assert manager.background_tournament_manager is None
        assert manager.advanced_analytics is not None
        assert manager.enhanced_opponent_manager is None

    def test_get_enhancement_status(self, test_evaluation_config):
        """Test enhancement status reporting."""
        manager = EnhancedEvaluationManager(
            config=test_evaluation_config,
            run_name="test_status",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=False,
        )

        status = manager.get_enhancement_status()
        assert status["background_tournaments"] is True
        assert status["advanced_analytics"] is True
        assert status["enhanced_opponents"] is False
        assert "analytics_output_dir" in status


class TestAdvancedAnalytics:
    """Test advanced analytics functionality."""

    def test_analytics_initialization(self):
        """Test analytics initialization."""
        analytics = AdvancedAnalytics(
            significance_level=0.01, min_practical_difference=0.1
        )

        assert abs(analytics.significance_level - 0.01) < 1e-9
        assert abs(analytics.min_practical_difference - 0.1) < 1e-9

    def test_performance_comparison(self, sample_game_results):
        """Test performance comparison between result sets."""
        analytics = AdvancedAnalytics()

        # Split results into baseline and comparison
        baseline_results = sample_game_results[:3]
        comparison_results = sample_game_results[3:]

        comparison = analytics.compare_performance(
            baseline_results=baseline_results,
            comparison_results=comparison_results,
            baseline_name="Baseline",
            comparison_name="Current",
        )

        assert comparison.baseline_name == "Baseline"
        assert comparison.comparison_name == "Current"
        assert isinstance(comparison.win_rate_difference, float)
        assert isinstance(comparison.statistical_tests, list)
        assert len(comparison.statistical_tests) > 0
        assert isinstance(comparison.confidence_interval, tuple)
        assert len(comparison.confidence_interval) == 2

    def test_trend_analysis(self, sample_evaluation_result):
        """Test trend analysis over time."""
        analytics = AdvancedAnalytics()

        # Create historical data
        historical_data = []
        base_time = datetime.now() - timedelta(days=30)

        for i in range(10):
            timestamp = base_time + timedelta(days=i * 3)
            # Modify win rate slightly for each entry
            modified_result = sample_evaluation_result
            modified_result.summary_stats.wins = 2 + i // 3  # Gradual improvement
            modified_result.summary_stats.total_games = 6
            historical_data.append((timestamp, modified_result))

        trend = analytics.analyze_trends(historical_data, "win_rate")

        assert trend.metric_name == "win_rate"
        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
        assert 0 <= trend.trend_strength <= 1
        assert trend.data_points == len(historical_data)

    def test_automated_report_generation(
        self, sample_evaluation_result, temp_analytics_dir
    ):
        """Test automated report generation."""
        analytics = AdvancedAnalytics()

        output_file = temp_analytics_dir / "test_report.json"

        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result, output_file=output_file
        )

        assert isinstance(report, dict)
        assert "report_metadata" in report
        assert "current_performance" in report
        assert "advanced_metrics" in report
        assert "insights_and_recommendations" in report

        # Check file was created
        assert output_file.exists()

        # Validate JSON content
        with open(output_file, "r") as f:
            saved_report = json.load(f)
        assert saved_report == report


class TestEnhancedOpponentManager:
    """Test enhanced opponent management functionality."""

    def test_opponent_manager_initialization(self, temp_analytics_dir):
        """Test opponent manager initialization."""
        data_file = temp_analytics_dir / "opponent_data.json"
        manager = EnhancedOpponentManager(
            opponent_data_file=data_file, target_win_rate=0.6
        )

        assert manager.opponent_data_file == data_file
        assert abs(manager.target_win_rate - 0.6) < 1e-9

    def test_register_opponents(self, temp_analytics_dir, test_opponents):
        """Test opponent registration."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )

        manager.register_opponents(test_opponents)

        assert len(manager.available_opponents) == len(test_opponents)
        assert len(manager.opponent_data) == len(test_opponents)

        for opponent in test_opponents:
            assert opponent.name in manager.opponent_data

    def test_opponent_selection_strategies(self, temp_analytics_dir, test_opponents):
        """Test different opponent selection strategies."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        manager.register_opponents(test_opponents)

        # Test different strategies
        strategies = [
            SelectionStrategy.RANDOM,
            SelectionStrategy.ADAPTIVE_DIFFICULTY,
            SelectionStrategy.DIVERSITY_MAXIMIZING,
        ]

        for strategy in strategies:
            selected = manager.select_opponent(
                strategy=strategy, agent_current_win_rate=0.6
            )
            assert selected is not None
            assert selected in test_opponents

    def test_performance_tracking(
        self, temp_analytics_dir, test_opponents, test_agent_info
    ):
        """Test opponent performance tracking."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        manager.register_opponents(test_opponents)

        # Create and update with game result
        opponent = test_opponents[0]
        game_result = GameResult(
            game_id="test_game",
            agent_info=test_agent_info,
            opponent_info=opponent,
            winner=0,  # Agent wins
            moves_count=25,
            duration_seconds=45.0,
            metadata={},
        )

        initial_games = manager.opponent_data[opponent.name].total_games
        initial_wins = manager.opponent_data[opponent.name].wins_against

        manager.update_performance(game_result)

        assert manager.opponent_data[opponent.name].total_games == initial_games + 1
        assert manager.opponent_data[opponent.name].wins_against == initial_wins + 1
        assert manager.opponent_data[opponent.name].last_played is not None

    def test_opponent_statistics(self, temp_analytics_dir, test_opponents):
        """Test opponent statistics generation."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        manager.register_opponents(test_opponents)

        stats = manager.get_opponent_statistics()

        assert isinstance(stats, dict)
        assert "total_opponents" in stats
        assert "total_games" in stats
        assert "average_win_rate" in stats
        assert "curriculum_level" in stats
        assert "current_strategy" in stats
        assert stats["total_opponents"] == len(test_opponents)


class TestIntegrationScenarios:
    """Test integration scenarios with enhanced features."""

    @pytest.mark.asyncio
    async def test_full_enhanced_evaluation_workflow(
        self,
        test_evaluation_config,
        test_agent_info,
        test_opponents,
        temp_analytics_dir,
    ):
        """Test complete enhanced evaluation workflow."""
        manager = EnhancedEvaluationManager(
            config=test_evaluation_config,
            run_name="integration_test",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=True,
            analytics_output_dir=temp_analytics_dir,
        )

        # Register opponents for enhanced selection
        manager.register_opponents_for_enhanced_selection(test_opponents)

        # Select adaptive opponent
        selected_opponent = manager.select_adaptive_opponent(
            current_win_rate=0.6, strategy="adaptive"
        )
        assert selected_opponent is not None

        # Start background tournament
        with patch(
            "keisei.evaluation.core.background_tournament.TournamentEvaluator"
        ) as mock_evaluator:
            # Configure the mock to simulate tournament behavior with delay
            from unittest.mock import AsyncMock

            mock_instance = mock_evaluator.return_value

            # Add delay to simulate longer running tournament
            async def delayed_games(*args, **kwargs):
                await asyncio.sleep(0.3)  # Delay to keep tournament running
                return ([], [])

            mock_instance._play_games_against_opponent = AsyncMock(
                side_effect=delayed_games
            )
            mock_instance._calculate_tournament_standings = lambda *args: {}

            tournament_id = await manager.start_background_tournament(
                agent_info=test_agent_info,
                opponents=test_opponents[:2],  # Smaller tournament for faster test
                tournament_name="integration_test",
            )

            assert tournament_id is not None

            # Give the tournament task time to start and update status to RUNNING
            await asyncio.sleep(0.2)

            # Check tournament progress
            progress = manager.get_tournament_progress(tournament_id)
            assert progress is not None

            # List active tournaments - should show the running tournament
            active = manager.list_active_tournaments()
            assert len(active) > 0

            # Cancel tournament for cleanup
            cancelled = await manager.cancel_tournament(tournament_id)
            assert cancelled is True

        # Test analytics report generation
        from keisei.evaluation.core.evaluation_context import EvaluationContext

        test_context = EvaluationContext(
            session_id="test",
            timestamp=datetime.now(),
            agent_info=test_agent_info,
            configuration=test_evaluation_config,  # Use the factory config
            environment_info={},
        )

        sample_result = EvaluationResult(
            context=test_context,
            games=[],
            summary_stats=SummaryStats.from_games([]),
            errors=[],
            analytics_data={},
        )

        report = manager.generate_analysis_report(sample_result)
        assert report is not None

        # Test opponent statistics
        stats = manager.get_opponent_statistics()
        assert stats["enhanced_features"] is True

        # Cleanup
        await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
