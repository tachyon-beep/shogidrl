"""
Test suite for enhanced evaluation features.

Tests the optional advanced features including background tournaments,
advanced analytics, and enhanced opponent management.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
from keisei.evaluation.core.background_tournament import BackgroundTournamentManager, TournamentStatus
from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
from keisei.evaluation.opponents.enhanced_manager import EnhancedOpponentManager, SelectionStrategy

from keisei.evaluation.core import (
    AgentInfo,
    EvaluationResult,
    GameResult,
    OpponentInfo,
    SummaryStats,
    create_evaluation_config,
    EvaluationStrategy
)


@pytest.fixture
def temp_analytics_dir():
    """Create temporary directory for analytics output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_evaluation_config():
    """Create mock evaluation configuration."""
    return create_evaluation_config(
        strategy=EvaluationStrategy.TOURNAMENT,
        num_games=4,
        num_games_per_opponent=2
    )


@pytest.fixture
def mock_agent_info():
    """Create mock agent info."""
    return AgentInfo(
        name="TestAgent",
        checkpoint_path="/path/to/agent.ptk",
        model_type="ppo_agent"
    )


@pytest.fixture
def mock_opponents():
    """Create mock opponents."""
    return [
        OpponentInfo(name="Opponent1", type="random"),
        OpponentInfo(name="Opponent2", type="heuristic"),
        OpponentInfo(name="Opponent3", type="ppo_agent", checkpoint_path="/path/to/opp3.ptk")
    ]


@pytest.fixture
def sample_game_results(mock_agent_info, mock_opponents):
    """Create sample game results for testing."""
    results = []
    for i, opponent in enumerate(mock_opponents):
        for game_num in range(2):
            result = GameResult(
                game_id=f"game_{i}_{game_num}",
                agent_info=mock_agent_info,
                opponent_info=opponent,
                winner=game_num % 2,  # Alternating wins
                moves_count=30 + i * 5,
                duration_seconds=60.0 + i * 10,
                metadata={"test_game": True}
            )
            results.append(result)
    return results


@pytest.fixture
def sample_evaluation_result(sample_game_results, mock_agent_info):
    """Create sample evaluation result."""
    from keisei.evaluation.core.evaluation_context import EvaluationContext
    from datetime import datetime
    
    # Create evaluation context with all required parameters
    context = EvaluationContext(
        session_id="test_session",
        timestamp=datetime.now(),
        agent_info=mock_agent_info,
        configuration=None,  # Can be None for test
        environment_info={}  # Required empty dict
    )
    
    summary_stats = SummaryStats.from_games(sample_game_results)
    return EvaluationResult(
        context=context,
        games=sample_game_results,
        summary_stats=summary_stats,
        analytics_data={"test": "data"},
        errors=[]
    )


class TestEnhancedEvaluationManager:
    """Test enhanced evaluation manager functionality."""
    
    def test_enhanced_manager_initialization(
        self, 
        mock_evaluation_config, 
        temp_analytics_dir
    ):
        """Test enhanced manager initialization with all features."""
        manager = EnhancedEvaluationManager(
            config=mock_evaluation_config,
            run_name="test_enhanced",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=True,
            analytics_output_dir=temp_analytics_dir
        )
        
        assert manager.enable_background_tournaments
        assert manager.enable_advanced_analytics
        assert manager.enable_enhanced_opponents
        assert manager.analytics_output_dir == temp_analytics_dir
    
    def test_enhanced_manager_selective_features(self, mock_evaluation_config):
        """Test enhanced manager with selective feature enabling."""
        manager = EnhancedEvaluationManager(
            config=mock_evaluation_config,
            run_name="test_selective",
            enable_background_tournaments=False,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=False
        )
        
        assert not manager.enable_background_tournaments
        assert manager.enable_advanced_analytics
        assert not manager.enable_enhanced_opponents
        assert manager.background_tournament_manager is None
        assert manager.advanced_analytics is not None
        assert manager.enhanced_opponent_manager is None
    
    def test_get_enhancement_status(self, mock_evaluation_config):
        """Test enhancement status reporting."""
        manager = EnhancedEvaluationManager(
            config=mock_evaluation_config,
            run_name="test_status",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=False
        )
        
        status = manager.get_enhancement_status()
        assert status["background_tournaments"] is True
        assert status["advanced_analytics"] is True
        assert status["enhanced_opponents"] is False
        assert "analytics_output_dir" in status


class TestBackgroundTournamentManager:
    """Test background tournament functionality."""
    
    @pytest.mark.asyncio
    async def test_tournament_manager_initialization(self, temp_analytics_dir):
        """Test tournament manager initialization."""
        tournament_dir = temp_analytics_dir / "tournaments"
        manager = BackgroundTournamentManager(
            max_concurrent_tournaments=2,
            result_storage_dir=tournament_dir
        )
        
        assert manager.max_concurrent_tournaments == 2
        assert manager.result_storage_dir == tournament_dir
        assert tournament_dir.exists()
    
    @pytest.mark.asyncio
    async def test_start_tournament_basic(
        self, 
        temp_analytics_dir, 
        mock_evaluation_config,
        mock_agent_info,
        mock_opponents
    ):
        """Test starting a basic tournament."""
        manager = BackgroundTournamentManager(
            result_storage_dir=temp_analytics_dir / "tournaments"
        )
        
        # Mock the tournament evaluator to avoid complex dependencies
        with patch('keisei.evaluation.strategies.tournament.TournamentEvaluator') as mock_evaluator:
            mock_evaluator_instance = MagicMock()
            mock_evaluator.return_value = mock_evaluator_instance
            
            async def mock_play_games(*args):
                return [], []
            
            mock_evaluator_instance._play_games_against_opponent = mock_play_games
            mock_evaluator_instance._calculate_tournament_standings = MagicMock(return_value={})
            
            tournament_id = await manager.start_tournament(
                tournament_config=mock_evaluation_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents,
                tournament_name="test_tournament"
            )
            
            assert tournament_id is not None
            assert "test_tournament" in tournament_id
            assert tournament_id in manager._active_tournaments
            
            # Wait a brief moment for tournament to start
            await asyncio.sleep(0.1)
            
            progress = manager.get_tournament_progress(tournament_id)
            assert progress is not None
            assert progress.tournament_id == tournament_id
    
    @pytest.mark.asyncio
    async def test_tournament_progress_tracking(
        self, 
        temp_analytics_dir,
        mock_evaluation_config,
        mock_agent_info,
        mock_opponents
    ):
        """Test tournament progress tracking."""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append({
                "tournament_id": progress.tournament_id,
                "status": progress.status,
                "completion": progress.completion_percentage
            })
        
        manager = BackgroundTournamentManager(
            result_storage_dir=temp_analytics_dir / "tournaments",
            progress_callback=progress_callback
        )
        
        with patch('keisei.evaluation.strategies.tournament.TournamentEvaluator'):
            tournament_id = await manager.start_tournament(
                tournament_config=mock_evaluation_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents[:1]  # Single opponent for faster test
            )
            
            # Allow some time for progress updates
            await asyncio.sleep(0.2)
            
            assert len(progress_updates) > 0
            assert any(update["tournament_id"] == tournament_id for update in progress_updates)
    
    @pytest.mark.asyncio
    async def test_cancel_tournament(
        self, 
        temp_analytics_dir,
        mock_evaluation_config,
        mock_agent_info,
        mock_opponents
    ):
        """Test tournament cancellation."""
        manager = BackgroundTournamentManager(
            result_storage_dir=temp_analytics_dir / "tournaments"
        )
        
        with patch('keisei.evaluation.strategies.tournament.TournamentEvaluator'):
            tournament_id = await manager.start_tournament(
                tournament_config=mock_evaluation_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents
            )
            
            # Cancel the tournament
            cancelled = await manager.cancel_tournament(tournament_id)
            assert cancelled is True
            
            # Check status
            await asyncio.sleep(0.1)
            progress = manager.get_tournament_progress(tournament_id)
            assert progress.status == TournamentStatus.CANCELLED


class TestAdvancedAnalytics:
    """Test advanced analytics functionality."""
    
    def test_analytics_initialization(self):
        """Test analytics initialization."""
        analytics = AdvancedAnalytics(
            significance_level=0.01,
            min_practical_difference=0.1
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
            comparison_name="Current"
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
        self, 
        sample_evaluation_result, 
        temp_analytics_dir
    ):
        """Test automated report generation."""
        analytics = AdvancedAnalytics()
        
        output_file = temp_analytics_dir / "test_report.json"
        
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result,
            output_file=output_file
        )
        
        assert isinstance(report, dict)
        assert "report_metadata" in report
        assert "current_performance" in report
        assert "advanced_metrics" in report
        assert "insights_and_recommendations" in report
        
        # Check file was created
        assert output_file.exists()
        
        # Validate JSON content
        with open(output_file, 'r') as f:
            saved_report = json.load(f)
        assert saved_report == report


class TestEnhancedOpponentManager:
    """Test enhanced opponent management functionality."""
    
    def test_opponent_manager_initialization(self, temp_analytics_dir):
        """Test opponent manager initialization."""
        data_file = temp_analytics_dir / "opponent_data.json"
        manager = EnhancedOpponentManager(
            opponent_data_file=data_file,
            target_win_rate=0.6
        )
        
        assert manager.opponent_data_file == data_file
        assert abs(manager.target_win_rate - 0.6) < 1e-9
    
    def test_register_opponents(self, temp_analytics_dir, mock_opponents):
        """Test opponent registration."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        
        manager.register_opponents(mock_opponents)
        
        assert len(manager.available_opponents) == len(mock_opponents)
        assert len(manager.opponent_data) == len(mock_opponents)
        
        for opponent in mock_opponents:
            assert opponent.name in manager.opponent_data
    
    def test_opponent_selection_strategies(self, temp_analytics_dir, mock_opponents):
        """Test different opponent selection strategies."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        manager.register_opponents(mock_opponents)
        
        # Test different strategies
        strategies = [
            SelectionStrategy.RANDOM,
            SelectionStrategy.ADAPTIVE_DIFFICULTY,
            SelectionStrategy.DIVERSITY_MAXIMIZING
        ]
        
        for strategy in strategies:
            selected = manager.select_opponent(
                strategy=strategy,
                agent_current_win_rate=0.6
            )
            assert selected is not None
            assert selected in mock_opponents
    
    def test_performance_tracking(
        self, 
        temp_analytics_dir, 
        mock_opponents, 
        mock_agent_info
    ):
        """Test opponent performance tracking."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        manager.register_opponents(mock_opponents)
        
        # Create and update with game result
        opponent = mock_opponents[0]
        game_result = GameResult(
            game_id="test_game",
            agent_info=mock_agent_info,
            opponent_info=opponent,
            winner=0,  # Agent wins
            moves_count=25,
            duration_seconds=45.0,
            metadata={}
        )
        
        initial_games = manager.opponent_data[opponent.name].total_games
        initial_wins = manager.opponent_data[opponent.name].wins_against
        
        manager.update_performance(game_result)
        
        assert manager.opponent_data[opponent.name].total_games == initial_games + 1
        assert manager.opponent_data[opponent.name].wins_against == initial_wins + 1
        assert manager.opponent_data[opponent.name].last_played is not None
    
    def test_opponent_statistics(self, temp_analytics_dir, mock_opponents):
        """Test opponent statistics generation."""
        manager = EnhancedOpponentManager(
            opponent_data_file=temp_analytics_dir / "opponent_data.json"
        )
        manager.register_opponents(mock_opponents)
        
        stats = manager.get_opponent_statistics()
        
        assert isinstance(stats, dict)
        assert "total_opponents" in stats
        assert "total_games" in stats
        assert "average_win_rate" in stats
        assert "curriculum_level" in stats
        assert "current_strategy" in stats
        assert stats["total_opponents"] == len(mock_opponents)


class TestIntegrationScenarios:
    """Test integration scenarios with enhanced features."""
    
    @pytest.mark.asyncio
    async def test_full_enhanced_evaluation_workflow(
        self,
        mock_evaluation_config,
        mock_agent_info,
        mock_opponents,
        temp_analytics_dir
    ):
        """Test complete enhanced evaluation workflow."""
        manager = EnhancedEvaluationManager(
            config=mock_evaluation_config,
            run_name="integration_test",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=True,
            analytics_output_dir=temp_analytics_dir
        )
        
        # Register opponents for enhanced selection
        manager.register_opponents_for_enhanced_selection(mock_opponents)
        
        # Select adaptive opponent
        selected_opponent = manager.select_adaptive_opponent(
            current_win_rate=0.6,
            strategy="adaptive"
        )
        assert selected_opponent is not None
        
        # Start background tournament
        with patch('keisei.evaluation.core.background_tournament.TournamentEvaluator') as mock_evaluator:
            # Configure the mock to simulate tournament behavior with delay
            from unittest.mock import AsyncMock
            mock_instance = mock_evaluator.return_value
            
            # Add delay to simulate longer running tournament
            async def delayed_games(*args, **kwargs):
                await asyncio.sleep(0.3)  # Delay to keep tournament running
                return ([], [])
            
            mock_instance._play_games_against_opponent = AsyncMock(side_effect=delayed_games)
            mock_instance._calculate_tournament_standings = lambda *args: {}
            
            tournament_id = await manager.start_background_tournament(
                agent_info=mock_agent_info,
                opponents=mock_opponents[:2],  # Smaller tournament for faster test
                tournament_name="integration_test"
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
            agent_info=mock_agent_info,
            configuration=mock_evaluation_config,  # Use the mock config
            environment_info={}
        )
        
        sample_result = EvaluationResult(
            context=test_context,
            games=[],
            summary_stats=SummaryStats.from_games([]),
            errors=[],
            analytics_data={}
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
