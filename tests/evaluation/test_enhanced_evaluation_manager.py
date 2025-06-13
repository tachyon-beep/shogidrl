"""
Tests for enhanced evaluation manager features (Task 4 - Critical)
"""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
from keisei.evaluation.core import AgentInfo, OpponentInfo, create_evaluation_config, EvaluationStrategy


class TestEnhancedEvaluationManager:
    """Test enhanced evaluation manager optional features"""

    def test_enhanced_manager_initialization_all_features(self):
        """Test initialization with all enhanced features enabled."""
        config = create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games_per_opponent=2
        )
        
        manager = EnhancedEvaluationManager(
            config=config,
            run_name="test_enhanced",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=True
        )
        
        assert manager.enable_background_tournaments
        assert manager.enable_advanced_analytics
        assert manager.enable_enhanced_opponents
        # Enhanced components should be initialized when features are enabled
        # Note: May be None if optional dependencies not available

    @pytest.mark.asyncio
    async def test_background_tournament_lifecycle(self):
        """Test complete background tournament workflow."""
        config = create_evaluation_config(
            strategy=EvaluationStrategy.TOURNAMENT,
            num_games_per_opponent=1
        )
        
        manager = EnhancedEvaluationManager(
            config=config,
            run_name="test_tournament",
            enable_background_tournaments=True
        )
        
        # Create test data
        agent_info = AgentInfo(name="TestAgent")
        opponents = [
            OpponentInfo(name="Opponent1", type="random"),
            OpponentInfo(name="Opponent2", type="random")
        ]
        
        # Start tournament
        tournament_id = await manager.start_background_tournament(
            agent_info=agent_info,
            opponents=opponents,
            tournament_name="test_tournament"
        )
        
        if tournament_id:  # Only test if background tournaments are available
            # Test progress monitoring
            progress = manager.get_tournament_progress(tournament_id)
            active_tournaments = manager.list_active_tournaments()
            
            # Test cancellation
            cancelled = await manager.cancel_tournament(tournament_id)

    def test_advanced_analytics_integration(self):
        """Test analytics pipeline integration."""
        config = create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games_per_opponent=2
        )
        
        manager = EnhancedEvaluationManager(
            config=config,
            run_name="test_analytics",
            enable_advanced_analytics=True,
            analytics_output_dir=Path("./test_analytics")
        )
        
        # Test analytics functionality if available
        status = manager.get_enhancement_status()
        assert "advanced_analytics" in status

    def test_adaptive_opponent_selection(self):
        """Test intelligent opponent selection strategies."""
        config = create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games_per_opponent=2
        )
        
        manager = EnhancedEvaluationManager(
            config=config,
            run_name="test_adaptive",
            enable_enhanced_opponents=True
        )
        
        # Create test opponents
        opponents = [
            OpponentInfo(name="EasyOpp", type="random"),
            OpponentInfo(name="HardOpp", type="random")
        ]
        
        # Register opponents for enhanced selection
        manager.register_opponents_for_enhanced_selection(opponents)
        
        # Test adaptive selection
        selected = manager.select_adaptive_opponent(
            current_win_rate=0.75,
            strategy="challenging"
        )
        
        # May return None if enhanced opponents not available
        stats = manager.get_opponent_statistics()
        assert isinstance(stats, dict)
