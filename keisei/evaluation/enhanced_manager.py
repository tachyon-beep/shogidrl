"""
Enhanced Evaluation Manager with Optional Advanced Features.

This module provides an enhanced evaluation manager that extends the base
EvaluationManager with optional advanced features like background tournaments,
advanced analytics, and enhanced opponent management.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .manager import EvaluationManager
from .core import AgentInfo, EvaluationResult, OpponentInfo

logger = logging.getLogger(__name__)


class EnhancedEvaluationManager(EvaluationManager):
    """
    Enhanced evaluation manager with optional advanced features.
    
    Features (when enabled):
    - Background tournament execution
    - Advanced analytics and reporting
    - Enhanced opponent management with adaptive selection
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        config,
        run_name: str,
        pool_size: int = 5,
        elo_registry_path: Optional[str] = None,
        enable_background_tournaments: bool = False,
        enable_advanced_analytics: bool = False,
        enable_enhanced_opponents: bool = False,
        analytics_output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None
    ):
        # Initialize base manager
        super().__init__(config, run_name, pool_size, elo_registry_path)
        
        # Enhanced feature flags
        self.enable_background_tournaments = enable_background_tournaments
        self.enable_advanced_analytics = enable_advanced_analytics
        self.enable_enhanced_opponents = enable_enhanced_opponents
        
        # Storage directory
        self.analytics_output_dir = analytics_output_dir or Path("./analytics_output")
        self.analytics_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced components (optional)
        self.background_tournament_manager = None
        self.advanced_analytics = None
        self.enhanced_opponent_manager = None
        
        # Initialize enhanced features
        self._initialize_enhanced_features(progress_callback)
    
    def _initialize_enhanced_features(self, progress_callback: Optional[Callable] = None):
        """Initialize enhanced features if enabled and available."""
        
        # Background Tournament Manager
        if self.enable_background_tournaments:
            try:
                from .core.background_tournament import BackgroundTournamentManager
                self.background_tournament_manager = BackgroundTournamentManager(
                    max_concurrent_tournaments=2,
                    progress_callback=progress_callback or self._default_progress_callback,
                    result_storage_dir=self.analytics_output_dir / "tournaments"
                )
                logger.info("Background tournament manager initialized")
            except ImportError as e:
                logger.warning(f"Background tournaments not available: {e}")
                self.enable_background_tournaments = False
        
        # Advanced Analytics
        if self.enable_advanced_analytics:
            try:
                from .analytics.advanced_analytics import AdvancedAnalytics
                self.advanced_analytics = AdvancedAnalytics(
                    significance_level=0.05,
                    min_practical_difference=0.05
                )
                logger.info("Advanced analytics initialized")
            except ImportError as e:
                logger.warning(f"Advanced analytics not available: {e}")
                self.enable_advanced_analytics = False
        
        # Enhanced Opponent Manager
        if self.enable_enhanced_opponents:
            try:
                from .opponents.enhanced_manager import EnhancedOpponentManager
                opponent_data_file = self.analytics_output_dir / "opponent_performance_data.json"
                self.enhanced_opponent_manager = EnhancedOpponentManager(
                    opponent_data_file=opponent_data_file,
                    target_win_rate=0.55
                )
                logger.info("Enhanced opponent manager initialized")
            except ImportError as e:
                logger.warning(f"Enhanced opponent management not available: {e}")
                self.enable_enhanced_opponents = False
    
    def _default_progress_callback(self, progress):
        """Default progress callback for tournament updates."""
        logger.info(
            f"Tournament {progress.tournament_id}: "
            f"{progress.completion_percentage:.1f}% complete "
            f"({progress.completed_games}/{progress.total_games} games)"
        )
        
        if progress.is_complete:
            logger.info(f"Tournament {progress.tournament_id} completed with status: {progress.status.value}")
    
    async def start_background_tournament(
        self,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
        tournament_name: Optional[str] = None,
        num_games_per_opponent: int = 2
    ) -> Optional[str]:
        """
        Start a background tournament.
        
        Args:
            agent_info: Agent to evaluate
            opponents: List of opponents
            tournament_name: Optional tournament name
            num_games_per_opponent: Games per opponent
            
        Returns:
            Tournament ID if started successfully, None otherwise
        """
        if not self.background_tournament_manager:
            logger.warning("Background tournaments not available")
            return None
        
        try:
            # Create tournament config
            tournament_config = self.config
            tournament_config.num_games_per_opponent = num_games_per_opponent
            
            tournament_id = await self.background_tournament_manager.start_tournament(
                tournament_config=tournament_config,
                agent_info=agent_info,
                opponents=opponents,
                tournament_name=tournament_name
            )
            
            logger.info(f"Started background tournament: {tournament_id}")
            return tournament_id
            
        except Exception as e:
            logger.error(f"Failed to start background tournament: {e}")
            return None
    
    def get_tournament_progress(self, tournament_id: str):
        """Get progress for a specific tournament."""
        if not self.background_tournament_manager:
            return None
        
        return self.background_tournament_manager.get_tournament_progress(tournament_id)
    
    def list_active_tournaments(self):
        """List all active tournaments."""
        if not self.background_tournament_manager:
            return []
        
        return self.background_tournament_manager.list_active_tournaments()
    
    async def cancel_tournament(self, tournament_id: str) -> bool:
        """Cancel a running tournament."""
        if not self.background_tournament_manager:
            return False
        
        return await self.background_tournament_manager.cancel_tournament(tournament_id)
    
    def generate_analysis_report(
        self,
        current_results: EvaluationResult,
        baseline_results: Optional[EvaluationResult] = None,
        historical_data: Optional[List] = None,
        output_file: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive analysis report.
        
        Args:
            current_results: Current evaluation results
            baseline_results: Optional baseline for comparison
            historical_data: Optional historical data for trend analysis
            output_file: Optional output file path
            
        Returns:
            Analysis report dict if successful, None otherwise
        """
        if not self.advanced_analytics:
            logger.warning("Advanced analytics not available")
            return None
        
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.analytics_output_dir / f"analysis_report_{timestamp}.json"
            
            report = self.advanced_analytics.generate_automated_report(
                current_results=current_results,
                baseline_results=baseline_results,
                historical_data=historical_data,
                output_file=output_file
            )
            
            logger.info(f"Analysis report generated: {output_file}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            return None
    
    def compare_performance(
        self,
        baseline_results: EvaluationResult,
        comparison_results: EvaluationResult,
        baseline_name: str = "Baseline",
        comparison_name: str = "Current"
    ):
        """
        Compare performance between two sets of results.
        
        Returns statistical comparison if advanced analytics available.
        """
        if not self.advanced_analytics:
            logger.warning("Advanced analytics not available for performance comparison")
            return None
        
        try:
            comparison = self.advanced_analytics.compare_performance(
                baseline_results.games,
                comparison_results.games,
                baseline_name,
                comparison_name
            )
            
            logger.info(f"Performance comparison completed: {comparison.recommendation}")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare performance: {e}")
            return None
    
    def register_opponents_for_enhanced_selection(self, opponents: List[OpponentInfo]):
        """Register opponents for enhanced selection strategies."""
        if not self.enhanced_opponent_manager:
            logger.warning("Enhanced opponent management not available")
            return
        
        try:
            self.enhanced_opponent_manager.register_opponents(opponents)
            logger.info(f"Registered {len(opponents)} opponents for enhanced selection")
        except Exception as e:
            logger.error(f"Failed to register opponents: {e}")
    
    def select_adaptive_opponent(
        self,
        current_win_rate: Optional[float] = None,
        strategy: Optional[str] = None
    ) -> Optional[OpponentInfo]:
        """
        Select opponent using enhanced adaptive strategies.
        
        Args:
            current_win_rate: Current agent win rate
            strategy: Selection strategy ("adaptive", "curriculum", "diversity", etc.)
            
        Returns:
            Selected opponent or None if not available
        """
        if not self.enhanced_opponent_manager:
            logger.warning("Enhanced opponent management not available")
            return None
        
        try:
            from .opponents.enhanced_manager import SelectionStrategy
            
            # Map string strategies to enum
            strategy_map = {
                "random": SelectionStrategy.RANDOM,
                "elo": SelectionStrategy.ELO_BASED,
                "adaptive": SelectionStrategy.ADAPTIVE_DIFFICULTY,
                "curriculum": SelectionStrategy.CURRICULUM_LEARNING,
                "diversity": SelectionStrategy.DIVERSITY_MAXIMIZING,
            }
            
            selection_strategy = strategy_map.get(strategy, SelectionStrategy.ADAPTIVE_DIFFICULTY)
            
            opponent = self.enhanced_opponent_manager.select_opponent(
                strategy=selection_strategy,
                agent_current_win_rate=current_win_rate
            )
            
            if opponent:
                logger.debug(f"Selected opponent: {opponent.name} using strategy: {strategy}")
            
            return opponent
            
        except Exception as e:
            logger.error(f"Failed to select adaptive opponent: {e}")
            return None
    
    def update_opponent_performance(self, game_result):
        """Update opponent performance data."""
        if not self.enhanced_opponent_manager:
            return
        
        try:
            self.enhanced_opponent_manager.update_performance(game_result)
        except Exception as e:
            logger.error(f"Failed to update opponent performance: {e}")
    
    def get_opponent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive opponent statistics."""
        if not self.enhanced_opponent_manager:
            return {"enhanced_features": False}
        
        try:
            stats = self.enhanced_opponent_manager.get_opponent_statistics()
            stats["enhanced_features"] = True
            return stats
        except Exception as e:
            logger.error(f"Failed to get opponent statistics: {e}")
            return {"enhanced_features": False, "error": str(e)}
    
    async def shutdown(self):
        """Shutdown enhanced components."""
        if self.background_tournament_manager:
            try:
                await self.background_tournament_manager.shutdown()
                logger.info("Background tournament manager shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down background tournaments: {e}")
    
    def get_enhancement_status(self) -> Dict[str, bool]:
        """Get status of all enhanced features."""
        return {
            "background_tournaments": self.enable_background_tournaments,
            "advanced_analytics": self.enable_advanced_analytics,
            "enhanced_opponents": self.enable_enhanced_opponents,
            "analytics_output_dir": str(self.analytics_output_dir)
        }
