#!/usr/bin/env python3
"""
Enhanced Evaluation Features Demonstration

This script demonstrates the optional advanced features implemented for the
Keisei evaluation system:
1. Background Tournament System
2. Advanced Analytics Pipeline  
3. Enhanced Opponent Management

Usage: python demo_enhanced_features.py
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main demonstration function."""
    print("🎯 Keisei Enhanced Evaluation Features Demonstration")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing Enhanced Feature Imports...")
    try:
        from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
        from keisei.evaluation.core.background_tournament import BackgroundTournamentManager
        from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics  
        from keisei.evaluation.opponents.enhanced_manager import EnhancedOpponentManager, SelectionStrategy
        from keisei.evaluation.core import (
            create_evaluation_config, 
            EvaluationStrategy, 
            AgentInfo, 
            OpponentInfo, 
            GameResult,
            EvaluationResult,
            SummaryStats
        )
        print("✅ All enhanced features imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test Enhanced Evaluation Manager
    print("\n2. Testing Enhanced Evaluation Manager...")
    try:
        config = create_evaluation_config(
            strategy=EvaluationStrategy.TOURNAMENT,
            num_games=10,
            num_games_per_opponent=2
        )
        
        manager = EnhancedEvaluationManager(
            config=config,
            run_name="demo_enhanced",
            enable_background_tournaments=True,
            enable_advanced_analytics=True,
            enable_enhanced_opponents=True
        )
        
        status = manager.get_enhancement_status()
        print(f"✅ Enhanced manager created with features: {status}")
        
    except Exception as e:
        print(f"❌ Enhanced manager error: {e}")
        return False
    
    # Test Background Tournament Manager
    print("\n3. Testing Background Tournament System...")
    try:
        tournament_manager = BackgroundTournamentManager(
            max_concurrent_tournaments=2,
            result_storage_dir=Path("./demo_tournaments")
        )
        
        print("✅ Background tournament manager created")
        print(f"   - Max concurrent tournaments: {tournament_manager.max_concurrent_tournaments}")
        print(f"   - Storage directory: {tournament_manager.result_storage_dir}")
        
        # Test tournament progress tracking
        from keisei.evaluation.core.background_tournament import TournamentProgress, TournamentStatus
        
        progress = TournamentProgress(
            tournament_id="demo_tournament",
            total_games=20,
            completed_games=12,
            status=TournamentStatus.RUNNING
        )
        
        print(f"   - Sample tournament progress: {progress.completion_percentage:.1f}% complete")
        
    except Exception as e:
        print(f"❌ Background tournament error: {e}")
        return False
    
    # Test Advanced Analytics
    print("\n4. Testing Advanced Analytics...")
    try:
        analytics = AdvancedAnalytics(
            significance_level=0.05,
            min_practical_difference=0.05
        )
        
        # Create sample data for testing
        agent_info = AgentInfo(name="DemoAgent", model_type="ppo_agent", checkpoint_path="/path/to/demo.pth")
        opponent1 = OpponentInfo(name="Opponent1", type="random")
        opponent2 = OpponentInfo(name="Opponent2", type="heuristic")
        
        # Generate sample game results
        baseline_games = []
        current_games = []
        
        for i in range(10):
            # Baseline games (50% win rate)
            baseline_game = GameResult(
                game_id=f"baseline_{i}",
                agent_info=agent_info,
                opponent_info=opponent1,
                winner=0 if i < 5 else 1,
                moves_count=25 + i,
                duration_seconds=45.0,
                metadata={"phase": "baseline"}
            )
            baseline_games.append(baseline_game)
            
            # Current games (70% win rate - improvement)
            current_game = GameResult(
                game_id=f"current_{i}",
                agent_info=agent_info,
                opponent_info=opponent2,
                winner=0 if i < 7 else 1,
                moves_count=20 + i,
                duration_seconds=40.0,
                metadata={"phase": "current"}
            )
            current_games.append(current_game)
        
        # Test performance comparison
        comparison = analytics.compare_performance(
            baseline_results=baseline_games,
            comparison_results=current_games,
            baseline_name="Baseline Model",
            comparison_name="Current Model"
        )
        
        print("✅ Advanced analytics created and tested")
        print(f"   - Win rate difference: {comparison.win_rate_difference:.1%}")
        print(f"   - Practical significance: {comparison.practical_significance}")
        print(f"   - Statistical tests: {len(comparison.statistical_tests)}")
        print(f"   - Recommendation: {comparison.recommendation[:80]}...")
        
        # Test trend analysis
        historical_data = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(10):
            timestamp = base_time + timedelta(days=i * 3)
            
            # Create improving results over time
            games = []
            win_rate = 0.4 + (i * 0.05)  # Gradual improvement from 40% to 85%
            for j in range(5):
                game = GameResult(
                    game_id=f"trend_{i}_{j}",
                    agent_info=agent_info,
                    opponent_info=opponent1,
                    winner=0 if j < (win_rate * 5) else 1,
                    moves_count=30,
                    duration_seconds=50.0,
                    metadata={"trend_test": True}
                )
                games.append(game)
            
            eval_result = EvaluationResult(
                context=EvaluationContext(
                    session_id=f"trend_session_{i}",
                    timestamp=timestamp,
                    agent_info=agent_info,
                    configuration=create_evaluation_config(EvaluationStrategy.SINGLE_OPPONENT),
                    environment_info={}
                ),
                games=games,
                summary_stats=SummaryStats.from_games(games),
                errors=[],
                analytics_data={}
            )
            
            historical_data.append((timestamp, eval_result))
        
        trend = analytics.analyze_trends(historical_data, "win_rate")
        print(f"   - Trend analysis: {trend.trend_direction} (strength: {trend.trend_strength:.2f})")
        
    except Exception as e:
        print(f"❌ Advanced analytics error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Enhanced Opponent Management
    print("\n5. Testing Enhanced Opponent Management...")
    try:
        opponent_manager = EnhancedOpponentManager(
            target_win_rate=0.55,
            difficulty_adaptation_rate=0.1
        )
        
        # Register sample opponents
        opponents = [
            OpponentInfo(name="EasyBot", type="random"),
            OpponentInfo(name="MediumBot", type="heuristic"),
            OpponentInfo(name="HardBot", type="ppo_agent", checkpoint_path="/path/to/hardbot.ptk"),
        ]
        
        opponent_manager.register_opponents(opponents)
        
        print("✅ Enhanced opponent manager created")
        print(f"   - Registered opponents: {len(opponents)}")
        
        # Test different selection strategies
        strategies = [
            SelectionStrategy.RANDOM,
            SelectionStrategy.ADAPTIVE_DIFFICULTY,
            SelectionStrategy.DIVERSITY_MAXIMIZING
        ]
        
        for strategy in strategies:
            selected = opponent_manager.select_opponent(
                strategy=strategy,
                agent_current_win_rate=0.6
            )
            print(f"   - {strategy.value} strategy selected: {selected.name if selected else 'None'}")
        
        # Test performance tracking
        test_game = GameResult(
            game_id="test_tracking",
            agent_info=agent_info,
            opponent_info=opponents[0],
            winner=0,
            moves_count=25,
            duration_seconds=45.0,
            metadata={"tracking_test": True}
        )
        
        opponent_manager.update_performance(test_game)
        stats = opponent_manager.get_opponent_statistics()
        print(f"   - Opponent statistics: {stats['total_opponents']} opponents tracked")
        
    except Exception as e:
        print(f"❌ Enhanced opponent management error: {e}")
        return False
    
    # Test Integration
    print("\n6. Testing Enhanced Manager Integration...")
    try:
        # Register opponents with enhanced manager
        manager.register_opponents_for_enhanced_selection(opponents)
        
        # Select adaptive opponent
        selected = manager.select_adaptive_opponent(
            current_win_rate=0.65,
            strategy="adaptive"
        )
        print(f"✅ Adaptive opponent selected: {selected.name if selected else 'None'}")
        
        # Generate analysis report
        sample_result = EvaluationResult(
            session_id="demo_report",
            games=current_games,
            summary_stats=SummaryStats.from_games(current_games),
            errors=[],
            analytics_data={"demo": True}
        )
        
        report = manager.generate_analysis_report(
            current_results=sample_result,
            baseline_results=EvaluationResult(
                session_id="demo_baseline",
                games=baseline_games,
                summary_stats=SummaryStats.from_games(baseline_games),
                errors=[],
                analytics_data={"demo": True}
            )
        )
        
        if report:
            print("✅ Analysis report generated successfully")
            print(f"   - Report sections: {list(report.keys())}")
            if "insights_and_recommendations" in report:
                insights = report["insights_and_recommendations"]
                print(f"   - Generated insights: {len(insights)} recommendations")
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False
    
    # Performance Summary
    print("\n7. Performance Summary...")
    print("📊 Enhanced Features Performance Benefits:")
    print("   • Background Tournaments: Non-blocking long-running evaluations")
    print("   • Advanced Analytics: Statistical significance testing & trend analysis")  
    print("   • Enhanced Opponents: Adaptive selection & curriculum learning")
    print("   • Automated Reports: Comprehensive analysis with actionable insights")
    
    print("\n🎉 All Enhanced Features Demonstrated Successfully!")
    print("\n📝 Usage Examples:")
    print("   1. EnhancedEvaluationManager for all-in-one enhanced evaluation")
    print("   2. BackgroundTournamentManager for async tournament execution") 
    print("   3. AdvancedAnalytics for statistical analysis and reporting")
    print("   4. EnhancedOpponentManager for adaptive opponent selection")
    
    return True


async def demo_background_tournament():
    """Demonstrate background tournament functionality."""
    print("\n🏆 Background Tournament Demo (Async)")
    
    try:
        from keisei.evaluation.core.background_tournament import BackgroundTournamentManager
        from keisei.evaluation.core import AgentInfo, OpponentInfo, create_evaluation_config, EvaluationStrategy
        
        manager = BackgroundTournamentManager(
            max_concurrent_tournaments=1,
            result_storage_dir=Path("./demo_tournaments")
        )
        
        # Create sample tournament data
        config = create_evaluation_config(
            strategy=EvaluationStrategy.TOURNAMENT,
            num_games_per_opponent=1  # Quick demo
        )
        
        agent = AgentInfo(name="DemoAgent", type="ppo_agent")
        opponents = [
            OpponentInfo(name="QuickOpp1", type="random"),
            OpponentInfo(name="QuickOpp2", type="random")
        ]
        
        print("Starting demo tournament...")
        
        # Note: This would normally start a real tournament
        # For demo purposes, we'll just show the setup
        print("✅ Tournament setup completed (demo mode)")
        print("   - In real usage, this would run games in background")
        print("   - Progress would be monitored via callbacks")
        print("   - Results would be saved automatically")
        
    except Exception as e:
        print(f"❌ Background tournament demo error: {e}")


if __name__ == "__main__":
    print("Starting Enhanced Features Demonstration...")
    print("This may take a moment to run all tests...\n")
    
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("🚀 Enhanced evaluation features are ready for production use!")
        print("   See enhanced_manager.py for usage examples")
        print("   Check analytics_output/ for generated reports")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some features may not be working correctly")
        print("   Check error messages above for details")
        print("=" * 60)
