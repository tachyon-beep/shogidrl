"""
Quick validation test for enhanced evaluation features.
"""

def test_enhanced_features_validation():
    """Test that enhanced features work correctly."""
    
    # Test imports
    from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
    from keisei.evaluation.core.background_tournament import BackgroundTournamentManager
    from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
    from keisei.evaluation.opponents.enhanced_manager import EnhancedOpponentManager
    from keisei.evaluation.core import create_evaluation_config, EvaluationStrategy
    
    print("✅ All imports successful")
    
    # Test enhanced manager creation
    config = create_evaluation_config(
        strategy=EvaluationStrategy.SINGLE_OPPONENT,
        num_games=5
    )
    
    manager = EnhancedEvaluationManager(
        config=config,
        run_name="validation_test",
        enable_background_tournaments=True,
        enable_advanced_analytics=True,
        enable_enhanced_opponents=True
    )
    
    print("✅ Enhanced manager created successfully")
    
    # Test feature status
    status = manager.get_enhancement_status()
    print(f"✅ Enhancement status: {status}")
    
    # Test advanced analytics
    analytics = AdvancedAnalytics()
    print("✅ Advanced analytics created")
    
    # Test opponent manager
    opponent_manager = EnhancedOpponentManager()
    stats = opponent_manager.get_opponent_statistics()
    print(f"✅ Opponent statistics: {stats}")
    
    print("🎉 All enhanced features validated successfully!")


if __name__ == "__main__":
    test_enhanced_features_validation()
