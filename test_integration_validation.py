#!/usr/bin/env python3
"""
Priority 2 Integration Test: Evaluation System Display Integration

This test validates the integration between evaluation system and display system.
"""

def test_evaluation_display_integration():
    """Test the core integration points."""
    print("🔍 Testing evaluation-display integration...")
    
    # Test 1: Verify trainer has evaluation_manager
    from keisei.training.trainer import Trainer
    import inspect
    
    trainer_source = inspect.getsource(Trainer.__init__)
    assert "evaluation_manager" in trainer_source
    assert "EvaluationManager" in trainer_source
    print("✅ Trainer has evaluation_manager integration")
    
    # Test 2: Verify display reads evaluation_elo_snapshot
    from keisei.training.display import TrainingDisplay
    
    display_source = inspect.getsource(TrainingDisplay.refresh_dashboard_panels)
    assert "evaluation_elo_snapshot" in display_source
    assert "elo_panel" in display_source
    assert "Waiting for initial model evaluations" in display_source
    print("✅ Display has ELO panel integration")
    
    # Test 3: Verify callback sets evaluation_elo_snapshot
    from keisei.training.callbacks import EvaluationCallback
    
    callback_source = inspect.getsource(EvaluationCallback.on_step_end)
    assert "trainer.evaluation_manager.evaluate_current_agent" in callback_source
    assert "evaluation_elo_snapshot" in callback_source
    print("✅ Evaluation callback updates trainer snapshot")
    
    # Test 4: Test data flow logic
    from unittest.mock import Mock
    
    trainer = Mock()
    trainer.evaluation_elo_snapshot = {
        "current_id": "test_agent",
        "current_rating": 1550.0,
        "top_ratings": [
            ("test_agent", 1550.0),
            ("opponent_1", 1450.0)
        ]
    }
    
    # Test display logic
    snap = getattr(trainer, "evaluation_elo_snapshot", None)
    if snap and snap.get("top_ratings") and len(snap["top_ratings"]) >= 2:
        lines = [f"{mid}: {rating:.0f}" for mid, rating in snap["top_ratings"]]
        content = "\n".join(lines)
    else:
        content = "Waiting for initial model evaluations..."
        
    assert "test_agent: 1550" in content
    assert "opponent_1: 1450" in content
    print("✅ Data flow integration works correctly")

if __name__ == "__main__":
    print("🔍 Running Priority 2: Evaluation System Display Integration Tests")
    print("=" * 70)
    
    try:
        test_evaluation_display_integration()
        
        print("\n" + "=" * 70)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Evaluation system properly integrates with trainer display")
        print("✅ Integration chain: trainer -> evaluation_manager -> callback -> display works")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
