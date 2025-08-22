#!/usr/bin/env python3
"""
Debug specific integration test failures
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime
sys.path.append('/home/john/keisei')

def debug_callback_integration():
    """Debug the callback integration test failure."""
    print("🔍 Debugging Callback Integration...")
    try:
        from keisei.training.callbacks import EvaluationCallback
        from keisei.config_schema import EvaluationConfig
        from keisei.training.metrics_manager import MetricsManager
        
        # Replicate exact test conditions
        mock_trainer = Mock()
        mock_trainer.metrics_manager = MetricsManager()
        mock_trainer.metrics_manager.global_timestep = 49999  # Just before trigger
        mock_trainer.agent = Mock()
        mock_trainer.agent.model = Mock()
        mock_trainer.evaluation_manager = Mock()
        mock_trainer.evaluation_manager.opponent_pool.sample.return_value = "/tmp/test_checkpoint.pt"
        mock_trainer.evaluation_manager.evaluate_current_agent.return_value = Mock()
        mock_trainer.evaluation_manager.evaluate_current_agent.return_value.summary_stats = Mock()
        mock_trainer.evaluation_manager.evaluate_current_agent.return_value.summary_stats.win_rate = 0.6
        mock_trainer.evaluation_manager.evaluate_current_agent.return_value.summary_stats.loss_rate = 0.3
        mock_trainer.run_name = "test_run"
        mock_trainer.log_both = Mock()
        
        eval_config = EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=50000
        )
        
        callback = EvaluationCallback(eval_config, interval=50000)
        
        # Test 1: Before trigger
        callback.on_step_end(mock_trainer)
        before_trigger_called = mock_trainer.evaluation_manager.evaluate_current_agent.called
        print(f"   Before trigger - evaluation called: {before_trigger_called}")
        
        # Reset mock
        mock_trainer.evaluation_manager.evaluate_current_agent.reset_mock()
        
        # Test 2: At trigger
        mock_trainer.metrics_manager.global_timestep = 50000
        callback.on_step_end(mock_trainer)
        at_trigger_called = mock_trainer.evaluation_manager.evaluate_current_agent.called
        print(f"   At trigger - evaluation called: {at_trigger_called}")
        
        # Test specific assertions
        assert not before_trigger_called, "Evaluation should not be called before trigger"
        assert at_trigger_called, "Evaluation should be called at trigger"
        assert mock_trainer.agent.model.eval.called, "Model should be set to eval mode"
        assert mock_trainer.agent.model.train.called, "Model should be set back to train mode"
        
        print("✅ Callback integration debug PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Callback integration debug FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_weight_management():
    """Debug the weight management test failure."""
    print("🔍 Debugging Weight Management...")
    try:
        from keisei.evaluation.core.model_manager import ModelWeightManager
        import torch
        import torch.nn as nn
        
        # Replicate exact test conditions
        weight_manager = ModelWeightManager(device="cpu", max_cache_size=3)
        
        class MockAgent:
            def __init__(self):
                self.model = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2)
                )
        
        agent = MockAgent()
        
        # Test weight extraction
        weights = weight_manager.extract_agent_weights(agent)
        print(f"   Weight extraction result: {type(weights)}, length: {len(weights)}")
        assert isinstance(weights, dict), "Weights should be a dictionary"
        assert len(weights) > 0, "Weights should not be empty"
        
        # Test cache functionality
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Create dummy checkpoint
            torch.save({"model_state_dict": agent.model.state_dict()}, f.name)
            print(f"   Created test checkpoint: {f.name}")
            
            # Test caching
            cached_weights = weight_manager.cache_opponent_weights("test_opponent", Path(f.name))
            print(f"   Cache result: {type(cached_weights)}, length: {len(cached_weights)}")
            assert isinstance(cached_weights, dict), "Cached weights should be a dictionary"
            
            # Test cache stats
            stats = weight_manager.get_cache_stats()
            print(f"   Cache stats: {stats}")
            assert "cache_size" in stats, "Cache stats should include cache_size"
            assert "hit_rate" in stats, "Cache stats should include hit_rate"
        
        print("✅ Weight management debug PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Weight management debug FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_move_validation():
    """Debug the move validation test failure."""
    print("🔍 Debugging Move Validation...")
    try:
        from keisei.shogi.shogi_game import ShogiGame
        
        # Test basic game functionality with correct method names
        game = ShogiGame()
        print(f"   Game created: {type(game)}")
        
        # Check available methods
        methods = [m for m in dir(game) if not m.startswith('_')]
        move_methods = [m for m in methods if 'move' in m.lower() or 'legal' in m.lower()]
        print(f"   Available move-related methods: {move_methods}")
        
        assert hasattr(game, 'reset'), "Game should have reset method"
        # Note: 'step' method doesn't exist, it's likely 'make_move'
        assert hasattr(game, 'make_move'), "Game should have make_move method"
        assert hasattr(game, 'get_legal_moves'), "Game should have get_legal_moves method"
        
        # Test game reset
        game.reset()
        print("   Game reset successful")
        
        # Test legal move generation with correct method name
        legal_moves = game.get_legal_moves()
        print(f"   Legal moves: {len(legal_moves)} found")
        assert len(legal_moves) > 0, "Game should have legal moves available"
        
        print("✅ Move validation debug PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Move validation debug FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("🔧 DEBUGGING INTEGRATION TEST FAILURES")
    print("="*50)
    
    results = []
    results.append(debug_callback_integration())
    results.append(debug_weight_management()) 
    results.append(debug_move_validation())
    
    print("\n" + "="*50)
    print("DEBUG SUMMARY")
    print("="*50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All debug tests PASSED - Issues are likely in test logic")
    else:
        print("❌ Some debug tests FAILED - Real integration issues found")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)