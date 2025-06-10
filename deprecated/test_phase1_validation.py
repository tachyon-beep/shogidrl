#!/usr/bin/env python3
"""
Phase 1 Validation Test for Keisei Evaluation System Refactor

This script validates that the Phase 1 implementation is working correctly,
including all core components, configuration, and basic strategy functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_phase_1_implementation():
    """Test all Phase 1 components."""
    
    print("🚀 Starting Phase 1 Validation Test")
    print("=" * 50)
    
    # Test 1: Core imports
    print("\n1. Testing core imports...")
    try:
        from keisei.evaluation.core import (
            EvaluationStrategy,
            EvaluationConfig,
            SingleOpponentConfig,
            BaseEvaluator,
            EvaluatorFactory,
            AgentInfo,
            OpponentInfo,
            EvaluationContext,
            GameResult,
            SummaryStats,
            EvaluationResult,
            create_agent_info,
            create_game_result
        )
        print("✓ All core imports successful")
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    # Test 2: Strategy imports
    print("\n2. Testing strategy imports...")
    try:
        from keisei.evaluation.strategies import SingleOpponentEvaluator
        print("✓ Strategy imports successful")
    except Exception as e:
        print(f"❌ Strategy import failed: {e}")
        return False
    
    # Test 3: Configuration creation
    print("\n3. Testing configuration creation...")
    try:
        # Test basic config
        config = EvaluationConfig(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=10
        )
        print(f"✓ Basic config created: {config.strategy.value}")
        
        # Test specific config
        single_config = SingleOpponentConfig(
            opponent_name="test_opponent",
            num_games=5
        )
        print(f"✓ SingleOpponent config created: {single_config.opponent_name}")
        
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False
    
    # Test 4: Data structure creation
    print("\n4. Testing data structure creation...")
    try:
        # Create agent info
        agent = create_agent_info(
            name="test_agent",
            checkpoint_path="/fake/path/model.pt"
        )
        print(f"✓ Agent info created: {agent.name}")
        
        # Create opponent info
        opponent = OpponentInfo(
            name="test_opponent",
            type="random"
        )
        print(f"✓ Opponent info created: {opponent.name}")
        
        # Create game result
        game_result = create_game_result(
            game_id="test_game_1",
            agent_info=agent,
            opponent_info=opponent,
            winner=0,  # Agent wins
            moves_count=50,
            duration_seconds=120.0
        )
        print(f"✓ Game result created: {game_result.game_id}")
        
    except Exception as e:
        print(f"❌ Data structure creation failed: {e}")
        return False
    
    # Test 5: Evaluator factory
    print("\n5. Testing evaluator factory...")
    try:
        # Check registered strategies
        strategies = EvaluatorFactory.list_strategies()
        print(f"✓ Registered strategies: {strategies}")
        
        # Create evaluator
        evaluator = EvaluatorFactory.create(single_config)
        print(f"✓ Evaluator created: {type(evaluator).__name__}")
        
    except Exception as e:
        print(f"❌ Evaluator factory failed: {e}")
        return False
    
    # Test 6: Full evaluation run (simulation)
    print("\n6. Testing full evaluation run...")
    try:
        # Run a small evaluation
        result = await evaluator.evaluate(agent)
        print(f"✓ Evaluation completed: {result.summary_stats.total_games} games")
        print(f"  - Win rate: {result.summary_stats.win_rate:.2f}")
        print(f"  - Game count: {len(result.games)}")
        print(f"  - Analytics: {len(result.analytics)} metrics")
        
    except Exception as e:
        print(f"❌ Full evaluation failed: {e}")
        return False
    
    # Test 7: Serialization
    print("\n7. Testing serialization...")
    try:
        # Test config serialization
        config_dict = single_config.to_dict()
        config_restored = SingleOpponentConfig.from_dict(config_dict)
        print(f"✓ Config serialization works")
        
        # Test result serialization  
        result_dict = result.to_dict()
        print(f"✓ Result serialization works")
        
        # Test W&B format
        wandb_dict = result.to_wandb_dict()
        print(f"✓ W&B format works: {len(wandb_dict)} metrics")
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        return False
    
    # Test 8: Configuration validation
    print("\n8. Testing configuration validation...")
    try:
        # Test configuration dictionary conversion
        config_dict = result.context.configuration.to_dict()
        print(f"✓ Configuration to_dict works: {len(config_dict)} fields")
        
        # Validate essential fields are present
        required_fields = ['strategy', 'num_games', 'max_concurrent_games']
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required field: {field}")
        print(f"✓ Required configuration fields present")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Phase 1 Validation Test PASSED!")
    print("=" * 50)
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Strategy: {result.context.configuration.strategy.value}")
    print(f"- Games played: {result.summary_stats.total_games}")
    print(f"- Win rate: {result.summary_stats.win_rate:.3f}")
    print(f"- Average game length: {result.summary_stats.avg_game_length:.1f} moves")
    print(f"- Total duration: {result.summary_stats.avg_duration_seconds:.1f}s average")
    print(f"- Analytics metrics: {len(result.analytics)}")
    print(f"- Errors: {len(result.errors)}")
    
    return True


def test_modern_compatibility():
    """Test that modern evaluation system works completely."""
    print("\n🔄 Testing Modern System Compatibility...")
    
    try:
        # Test that the new system imports work
        from keisei.evaluation.core import EvaluationStrategy, create_evaluation_config
        from keisei.evaluation.strategies import SingleOpponentEvaluator
        print("✓ Modern imports work")
        
        # Test that we can create configs without legacy conversion
        config = create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=1,
            opponent_name="test"
        )
        print("✓ Modern config creation works")
        return True
    except Exception as e:
        print(f"❌ Modern system error: {e}")
        return False


if __name__ == "__main__":
    print("Keisei Evaluation System - Phase 1 Validation")
    print("Testing refactored evaluation architecture...")
    
    try:
        # Run the async test
        success = asyncio.run(test_phase_1_implementation())
        
        # Test modern compatibility
        modern_success = test_modern_compatibility()
        
        if success and modern_success:
            print("\n✅ ALL TESTS PASSED - Phase 1 implementation is ready!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed - check implementation")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
