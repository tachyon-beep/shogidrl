"""
Test script to validate Phase 1 implementation of the evaluation system refactor.

This script tests the core infrastructure components and the single opponent evaluator
to ensure everything is working correctly before proceeding to Phase 2.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from keisei.evaluation.core import (
    AgentInfo,
    BaseEvaluator,
    EvaluationConfig,
    EvaluationContext,
    EvaluationStrategy,
    EvaluatorFactory,
    OpponentInfo,
    SingleOpponentConfig,
    create_agent_info,
    evaluate_agent,
)
from keisei.evaluation.strategies import SingleOpponentEvaluator


async def test_core_data_structures():
    """Test core data structures and serialization."""
    print("Testing core data structures...")

    # Test AgentInfo
    agent = AgentInfo(
        name="test_agent",
        checkpoint_path="/path/to/model.pt",
        model_type="resnet",
        training_timesteps=100000,
        version="1.0",
        metadata={"test": True},
    )
    assert agent.name == "test_agent"
    assert agent.checkpoint_path == "/path/to/model.pt"
    print("✓ AgentInfo creation successful")

    # Test OpponentInfo
    opponent = OpponentInfo(
        name="random_opponent",
        type="random",
        difficulty_level=0.5,
        metadata={"seed": 42},
    )
    assert opponent.name == "random_opponent"
    assert opponent.type == "random"
    print("✓ OpponentInfo creation successful")

    # Test configuration
    config = SingleOpponentConfig(
        opponent_name="test_opponent",
        num_games=10,
        max_concurrent_games=2,
        play_as_both_colors=True,
    )
    assert config.strategy == EvaluationStrategy.SINGLE_OPPONENT
    assert config.opponent_name == "test_opponent"
    print("✓ SingleOpponentConfig creation successful")

    # Test config serialization
    config_dict = config.to_dict()
    restored_config = EvaluationConfig.from_dict(config_dict)
    assert restored_config.num_games == config.num_games
    print("✓ Configuration serialization successful")


async def test_evaluator_factory():
    """Test the evaluator factory system."""
    print("\nTesting evaluator factory...")

    # Check that SingleOpponentEvaluator is registered
    strategies = EvaluatorFactory.list_strategies()
    print(f"Registered strategies: {strategies}")
    assert EvaluationStrategy.SINGLE_OPPONENT.value in strategies
    print("✓ SingleOpponentEvaluator is registered")

    # Create evaluator through factory
    config = SingleOpponentConfig(
        opponent_name="test_opponent", num_games=5, max_concurrent_games=1
    )

    evaluator = EvaluatorFactory.create(config)
    assert isinstance(evaluator, SingleOpponentEvaluator)
    assert evaluator.config.opponent_name == "test_opponent"
    print("✓ Evaluator factory creation successful")


async def test_single_opponent_evaluation():
    """Test running a single opponent evaluation."""
    print("\nTesting single opponent evaluation...")

    # Create test agent
    agent = create_agent_info(
        name="test_agent_v1", checkpoint_path="/tmp/fake_model.pt"
    )

    # Create configuration
    config = SingleOpponentConfig(
        opponent_name="random_bot",
        num_games=6,  # Small number for quick testing
        max_concurrent_games=2,
        play_as_both_colors=True,
        randomize_positions=False,  # For consistent testing
        random_seed=42,
    )

    # Run evaluation
    try:
        result = await evaluate_agent(agent, config)

        # Validate results
        assert result.summary_stats.total_games == 6
        assert len(result.games) == 6
        assert result.context.agent_info.name == "test_agent_v1"

        # Check color distribution
        first_player_games = sum(
            1
            for g in result.games
            if not g.opponent_info.metadata.get("agent_plays_second", False)
        )
        second_player_games = len(result.games) - first_player_games

        print(f"  Games as first player: {first_player_games}")
        print(f"  Games as second player: {second_player_games}")
        print(f"  Win rate: {result.summary_stats.win_rate:.3f}")
        print(f"  Avg game length: {result.summary_stats.avg_game_length:.1f}")

        # Basic validation
        assert first_player_games > 0
        assert second_player_games > 0  # Should have color balancing
        assert result.summary_stats.win_rate >= 0.0
        assert result.summary_stats.win_rate <= 1.0

        print("✓ Single opponent evaluation successful")
        return result

    except Exception as e:
        print(f"✗ Single opponent evaluation failed: {e}")
        raise


async def test_analytics_and_reporting():
    """Test analytics calculation and reporting features."""
    print("\nTesting analytics and reporting...")

    # Use a configuration that will generate analytics
    config = SingleOpponentConfig(
        opponent_name="analytics_test_opponent",
        num_games=8,
        max_concurrent_games=1,
        play_as_both_colors=True,
    )

    agent = create_agent_info("analytics_test_agent")
    result = await evaluate_agent(agent, config)

    # Check analytics
    analytics = result.analytics
    print(f"  Analytics keys: {list(analytics.keys())}")

    expected_keys = [
        "first_player_win_rate",
        "second_player_win_rate",
        "min_game_length",
        "max_game_length",
        "median_game_length",
    ]

    for key in expected_keys:
        if key in analytics:
            print(f"  {key}: {analytics[key]}")

    # Test serialization
    result_dict = result.to_dict()
    assert "analytics" in result_dict
    assert "summary_stats" in result_dict
    print("✓ Analytics and serialization successful")


async def test_error_handling():
    """Test error handling capabilities."""
    print("\nTesting error handling...")

    # Test invalid configuration
    try:
        config = SingleOpponentConfig(
            opponent_name="", num_games=5  # Empty name should cause validation error
        )
        agent = create_agent_info("test_agent")
        await evaluate_agent(agent, config)
        assert False, "Should have raised validation error"
    except ValueError as e:
        print(f"✓ Correctly caught validation error: {e}")

    # Test invalid agent
    try:
        config = SingleOpponentConfig(opponent_name="valid_opponent", num_games=5)
        agent = AgentInfo(name="")  # Empty name
        await evaluate_agent(agent, config)
        assert False, "Should have raised validation error"
    except ValueError:
        print("✓ Correctly caught agent validation error")


async def test_legacy_compatibility():
    """Test legacy compatibility features."""
    print("\nTesting legacy compatibility...")

    # Test legacy config conversion
    legacy_config = {
        "num_games": 20,
        "max_workers": 4,
        "randomize": True,
        "save_games": True,
        "log_to_wandb": False,
        "opponent_name": "legacy_opponent",
    }

    from keisei.evaluation.core.evaluation_config import from_legacy_config

    new_config = from_legacy_config(legacy_config)

    assert new_config.num_games == 20
    assert new_config.max_concurrent_games == 4
    assert new_config.randomize_positions == True
    print("✓ Legacy config conversion successful")

    # Test legacy format export
    config = SingleOpponentConfig(
        opponent_name="test", num_games=10, max_concurrent_games=2
    )
    evaluator = EvaluatorFactory.create(config)
    legacy_format = evaluator.to_legacy_format()

    assert legacy_format["num_games"] == 10
    assert legacy_format["max_workers"] == 2
    assert legacy_format["strategy"] == "single_opponent"
    print("✓ Legacy format export successful")


async def main():
    """Run all Phase 1 validation tests."""
    print("=== Phase 1 Evaluation System Validation ===\n")

    try:
        await test_core_data_structures()
        await test_evaluator_factory()
        result = await test_single_opponent_evaluation()
        await test_analytics_and_reporting()
        await test_error_handling()
        await test_legacy_compatibility()

        print("\n=== Phase 1 Validation Summary ===")
        print("✓ All core infrastructure tests passed")
        print("✓ SingleOpponentEvaluator is working correctly")
        print("✓ Data structures and serialization working")
        print("✓ Analytics and reporting functional")
        print("✓ Error handling and validation working")
        print("✓ Legacy compatibility maintained")
        print("\n🎉 Phase 1 implementation is ready!")
        print("Ready to proceed to Phase 2: Strategy implementations")

        return True

    except Exception as e:
        print(f"\n❌ Phase 1 validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
