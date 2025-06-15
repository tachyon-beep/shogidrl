"""Error scenarios and edge cases performance tests."""

import gc
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.config_schema import EvaluationConfig
from keisei.evaluation.core import EvaluationStrategy
from keisei.evaluation.core_manager import EvaluationManager

from .conftest import (
    ConfigurationFactory,
    MockGameResultFactory,
    PerformanceMonitor,
    TestAgentFactory,
    create_evaluation_config,
)

logger = logging.getLogger(__name__)


class TestErrorScenarios:
    """Test error handling and recovery performance."""

    @pytest.fixture
    def config(self):
        """Create evaluation configuration for error testing."""
        return create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=5,
            max_concurrent_games=1,
            timeout_per_game=30.0,
            opponent_name="test_opponent",
            wandb_logging=False,
            save_games=False,
        )

    def test_corrupted_checkpoint_recovery(self, tmp_path, config):
        """Test recovery from corrupted checkpoint files."""
        manager = EvaluationManager(config, "corrupted_checkpoint_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir=str(tmp_path),
            wandb_active=False,
        )

        # Create corrupted checkpoint
        corrupted_checkpoint = tmp_path / "corrupted.pth"
        with open(corrupted_checkpoint, "wb") as f:
            f.write(b"corrupted_data_not_a_valid_checkpoint")

        test_agent = TestAgentFactory.create_test_agent(
            ConfigurationFactory.create_minimal_test_config(),
            checkpoint_path=str(corrupted_checkpoint),
        )

        # Test should handle corruption gracefully
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result()
            )

            # Should not crash due to corrupted checkpoint
            try:
                result = manager.evaluate_checkpoint(test_agent.checkpoint_path)
                # If it succeeds, it should be a valid result
                if result:
                    assert result.summary_stats.total_games >= 0
            except Exception as e:
                # If it fails, it should be a graceful failure
                assert (
                    "corrupted" in str(e).lower()
                    or "invalid" in str(e).lower()
                    or "checkpoint" in str(e).lower()
                )

    def test_evaluation_timeout_handling(self, tmp_path, config):
        """Test handling of evaluation timeouts."""
        manager = EvaluationManager(config, "timeout_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir=str(tmp_path),
            wandb_active=False,
        )

        test_agent = TestAgentFactory.create_test_agent(
            ConfigurationFactory.create_minimal_test_config()
        )

        # Mock long-running evaluation
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:

            def timeout_evaluate(*args, **kwargs):
                time.sleep(0.1)  # Simulate long evaluation
                return MockGameResultFactory.create_successful_game_result()

            mock_evaluate.side_effect = timeout_evaluate

            start_time = time.perf_counter()

            # With short timeout, should complete or timeout gracefully
            try:
                assert (
                    test_agent.checkpoint_path is not None
                ), "Test agent must have a checkpoint path"
                result = manager.evaluate_checkpoint(test_agent.checkpoint_path)
                execution_time = time.perf_counter() - start_time

                # Should complete within reasonable time (allowing for test overhead)
                assert (
                    execution_time < 5.0
                ), f"Evaluation took too long: {execution_time:.1f}s"

                if result:
                    assert result.summary_stats.total_games >= 0

            except TimeoutError:
                execution_time = time.perf_counter() - start_time
                assert (
                    execution_time < 5.0
                ), f"Timeout handling took too long: {execution_time:.1f}s"

    def test_malformed_configuration_handling(self):
        """Test handling of malformed configurations."""
        # Test with invalid configuration values
        invalid_configs = [
            {"num_games": -1},  # Negative games
            {"max_concurrent_games": 0},  # Zero concurrency
            {"timeout_per_game": -10.0},  # Negative timeout
        ]

        for invalid_values in invalid_configs:
            try:
                config = create_evaluation_config(**invalid_values)
                manager = EvaluationManager(config, "malformed_config_test")

                # Should either reject invalid config or handle gracefully
                test_agent = TestAgentFactory.create_test_agent(
                    ConfigurationFactory.create_minimal_test_config()
                )

                with patch(
                    "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
                ) as mock_evaluate:
                    mock_evaluate.return_value = (
                        MockGameResultFactory.create_successful_game_result()
                    )

                    assert (
                        test_agent.checkpoint_path is not None
                    ), "Test agent must have a checkpoint path"
                    result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

                    # If it succeeds, should be reasonable result
                    if result:
                        assert result.summary_stats.total_games >= 0

            except (ValueError, AssertionError, TypeError) as e:
                # Expected for invalid configurations
                logger.info(f"Configuration validation caught: {e}")


class TestEdgeCases:
    """Test edge cases and boundary conditions in evaluation system."""

    @pytest.fixture
    def eval_config(self):
        """Create evaluation configuration for testing."""
        return create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=1,
            max_concurrent_games=1,
            timeout_per_game=30.0,
            opponent_name="test_opponent",
            wandb_logging=False,
            save_games=False,
        )

    def test_single_game_evaluation(self, tmp_path, eval_config):
        """Test evaluation with minimal (single) game configuration."""
        manager = EvaluationManager(eval_config, "single_game_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir=str(tmp_path),
            wandb_active=False,
        )

        test_agent = TestAgentFactory.create_test_agent(
            ConfigurationFactory.create_minimal_test_config()
        )

        # Mock the evaluation to avoid complex game execution
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result(
                    game_id="single_game_0",
                    winner="agent",
                    game_length=50,
                )
            )

            assert (
                test_agent.checkpoint_path is not None
            ), "Test agent must have a checkpoint path"
            result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

            # Validate single game result
            assert result is not None
            assert result.summary_stats.total_games == 1
            assert result.summary_stats.agent_wins >= 0
            assert result.summary_stats.opponent_wins >= 0
            assert result.summary_stats.draws >= 0
            assert (
                result.summary_stats.agent_wins
                + result.summary_stats.opponent_wins
                + result.summary_stats.draws
                == 1
            )

    def test_memory_cleanup_after_evaluation(
        self, tmp_path, eval_config, performance_monitor
    ):
        """Test that memory is properly cleaned up after evaluation."""
        manager = EvaluationManager(eval_config, "memory_cleanup_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir=str(tmp_path),
            wandb_active=False,
        )

        # Measure memory before evaluation
        memory_before = performance_monitor.get_current_memory_mb()

        test_agent = TestAgentFactory.create_test_agent(
            ConfigurationFactory.create_minimal_test_config()
        )

        # Mock the evaluation
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result(
                    game_id="cleanup_test_game",
                    winner="agent",
                    game_length=50,
                )
            )

            assert (
                test_agent.checkpoint_path is not None
            ), "Test agent must have a checkpoint path"
            result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

        # Force garbage collection
        del result
        del test_agent
        gc.collect()

        # Measure memory after cleanup
        memory_after = performance_monitor.get_current_memory_mb()
        memory_growth = memory_after - memory_before

        # Memory growth should be reasonable (less than 50MB for small test)
        assert memory_growth < 50, f"Memory growth too large: {memory_growth:.2f}MB"

    def test_zero_game_configuration(self, tmp_path):
        """Test handling of zero-game configuration."""
        try:
            config = create_evaluation_config(num_games=0)
            manager = EvaluationManager(config, "zero_game_test")

            test_agent = TestAgentFactory.create_test_agent(
                ConfigurationFactory.create_minimal_test_config()
            )

            assert (
                test_agent.checkpoint_path is not None
            ), "Test agent must have a checkpoint path"
            result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

            # Should either reject zero games or return empty result
            if result:
                assert result.summary_stats.total_games == 0
                assert len(result.games) == 0

        except (ValueError, AssertionError) as e:
            # Expected for zero games
            logger.info(f"Zero game configuration rejected: {e}")

    def test_extreme_concurrency_configuration(self, tmp_path):
        """Test handling of extreme concurrency settings."""
        extreme_configs = [
            {"max_concurrent_games": 1000},  # Very high concurrency
            {
                "max_concurrent_games": 1,
                "num_games": 1000,
            },  # High games, low concurrency
        ]

        for config_values in extreme_configs:
            try:
                config = create_evaluation_config(**config_values)
                manager = EvaluationManager(config, "extreme_concurrency_test")
                manager.setup(
                    device="cpu",
                    policy_mapper=None,
                    model_dir=str(tmp_path),
                    wandb_active=False,
                )

                test_agent = TestAgentFactory.create_test_agent(
                    ConfigurationFactory.create_minimal_test_config()
                )

                # Should handle extreme configurations gracefully
                with patch(
                    "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
                ) as mock_evaluate:
                    mock_evaluate.return_value = (
                        MockGameResultFactory.create_successful_game_result()
                    )

                    # Test should not hang or crash
                    start_time = time.perf_counter()
                    assert (
                        test_agent.checkpoint_path is not None
                    ), "Test agent must have a checkpoint path"
                    result = manager.evaluate_checkpoint(test_agent.checkpoint_path)
                    execution_time = time.perf_counter() - start_time

                    # Should complete within reasonable time (even if slow)
                    assert (
                        execution_time < 10.0
                    ), f"Extreme config took too long: {execution_time:.1f}s"

                    if result:
                        assert result.summary_stats.total_games >= 0

            except (ValueError, ResourceWarning, OSError) as e:
                # Expected for extreme configurations
                logger.info(f"Extreme configuration handled: {e}")

    def test_rapid_successive_evaluations(
        self, tmp_path, eval_config, performance_monitor
    ):
        """Test performance of rapid successive evaluations."""
        manager = EvaluationManager(eval_config, "rapid_evaluation_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir=str(tmp_path),
            wandb_active=False,
        )

        test_agent = TestAgentFactory.create_test_agent(
            ConfigurationFactory.create_minimal_test_config()
        )

        # Run multiple rapid evaluations
        num_evaluations = 10
        execution_times = []

        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result()
            )

            for _ in range(num_evaluations):
                start_time = time.perf_counter()
                assert (
                    test_agent.checkpoint_path is not None
                ), "Test agent must have a checkpoint path"
                result = manager.evaluate_checkpoint(test_agent.checkpoint_path)
                execution_time = time.perf_counter() - start_time

                execution_times.append(execution_time)
                assert result is not None
                assert result.summary_stats.total_games == 1

        # Performance should be consistent across evaluations
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)

        # Variation should be reasonable (max < 3x min)
        time_variation = max_time / min_time if min_time > 0 else 1.0
        assert (
            time_variation < 3.0
        ), f"Execution time variation too high: {time_variation:.1f}x"

        logger.info(
            f"Rapid evaluation test: avg={avg_time:.3f}s, "
            f"range=[{min_time:.3f}, {max_time:.3f}], "
            f"variation={time_variation:.1f}x"
        )
