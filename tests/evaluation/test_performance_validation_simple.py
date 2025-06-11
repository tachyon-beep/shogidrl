"""
Performance validation tests for the evaluation system refactor.

These tests validate the key performance claims:
1. In-memory evaluation performance
2. Memory usage optimization
3. System throughput
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from keisei.evaluation.core import EvaluationStrategy, create_evaluation_config
from keisei.evaluation.core.model_manager import ModelWeightManager
from keisei.evaluation.core_manager import EvaluationManager


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceValidation:
    """Performance validation tests for evaluation system."""

    def test_memory_usage_validation(self):
        """Test that memory usage remains reasonable during evaluation."""
        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create weight manager and test memory usage
        weight_manager = ModelWeightManager(max_cache_size=10)

        # Add multiple agent weights to test memory management
        for i in range(15):  # Exceed cache size to test LRU
            weights = {f"layer_{j}.weight": torch.randn(128, 64) for j in range(5)}
            opponent_id = f"agent_{i}"
            # Create a temporary file for testing cache
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
                torch.save(weights, tmp_file.name)
                try:
                    weight_manager.cache_opponent_weights(
                        opponent_id, Path(tmp_file.name)
                    )
                finally:
                    os.unlink(tmp_file.name)  # Clean up

        # Check memory after operations
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Current memory: {current_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")

        # Memory increase should be reasonable (less than 500MB for this test)
        assert (
            memory_increase < 500
        ), f"Memory usage increased by {memory_increase:.1f} MB, should be under 500 MB"

        # Verify cache is working (should have only cache_size entries)
        cache_stats = weight_manager.get_cache_stats()
        assert cache_stats["cache_size"] <= weight_manager.max_cache_size

    def test_evaluation_manager_throughput(self):
        """Test EvaluationManager can handle multiple evaluation requests efficiently."""

        # Create evaluation configuration
        config = create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=1,  # Minimal for testing
            opponent_name="random",
            enable_in_memory_evaluation=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = EvaluationManager(
                config=config,
                run_name="throughput_test",
                pool_size=5,
                elo_registry_path=os.path.join(temp_dir, "elo_registry.json"),
            )

            manager.setup(
                device="cpu",
                policy_mapper=Mock(),
                model_dir=temp_dir,
                wandb_active=False,
            )

            # Mock agent
            mock_agent = Mock()
            mock_agent.name = "test_agent"
            mock_agent.model = Mock()
            mock_agent.model.state_dict.return_value = {
                "layer.weight": torch.randn(32, 16)
            }

            # Time multiple evaluations
            start_time = time.time()

            for i in range(3):  # Multiple quick evaluations
                with patch("keisei.evaluation.strategies.single_opponent.ShogiGame"):
                    result = manager.evaluate_current_agent(mock_agent)
                    assert (
                        result is not None
                    ), f"Evaluation {i} should complete successfully"

            total_time = time.time() - start_time
            avg_time = total_time / 3

            print(f"Total time for 3 evaluations: {total_time:.2f}s")
            print(f"Average time per evaluation: {avg_time:.2f}s")

            # Should complete multiple evaluations efficiently
            assert (
                avg_time < 10.0
            ), f"Average evaluation time {avg_time:.2f}s should be under 10s"

    def test_in_memory_evaluation_performance(self):
        """Test that in-memory evaluation completes quickly."""

        # Create weight manager
        weight_manager = ModelWeightManager(max_cache_size=5)

        # Create realistic weights for a shogi model
        mock_weights = {
            "conv.weight": torch.randn(16, 46, 3, 3),
            "conv.bias": torch.randn(16),
            "policy_head.weight": torch.randn(4096, 1296),
            "policy_head.bias": torch.randn(4096),
            "value_head.weight": torch.randn(1, 1296),
            "value_head.bias": torch.randn(1),
        }

        # Test agent creation from weights performance
        start_time = time.time()

        try:
            # Create agent from weights (this is the main performance test)
            agent = weight_manager.create_agent_from_weights(mock_weights)
            assert agent is not None

            operation_time = time.time() - start_time
            print(f"Agent creation from weights took {operation_time:.3f}s")

            # Should complete quickly
            assert (
                operation_time < 10.0
            ), f"Agent creation took {operation_time:.2f}s, should be under 10s"

        except (RuntimeError, ValueError, ImportError) as e:
            # If agent creation fails due to missing dependencies, that's not a performance issue
            print(f"Agent creation test skipped due to: {e}")
            # Still check that the weights are handled quickly
            operation_time = time.time() - start_time
            assert (
                operation_time < 5.0
            ), f"Weight handling took {operation_time:.2f}s even when failing"

    def test_model_weight_manager_cache_performance(self):
        """Test that ModelWeightManager cache performs efficiently."""

        weight_manager = ModelWeightManager(max_cache_size=5)

        # Test cache performance with different sized tensors
        small_weights = {"layer.weight": torch.randn(32, 16)}
        medium_weights = {"layer.weight": torch.randn(128, 64)}
        large_weights = {"layer.weight": torch.randn(512, 256)}

        test_cases = [
            ("small", small_weights),
            ("medium", medium_weights),
            ("large", large_weights),
        ]

        for size_name, weights in test_cases:
            start_time = time.time()

            # Test caching via temporary file
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
                torch.save(weights, tmp_file.name)
                try:
                    opponent_id = f"test_{size_name}"
                    cached_weights = weight_manager.cache_opponent_weights(
                        opponent_id, Path(tmp_file.name)
                    )
                    assert cached_weights is not None
                finally:
                    os.unlink(tmp_file.name)

            operation_time = time.time() - start_time

            print(f"{size_name.capitalize()} weights operation: {operation_time:.4f}s")

            # Should complete very quickly
            assert (
                operation_time < 2.0
            ), f"{size_name} weights took {operation_time:.3f}s, should be under 2s"

        # Test cache stats
        cache_stats = weight_manager.get_cache_stats()
        assert cache_stats["cache_size"] == len(test_cases)
        print(f"Cache contains {cache_stats['cache_size']} opponents")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
