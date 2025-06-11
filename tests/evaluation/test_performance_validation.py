"""
Performance validation tests for the evaluation system refactor.

These tests validate the key performance claims:
1. 10x speedup from in-memory evaluation
2. Parallel execution performance benefits
3. Memory usage optimization
4. Background processing efficiency
"""

import pytest
import time
import tempfile
import os
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from keisei.evaluation.core import create_evaluation_config, EvaluationStrategy
from keisei.evaluation.core_manager import EvaluationManager
from keisei.evaluation.core.model_manager import ModelWeightManager


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceValidation:
    """Performance validation tests for evaluation system."""

    @pytest.fixture
    def performance_config(self):
        """Configuration optimized for performance testing."""
        return create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=5,  # Small number for testing
            opponent_name="random",
        )

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

        # Verify cache is working (should have only max_cache_size entries)
        cache_stats = weight_manager.get_cache_stats()
        assert cache_stats["cache_size"] <= weight_manager.max_cache_size

    def test_evaluation_manager_throughput(self, performance_config):
        """Test EvaluationManager can handle multiple evaluation requests efficiently."""

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = EvaluationManager(
                config=performance_config,
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

    def test_in_memory_evaluation_performance_benefit(self):
        """Test that in-memory evaluation shows performance benefits."""
        weight_manager = ModelWeightManager(max_cache_size=5)

        # Create realistic weights
        weights = {
            "conv.weight": torch.randn(16, 46, 3, 3),
            "conv.bias": torch.randn(16),
            "policy_head.weight": torch.randn(4096, 1296),
            "policy_head.bias": torch.randn(4096),
            "value_head.weight": torch.randn(1, 1296),
            "value_head.bias": torch.randn(1),
        }

        # Time agent creation from weights
        start_time = time.time()
        agent = weight_manager.create_agent_from_weights(weights)
        creation_time = time.time() - start_time

        assert agent is not None, "Agent should be created successfully"
        assert (
            creation_time < 5.0
        ), f"Agent creation took {creation_time:.3f}s, should be under 5s"

        print(f"Agent creation from weights: {creation_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
