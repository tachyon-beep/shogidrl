"""
Tests for ModelWeightManager in-memory evaluation functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.core.model_manager import ModelWeightManager


class TestModelWeightManager:
    """Test ModelWeightManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ModelWeightManager(device="cpu", max_cache_size=3)

    def create_mock_agent(self):
        """Create a mock PPOAgent with a model."""
        agent = Mock(spec=PPOAgent)
        agent.model = Mock()

        # Create some dummy weights
        weights = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 3),
            "layer2.bias": torch.randn(5),
        }
        agent.model.state_dict.return_value = weights
        return agent

    def test_extract_agent_weights(self):
        """Test extracting weights from an agent."""
        agent = self.create_mock_agent()

        weights = self.manager.extract_agent_weights(agent)

        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert "layer1.weight" in weights
        assert "layer1.bias" in weights
        assert "layer2.weight" in weights
        assert "layer2.bias" in weights

        # Verify weights are cloned and on CPU
        for _, tensor in weights.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == "cpu"

    def test_extract_agent_weights_no_model(self):
        """Test error when agent has no model."""
        agent = Mock()
        agent.model = None

        with pytest.raises(ValueError, match="Agent must have a model attribute"):
            self.manager.extract_agent_weights(agent)

    def test_cache_opponent_weights(self):
        """Test caching opponent weights from checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Create a mock checkpoint
            checkpoint = {
                "model_state_dict": {
                    "weight1": torch.randn(5, 5),
                    "bias1": torch.randn(5),
                }
            }
            torch.save(checkpoint, tmp_path)

            try:
                weights = self.manager.cache_opponent_weights("test_opponent", tmp_path)

                assert isinstance(weights, dict)
                assert len(weights) == 2
                assert "weight1" in weights
                assert "bias1" in weights

                # Verify cache is populated
                cache_stats = self.manager.get_cache_stats()
                assert "test_opponent" in cache_stats["cached_opponents"]
                assert cache_stats["cache_size"] == 1

            finally:
                tmp_path.unlink()

    def test_cache_opponent_weights_file_not_found(self):
        """Test error when checkpoint file doesn't exist."""
        non_existent_path = Path("/nonexistent/file.pt")

        with pytest.raises(RuntimeError, match="Checkpoint loading failed"):
            self.manager.cache_opponent_weights("test", non_existent_path)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                checkpoint = {"model_state_dict": {"weight": torch.randn(2, 2)}}
                torch.save(checkpoint, tmp_path)

                try:
                    self.manager.cache_opponent_weights(f"opponent_{i}", tmp_path)
                finally:
                    tmp_path.unlink()

        cache_stats = self.manager.get_cache_stats()
        assert cache_stats["cache_size"] == 3
        assert cache_stats["cache_order"] == ["opponent_0", "opponent_1", "opponent_2"]

        # Add one more - should evict the oldest
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {"model_state_dict": {"weight": torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)

            try:
                self.manager.cache_opponent_weights("opponent_3", tmp_path)
            finally:
                tmp_path.unlink()

        cache_stats = self.manager.get_cache_stats()
        assert cache_stats["cache_size"] == 3
        assert "opponent_0" not in cache_stats["cached_opponents"]
        assert "opponent_3" in cache_stats["cached_opponents"]
        assert cache_stats["cache_order"] == ["opponent_1", "opponent_2", "opponent_3"]

    def test_cache_reuse(self):
        """Test that cached weights are reused correctly."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {"model_state_dict": {"weight": torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)

            try:
                # First call should load from file
                weights1 = self.manager.cache_opponent_weights("test", tmp_path)

                # Second call should use cache
                weights2 = self.manager.cache_opponent_weights("test", tmp_path)

                # Should be the same weights (same references)
                assert weights1 is weights2

                # Should move to end of LRU order
                cache_stats = self.manager.get_cache_stats()
                assert cache_stats["cache_order"] == ["test"]

            finally:
                tmp_path.unlink()

    def test_get_cache_stats(self):
        """Test cache statistics."""
        stats = self.manager.get_cache_stats()

        assert stats["cache_size"] == 0
        assert stats["max_cache_size"] == 3
        assert stats["cached_opponents"] == []
        assert stats["cache_order"] == []

        # Add some cached weights
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {"model_state_dict": {"weight": torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)

            try:
                self.manager.cache_opponent_weights("test", tmp_path)

                stats = self.manager.get_cache_stats()
                assert stats["cache_size"] == 1
                assert stats["cached_opponents"] == ["test"]
                assert stats["cache_order"] == ["test"]

            finally:
                tmp_path.unlink()

    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some cached weights
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {"model_state_dict": {"weight": torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)

            try:
                self.manager.cache_opponent_weights("test", tmp_path)
                assert len(self.manager._weight_cache) == 1

                self.manager.clear_cache()
                assert len(self.manager._weight_cache) == 0
                assert len(self.manager._cache_order) == 0

            finally:
                tmp_path.unlink()

    def test_get_memory_usage(self):
        """Test memory usage reporting for cached weights."""
        # Manually cache two dummy weight dicts using public interface
        small_weights = {"a": torch.zeros(10, 10)}
        large_weights = {"b": torch.zeros(100, 100)}

        # Create temporary files to test realistic scenario
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp1:
            torch.save({"model_state_dict": small_weights}, tmp1.name)
            small_path = Path(tmp1.name)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp2:
            torch.save({"model_state_dict": large_weights}, tmp2.name)
            large_path = Path(tmp2.name)

        try:
            # Cache weights through public interface
            self.manager.cache_opponent_weights("small", small_path)
            self.manager.cache_opponent_weights("large", large_path)

            # Calculate usage
            stats = self.manager.get_memory_usage()
            # Total should equal sum of individual sizes
            total = stats["total_mb"]
            by_opponent = stats["by_opponent_mb"]
            assert "small" in by_opponent and "large" in by_opponent
            assert abs(total - (by_opponent["small"] + by_opponent["large"])) < 1e-6

        finally:
            small_path.unlink(missing_ok=True)
            large_path.unlink(missing_ok=True)

    def test_create_agent_from_weights_success(self):
        """Test successful agent reconstruction from valid weights."""
        # Create realistic weights that match ActorCritic architecture
        weights = {
            "conv.weight": torch.randn(16, 46, 3, 3),
            "conv.bias": torch.randn(16),
            "policy_head.weight": torch.randn(4096, 1296),
            "policy_head.bias": torch.randn(4096),
            "value_head.weight": torch.randn(1, 1296),
            "value_head.bias": torch.randn(1),
        }

        # Recreate agent from weights
        new_agent = self.manager.create_agent_from_weights(weights)
        assert isinstance(new_agent, PPOAgent)
        # Model should exist and have weights loaded
        new_weights = new_agent.model.state_dict()
        assert len(new_weights) > 0  # Should have some weights loaded

    def test_create_agent_from_weights_invalid(self):
        """Test error raised when reconstructing agent with invalid weights."""
        # Empty weights should fail
        with pytest.raises((RuntimeError, ValueError)):
            self.manager.create_agent_from_weights({})

    def test_create_agent_from_weights_architecture_inference(self):
        """Test agent reconstruction with architecture inference from weights."""
        # Test different weight configurations to verify architecture inference
        test_cases = [
            # Standard ResNet architecture
            {
                "conv.weight": torch.randn(16, 46, 3, 3),  # 46 input channels
                "conv.bias": torch.randn(16),
                "policy_head.weight": torch.randn(4096, 1296),
                "policy_head.bias": torch.randn(4096),
                "value_head.weight": torch.randn(1, 1296),
                "value_head.bias": torch.randn(1),
            },
            # Different input channels
            {
                "conv.weight": torch.randn(16, 32, 3, 3),  # 32 input channels
                "conv.bias": torch.randn(16),
                "policy_head.weight": torch.randn(2048, 1296),
                "policy_head.bias": torch.randn(2048),
                "value_head.weight": torch.randn(1, 1296),
                "value_head.bias": torch.randn(1),
            },
        ]

        for i, weights in enumerate(test_cases):
            # Should infer architecture from weights and create working agent
            agent = self.manager.create_agent_from_weights(weights)
            assert isinstance(agent, PPOAgent), f"Test case {i}: Should create PPOAgent"

            # Verify model exists and has expected structure
            assert hasattr(agent, "model"), f"Test case {i}: Agent should have model"
            model_weights = agent.model.state_dict()
            assert len(model_weights) > 0, f"Test case {i}: Model should have weights"

            # Verify agent can be used (basic functionality test)
            assert hasattr(
                agent, "select_action"
            ), f"Test case {i}: Agent should have select_action method"

    def test_memory_usage_tracking_under_limits(self):
        """Test memory usage stays within claimed limits during operations."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform operations that should stay within memory limits
        checkpoint_files = []
        try:
            # Cache multiple opponent weights (realistic sizes)
            for i in range(8):  # More than cache size to test LRU
                weights = {
                    "conv.weight": torch.randn(32, 46, 3, 3),
                    "conv.bias": torch.randn(32),
                    "policy_head.weight": torch.randn(2048, 1296),
                    "policy_head.bias": torch.randn(2048),
                    "value_head.weight": torch.randn(1, 1296),
                    "value_head.bias": torch.randn(1),
                }

                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    checkpoint = {"model_state_dict": weights}
                    torch.save(checkpoint, tmp.name)
                    checkpoint_files.append(Path(tmp.name))

                # Cache the weights
                self.manager.cache_opponent_weights(f"opponent_{i}", Path(tmp.name))

                # Create agents from weights (memory intensive operation)
                agent = self.manager.create_agent_from_weights(weights)
                del agent  # Cleanup

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (under 500MB as claimed in docs)
            assert (
                memory_increase < 500
            ), f"Memory increased by {memory_increase:.1f} MB, should be under 500MB"

            # Get manager's internal memory usage tracking
            usage_stats = self.manager.get_memory_usage()
            assert usage_stats["total_mb"] > 0, "Should track some memory usage"
            assert (
                "by_opponent_mb" in usage_stats
            ), "Should provide per-opponent breakdown"

        finally:
            # Cleanup
            for file in checkpoint_files:
                file.unlink(missing_ok=True)

    def test_cache_performance_benchmark(self):
        """Test cache operations complete within 2s as per remediation plan."""
        import time

        # Create test weights
        weights = {
            "conv.weight": torch.randn(16, 46, 3, 3),
            "conv.bias": torch.randn(16),
            "policy_head.weight": torch.randn(1024, 1296),
            "policy_head.bias": torch.randn(1024),
            "value_head.weight": torch.randn(1, 1296),
            "value_head.bias": torch.randn(1),
        }

        checkpoint_files = []
        try:
            # Create checkpoints for performance testing
            for i in range(10):
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    checkpoint = {"model_state_dict": weights}
                    torch.save(checkpoint, tmp.name)
                    checkpoint_files.append(Path(tmp.name))

            # Benchmark cache operations
            start_time = time.perf_counter()

            # Perform cache operations
            for i, checkpoint_file in enumerate(checkpoint_files):
                self.manager.cache_opponent_weights(f"perf_test_{i}", checkpoint_file)
                # Access cached weights (should be fast)
                self.manager.cache_opponent_weights(f"perf_test_{i}", checkpoint_file)

            # Multiple cache hits
            for _ in range(50):
                self.manager.get_cache_stats()
                self.manager.get_memory_usage()

            elapsed = time.perf_counter() - start_time

            # Should complete within 2 seconds as per remediation plan
            assert (
                elapsed < 2.0
            ), f"Cache operations took {elapsed:.3f}s, should be under 2s"

        finally:
            # Cleanup
            for file in checkpoint_files:
                file.unlink(missing_ok=True)
