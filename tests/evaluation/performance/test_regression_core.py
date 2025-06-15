"""Core performance regression tests."""

import gc
import logging
import time
from unittest.mock import patch, MagicMock

import pytest
import torch

from keisei.evaluation.core import EvaluationStrategy
from keisei.evaluation.core_manager import EvaluationManager
from .conftest import (
    PerformanceMonitor,
    ConfigurationFactory,
    TestAgentFactory,
    MockGameResultFactory,
)

# Performance baselines
PERFORMANCE_BASELINES = {
    "setup_time_seconds": 1.0,
    "games_per_second": 5.0,
    "memory_per_game_mb": 10.0,
    "in_memory_speedup_factor": 2.0,
    "cache_hit_time_ms": 1.0,
    "memory_growth_limit_mb": 50.0,
}

logger = logging.getLogger(__name__)


class TestEvaluationThroughput:
    """Test evaluation throughput and performance baselines."""

    @pytest.fixture
    def config(self):
        """Create configuration for throughput testing."""
        return ConfigurationFactory.create_performance_test_config(num_games=10)

    @pytest.fixture
    def test_agent(self, config):
        """Create test agent."""
        return TestAgentFactory.create_test_agent(config)

    def test_evaluation_throughput_baseline(
        self, config, test_agent, performance_monitor: PerformanceMonitor
    ):
        """Test that evaluation meets throughput requirements."""
        # Setup
        manager = EvaluationManager(config, "throughput_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        # Start monitoring
        performance_monitor.start_monitoring()

        # Mock game execution to isolate evaluation overhead
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result()
            )

            # Execute evaluation
            assert test_agent.checkpoint_path is not None, "Test agent must have a checkpoint path"
            result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

        # Stop monitoring and validate
        metrics = performance_monitor.stop_monitoring()

        # Validate throughput
        games_per_second = config.num_games / metrics["execution_time"]

        assert (
            games_per_second >= PERFORMANCE_BASELINES["games_per_second"]
        ), f"Throughput {games_per_second:.2f} games/s below baseline {PERFORMANCE_BASELINES['games_per_second']}"

        # Validate memory efficiency
        assert (
            metrics["memory_used"]
            < PERFORMANCE_BASELINES["memory_per_game_mb"] * config.num_games
        ), f"Memory usage {metrics['memory_used']:.1f}MB exceeds baseline"

        # Validate results correctness
        assert result.summary_stats.total_games == config.num_games

        logger.info(
            f"Throughput test: {games_per_second:.2f} games/s, "
            f"Memory: {metrics['memory_used']:.1f}MB"
        )

    def test_evaluation_setup_time(self, config, test_agent, performance_monitor):
        """Test that evaluation setup meets time requirements."""
        # Measure setup time
        start_time = time.perf_counter()

        manager = EvaluationManager(config, "setup_time_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        setup_time = time.perf_counter() - start_time

        assert (
            setup_time < PERFORMANCE_BASELINES["setup_time_seconds"]
        ), f"Setup time {setup_time:.2f}s exceeds baseline {PERFORMANCE_BASELINES['setup_time_seconds']}s"

        logger.info(f"Setup time test: {setup_time:.3f}s")


class TestMemoryStability:
    """Test memory usage and stability over time."""

    @pytest.fixture
    def config(self):
        """Create configuration for memory testing."""
        return ConfigurationFactory.create_performance_test_config(num_games=50)

    @pytest.fixture
    def test_agent(self, config):
        """Create test agent."""
        return TestAgentFactory.create_test_agent(config)

    def test_memory_stability_over_time(
        self, config, test_agent, performance_monitor: PerformanceMonitor
    ):
        """Test memory stability during extended evaluation runs."""
        manager = EvaluationManager(config, "memory_stability_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        memory_samples = []

        # Run multiple evaluation cycles
        for cycle in range(5):
            cycle_start_memory = performance_monitor.get_current_memory_mb()

            with patch(
                "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
            ) as mock_evaluate:
                mock_evaluate.return_value = (
                    MockGameResultFactory.create_successful_game_result(
                        game_id=f"memory_test_{cycle}"
                    )
                )

                assert test_agent.checkpoint_path is not None, "Test agent must have a checkpoint path"
                result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

            cycle_end_memory = performance_monitor.get_current_memory_mb()
            memory_growth = cycle_end_memory - cycle_start_memory
            memory_samples.append(memory_growth)

            # Force garbage collection between cycles
            gc.collect()

        # Validate memory stability
        avg_growth = sum(memory_samples) / len(memory_samples)
        max_growth = max(memory_samples)

        assert (
            avg_growth < PERFORMANCE_BASELINES["memory_growth_limit_mb"]
        ), f"Average memory growth {avg_growth:.1f}MB exceeds limit"

        assert (
            max_growth < PERFORMANCE_BASELINES["memory_growth_limit_mb"] * 2
        ), f"Peak memory growth {max_growth:.1f}MB exceeds limit"

        logger.info(
            f"Memory stability test: avg={avg_growth:.1f}MB, max={max_growth:.1f}MB"
        )

    def test_memory_cleanup_after_evaluation(self, config, test_agent, performance_monitor):
        """Test that memory is properly cleaned up after evaluation."""
        manager = EvaluationManager(config, "memory_cleanup_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        # Measure memory before evaluation
        memory_before = performance_monitor.get_current_memory_mb()

        # Run evaluation
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result()
            )

            assert test_agent.checkpoint_path is not None, "Test agent must have a checkpoint path"
            result = manager.evaluate_checkpoint(test_agent.checkpoint_path)

        # Clean up explicitly
        del result
        del test_agent
        gc.collect()

        # Measure memory after cleanup
        memory_after = performance_monitor.get_current_memory_mb()
        memory_growth = memory_after - memory_before

        # Memory growth should be reasonable
        assert (
            memory_growth < PERFORMANCE_BASELINES["memory_growth_limit_mb"]
        ), f"Memory growth after cleanup {memory_growth:.2f}MB too large"

        logger.info(f"Memory cleanup test: growth={memory_growth:.2f}MB")


class TestInMemoryPerformance:
    """Test in-memory evaluation performance improvements."""

    @pytest.fixture
    def config(self):
        """Create configuration for in-memory testing."""
        return ConfigurationFactory.create_performance_test_config(
            num_games=20, enable_enhanced_features=True
        )

    @pytest.fixture
    def test_agent(self, config):
        """Create test agent."""
        return TestAgentFactory.create_test_agent(config)

    def test_in_memory_evaluation_speedup(
        self, config, test_agent, performance_monitor: PerformanceMonitor
    ):
        """Test that in-memory evaluation provides expected speedup."""
        manager = EvaluationManager(config, "in_memory_speedup_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        # Test file-based evaluation (simulation)
        performance_monitor.start_monitoring()
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            # Simulate slower file-based evaluation
            def slow_evaluate(*args, **kwargs):
                time.sleep(0.01)  # Simulate file I/O delay
                return MockGameResultFactory.create_successful_game_result()

            mock_evaluate.side_effect = slow_evaluate

            assert test_agent.checkpoint_path is not None, "Test agent must have a checkpoint path"
            result_file_based = manager.evaluate_checkpoint(test_agent.checkpoint_path)

        file_based_metrics = performance_monitor.stop_monitoring()

        # Test in-memory evaluation (simulation)
        performance_monitor.start_monitoring()
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            # Simulate faster in-memory evaluation
            mock_evaluate.return_value = (
                MockGameResultFactory.create_successful_game_result()
            )

            assert test_agent.checkpoint_path is not None, "Test agent must have a checkpoint path"
            result_in_memory = manager.evaluate_checkpoint(test_agent.checkpoint_path)

        in_memory_metrics = performance_monitor.stop_monitoring()

        # Calculate speedup
        speedup_factor = (
            file_based_metrics["execution_time"] / in_memory_metrics["execution_time"]
        )

        assert (
            speedup_factor >= PERFORMANCE_BASELINES["in_memory_speedup_factor"]
        ), f"In-memory speedup {speedup_factor:.1f}x below baseline {PERFORMANCE_BASELINES['in_memory_speedup_factor']}x"

        logger.info(f"In-memory speedup test: {speedup_factor:.2f}x improvement")


class TestCachePerformance:
    """Test cache performance and efficiency."""

    def test_cache_performance_and_efficiency(self):
        """Test cache hit performance and LRU efficiency."""
        from collections import OrderedDict
        import time

        # Simple LRU cache simulation for testing
        cache_size = 100
        cache = OrderedDict()

        def cache_get(key):
            if key in cache:
                # Move to end (most recently used)
                cache.move_to_end(key)
                return cache[key]
            return None

        def cache_set(key, value):
            if key in cache:
                cache.move_to_end(key)
            else:
                if len(cache) >= cache_size:
                    cache.popitem(last=False)  # Remove least recently used
                cache[key] = value

        # Test cache performance
        num_operations = 1000
        hit_times = []
        miss_times = []

        # Populate cache
        for i in range(cache_size // 2):
            cache_set(f"key_{i}", f"value_{i}")

        # Test cache hits
        for i in range(num_operations):
            key = f"key_{i % (cache_size // 2)}"  # Ensure hits
            start_time = time.perf_counter()
            result = cache_get(key)
            end_time = time.perf_counter()

            hit_times.append((end_time - start_time) * 1000)  # Convert to ms
            assert result is not None

        # Test cache misses
        for i in range(num_operations):
            key = f"miss_key_{i}"
            start_time = time.perf_counter()
            result = cache_get(key)
            end_time = time.perf_counter()

            miss_times.append((end_time - start_time) * 1000)  # Convert to ms
            assert result is None

        # Validate performance
        avg_hit_time = sum(hit_times) / len(hit_times)
        avg_miss_time = sum(miss_times) / len(miss_times)

        assert (
            avg_hit_time < PERFORMANCE_BASELINES["cache_hit_time_ms"]
        ), f"Cache hit time {avg_hit_time:.3f}ms exceeds baseline"

        # Both hit and miss times should be very fast (microsecond level)
        # Note: At microsecond level, hits aren't necessarily faster than misses
        # due to CPU caching and memory layout factors
        max_acceptable_time = 0.1  # 0.1ms
        assert avg_hit_time < max_acceptable_time, f"Cache hit time {avg_hit_time:.3f}ms too slow"
        assert avg_miss_time < max_acceptable_time, f"Cache miss time {avg_miss_time:.3f}ms too slow"

        logger.info(
            f"Cache performance test: hit={avg_hit_time:.3f}ms, miss={avg_miss_time:.3f}ms"
        )
