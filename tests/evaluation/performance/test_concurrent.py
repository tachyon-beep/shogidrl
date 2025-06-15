"""Concurrent evaluation performance tests."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from keisei.evaluation.core_manager import EvaluationManager

from .conftest import (
    ConfigurationFactory,
    MockGameResultFactory,
    PerformanceMonitor,
    TestAgentFactory,
)

logger = logging.getLogger(__name__)


class TestConcurrentEvaluation:
    """Test concurrent evaluation performance and overhead."""

    @pytest.fixture
    def sequential_config(self):
        """Create configuration for sequential evaluation."""
        return ConfigurationFactory.create_performance_test_config(
            num_games=20, parallel_execution=False
        )

    @pytest.fixture
    def parallel_config(self):
        """Create configuration for parallel evaluation."""
        return ConfigurationFactory.create_performance_test_config(
            num_games=20, parallel_execution=True
        )

    @pytest.fixture
    def test_agent_sequential(self, sequential_config):
        """Create test agent for sequential testing."""
        return TestAgentFactory.create_test_agent(sequential_config)

    @pytest.fixture
    def test_agent_parallel(self, parallel_config):
        """Create test agent for parallel testing."""
        return TestAgentFactory.create_test_agent(parallel_config)

    def test_concurrent_evaluation_overhead(
        self,
        sequential_config,
        parallel_config,
        test_agent_sequential,
        test_agent_parallel,
        performance_monitor: PerformanceMonitor,
    ):
        """Test concurrent evaluation performance vs sequential."""
        # Test sequential evaluation
        sequential_manager = EvaluationManager(sequential_config, "sequential_test")
        sequential_manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        performance_monitor.start_monitoring()
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            # Simulate some processing time
            def sequential_evaluate(*_args, **_kwargs):
                time.sleep(0.005)  # 5ms per game
                return MockGameResultFactory.create_successful_game_result()

            mock_evaluate.side_effect = sequential_evaluate

            sequential_result = sequential_manager.evaluate_checkpoint(
                test_agent_sequential.checkpoint_path
            )

        sequential_metrics = performance_monitor.stop_monitoring()

        # Test parallel evaluation
        parallel_manager = EvaluationManager(parallel_config, "parallel_test")
        parallel_manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        performance_monitor.start_monitoring()
        with patch(
            "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
        ) as mock_evaluate:
            # Simulate concurrent processing
            def parallel_evaluate(*_args, **_kwargs):
                time.sleep(0.005)  # Same 5ms per game
                return MockGameResultFactory.create_successful_game_result()

            mock_evaluate.side_effect = parallel_evaluate

            parallel_result = parallel_manager.evaluate_checkpoint(
                test_agent_parallel.checkpoint_path
            )

        parallel_metrics = performance_monitor.stop_monitoring()

        # Validate results
        assert sequential_result.summary_stats.total_games == 20
        assert parallel_result.summary_stats.total_games == 20

        # Calculate performance improvement
        speedup = (
            sequential_metrics["execution_time"] / parallel_metrics["execution_time"]
        )

        # Parallel should provide some speedup (though overhead may limit it in test environments)
        # Note: CI environments often don't show parallel benefits due to resource constraints
        assert (
            speedup >= 0.8
        ), f"Parallel performance {speedup:.1f}x significantly degraded"

        # Memory overhead should be reasonable
        memory_overhead = (
            parallel_metrics["memory_used"] - sequential_metrics["memory_used"]
        )
        assert (
            memory_overhead < 20.0
        ), f"Parallel memory overhead {memory_overhead:.1f}MB too high"

        logger.info(
            "Concurrent evaluation test: %.2fx speedup, memory overhead: %.1fMB",
            speedup,
            memory_overhead,
        )

    def test_parallel_executor_scalability(self, performance_monitor):
        """Test parallel executor scalability with varying worker counts."""
        num_tasks = 100
        task_duration = 0.001  # 1ms per task

        def mock_task(task_id):
            """Mock CPU-bound task."""
            time.sleep(task_duration)
            return f"result_{task_id}"

        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        execution_times = {}

        for workers in worker_counts:
            performance_monitor.start_monitoring()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(mock_task, i) for i in range(num_tasks)]
                results = [future.result() for future in as_completed(futures)]

            metrics = performance_monitor.stop_monitoring()
            execution_times[workers] = metrics["execution_time"]

            assert len(results) == num_tasks

        # Validate scaling behavior
        single_thread_time = execution_times[1]
        best_parallel_time = min(execution_times[w] for w in worker_counts[1:])

        speedup = single_thread_time / best_parallel_time
        assert speedup >= 1.5, f"Parallel speedup {speedup:.1f}x insufficient"

        # Find the worker count that achieved the best time
        best_worker_count = min(
            execution_times.keys(), key=lambda w: execution_times[w]
        )

        logger.info(
            "Parallel scalability test: best speedup %.2fx with %d workers",
            speedup,
            best_worker_count,
        )

    def test_resource_contention_handling(self, parallel_config, performance_monitor):
        """Test handling of resource contention in parallel evaluation."""
        manager = EvaluationManager(parallel_config, "contention_test")
        manager.setup(
            device="cpu",
            policy_mapper=None,
            model_dir="./test_models",
            wandb_active=False,
        )

        # Simulate high resource contention
        contention_levels = [0.001, 0.005, 0.01, 0.02]  # Different delay levels
        execution_times = []

        for contention_delay in contention_levels:
            test_agent = TestAgentFactory.create_test_agent(parallel_config)

            performance_monitor.start_monitoring()

            with patch(
                "keisei.evaluation.strategies.single_opponent.SingleOpponentEvaluator.evaluate_step"
            ) as mock_evaluate:

                def contention_evaluate(*_args, delay=contention_delay, **_kwargs):
                    time.sleep(delay)
                    return MockGameResultFactory.create_successful_game_result()

                mock_evaluate.side_effect = contention_evaluate

                # Ensure we have a valid checkpoint path
                checkpoint_path = test_agent.checkpoint_path or "dummy_checkpoint.pth"
                result = manager.evaluate_checkpoint(checkpoint_path)

            metrics = performance_monitor.stop_monitoring()
            execution_times.append(metrics["execution_time"])

            assert result.summary_stats.total_games == 20

        # Execution time should scale roughly linearly with contention
        # (though parallel overhead may cause some deviation)
        time_ratios = [
            execution_times[i] / execution_times[0]
            for i in range(1, len(execution_times))
        ]
        delay_ratios = [
            contention_levels[i] / contention_levels[0]
            for i in range(1, len(contention_levels))
        ]

        # Check that time scaling is reasonable (within 2x of expected)
        for time_ratio, delay_ratio in zip(time_ratios, delay_ratios):
            assert (
                time_ratio <= delay_ratio * 2
            ), f"Time scaling {time_ratio:.1f} vs delay ratio {delay_ratio:.1f} too high"

        logger.info(
            "Resource contention test: time ratios %s, delay ratios %s",
            time_ratios,
            delay_ratios,
        )
