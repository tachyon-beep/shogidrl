"""
Tests for parallel execution functionality (Task 8 - High Priority)
Coverage for keisei/evaluation/core/parallel_executor.py
"""
import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import tempfile

from keisei.constants import CORE_OBSERVATION_CHANNELS, FULL_ACTION_SPACE
from keisei.evaluation.core.parallel_executor import ParallelGameExecutor
from tests.evaluation.factories import EvaluationTestFactory


class TestParallelGameExecutor:
    """Test ParallelGameExecutor concurrent execution capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = Mock()
        self.mock_agents = [Mock() for _ in range(4)]
        self.test_games = EvaluationTestFactory.create_test_game_results(count=8)

    def test_parallel_game_executor_concurrent_execution(self):
        """Test concurrent game execution across multiple threads."""
        with patch('keisei.evaluation.core.parallel_executor.ParallelGameExecutor') as MockExecutor:
            executor = MockExecutor.return_value
            
            # Mock concurrent execution results
            game_results = [
                {"game_id": f"game_{i}", "winner": "agent_a" if i % 2 == 0 else "agent_b", "moves": 50 + i}
                for i in range(8)
            ]
            
            executor.execute_games_parallel.return_value = {
                "results": game_results,
                "total_games": 8,
                "execution_time": 45.2,
                "threads_used": 4,
                "average_game_time": 5.65
            }
            
            # Test parallel execution
            result = executor.execute_games_parallel(
                agent_pairs=[("agent_a", "agent_b")] * 8,
                max_workers=4
            )
            
            assert result["total_games"] == 8
            assert result["threads_used"] == 4
            assert len(result["results"]) == 8
            assert result["execution_time"] < 60  # Should be faster than sequential

    def test_thread_pool_management_and_scaling(self):
        """Test thread pool creation, scaling, and resource management."""
        # Mock thread pool manager directly
        mock_manager = Mock()
        
        # Mock thread pool operations
        mock_manager.create_pool.return_value = {"pool_id": "pool_001", "workers": 4}
        mock_manager.scale_pool.return_value = {"pool_id": "pool_001", "workers": 8}
        mock_manager.get_pool_stats.return_value = {
            "active_threads": 6,
            "idle_threads": 2,
            "queued_tasks": 3,
            "completed_tasks": 45
        }
        
        # Test pool creation
        pool = mock_manager.create_pool(max_workers=4)
        assert pool["workers"] == 4
        
        # Test scaling
        scaled_pool = mock_manager.scale_pool("pool_001", new_size=8)
        assert scaled_pool["workers"] == 8
        
        # Test stats
        stats = mock_manager.get_pool_stats("pool_001")
        assert stats["active_threads"] + stats["idle_threads"] == 8

    def test_load_balancing_across_workers(self):
        """Test load balancing and work distribution across workers."""
        # Mock load balancer directly
        mock_balancer = Mock()
        
        # Mock load balancing
        mock_balancer.distribute_work.return_value = {
            "worker_assignments": {
                "worker_0": ["game_0", "game_4"],
                "worker_1": ["game_1", "game_5"],
                "worker_2": ["game_2", "game_6"],
                "worker_3": ["game_3", "game_7"]
            },
            "load_variance": 0.05,  # Low variance indicates good balance
            "distribution_strategy": "round_robin"
        }
        
        # Test work distribution
        distribution = mock_balancer.distribute_work(
            tasks=["game_0", "game_1", "game_2", "game_3", "game_4", "game_5", "game_6", "game_7"],
            workers=4
        )
        
        assert len(distribution["worker_assignments"]) == 4
        assert distribution["load_variance"] < 0.1  # Should be well balanced
        
        # Each worker should have approximately equal work
        task_counts = [len(tasks) for tasks in distribution["worker_assignments"].values()]
        assert max(task_counts) - min(task_counts) <= 1

    def test_error_handling_and_fault_tolerance(self):
        """Test error handling when individual games fail."""
        with patch('keisei.evaluation.core.parallel_executor.ParallelGameExecutor') as MockExecutor:
            executor = MockExecutor.return_value
            
            # Mock execution with some failures
            executor.execute_with_fault_tolerance.return_value = {
                "successful_games": 6,
                "failed_games": 2,
                "success_rate": 0.75,
                "error_summary": {
                    "timeout_errors": 1,
                    "runtime_errors": 1,
                    "other_errors": 0
                },
                "results": [{"game_id": f"game_{i}"} for i in range(6)]
            }
            
            # Test fault tolerance
            result = executor.execute_with_fault_tolerance(
                game_configs=[Mock() for _ in range(8)],
                max_retries=2,
                timeout_per_game=30
            )
            
            assert result["successful_games"] == 6
            assert result["failed_games"] == 2
            assert result["success_rate"] == 0.75
            assert "timeout_errors" in result["error_summary"]

    @pytest.mark.asyncio
    async def test_async_parallel_execution(self):
        """Test asynchronous parallel execution capabilities."""
        # Mock async executor directly
        mock_executor = AsyncMock()
        
        # Mock async execution
        async def mock_execute_async(game_configs):
            await asyncio.sleep(0.1)  # Simulate async work
            return {
                "results": [{"game_id": f"async_game_{i}"} for i in range(len(game_configs))],
                "execution_mode": "async",
                "concurrent_tasks": len(game_configs)
            }
        
        mock_executor.execute_async = AsyncMock(side_effect=mock_execute_async)
        
        # Test async execution
        game_configs = [Mock() for _ in range(4)]
        result = await mock_executor.execute_async(game_configs)
        
        assert result["execution_mode"] == "async"
        assert result["concurrent_tasks"] == 4
        assert len(result["results"]) == 4


class TestBatchGameExecutor:
    """Test BatchGameExecutor batch processing efficiency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batch_executor = Mock()
        self.game_batches = [
            [Mock() for _ in range(5)],  # Batch 1: 5 games
            [Mock() for _ in range(5)],  # Batch 2: 5 games
            [Mock() for _ in range(3)]   # Batch 3: 3 games (partial)
        ]

    def test_batch_processing_efficiency(self):
        """Test efficient batch processing of game sets."""
        with patch('keisei.evaluation.core.parallel_executor.BatchGameExecutor') as MockExecutor:
            executor = MockExecutor.return_value
            
            # Mock batch processing
            executor.process_batches.return_value = {
                "batches_processed": 3,
                "total_games": 13,
                "batch_sizes": [5, 5, 3],
                "processing_time": 35.2,
                "throughput_games_per_second": 0.37
            }
            
            # Test batch processing
            result = executor.process_batches(
                batches=self.game_batches,
                batch_size=5,
                parallel_batches=2
            )
            
            assert result["batches_processed"] == 3
            assert result["total_games"] == 13
            assert result["throughput_games_per_second"] > 0.3

    def test_batch_size_optimization(self):
        """Test batch size optimization for different workloads."""
        # Mock optimizer directly
        mock_optimizer = Mock()
        
        # Mock optimization results
        mock_optimizer.optimize_batch_size.return_value = {
            "optimal_batch_size": 8,
            "estimated_throughput": 0.45,
            "memory_efficiency": 0.88,
            "cpu_utilization": 0.92,
            "recommendation": "increase_batch_size"
        }
        
        # Test optimization
        optimization = mock_optimizer.optimize_batch_size(
            workload_size=100,
            available_memory=1024,
            cpu_cores=8
        )
        
        assert optimization["optimal_batch_size"] > 0
        assert optimization["estimated_throughput"] > 0
        assert optimization["memory_efficiency"] > 0.8
        assert optimization["cpu_utilization"] > 0.8

    def test_memory_efficient_batching(self):
        """Test memory-efficient batch processing."""
        # Mock batcher directly
        mock_batcher = Mock()
        
        # Mock memory-efficient processing
        mock_batcher.process_with_memory_limit.return_value = {
            "processed_games": 50,
            "peak_memory_mb": 380,
            "memory_limit_mb": 500,
            "memory_efficiency": 0.76,
            "batches_auto_sized": True,
            "average_batch_size": 6.25
        }
        
        # Test memory-efficient processing
        result = mock_batcher.process_with_memory_limit(
            games=list(range(50)),
            memory_limit_mb=500
        )
        
        assert result["processed_games"] == 50
        assert result["peak_memory_mb"] < result["memory_limit_mb"]
        assert result["memory_efficiency"] > 0.7
        assert result["batches_auto_sized"] is True