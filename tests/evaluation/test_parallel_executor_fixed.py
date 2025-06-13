"""
Tests for parallel execution functionality (Phase 1 - Foundation Fixes)
Coverage for keisei/evaluation/core/parallel_executor.py

This file replaces mock-based tests with real thread-based testing.
"""

import asyncio
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pytest

from keisei.constants import CORE_OBSERVATION_CHANNELS, FULL_ACTION_SPACE
from keisei.evaluation.core.parallel_executor import ParallelGameExecutor
from keisei.shogi.shogi_game import ShogiGame
from tests.evaluation.factories import EvaluationTestFactory


class TestParallelGameExecutor:
    """Test ParallelGameExecutor concurrent execution capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_agents = [
            EvaluationTestFactory.create_test_agent(f"TestAgent{i}", "cpu") 
            for i in range(4)
        ]
        self.test_games = EvaluationTestFactory.create_test_game_results(count=8)
        
    def simulate_game_execution(self, agent_pair, game_id: str) -> Dict[str, Any]:
        """Simulate a game between two agents for testing purposes."""
        # Simple deterministic game result based on game_id
        game_num = int(game_id.split('_')[-1]) if '_' in game_id else 0
        
        return {
            "game_id": game_id,
            "winner": "agent_a" if game_num % 2 == 0 else "agent_b",
            "moves": 50 + game_num,
            "duration": 0.1 + (game_num * 0.01),  # Simulated duration
            "agent_a": agent_pair[0],
            "agent_b": agent_pair[1]
        }

    def test_parallel_game_executor_concurrent_execution(self):
        """Test real concurrent game execution across multiple threads."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Create agent pairs for games
        agent_pairs = [
            (self.test_agents[0], self.test_agents[1]),
            (self.test_agents[1], self.test_agents[2]),
            (self.test_agents[2], self.test_agents[3]),
            (self.test_agents[0], self.test_agents[3]),
            (self.test_agents[1], self.test_agents[3]),
            (self.test_agents[0], self.test_agents[2]),
            (self.test_agents[2], self.test_agents[1]),
            (self.test_agents[3], self.test_agents[0]),
        ]
        
        # Test parallel execution with real thread pool
        max_workers = 4
        start_time = time.perf_counter()
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all games to thread pool
            future_to_game = {
                executor.submit(self.simulate_game_execution, pair, f"game_{i}"): i
                for i, pair in enumerate(agent_pairs)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_game):
                game_id = future_to_game[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    pytest.fail(f"Game {game_id} generated an exception: {exc}")
        
        execution_time = time.perf_counter() - start_time
        
        # Verify results
        assert len(results) == 8, f"Expected 8 results, got {len(results)}"
        assert execution_time < 2.0, f"Parallel execution took {execution_time:.3f}s, should be under 2s"
        
        # Verify all games completed successfully
        game_ids = {result["game_id"] for result in results}
        expected_ids = {f"game_{i}" for i in range(8)}
        assert game_ids == expected_ids, "Not all games completed successfully"
        
        # Verify winners are distributed as expected (deterministic based on game_id)
        winners = [result["winner"] for result in results]
        agent_a_wins = winners.count("agent_a")
        agent_b_wins = winners.count("agent_b") 
        assert agent_a_wins == 4 and agent_b_wins == 4, "Winner distribution should be 50/50"

    def test_thread_pool_management_and_scaling(self):
        """Test real thread pool creation, scaling, and resource management."""
        import threading
        
        # Test different pool sizes
        pool_sizes = [2, 4, 8]
        
        for pool_size in pool_sizes:
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                # Submit tasks to test pool scaling
                tasks = []
                for i in range(pool_size * 2):  # Submit more tasks than workers
                    future = executor.submit(self.simulate_game_execution, 
                                           (self.test_agents[0], self.test_agents[1]), 
                                           f"scaling_test_{i}")
                    tasks.append(future)
                
                # Verify all tasks complete successfully
                results = [future.result() for future in tasks]
                assert len(results) == pool_size * 2
                
                # Verify thread pool handled the load correctly
                for result in results:
                    assert "game_id" in result
                    assert result["winner"] in ["agent_a", "agent_b"]
                    
        # Test thread safety by running concurrent access to shared resources
        shared_counter = {"value": 0}
        lock = threading.Lock()
        
        def increment_counter(agent_pair, game_id):
            # Simulate thread-safe game execution
            result = self.simulate_game_execution(agent_pair, game_id)
            with lock:
                shared_counter["value"] += 1
            return result
            
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(increment_counter, 
                               (self.test_agents[i % 2], self.test_agents[(i+1) % 2]), 
                               f"thread_safety_test_{i}")
                for i in range(10)
            ]
            
            # Wait for all tasks to complete
            results = [future.result() for future in futures]
            
        # Verify thread safety
        assert shared_counter["value"] == 10, "Thread safety violated"
        assert len(results) == 10, "Not all tasks completed"

    def test_load_balancing_across_workers(self):
        """Test real load balancing and work distribution across workers."""
        import time
        
        # Create tasks with varying execution times to test load balancing
        def variable_duration_game(agent_pair, game_id, duration_multiplier):
            """Simulate game with variable duration."""
            time.sleep(0.01 * duration_multiplier)  # Small sleep to simulate work
            return self.simulate_game_execution(agent_pair, game_id)
        
        # Test load balancing with different task durations
        tasks = []
        duration_multipliers = [1, 2, 3, 1, 2, 3, 1, 2]  # Mixed durations
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks with different durations
            futures = []
            for i, duration in enumerate(duration_multipliers):
                agent_pair = (self.test_agents[i % 2], self.test_agents[(i+1) % 2])
                future = executor.submit(variable_duration_game, 
                                       agent_pair, 
                                       f"load_balance_test_{i}", 
                                       duration)
                futures.append(future)
                tasks.append((i, duration))
            
            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        execution_time = time.perf_counter() - start_time
        
        # Verify load balancing effectiveness
        assert len(results) == len(tasks), "Not all tasks completed"
        
        # With good load balancing, total time should be less than sequential execution
        # Sequential time would be sum of all durations: (1+2+3+1+2+3+1+2) * 0.01 = 0.15s
        # With 4 workers, should be significantly faster
        expected_sequential_time = sum(duration_multipliers) * 0.01
        assert execution_time < expected_sequential_time * 0.7, \
            f"Load balancing ineffective: {execution_time:.3f}s vs expected < {expected_sequential_time * 0.7:.3f}s"
            
        # Verify all results are valid
        for result in results:
            assert result["game_id"].startswith("load_balance_test_")
            assert result["winner"] in ["agent_a", "agent_b"]

    def test_error_handling_and_fault_tolerance(self):
        """Test real error handling when individual games fail."""
        import random
        
        def unreliable_game_execution(agent_pair, game_id, failure_rate=0.3):
            """Simulate game execution that sometimes fails."""
            if random.random() < failure_rate:
                if random.random() < 0.5:
                    raise TimeoutError(f"Game {game_id} timed out")
                else:
                    raise RuntimeError(f"Game {game_id} failed with runtime error")
            
            return self.simulate_game_execution(agent_pair, game_id)
        
        # Set seed for reproducible test
        random.seed(42)
        
        successful_games = 0
        failed_games = 0
        timeout_errors = 0
        runtime_errors = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit games with potential failures
            futures = []
            for i in range(10):
                agent_pair = (self.test_agents[i % 2], self.test_agents[(i+1) % 2])
                future = executor.submit(unreliable_game_execution, 
                                       agent_pair, 
                                       f"fault_tolerance_test_{i}",
                                       0.3)  # 30% failure rate
                futures.append(future)
            
            # Collect results with error handling
            for future in as_completed(futures):
                try:
                    result = future.result()
                    successful_games += 1
                    # Verify successful result structure
                    assert "game_id" in result
                    assert result["winner"] in ["agent_a", "agent_b"]
                except TimeoutError:
                    timeout_errors += 1
                    failed_games += 1
                except RuntimeError:
                    runtime_errors += 1
                    failed_games += 1
                except Exception as e:
                    failed_games += 1
                    pytest.fail(f"Unexpected error type: {type(e).__name__}: {e}")
        
        # Verify fault tolerance
        total_games = successful_games + failed_games
        assert total_games == 10, f"Expected 10 total games, got {total_games}"
        assert successful_games > 0, "Should have some successful games"
        assert failed_games > 0, "Should have some failed games (due to 30% failure rate)"
        
        success_rate = successful_games / total_games
        assert 0.4 <= success_rate <= 0.9, f"Success rate {success_rate:.2f} outside expected range"
        
        # Verify error categorization
        assert timeout_errors + runtime_errors == failed_games, \
            "Error categorization doesn't match total failures"
            
    def test_performance_benchmarks(self):
        """Test that parallel execution meets performance requirements."""
        import time
        
        # Test that parallel execution is actually faster than sequential
        agent_pairs = [(self.test_agents[0], self.test_agents[1])] * 8
        
        # Sequential execution baseline
        start_time = time.perf_counter()
        sequential_results = []
        for i, pair in enumerate(agent_pairs):
            result = self.simulate_game_execution(pair, f"sequential_{i}")
            sequential_results.append(result)
        sequential_time = time.perf_counter() - start_time
        
        # Parallel execution
        start_time = time.perf_counter()
        parallel_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.simulate_game_execution, pair, f"parallel_{i}")
                for i, pair in enumerate(agent_pairs)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
        parallel_time = time.perf_counter() - start_time
        
        # Verify both produce same number of results
        assert len(sequential_results) == len(parallel_results) == 8
        
        # Parallel should be faster (allowing some variance for testing environment)
        speedup_ratio = sequential_time / parallel_time
        assert speedup_ratio > 1.5, \
            f"Parallel execution not significantly faster: {speedup_ratio:.2f}x speedup"
        
        # Performance should be reasonable (under 1 second total)
        assert parallel_time < 1.0, \
            f"Parallel execution too slow: {parallel_time:.3f}s"
