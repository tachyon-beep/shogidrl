"""
Test suite for performance profiling utilities.

Tests the profiling and monitoring functionality in keisei.utils.profiling.
"""

import threading
import time
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from keisei.utils.profiling import (
    PerformanceMonitor,
    memory_usage_mb,
    perf_monitor,
    profile_code_block,
    profile_function,
    profile_game_operation,
    profile_training_step,
    run_profiler,
)


class TestPerformanceMonitor:
    """Test PerformanceMonitor class functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.monitor = PerformanceMonitor()

    def test_monitor_initialization(self):
        """Test that PerformanceMonitor initializes correctly."""
        assert hasattr(self.monitor, "timings")
        assert hasattr(self.monitor, "counters")
        assert isinstance(self.monitor.timings, dict)
        assert isinstance(self.monitor.counters, dict)
        assert len(self.monitor.timings) == 0
        assert len(self.monitor.counters) == 0

    def test_time_operation_context_manager(self):
        """Test timing operations with context manager."""
        operation_name = "test_operation"

        with self.monitor.time_operation(operation_name):
            time.sleep(0.001)  # Small sleep for measurable time

        # Should have recorded the timing
        assert operation_name in self.monitor.timings
        assert len(self.monitor.timings[operation_name]) == 1
        assert self.monitor.timings[operation_name][0] > 0

    def test_multiple_timing_operations(self):
        """Test multiple timing operations."""
        for i in range(3):
            with self.monitor.time_operation("test_op"):
                time.sleep(0.001)

        assert "test_op" in self.monitor.timings
        assert len(self.monitor.timings["test_op"]) == 3
        assert all(t > 0 for t in self.monitor.timings["test_op"])

    def test_increment_counter(self):
        """Test counter functionality."""
        counter_name = "test_counter"

        # Test default increment
        self.monitor.increment_counter(counter_name)
        assert self.monitor.counters[counter_name] == 1

        # Test custom increment
        self.monitor.increment_counter(counter_name, 5)
        assert self.monitor.counters[counter_name] == 6

        # Test new counter
        self.monitor.increment_counter("new_counter", 10)
        assert self.monitor.counters["new_counter"] == 10

    def test_get_stats(self):
        """Test statistics generation."""
        # Add some timing data
        with self.monitor.time_operation("op1"):
            time.sleep(0.001)
        with self.monitor.time_operation("op1"):
            time.sleep(0.001)

        # Add counter data
        self.monitor.increment_counter("counter1", 5)

        stats = self.monitor.get_stats()

        # Should have timing statistics
        assert "op1_avg" in stats
        assert "op1_min" in stats
        assert "op1_max" in stats
        assert "op1_total" in stats
        assert "op1_count" in stats

        # Should have counter statistics
        assert "counter1" in stats
        assert stats["counter1"] == 5

        # Verify timing calculations
        assert stats["op1_count"] == 2
        assert stats["op1_avg"] > 0
        assert stats["op1_min"] > 0
        assert stats["op1_max"] > 0
        assert stats["op1_total"] > 0

    def test_reset(self):
        """Test resetting collected data."""
        # Add some data
        with self.monitor.time_operation("test"):
            time.sleep(0.001)
        self.monitor.increment_counter("test_counter")

        # Verify data exists
        assert len(self.monitor.timings) > 0
        assert len(self.monitor.counters) > 0

        # Reset and verify
        self.monitor.reset()
        assert len(self.monitor.timings) == 0
        assert len(self.monitor.counters) == 0

    def test_print_summary(self):
        """Test summary printing functionality."""
        # Add test data
        with self.monitor.time_operation("test_op"):
            time.sleep(0.001)
        self.monitor.increment_counter("test_counter", 3)

        # Capture output
        with patch("builtins.print") as mock_print:
            self.monitor.print_summary()

            # Should have printed something
            assert mock_print.called

            # Check that relevant information was printed
            printed_text = "".join(str(call) for call in mock_print.call_args_list)
            assert "Performance Summary" in printed_text
            assert "test_op" in printed_text
            assert "test_counter" in printed_text


class TestProfileDecorators:
    """Test profiling decorators."""

    def setup_method(self):
        """Setup test environment."""
        perf_monitor.reset()

    def test_profile_function_decorator(self):
        """Test function profiling decorator."""

        @profile_function
        def test_function(x, y):
            time.sleep(0.001)
            return x + y

        result = test_function(1, 2)

        # Function should work normally
        assert result == 3

        # Should have been profiled
        stats = perf_monitor.get_stats()
        function_key = f"{test_function.__module__}.{test_function.__name__}"
        assert f"{function_key}_count" in stats
        assert stats[f"{function_key}_count"] == 1

    def test_profile_training_step_decorator(self):
        """Test training step profiling decorator."""

        @profile_training_step
        def training_step(batch_data):
            time.sleep(0.001)
            return {"loss": 0.5}

        result = training_step({"data": "test"})

        # Function should work normally
        assert result["loss"] == 0.5

        # Should have been profiled
        stats = perf_monitor.get_stats()
        assert "training_steps_completed" in stats
        assert stats["training_steps_completed"] == 1
        assert "training_step_training_step_count" in stats

    def test_profile_game_operation_decorator(self):
        """Test game operation profiling decorator."""

        @profile_game_operation("move_generation")
        def generate_moves():
            time.sleep(0.001)
            return ["move1", "move2"]

        result = generate_moves()

        # Function should work normally
        assert result == ["move1", "move2"]

        # Should have been profiled
        stats = perf_monitor.get_stats()
        assert "move_generation_count" in stats
        assert stats["move_generation_count"] == 1
        assert "move_generation_count" in stats

    def test_profile_code_block_context_manager(self):
        """Test code block profiling context manager."""
        with profile_code_block("test_block"):
            time.sleep(0.001)
            result = 42

        # Should have been profiled
        stats = perf_monitor.get_stats()
        assert "test_block_count" in stats
        assert stats["test_block_count"] == 1


class TestAdvancedProfiling:
    """Test advanced profiling functionality."""

    def test_run_profiler(self):
        """Test cProfile integration."""

        def test_function():
            return sum(i**2 for i in range(100))

        result, profile_output = run_profiler(test_function)

        # Should return correct result
        expected = sum(i**2 for i in range(100))
        assert result == expected

        # Should have profile output
        assert isinstance(profile_output, str)
        assert len(profile_output) > 0
        assert "function calls" in profile_output.lower()

    def test_run_profiler_with_arguments(self):
        """Test cProfile with function arguments."""

        def add_numbers(a, b, c=0):
            return a + b + c

        result, profile_output = run_profiler(add_numbers, 1, 2, c=3)

        assert result == 6
        assert isinstance(profile_output, str)

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        initial_memory = memory_usage_mb()

        # Should return a positive number
        assert isinstance(initial_memory, float)
        assert initial_memory > 0

        # Create some memory usage
        large_list = [i for i in range(10000)]
        current_memory = memory_usage_mb()

        # Memory should have increased (though this might be flaky)
        assert current_memory >= initial_memory

        # Clean up
        del large_list


class TestProfilingIntegration:
    """Test profiling integration with real components."""

    def setup_method(self):
        """Setup test environment."""
        perf_monitor.reset()

    def test_global_monitor_instance(self):
        """Test that global monitor instance works correctly."""
        # The global instance should be accessible
        assert perf_monitor is not None
        assert isinstance(perf_monitor, PerformanceMonitor)

        # Should work with timing operations
        with perf_monitor.time_operation("global_test"):
            time.sleep(0.001)

        stats = perf_monitor.get_stats()
        assert "global_test_count" in stats

    def test_concurrent_profiling(self):
        """Test profiling with concurrent operations."""

        def worker_function(worker_id):
            with perf_monitor.time_operation(f"worker_{worker_id}"):
                time.sleep(0.001)
                perf_monitor.increment_counter("worker_operations")

        # Run multiple workers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        stats = perf_monitor.get_stats()
        assert stats["worker_operations"] == 3

    def test_profiling_error_handling(self):
        """Test profiling behavior with exceptions."""

        @profile_function
        def function_with_error():
            time.sleep(0.001)
            raise ValueError("Test error")

        # Function should still raise the exception
        with pytest.raises(ValueError):
            function_with_error()

        # But profiling should still have recorded the timing
        stats = perf_monitor.get_stats()
        function_key = (
            f"{function_with_error.__module__}.{function_with_error.__name__}"
        )
        assert f"{function_key}_count" in stats

    def test_nested_profiling_operations(self):
        """Test nested profiling operations."""

        @profile_function
        def outer_function():
            with perf_monitor.time_operation("inner_operation"):
                time.sleep(0.001)
            return "result"

        result = outer_function()

        assert result == "result"

        stats = perf_monitor.get_stats()

        # Should have both outer and inner operations
        outer_key = f"{outer_function.__module__}.{outer_function.__name__}"
        assert f"{outer_key}_count" in stats
        assert "inner_operation_count" in stats

    def test_profiling_with_real_training_simulation(self):
        """Test profiling with simulated training components."""

        @profile_training_step
        def simulated_training_step(batch):
            # Simulate forward pass
            with perf_monitor.time_operation("forward_pass"):
                time.sleep(0.001)

            # Simulate backward pass
            with perf_monitor.time_operation("backward_pass"):
                time.sleep(0.001)

            return {"loss": 0.5}

        # Simulate multiple training steps
        for i in range(3):
            result = simulated_training_step({"batch": i})
            assert result["loss"] == 0.5

        stats = perf_monitor.get_stats()

        # Should track training steps
        assert stats["training_steps_completed"] == 3

        # Should track sub-operations
        assert stats["forward_pass_count"] == 3
        assert stats["backward_pass_count"] == 3

        # Should have timing information
        assert "forward_pass_avg" in stats
        assert "backward_pass_avg" in stats


class TestProfilingEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup test environment."""
        self.monitor = PerformanceMonitor()

    def test_empty_statistics(self):
        """Test behavior with no collected data."""
        stats = self.monitor.get_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0

        # Should handle empty summary gracefully
        with patch("builtins.print") as mock_print:
            self.monitor.print_summary()
            assert mock_print.called

    def test_rapid_consecutive_operations(self):
        """Test rapid consecutive timing operations."""
        for i in range(100):
            with self.monitor.time_operation("rapid_op"):
                pass  # No sleep, minimal time

        stats = self.monitor.get_stats()
        assert stats["rapid_op_count"] == 100
        assert stats["rapid_op_avg"] >= 0

    def test_long_operation_names(self):
        """Test with very long operation names."""
        long_name = "very_long_operation_name_" * 10

        with self.monitor.time_operation(long_name):
            time.sleep(0.001)

        stats = self.monitor.get_stats()
        assert f"{long_name}_count" in stats

    @pytest.mark.parametrize(
        "operation_name",
        [
            "op-with-dashes",
            "op_with_underscores", 
            "op.with.dots",
            "op with spaces",
        ],
        ids=["dashes", "underscores", "dots", "spaces"],
    )
    def test_special_characters_in_names(self, operation_name):
        """Test operation names with special characters."""
        with self.monitor.time_operation(operation_name):
            time.sleep(0.001)

        stats = self.monitor.get_stats()
        # Operation should be recorded with the exact name provided (no normalization)
        expected_count_key = f"{operation_name}_count"
        assert expected_count_key in stats, f"No stats found for operation '{operation_name}'. Available keys: {list(stats.keys())}"
        assert stats[expected_count_key] == 1


@pytest.mark.performance
class TestProfilingPerformance:
    """Test profiling overhead and performance."""

    def test_profiling_overhead(self):
        """Test that profiling doesn't add significant overhead."""

        def simple_operation():
            return sum(range(1000))

        # Time without profiling
        start_time = time.perf_counter()
        for _ in range(100):
            simple_operation()
        unmonitored_time = time.perf_counter() - start_time

        # Time with profiling
        monitor = PerformanceMonitor()
        start_time = time.perf_counter()
        for _ in range(100):
            with monitor.time_operation("test"):
                simple_operation()
        monitored_time = time.perf_counter() - start_time

        # Overhead should be reasonable (less than 100% increase)
        overhead_ratio = monitored_time / unmonitored_time
        assert (
            overhead_ratio < 2.0
        ), f"Profiling overhead too high: {overhead_ratio:.2f}x"

    def test_memory_efficiency(self):
        """Test that profiling doesn't leak memory."""
        monitor = PerformanceMonitor()

        # Generate lots of profiling data
        for i in range(1000):
            with monitor.time_operation(f"op_{i % 10}"):  # Reuse operation names
                pass
            monitor.increment_counter(f"counter_{i % 5}")

        # Memory usage should be reasonable
        stats = monitor.get_stats()

        # Should have consolidated data, not 1000 separate operations
        timing_keys = [k for k in stats.keys() if k.endswith("_count")]
        assert len(timing_keys) <= 20  # Should reuse operation names


if __name__ == "__main__":
    pytest.main([__file__])
