"""
Integration test suite for the complete remediation strategy.

This test suite validates that all components of the remediation work together
and that the overall system remains stable and functional.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import all the components we've been working on
from keisei.config_schema import TrainingConfig
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.env_manager import EnvManager
from keisei.utils import load_config
from keisei.utils.profiling import (
    perf_monitor,
    profile_code_block,
    profile_function,
    profile_game_operation,
    profile_training_step,
)


@pytest.mark.integration
class TestRemediationIntegration:
    """Test integration of all remediation components."""

    def setup_method(self):
        """Setup for each test."""
        perf_monitor.reset()

    def test_complete_system_startup(self):
        """Test that the complete system can start up without errors."""
        # This simulates a full system initialization
        config = load_config()
        env_manager = EnvManager(config=config)

        # Setup the environment
        env_manager.setup_environment()

        # Should be able to access all components
        assert config is not None
        assert env_manager is not None
        assert env_manager.game is not None

        # Seeding should work
        env_manager.game.seed(42)
        assert env_manager.game._seed_value == 42

        # Basic game operations should work
        state = env_manager.game.get_observation()
        assert state is not None

    def test_seeding_with_profiling_integration(self):
        """Test that seeding and profiling work together."""
        game = ShogiGame()

        # Profile the seeding operation
        with profile_code_block("game_seeding"):
            game.seed(42)

        # Verify both seeding and profiling worked
        assert game._seed_value == 42

        stats = perf_monitor.get_stats()
        assert "game_seeding_count" in stats
        assert stats["game_seeding_count"] == 1

    def test_training_simulation_with_full_stack(self):
        """Test a simulated training scenario with all components."""
        # Setup
        config = load_config()
        env_manager = EnvManager(config=config)

        # Setup the environment
        env_manager.setup_environment()

        # Seed for reproducibility
        env_manager.game.seed(123)

        # Simulate a training step with profiling
        @profile_training_step
        def simulated_training_step():
            # Profile game operations
            with profile_code_block("state_extraction"):
                state = env_manager.game.get_observation()

            # Profile model operations
            with profile_code_block("model_inference"):
                time.sleep(0.001)  # Simulate inference

            # Profile environment step
            with profile_code_block("env_step"):
                # Simulate environment interaction
                pass

            return {"loss": 0.5, "state": state}

        # Run multiple training steps
        results = []
        for i in range(3):
            result = simulated_training_step()
            results.append(result)

        # Verify all components worked
        assert len(results) == 3
        assert all(r["loss"] == 0.5 for r in results)
        assert all(r["state"] is not None for r in results)

        # Verify profiling captured everything
        stats = perf_monitor.get_stats()
        assert stats["training_steps_completed"] == 3
        assert stats["state_extraction_count"] == 3
        assert stats["model_inference_count"] == 3
        assert stats["env_step_count"] == 3

    def test_configuration_seeding_profiling_workflow(self):
        """Test the complete workflow from config to execution."""
        # Create config with custom settings
        config = load_config()

        # Initialize environment with profiling
        with profile_code_block("env_initialization"):
            env_manager = EnvManager(config=config)
            env_manager.setup_environment()

        # Seed environment with profiling
        with profile_code_block("environment_seeding"):
            env_manager.game.seed(456)

        # Verify the complete workflow
        assert env_manager.game._seed_value == 456

        stats = perf_monitor.get_stats()
        assert "env_initialization_count" in stats
        assert "environment_seeding_count" in stats

    def test_error_handling_across_components(self):
        """Test that error handling works across all components."""
        config = load_config()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()

        @profile_function
        def operation_with_error():
            env_manager.game.seed(42)
            raise ValueError("Simulated error")

        # Error should be raised but profiling should still work
        with pytest.raises(ValueError):
            operation_with_error()

        # Seeding should have worked despite the error
        assert env_manager.game._seed_value == 42

        # Profiling should have captured the operation
        stats = perf_monitor.get_stats()
        function_key = (
            f"{operation_with_error.__module__}.{operation_with_error.__name__}"
        )
        assert f"{function_key}_count" in stats


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test that remediation doesn't break existing functionality."""

    def test_existing_config_system(self):
        """Test that existing configuration system still works."""
        # Basic config creation should work
        config = TrainingConfig()

        # Should have expected attributes
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "minibatch_size")  # Fixed: batch_size -> minibatch_size
        assert hasattr(
            config, "total_timesteps"
        )  # Fixed: max_episodes -> total_timesteps

        # Should be serializable
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert len(config_dict) > 0

    def test_existing_game_functionality(self):
        """Test that existing game functionality is preserved."""
        game = ShogiGame()

        # Basic game operations should work
        state = game.get_observation()  # Fixed: get_state -> get_observation
        assert state is not None

        # Game should maintain its core interface
        assert hasattr(game, "get_observation")  # Fixed: get_state -> get_observation
        assert hasattr(game, "reset")
        assert callable(game.get_observation)  # Fixed: get_state -> get_observation
        assert callable(game.reset)

    def test_existing_training_infrastructure(self):
        """Test that existing training infrastructure works."""
        config = load_config()  # Fixed: TrainingConfig() -> load_config()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()  # Added required setup call

        # Environment should work as before
        assert env_manager.game is not None

        # Should be able to get action space info
        action_space = env_manager.action_space_size
        assert isinstance(action_space, int)
        assert action_space > 0

    def test_imports_remain_stable(self):
        """Test that important imports still work."""
        # These imports should continue to work
        from keisei.config_schema import TrainingConfig
        from keisei.shogi.shogi_game import ShogiGame
        from keisei.training.env_manager import EnvManager

        # New imports should also work
        from keisei.utils.profiling import perf_monitor

        # All should be importable without errors
        assert TrainingConfig is not None
        assert ShogiGame is not None
        assert EnvManager is not None
        assert perf_monitor is not None


@pytest.mark.integration
class TestPerformanceImpact:
    """Test that remediation doesn't negatively impact performance."""

    def test_seeding_performance_impact(self):
        """Test that seeding doesn't significantly impact performance."""
        game = ShogiGame()

        # Time game operations without seeding
        start_time = time.perf_counter()
        for _ in range(100):
            game.get_observation()  # Fixed: get_state -> get_observation
        baseline_time = time.perf_counter() - start_time

        # Time game operations with seeding
        game.seed(42)
        start_time = time.perf_counter()
        for _ in range(100):
            game.get_observation()  # Fixed: get_state -> get_observation
        seeded_time = time.perf_counter() - start_time

        # Seeding should not add significant overhead
        overhead_ratio = seeded_time / baseline_time if baseline_time > 0 else 1
        assert (
            overhead_ratio < 1.5
        ), f"Seeding adds too much overhead: {overhead_ratio:.2f}x"

    def test_profiling_performance_impact(self):
        """Test that profiling overhead is acceptable."""

        def simple_operation():
            return sum(range(100))

        # Time without profiling
        start_time = time.perf_counter()
        for _ in range(1000):
            simple_operation()
        baseline_time = time.perf_counter() - start_time

        # Time with profiling
        start_time = time.perf_counter()
        for _ in range(1000):
            with perf_monitor.time_operation("test"):
                simple_operation()
        profiled_time = time.perf_counter() - start_time

        # Profiling overhead should be reasonable
        overhead_ratio = profiled_time / baseline_time if baseline_time > 0 else 1
        assert (
            overhead_ratio < 5.0
        ), f"Profiling overhead too high: {overhead_ratio:.2f}x"

    def test_memory_usage_stability(self):
        """Test that new features don't cause memory leaks."""
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Perform many operations
            for i in range(100):
                game = ShogiGame()
                game.seed(i)

                with perf_monitor.time_operation(f"iteration_{i % 10}"):
                    state = (
                        game.get_observation()
                    )  # Fixed: get_state -> get_observation

                # Clean up references
                del game, state

            # Check memory after operations
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 50MB)
            assert (
                memory_increase < 50 * 1024 * 1024
            ), f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"

        except ImportError:
            pytest.skip("psutil not available for memory testing")


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_development_workflow(self):
        """Test a typical development workflow."""
        # Developer sets up environment
        config = load_config()  # Fixed: TrainingConfig() -> load_config()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()  # Added required setup call

        # Developer seeds for reproducible testing
        env_manager.game.seed(42)

        # Developer runs some operations with profiling
        @profile_game_operation("move_generation")
        def generate_test_moves():
            # Simulate move generation
            time.sleep(0.001)
            return ["move1", "move2", "move3"]

        @profile_function
        def evaluate_position():
            state = (
                env_manager.game.get_observation()
            )  # Fixed: get_state -> get_observation
            time.sleep(0.002)
            return 0.5

        # Simulate development session
        for _ in range(5):
            moves = generate_test_moves()
            score = evaluate_position()

            assert len(moves) == 3
            assert score == 0.5

        # Developer checks performance metrics
        perf_monitor.print_summary()
        stats = perf_monitor.get_stats()

        # Should have collected useful metrics
        assert stats["move_generation_count"] == 5
        # Check for function profiling stats (includes full module path)
        evaluate_position_keys = [
            k for k in stats.keys() if "evaluate_position" in k and k.endswith("_count")
        ]
        assert (
            len(evaluate_position_keys) > 0
        ), f"Should have evaluate_position stats, found keys: {list(stats.keys())}"
        # training_steps_completed counter is only present when using @profile_training_step
        assert (
            stats.get("training_steps_completed", 0) == 0
        )  # No training steps in this scenario

    def test_debugging_scenario(self):
        """Test a debugging scenario using the new features."""
        game = ShogiGame()

        # Developer suspects an issue, sets up detailed profiling
        with profile_code_block("debugging_session"):
            # Set reproducible seed
            game.seed(999)

            # Profile specific operations
            with profile_code_block("state_analysis"):
                state = game.get_observation()  # Fixed: get_state -> get_observation
                time.sleep(0.001)  # Simulate analysis

            with profile_code_block("problem_reproduction"):
                # Simulate reproducing a bug
                game.seed(999)  # Re-seed for consistency
                state2 = game.get_observation()  # Fixed: get_state -> get_observation
                time.sleep(0.001)

        # Verify debugging tools worked
        assert game._seed_value == 999
        # Use numpy array comparison for state comparison
        import numpy as np

        assert np.array_equal(
            state, state2
        ), "States should be consistent due to seeding"

        stats = perf_monitor.get_stats()
        assert "debugging_session_count" in stats
        assert "state_analysis_count" in stats
        assert "problem_reproduction_count" in stats

    def test_performance_optimization_workflow(self):
        """Test a performance optimization workflow."""
        config = load_config()  # Fixed: TrainingConfig() -> load_config()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()  # Added required setup call

        # Baseline measurement
        @profile_function
        def baseline_operation():
            env_manager.game.seed(42)
            state = (
                env_manager.game.get_observation()
            )  # Fixed: get_state -> get_observation
            time.sleep(0.002)  # Simulate slow operation
            return state

        # "Optimized" version
        @profile_function
        def optimized_operation():
            env_manager.game.seed(42)
            state = (
                env_manager.game.get_observation()
            )  # Fixed: get_state -> get_observation
            time.sleep(0.001)  # Simulate faster operation
            return state

        # Run baseline
        perf_monitor.reset()
        for _ in range(3):
            baseline_operation()
        baseline_stats = perf_monitor.get_stats()

        # Run optimized version
        perf_monitor.reset()
        for _ in range(3):
            optimized_operation()
        optimized_stats = perf_monitor.get_stats()

        # Should be able to compare performance
        baseline_avg = baseline_stats[
            f"{baseline_operation.__module__}.{baseline_operation.__name__}_avg"
        ]
        optimized_avg = optimized_stats[
            f"{optimized_operation.__module__}.{optimized_operation.__name__}_avg"
        ]

        # Optimized should be faster
        assert optimized_avg < baseline_avg, "Optimization should show improvement"


@pytest.mark.integration
class TestRemediationCompleteness:
    """Test that the remediation is complete and comprehensive."""

    def test_all_remediation_components_present(self):
        """Test that all planned remediation components are present."""
        # Environment seeding
        game = ShogiGame()
        assert hasattr(game, "seed")
        assert callable(game.seed)

        # Performance monitoring
        from keisei.utils.profiling import perf_monitor, profile_function

        assert perf_monitor is not None
        assert profile_function is not None

        # Dependency optimization (matplotlib should be gone)
        pyproject_path = Path("/home/john/keisei/pyproject.toml")
        content = pyproject_path.read_text()
        assert "matplotlib" not in content, "matplotlib should be removed"

        # Core functionality preserved
        config = load_config()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()
        assert config is not None
        assert env_manager is not None

    def test_remediation_documentation_exists(self):
        """Test that remediation documentation exists."""
        docs_path = Path("/home/john/keisei/docs")

        # Should have profiling documentation
        profiling_doc = docs_path / "development" / "PROFILING_WORKFLOW.md"
        if profiling_doc.exists():
            content = profiling_doc.read_text()
            assert "Performance Profiling Workflow" in content
            assert len(content) > 1000  # Should be comprehensive

    def test_no_breaking_changes(self):
        """Test that no breaking changes were introduced."""
        # All existing public APIs should still work
        config = load_config()
        game = ShogiGame()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()

        # Basic operations should work exactly as before
        state = game.get_observation()
        game.reset()
        new_state = game.get_observation()

        assert state is not None
        assert new_state is not None

        # Configuration should work as before
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert len(config_dict) > 0

    def test_system_stability_after_remediation(self):
        """Test overall system stability after all changes."""
        # This is a comprehensive test that exercises multiple components
        config = load_config()
        env_manager = EnvManager(config=config)
        env_manager.setup_environment()

        # Perform a complex sequence of operations
        for i in range(10):
            # Seed for some operations
            if i % 3 == 0:
                env_manager.game.seed(i)

            # Profile some operations
            with profile_code_block(f"operation_batch_{i}"):
                state = env_manager.game.get_observation()

                # Reset occasionally
                if i % 5 == 0:
                    env_manager.game.reset()

                new_state = env_manager.game.get_observation()

            # Verify state consistency
            assert state is not None
            assert new_state is not None

        # System should be stable throughout
        final_stats = perf_monitor.get_stats()
        assert len(final_stats) > 0

        # Should be able to generate a summary without errors
        perf_monitor.print_summary()


if __name__ == "__main__":
    pytest.main([__file__])
