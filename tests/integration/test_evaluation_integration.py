"""
Integration tests for evaluation system integration.
Tests the complete integration of async evaluation with training system.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from keisei.config_schema import AppConfig, EvaluationConfig
from keisei.evaluation.core_manager import EvaluationManager
from keisei.evaluation.performance_manager import (
    EvaluationPerformanceManager,
    EvaluationPerformanceSLA,
    ResourceMonitor
)
from keisei.training.callbacks import AsyncEvaluationCallback
from keisei.training.callback_manager import CallbackManager
from keisei.training.session_manager import SessionManager


class MockTrainer:
    """Mock trainer for testing callbacks."""
    
    def __init__(self):
        self.agent = Mock()
        self.agent.model = Mock()
        self.evaluation_manager = Mock()
        self.evaluation_manager.opponent_pool = Mock()
        self.evaluation_manager.opponent_pool.sample.return_value = "/mock/checkpoint.pt"
        self.evaluation_manager.evaluate_current_agent_async = AsyncMock()
        self.metrics_manager = Mock()
        # Set global_timestep so that (timestep + 1) % interval == 0 for interval=1000
        self.metrics_manager.global_timestep = 999  # So step becomes 1000, and 1000 % 1000 == 0
        self.run_name = "test_run"
        self.log_both = Mock()


class TestAsyncEvaluationIntegration:
    """Test async evaluation integration with training system."""
    
    @pytest.mark.asyncio
    async def test_async_evaluation_callback_execution(self):
        """Test that async evaluation callback executes properly."""
        # Setup - create config without elo_registry_path to avoid Elo registry issues
        eval_config = EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=1000,
            num_games=5,
            elo_registry_path=None  # Disable Elo registry to avoid file system issues
        )
        callback = AsyncEvaluationCallback(eval_config, interval=1000)
        trainer = MockTrainer()
        
        # Mock evaluation result
        mock_result = Mock()
        mock_result.summary_stats = Mock()
        mock_result.summary_stats.win_rate = 0.6
        mock_result.summary_stats.total_games = 5
        mock_result.summary_stats.avg_game_length = 50.0
        trainer.evaluation_manager.evaluate_current_agent_async.return_value = mock_result
        
        # Execute
        result = await callback.on_step_end_async(trainer)
        
        # Verify
        assert result is not None
        assert "evaluation/win_rate" in result
        assert result["evaluation/win_rate"] == 0.6
        assert result["evaluation/total_games"] == 5
        trainer.evaluation_manager.evaluate_current_agent_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_evaluation_callback_skips_when_disabled(self):
        """Test that callback skips when periodic evaluation is disabled."""
        # Setup
        eval_config = EvaluationConfig(
            enable_periodic_evaluation=False,
            evaluation_interval_timesteps=1000,
            elo_registry_path=None
        )
        callback = AsyncEvaluationCallback(eval_config, interval=1000)
        trainer = MockTrainer()
        
        # Execute
        result = await callback.on_step_end_async(trainer)
        
        # Verify
        assert result is None
        trainer.evaluation_manager.evaluate_current_agent_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_async_evaluation_callback_error_handling(self):
        """Test that callback handles errors gracefully."""
        # Setup
        eval_config = EvaluationConfig(
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=1000,
            elo_registry_path=None
        )
        callback = AsyncEvaluationCallback(eval_config, interval=1000)
        trainer = MockTrainer()
        
        # Mock evaluation failure
        trainer.evaluation_manager.evaluate_current_agent_async.side_effect = Exception("Evaluation failed")
        
        # Execute
        result = await callback.on_step_end_async(trainer)
        
        # Verify
        assert result is None  # Should return None on error
        trainer.log_both.assert_called()  # Should log error


class TestCallbackManagerIntegration:
    """Test callback manager async integration."""
    
    def test_callback_manager_async_setup(self):
        """Test that callback manager can setup async callbacks."""
        # Setup
        config = Mock()
        config.evaluation = EvaluationConfig(elo_registry_path=None)
        config.training = Mock()
        config.training.steps_per_epoch = 2048
        config.training.checkpoint_interval_timesteps = 4096  # Make it divisible by steps_per_epoch
        
        callback_manager = CallbackManager(config, "test_model_dir")
        
        # Execute
        async_callbacks = callback_manager.setup_async_callbacks()
        
        # Verify
        assert len(async_callbacks) == 1
        assert isinstance(async_callbacks[0], AsyncEvaluationCallback)
    
    @pytest.mark.asyncio
    async def test_callback_manager_async_execution(self):
        """Test that callback manager can execute async callbacks."""
        # Setup
        config = Mock()
        config.evaluation = EvaluationConfig(
            elo_registry_path=None,
            enable_periodic_evaluation=True,
            evaluation_interval_timesteps=2048  # Match steps_per_epoch to trigger evaluation
        )
        config.training = Mock()
        config.training.steps_per_epoch = 2048
        
        callback_manager = CallbackManager(config, "test_model_dir")
        callback_manager.setup_async_callbacks()
        
        # Mock trainer with proper timestep alignment
        trainer = MockTrainer()
        trainer.metrics_manager.global_timestep = 2047  # So step becomes 2048, and 2048 % 2048 == 0
        
        mock_result = Mock()
        mock_result.summary_stats = Mock()
        mock_result.summary_stats.win_rate = 0.7
        mock_result.summary_stats.total_games = 5
        mock_result.summary_stats.avg_game_length = 45.0
        trainer.evaluation_manager.evaluate_current_agent_async.return_value = mock_result
        
        # Execute
        result = await callback_manager.execute_step_callbacks_async(trainer)
        
        # Verify
        assert result is not None
        assert "evaluation/win_rate" in result
        assert result["evaluation/win_rate"] == 0.7
    
    def test_use_async_evaluation_switch(self):
        """Test switching to async evaluation mode."""
        # Setup
        config = Mock()
        config.evaluation = EvaluationConfig(elo_registry_path=None)
        config.training = Mock()
        config.training.steps_per_epoch = 2048
        config.training.checkpoint_interval_timesteps = 4096  # Make it divisible by steps_per_epoch
        
        callback_manager = CallbackManager(config, "test_model_dir")
        callback_manager.setup_default_callbacks()
        
        # Verify sync callback is present initially
        assert callback_manager.has_async_callbacks() == False
        assert len(callback_manager.get_callbacks()) > 0
        
        # Switch to async
        callback_manager.use_async_evaluation()
        
        # Verify switch
        assert callback_manager.has_async_callbacks() == True
        assert len(callback_manager.get_async_callbacks()) > 0


class TestPerformanceManagerIntegration:
    """Test performance manager integration."""
    
    @pytest.mark.asyncio
    async def test_performance_manager_safeguards(self):
        """Test that performance manager applies safeguards."""
        # Setup
        performance_manager = EvaluationPerformanceManager(
            max_concurrent=2,
            timeout_seconds=10
        )
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate = AsyncMock()
        mock_result = Mock()
        mock_evaluator.evaluate.return_value = mock_result
        
        agent_info = Mock()
        context = Mock()
        
        # Execute
        result = await performance_manager.run_evaluation_with_safeguards(
            mock_evaluator, agent_info, context
        )
        
        # Verify
        assert result == mock_result
        mock_evaluator.evaluate.assert_called_once_with(agent_info, context)
    
    @pytest.mark.asyncio
    async def test_performance_manager_timeout(self):
        """Test that performance manager enforces timeouts."""
        # Setup
        performance_manager = EvaluationPerformanceManager(
            max_concurrent=1,
            timeout_seconds=0.1  # Very short timeout
        )
        
        # Mock evaluator that takes too long
        mock_evaluator = Mock()
        async def slow_evaluate(*args, **kwargs):
            await asyncio.sleep(1)  # Takes longer than timeout
            return Mock()
        
        mock_evaluator.evaluate = slow_evaluate
        
        agent_info = Mock()
        context = Mock()
        
        # Execute and verify timeout
        with pytest.raises(Exception):  # Should raise timeout error
            await performance_manager.run_evaluation_with_safeguards(
                mock_evaluator, agent_info, context
            )
    
    def test_performance_sla_validation(self):
        """Test performance SLA validation."""
        sla_monitor = EvaluationPerformanceSLA()
        
        # Test passing metrics
        good_metrics = {
            "evaluation_latency_ms": 1000,  # Under 5000 limit
            "memory_overhead_mb": 200,      # Under 500 limit
        }
        assert sla_monitor.validate_performance_sla(good_metrics) == True
        
        # Test failing metrics
        bad_metrics = {
            "evaluation_latency_ms": 10000,  # Over 5000 limit
            "memory_overhead_mb": 200,
        }
        assert sla_monitor.validate_performance_sla(bad_metrics) == False
    
    def test_resource_monitor(self):
        """Test resource monitoring functionality."""
        monitor = ResourceMonitor()
        
        # Test basic monitoring
        monitor.start_monitoring()
        memory_usage = monitor.get_memory_usage()
        cpu_percent = monitor.get_cpu_percent()
        
        assert isinstance(memory_usage, (int, float))
        assert memory_usage > 0
        assert isinstance(cpu_percent, (int, float))
        assert cpu_percent >= 0


class TestSessionManagerEvaluationIntegration:
    """Test session manager evaluation logging integration."""
    
    def test_setup_evaluation_logging(self):
        """Test setting up evaluation logging in WandB."""
        # Setup
        config = Mock()
        config.env = Mock()
        config.env.seed = 42
        config.logging = Mock()
        config.logging.run_name = None
        config.training = Mock()
        config.training.model_type = "resnet_tower"  # Provide proper model type string
        config.training.input_features = "core_46_channels"  # Provide proper input features string
        config.wandb = Mock()
        config.wandb.run_name_prefix = "keisei"  # Provide proper wandb config
        
        args = Mock()
        args.run_name = None
        
        eval_config = EvaluationConfig(
            strategy="single_opponent",
            num_games=10,
            enable_periodic_evaluation=True,
            elo_registry_path=None
        )
        
        session_manager = SessionManager(config, args)
        session_manager._is_wandb_active = True
        
        # Mock wandb
        with patch('wandb.run', Mock()):
            with patch('wandb.config', Mock()) as mock_wandb_config:
                # Execute
                session_manager.setup_evaluation_logging(eval_config)
                
                # Verify WandB config was updated
                mock_wandb_config.update.assert_called()
                call_args = mock_wandb_config.update.call_args[0][0]
                assert "evaluation/strategy" in call_args
                assert call_args["evaluation/strategy"] == "single_opponent"
                assert call_args["evaluation/num_games"] == 10
    
    def test_log_evaluation_metrics(self):
        """Test logging evaluation metrics to WandB."""
        # Setup
        config = Mock()
        config.env = Mock()
        config.logging = Mock()
        config.logging.run_name = None
        config.training = Mock()
        config.training.model_type = "resnet_tower"
        config.training.input_features = "core_46_channels"
        config.wandb = Mock()
        config.wandb.run_name_prefix = "keisei"
        
        args = Mock()
        session_manager = SessionManager(config, args)
        session_manager._is_wandb_active = True
        
        # Create a proper mock result that behaves like the expected object
        class MockEvalResult:
            def __init__(self):
                self.summary_stats = Mock()
                self.summary_stats.win_rate = 0.75
                self.summary_stats.total_games = 20
                self.summary_stats.avg_game_length = 60.0
                self.win_rate = 0.75
                self.total_games = 20
                self.avg_game_length = 60.0
            
            def __iter__(self):
                return iter([
                    ("win_rate", 0.75),
                    ("total_games", 20),
                    ("avg_game_length", 60.0)
                ])
        
        mock_result = MockEvalResult()
        
        # Mock wandb
        with patch('wandb.run', Mock()):
            with patch('wandb.log') as mock_wandb_log:
                # Execute
                session_manager.log_evaluation_metrics(mock_result, step=1000)
                
                # Verify metrics were logged
                mock_wandb_log.assert_called_once()
                logged_metrics = mock_wandb_log.call_args[1]  # kwargs
                assert logged_metrics["step"] == 1000
                
                # Check the metrics content
                metrics = mock_wandb_log.call_args[0][0]  # first positional arg
                assert "evaluation/win_rate" in metrics
                assert metrics["evaluation/win_rate"] == 0.75
                assert metrics["evaluation/total_games"] == 20
    
    def test_log_evaluation_performance(self):
        """Test logging performance metrics to WandB."""
        # Setup
        config = Mock()
        config.env = Mock()
        config.logging = Mock()
        config.training = Mock()
        config.training.model_type = "resnet_tower"
        config.training.input_features = "core_46_channels"
        config.wandb = Mock()
        config.wandb.run_name_prefix = "keisei"
        args = Mock()
        
        session_manager = SessionManager(config, args)
        session_manager._is_wandb_active = True
        
        performance_metrics = {
            "latency_ms": 2000,
            "memory_overhead_mb": 300,
            "cpu_utilization_percent": 45.5
        }
        
        # Mock wandb
        with patch('wandb.run', Mock()):
            with patch('wandb.log') as mock_wandb_log:
                # Execute
                session_manager.log_evaluation_performance(performance_metrics, step=1500)
                
                # Verify
                mock_wandb_log.assert_called_once()
                logged_metrics = mock_wandb_log.call_args[0][0]
                assert "evaluation/performance/latency_ms" in logged_metrics
                assert logged_metrics["evaluation/performance/latency_ms"] == 2000


class TestEvaluationManagerAsyncIntegration:
    """Test evaluation manager async integration fixes."""
    
    @pytest.mark.asyncio
    async def test_evaluate_checkpoint_async(self):
        """Test async checkpoint evaluation."""
        # Setup temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            
            # Create a mock checkpoint data
            import torch
            mock_data = {"model_state_dict": {}, "metadata": {"training_step": 1000}}
            torch.save(mock_data, tmp_file.name)
        
        try:
            eval_config = EvaluationConfig(num_games=2, elo_registry_path=None)
            manager = EvaluationManager(eval_config, "test_run")
            manager.setup("cpu", None, str(Path(checkpoint_path).parent), False)
            
            # Mock the evaluator
            with patch('keisei.evaluation.core_manager.EvaluatorFactory') as mock_factory:
                mock_evaluator = Mock()
                mock_evaluator.evaluate = AsyncMock()
                mock_result = Mock()
                mock_evaluator.evaluate.return_value = mock_result
                mock_factory.create.return_value = mock_evaluator
                
                # Execute
                result = await manager.evaluate_checkpoint_async(checkpoint_path)
                
                # Verify
                assert result == mock_result
                mock_evaluator.evaluate.assert_called_once()
                
        finally:
            # Cleanup
            Path(checkpoint_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_evaluate_current_agent_async(self):
        """Test async current agent evaluation."""
        # Setup
        eval_config = EvaluationConfig(num_games=2, elo_registry_path=None)
        manager = EvaluationManager(eval_config, "test_run")
        manager.setup("cpu", None, "/mock/model/dir", False)
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.eval = Mock()
        mock_agent.model.train = Mock()
        mock_agent.name = "test_agent"
        
        # Mock the evaluator
        with patch('keisei.evaluation.core_manager.EvaluatorFactory') as mock_factory:
            mock_evaluator = Mock()
            mock_evaluator.evaluate = AsyncMock()
            mock_result = Mock()
            mock_evaluator.evaluate.return_value = mock_result
            mock_factory.create.return_value = mock_evaluator
            
            # Execute
            result = await manager.evaluate_current_agent_async(mock_agent)
            
            # Verify
            assert result == mock_result
            mock_evaluator.evaluate.assert_called_once()
            mock_agent.model.eval.assert_called_once()
            mock_agent.model.train.assert_called_once()


class TestResourceContentionPrevention:
    """Test resource contention prevention between training and evaluation."""
    
    @pytest.mark.asyncio
    async def test_concurrent_training_evaluation_simulation(self):
        """Simulate concurrent training and evaluation to test resource safety."""
        
        async def mock_training_task():
            """Simulate training workload."""
            for i in range(10):
                await asyncio.sleep(0.01)  # Simulate training step
                yield f"training_step_{i}"
        
        async def mock_evaluation_task():
            """Simulate evaluation workload."""
            await asyncio.sleep(0.05)  # Simulate evaluation time
            return {"win_rate": 0.6, "total_games": 5}
        
        # Execute concurrent tasks
        training_gen = mock_training_task()
        evaluation_task = asyncio.create_task(mock_evaluation_task())
        
        # Collect some training steps while evaluation runs
        training_steps = []
        async for step in training_gen:
            training_steps.append(step)
            if len(training_steps) >= 5:
                break
        
        # Wait for evaluation to complete
        evaluation_result = await evaluation_task
        
        # Verify both completed successfully
        assert len(training_steps) == 5
        assert evaluation_result["win_rate"] == 0.6
        assert all("training_step_" in step for step in training_steps)
    
    def test_performance_manager_concurrency_limit(self):
        """Test that performance manager properly limits concurrency."""
        performance_manager = EvaluationPerformanceManager(max_concurrent=2)
        
        # Verify semaphore is properly configured
        assert performance_manager.semaphore._value == 2
        
        # Test timeout configuration
        assert performance_manager.timeout == 300  # Default timeout
    
    @pytest.mark.asyncio
    async def test_async_callback_no_event_loop_conflict(self):
        """Test that async callbacks don't create event loop conflicts."""
        # This test verifies the fix for the asyncio.run() anti-pattern
        
        # Setup callback in simulated async context
        eval_config = EvaluationConfig(
            enable_periodic_evaluation=True, 
            elo_registry_path=None
        )
        callback = AsyncEvaluationCallback(eval_config, interval=1000)
        trainer = MockTrainer()
        
        # Mock a successful evaluation
        mock_result = Mock()
        mock_result.summary_stats = Mock()
        mock_result.summary_stats.win_rate = 0.8
        mock_result.summary_stats.total_games = 3
        mock_result.summary_stats.avg_game_length = 40.0
        trainer.evaluation_manager.evaluate_current_agent_async.return_value = mock_result
        
        # Execute within async context (should not create event loop conflict)
        result = await callback.on_step_end_async(trainer)
        
        # Verify execution completed successfully
        assert result is not None
        assert "evaluation/win_rate" in result
        assert result["evaluation/win_rate"] == 0.8
        
        # Verify no event loop conflicts by ensuring async call was made
        trainer.evaluation_manager.evaluate_current_agent_async.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])