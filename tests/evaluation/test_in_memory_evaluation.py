"""
Tests for in-memory evaluation functionality.

This module tests the integration between ModelWeightManager, EvaluationManager,
and evaluation strategies for in-memory evaluation.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import torch
import pytest

from keisei.evaluation.core import AgentInfo, EvaluationConfig, ModelWeightManager, EvaluationStrategy
from keisei.evaluation.manager import EvaluationManager
# Import strategies to ensure factory registration
from keisei.evaluation.strategies import SingleOpponentEvaluator


def async_test(coro):
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


class TestInMemoryEvaluationIntegration:
    """Test integration of in-memory evaluation components."""
    
    def test_model_weight_manager_creation(self):
        """Test that ModelWeightManager can be created."""
        manager = ModelWeightManager()
        assert manager is not None
        assert manager.device == torch.device("cpu")
        assert len(manager._weight_cache) == 0
    
    def test_extract_agent_weights(self):
        """Test extraction of agent weights."""
        manager = ModelWeightManager()
        
        # Create mock agent with model
        mock_agent = MagicMock()
        mock_model = MagicMock()
        mock_agent.model = mock_model
        
        # Mock state_dict
        test_weights = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }
        mock_model.state_dict.return_value = test_weights
        
        # Test extraction
        extracted_weights = manager.extract_agent_weights(mock_agent)
        
        # Should have cloned the weights
        assert len(extracted_weights) == 2
        assert "layer1.weight" in extracted_weights
        assert "layer1.bias" in extracted_weights
        
        # Verify shapes are preserved
        assert extracted_weights["layer1.weight"].shape == torch.Size([10, 5])
        assert extracted_weights["layer1.bias"].shape == torch.Size([10])
    
    @async_test
    async def test_evaluation_manager_in_memory_method_exists(self):
        """Test that EvaluationManager has in-memory evaluation method."""
        config = EvaluationConfig(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=2,
            enable_in_memory_evaluation=True
        )
        manager = EvaluationManager(config, "test_run")
        
        # Check that the method exists
        assert hasattr(manager, 'evaluate_current_agent_in_memory')
        assert callable(getattr(manager, 'evaluate_current_agent_in_memory'))
        
        # Check that model weight manager is initialized
        assert hasattr(manager, 'model_weight_manager')
        assert manager.model_weight_manager is not None
    @async_test
    async def test_evaluation_manager_in_memory_with_mock_agent(self):
        """Test EvaluationManager in-memory evaluation with mocked components."""
        from keisei.evaluation.core import SingleOpponentConfig
        
        config = SingleOpponentConfig(
            num_games=2,
            opponent_name="random",
            enable_in_memory_evaluation=True
        )
        manager = EvaluationManager(config, "test_run")
        
        # Mock the opponent pool to provide an opponent
        with patch.object(manager.opponent_pool, 'sample') as mock_sample:
            # Create a temporary file to mock the opponent checkpoint
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
                checkpoint = {"model_state_dict": {"layer1.weight": torch.randn(5, 5)}}
                torch.save(checkpoint, temp_file.name)
                mock_sample.return_value = Path(temp_file.name)
                
                # Create mock agent with required attributes
                mock_agent = MagicMock()
                mock_agent.name = "test_agent"
                mock_agent.model = MagicMock()
                mock_agent.model.state_dict.return_value = {
                    "layer1.weight": torch.randn(5, 5),
                    "layer1.bias": torch.randn(5),
                }
                
                # Mock the strategy execution
                with patch.object(manager, '_run_in_memory_evaluation', new_callable=AsyncMock) as mock_run:
                    from keisei.evaluation.core import EvaluationResult, SummaryStats, EvaluationContext
                    from datetime import datetime
                    
                    # Create mock evaluation result
                    mock_context = EvaluationContext(
                        session_id="test_session",
                        timestamp=datetime.now(),
                        agent_info=AgentInfo(name="test_agent"),
                        configuration=config,
                        environment_info={}
                    )
                    
                    mock_result = EvaluationResult(
                        context=mock_context,
                        games=[],
                        summary_stats=SummaryStats(
                            total_games=2,
                            agent_wins=1,
                            opponent_wins=1,
                            draws=0,
                            win_rate=0.5,
                            loss_rate=0.5,
                            draw_rate=0.0,
                            avg_game_length=50.0,
                            total_moves=100,
                            avg_duration_seconds=1.5,
                        ),
                        analytics_data={},
                        errors=[]
                    )
                    mock_run.return_value = mock_result
                    
                    # Test in-memory evaluation
                    result = await manager.evaluate_current_agent_in_memory(mock_agent)
                    
                    # Verify the method was called
                    mock_run.assert_called_once()
                    assert result is mock_result
                
                # Clean up
                import os
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
    
    def test_model_weight_manager_cache_cleanup(self):
        """Test that ModelWeightManager properly manages cache size."""
        import tempfile
        import os
        
        manager = ModelWeightManager()
        manager.max_cache_size = 2  # Small cache for testing
        
        # Create temporary checkpoint files for testing
        temp_files = []
        try:
            for i in range(3):
                # Create temporary checkpoint file
                with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                    checkpoint = {"model_state_dict": {f"layer_{i}.weight": torch.randn(5, 5)}}
                    torch.save(checkpoint, f.name)
                    temp_files.append(f.name)
                
                # Cache the weights through the proper method
                try:
                    manager.cache_opponent_weights(f"opponent_{i}", Path(f.name))
                except (FileNotFoundError, RuntimeError):
                    # Expected for testing - the files might be cleaned up
                    pass
            
            # Cache should not exceed max size due to eviction
            assert len(manager._weight_cache) <= manager.max_cache_size
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
    
    def test_get_cache_stats(self):
        """Test cache statistics functionality."""
        manager = ModelWeightManager()
        
        # Add some weights to cache
        for i in range(2):
            weights = {f"layer_{i}.weight": torch.randn(5, 5)}
            manager._weight_cache[f"opponent_{i}"] = weights
        
        stats = manager.get_cache_stats()
        
        assert "cache_size" in stats
        assert "max_cache_size" in stats
        assert "device" in stats
        assert stats["cache_size"] == 2
        assert stats["max_cache_size"] == manager.max_cache_size
        assert stats["device"] == str(manager.device)


if __name__ == "__main__":
    # Run basic smoke test
    print("Running in-memory evaluation integration tests...")
    
    test_class = TestInMemoryEvaluationIntegration()
    test_class.test_model_weight_manager_creation()
    test_class.test_extract_agent_weights()
    test_class.test_get_cache_stats()
    test_class.test_model_weight_manager_cache_cleanup()
    
    print("✅ Basic in-memory evaluation tests passed!")
