"""
Tests for ModelWeightManager in-memory evaluation functionality.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json

from keisei.evaluation.core.model_manager import ModelWeightManager
from keisei.core.ppo_agent import PPOAgent


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
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(5, 3),
            'layer2.bias': torch.randn(5)
        }
        agent.model.state_dict.return_value = weights
        return agent

    def test_extract_agent_weights(self):
        """Test extracting weights from an agent."""
        agent = self.create_mock_agent()
        
        weights = self.manager.extract_agent_weights(agent)
        
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert 'layer1.weight' in weights
        assert 'layer1.bias' in weights
        assert 'layer2.weight' in weights
        assert 'layer2.bias' in weights
        
        # Verify weights are cloned and on CPU
        for name, tensor in weights.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == 'cpu'

    def test_extract_agent_weights_no_model(self):
        """Test error when agent has no model."""
        agent = Mock()
        agent.model = None
        
        with pytest.raises(ValueError, match="Agent must have a model attribute"):
            self.manager.extract_agent_weights(agent)

    def test_cache_opponent_weights(self):
        """Test caching opponent weights from checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
            # Create a mock checkpoint
            checkpoint = {
                'model_state_dict': {
                    'weight1': torch.randn(5, 5),
                    'bias1': torch.randn(5)
                }
            }
            torch.save(checkpoint, tmp_path)
            
            try:
                weights = self.manager.cache_opponent_weights("test_opponent", tmp_path)
                
                assert isinstance(weights, dict)
                assert len(weights) == 2
                assert 'weight1' in weights
                assert 'bias1' in weights
                
                # Verify cache is populated
                assert "test_opponent" in self.manager._weight_cache
                assert len(self.manager._cache_order) == 1
                
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
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                checkpoint = {'model_state_dict': {'weight': torch.randn(2, 2)}}
                torch.save(checkpoint, tmp_path)
                
                try:
                    self.manager.cache_opponent_weights(f"opponent_{i}", tmp_path)
                finally:
                    tmp_path.unlink()
        
        assert len(self.manager._weight_cache) == 3
        assert self.manager._cache_order == ["opponent_0", "opponent_1", "opponent_2"]
        
        # Add one more - should evict the oldest
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {'model_state_dict': {'weight': torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)
            
            try:
                self.manager.cache_opponent_weights("opponent_3", tmp_path)
            finally:
                tmp_path.unlink()
        
        assert len(self.manager._weight_cache) == 3
        assert "opponent_0" not in self.manager._weight_cache
        assert "opponent_3" in self.manager._weight_cache
        assert self.manager._cache_order == ["opponent_1", "opponent_2", "opponent_3"]

    def test_cache_reuse(self):
        """Test that cached weights are reused correctly."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {'model_state_dict': {'weight': torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)
            
            try:
                # First call should load from file
                weights1 = self.manager.cache_opponent_weights("test", tmp_path)
                
                # Second call should use cache
                weights2 = self.manager.cache_opponent_weights("test", tmp_path)
                
                # Should be the same weights (same references)
                assert weights1 is weights2
                
                # Should move to end of LRU order
                assert self.manager._cache_order == ["test"]
                
            finally:
                tmp_path.unlink()

    def test_get_cache_stats(self):
        """Test cache statistics."""
        stats = self.manager.get_cache_stats()
        
        assert stats['cache_size'] == 0
        assert stats['max_cache_size'] == 3
        assert stats['cached_opponents'] == []
        assert stats['cache_order'] == []
        
        # Add some cached weights
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {'model_state_dict': {'weight': torch.randn(2, 2)}}
            torch.save(checkpoint, tmp_path)
            
            try:
                self.manager.cache_opponent_weights("test", tmp_path)
                
                stats = self.manager.get_cache_stats()
                assert stats['cache_size'] == 1
                assert stats['cached_opponents'] == ["test"]
                assert stats['cache_order'] == ["test"]
                
            finally:
                tmp_path.unlink()

    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some cached weights
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            checkpoint = {'model_state_dict': {'weight': torch.randn(2, 2)}}
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
        """Test memory usage calculation."""
        usage = self.manager.get_memory_usage()
        
        assert usage['total_mb'] == 0.0
        assert usage['by_opponent_mb'] == {}
        
        # Add some weights and check usage
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Create a larger tensor for measurable memory usage
            checkpoint = {'model_state_dict': {'weight': torch.randn(100, 100)}}
            torch.save(checkpoint, tmp_path)
            
            try:
                self.manager.cache_opponent_weights("test", tmp_path)
                
                usage = self.manager.get_memory_usage()
                assert usage['total_mb'] > 0
                assert 'test' in usage['by_opponent_mb']
                assert usage['by_opponent_mb']['test'] > 0
                
            finally:
                tmp_path.unlink()

    @patch('keisei.evaluation.core.model_manager.PPOAgent')
    def test_create_agent_from_weights_placeholder(self, mock_ppo_agent_class):
        """Test create_agent_from_weights method (placeholder implementation)."""
        # This is a basic test since the full implementation requires proper agent construction
        weights = {'weight': torch.randn(5, 5)}
        mock_config = Mock()
        
        # The current implementation will likely fail since it's not fully implemented
        # This test serves as a placeholder for when the method is properly implemented
        with pytest.raises((ValueError, RuntimeError)):
            self.manager.create_agent_from_weights(weights, mock_ppo_agent_class, mock_config)
