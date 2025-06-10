"""
Model weight management for in-memory evaluation.

This module provides functionality to extract, cache, and manage model weights
for efficient in-memory evaluation without file I/O overhead.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from keisei.core.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class ModelWeightManager:
    """Manages model weights for in-memory evaluation."""

    def __init__(self, device: str = "cpu", max_cache_size: int = 5):
        """
        Initialize the ModelWeightManager.

        Args:
            device: Device to store cached weights on
            max_cache_size: Maximum number of opponent weights to cache
        """
        self.device = torch.device(device)
        self.max_cache_size = max_cache_size
        self._weight_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: List[str] = []  # For LRU eviction

    def extract_agent_weights(self, agent: PPOAgent) -> Dict[str, torch.Tensor]:
        """
        Extract and clone current agent weights for evaluation.

        Args:
            agent: PPOAgent instance to extract weights from

        Returns:
            Dictionary of cloned model weights on CPU

        Raises:
            ValueError: If agent doesn't have a model attribute
        """
        if not hasattr(agent, 'model') or agent.model is None:
            raise ValueError("Agent must have a model attribute")

        # Clone weights to avoid interference with training
        weights = {}
        for name, param in agent.model.state_dict().items():
            weights[name] = param.clone().detach().cpu()

        logger.debug(f"Extracted {len(weights)} weight tensors from agent")
        return weights

    def cache_opponent_weights(self, opponent_id: str, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
        """
        Load and cache opponent weights from checkpoint.

        Args:
            opponent_id: Unique identifier for the opponent
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary of model weights

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        # Check if already cached
        if opponent_id in self._weight_cache:
            # Move to end for LRU
            self._cache_order.remove(opponent_id)
            self._cache_order.append(opponent_id)
            logger.debug(f"Using cached weights for opponent {opponent_id}")
            return self._weight_cache[opponent_id]

        # Evict oldest if cache is full
        while len(self._weight_cache) >= self.max_cache_size:
            oldest_id = self._cache_order.pop(0)
            del self._weight_cache[oldest_id]
            logger.debug(f"Evicted cached weights for opponent {oldest_id}")

        # Load checkpoint
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model weights from checkpoint
            if 'model_state_dict' in checkpoint:
                weights = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                weights = checkpoint['state_dict']
            else:
                weights = checkpoint

            # Ensure weights are on CPU
            cpu_weights = {}
            for name, param in weights.items():
                if isinstance(param, torch.Tensor):
                    cpu_weights[name] = param.cpu()
                else:
                    cpu_weights[name] = param

            # Cache the weights
            self._weight_cache[opponent_id] = cpu_weights
            self._cache_order.append(opponent_id)
            
            logger.debug(f"Cached weights for opponent {opponent_id} ({len(cpu_weights)} tensors)")
            return cpu_weights

        except Exception as e:
            logger.error(f"Failed to load weights from {checkpoint_path}: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    def create_agent_from_weights(
        self,
        weights: Dict[str, torch.Tensor],
        agent_class=PPOAgent,
        config: Any = None,
        device: Optional[str] = None
    ) -> PPOAgent:
        """
        Create an agent instance from weights.

        Args:
            weights: Dictionary of model weights
            agent_class: Agent class to instantiate
            config: Configuration for agent creation
            device: Device to place the agent on

        Returns:
            Agent instance with loaded weights

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If agent creation fails
        """
        # This is a placeholder implementation - full implementation requires:
        # 1. Proper model reconstruction from weights
        # 2. Agent-specific configuration handling
        # 3. Model architecture compatibility checks
        
        raise RuntimeError(
            "create_agent_from_weights is not yet fully implemented. "
            "This method requires proper model reconstruction from state dictionaries, "
            "which needs coordination with the actual agent and model architecture."
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._weight_cache),
            'max_cache_size': self.max_cache_size,
            'device': str(self.device),
            'cached_opponents': list(self._weight_cache.keys()),
            'cache_order': self._cache_order.copy()
        }

    def clear_cache(self) -> None:
        """Clear the weight cache."""
        self._weight_cache.clear()
        self._cache_order.clear()
        logger.debug("Cleared weight cache")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get estimated memory usage of cached weights.

        Returns:
            Dictionary with memory usage statistics in MB
        """
        total_size = 0
        sizes_by_opponent = {}

        for opponent_id, weights in self._weight_cache.items():
            opponent_size = 0
            for name, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    opponent_size += tensor.numel() * tensor.element_size()
            
            sizes_by_opponent[opponent_id] = opponent_size / (1024 * 1024)  # Convert to MB
            total_size += opponent_size

        return {
            'total_mb': total_size / (1024 * 1024),
            'by_opponent_mb': sizes_by_opponent
        }
