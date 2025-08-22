"""
Abstract base evaluator class for the Keisei Shogi evaluation system.

This module defines the base interface that all evaluation strategy implementations
must follow, providing a consistent API for running evaluations.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch

from keisei.config_schema import EvaluationConfig
from .evaluation_context import AgentInfo, EvaluationContext, OpponentInfo
from .evaluation_result import EvaluationResult, GameResult

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation strategies.

    This class defines the interface that all evaluation implementations must follow,
    providing consistency across different evaluation approaches and enabling
    polymorphic usage in the training system.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator with configuration.

        Args:
            config: Evaluation configuration object
        """
        self.config = config
        self.context: Optional[EvaluationContext] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for this evaluator."""
        level = getattr(logging, self.config.log_level.upper())
        self.logger.setLevel(level)

    def set_runtime_context(
        self, 
        policy_mapper=None, 
        device: str = None, 
        model_dir: str = None,
        wandb_active: bool = False,
        **kwargs
    ) -> None:
        """
        Set runtime context from training system.
        
        This enables evaluators to access shared training infrastructure
        like the policy mapper, device configuration, and model directories.
        
        Args:
            policy_mapper: Shared PolicyOutputMapper from training
            device: Target device for evaluation
            model_dir: Directory containing model checkpoints
            wandb_active: Whether Weights & Biases logging is active
            **kwargs: Additional runtime context parameters
        """
        if policy_mapper is not None:
            self.policy_mapper = policy_mapper
        if device is not None:
            self.device = device
        if model_dir is not None:
            self.model_dir = model_dir
        self.wandb_active = wandb_active
        
        # Store any additional context
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.logger.debug(
            "Runtime context set: device=%s, model_dir=%s, wandb_active=%s", 
            getattr(self, 'device', 'None'),
            getattr(self, 'model_dir', 'None'), 
            wandb_active
        )

    @abstractmethod
    async def evaluate(
        self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """
        Run evaluation with the specified agent.

        Args:
            agent_info: Information about the agent to evaluate
            context: Optional evaluation context (created if not provided)

        Returns:
            Complete evaluation results
        """
        pass

    async def evaluate_in_memory(
        self,
        agent_info: AgentInfo,
        context: Optional[EvaluationContext] = None,
        *,
        agent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_info: Optional[OpponentInfo] = None,
    ) -> EvaluationResult:
        """
        Run evaluation using in-memory weights (optional optimization).

        Default implementation falls back to regular evaluation.
        Subclasses can override for performance optimization.

        Args:
            agent_info: Information about the agent to evaluate
            context: Optional evaluation context
            agent_weights: Pre-extracted agent model weights
            opponent_weights: Pre-extracted opponent model weights
            opponent_info: Opponent information

        Returns:
            Complete evaluation results
        """
        # Default implementation: fallback to regular evaluation
        return await self.evaluate(agent_info, context)

    @abstractmethod
    async def evaluate_step(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """
        Evaluate a single step (game) against an opponent.

        Args:
            agent_info: Information about the agent being evaluated
            opponent_info: Information about the opponent
            context: Current evaluation context

        Returns:
            Result of the single game
        """
        pass

    def setup_context(
        self, agent_info: AgentInfo, base_context: Optional[EvaluationContext] = None
    ) -> EvaluationContext:
        """
        Set up evaluation context for a run.

        Args:
            agent_info: Information about the agent to evaluate
            base_context: Optional base context to extend

        Returns:
            Configured evaluation context
        """
        if base_context is not None:
            context = base_context
        else:
            # Create new context with required fields
            context = EvaluationContext(
                session_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                agent_info=agent_info,
                configuration=self.config,
                environment_info={},
            )

        # Set up output directory in metadata if needed
        if self.config.save_path:
            output_dir = Path(self.config.save_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            context.metadata["output_dir"] = str(output_dir)

        # Initialize random seed if specified
        if self.config.random_seed is not None:
            import random

            import numpy as np

            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            context.metadata["random_seed"] = self.config.random_seed

        self.context = context
        return context

    @abstractmethod
    def get_opponents(self, context: EvaluationContext) -> List[OpponentInfo]:
        """
        Get list of opponents for this evaluation strategy.

        Each evaluation strategy must implement this method to define
        which opponents the agent will be evaluated against.

        Args:
            context: Current evaluation context

        Returns:
            List of opponent information objects
        """
        pass

    async def run_game(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
        game_index: int = 0,
    ) -> GameResult:
        """
        Run a single game between agent and opponent.

        Args:
            agent_info: Information about the agent
            opponent_info: Information about the opponent
            context: Evaluation context
            game_index: Index of this game in the evaluation

        Returns:
            Game result
        """
        logger.debug(
            f"Running game {game_index}: {agent_info.name} vs {opponent_info.name}"
        )

        try:
            # This would integrate with the actual game engine
            # For now, this is a placeholder that delegates to evaluate_step
            result = await self.evaluate_step(agent_info, opponent_info, context)
            return result

        except Exception as e:
            logger.error(f"Error in game {game_index}: {e}")
            # Return a failed game result
            return GameResult(
                game_id=f"game_{game_index}",
                winner=None,  # Error case
                moves_count=0,
                duration_seconds=0.0,
                agent_info=agent_info,
                opponent_info=opponent_info,
                metadata={"error": str(e), "game_index": game_index},
            )

    async def run_concurrent_games(
        self,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
        context: EvaluationContext,
        max_concurrent: int = 4,
    ) -> AsyncIterator[GameResult]:
        """
        Run multiple games concurrently with controlled parallelism.

        Args:
            agent_info: Information about the agent
            opponents: List of opponents to play against
            context: Evaluation context
            max_concurrent: Maximum number of concurrent games

        Yields:
            Game results as they complete
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(
            opponent: OpponentInfo, game_idx: int
        ) -> GameResult:
            async with semaphore:
                return await self.run_game(agent_info, opponent, context, game_idx)

        # Create tasks for all games
        tasks = []
        game_index = 0

        for opponent in opponents:
            task = asyncio.create_task(run_with_semaphore(opponent, game_index))
            tasks.append(task)
            game_index += 1

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result

    def validate_agent(self, agent_info: AgentInfo) -> bool:
        """
        Validate that the agent is properly configured for evaluation.

        Args:
            agent_info: Agent information to validate

        Returns:
            True if agent is valid, False otherwise
        """
        if not agent_info.name:
            logger.error("Agent name is required")
            return False

        # Only validate checkpoint path if it's provided and not for in-memory evaluation
        if agent_info.checkpoint_path:
            try:
                checkpoint_path = Path(agent_info.checkpoint_path)
                if not checkpoint_path.exists():
                    # Check if this might be in-memory evaluation by looking for agent instance
                    if not agent_info.metadata.get("agent_instance"):
                        logger.error(
                            f"Agent checkpoint path does not exist: {agent_info.checkpoint_path}"
                        )
                        return False
                    else:
                        logger.debug(
                            "Checkpoint path validation skipped for in-memory evaluation"
                        )
            except (OSError, ValueError) as e:
                logger.error(f"Invalid checkpoint path: {agent_info.checkpoint_path}, error: {e}")
                return False

        return True

    def validate_config(self) -> bool:
        """
        Validate the evaluator configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Basic validation is done in config __post_init__
            # Subclasses can override for strategy-specific validation
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def log_evaluation_start(self, agent_info: AgentInfo, context: EvaluationContext):
        """Log the start of an evaluation run."""
        # Handle both string and enum strategy types
        strategy_name = self.config.strategy.value if hasattr(self.config.strategy, 'value') else self.config.strategy
        logger.info(
            f"Starting {strategy_name} evaluation for agent: {agent_info.name}"
        )
        logger.info(
            f"Configuration: {self.config.num_games} games, "
            f"{self.config.max_concurrent_games} concurrent"
        )
        if "output_dir" in context.metadata:
            logger.info(f"Output directory: {context.metadata['output_dir']}")

    def log_evaluation_complete(self, result: EvaluationResult):
        """Log the completion of an evaluation run."""
        stats = result.summary_stats
        logger.info(f"Evaluation complete: {stats.total_games} games played")
        logger.info(
            f"Results: {stats.agent_wins}W-{stats.opponent_wins}L-{stats.draws}D "
            f"(Win rate: {stats.win_rate:.3f})"
        )
        if stats.avg_game_length:
            logger.info(f"Average game length: {stats.avg_game_length:.1f} moves")


class EvaluatorFactory:
    """Factory for creating evaluator instances based on strategy type."""

    _evaluators: Dict[str, type] = {}

    @classmethod
    def register(cls, strategy: str, evaluator_class: type):
        """
        Register an evaluator class for a strategy.

        Args:
            strategy: Strategy name
            evaluator_class: Evaluator class to register
        """
        cls._evaluators[strategy] = evaluator_class

    @classmethod
    def create(cls, config: EvaluationConfig) -> BaseEvaluator:
        """
        Create evaluator instance based on configuration.

        Args:
            config: Evaluation configuration

        Returns:
            Configured evaluator instance
        """
        # Handle both enum and string strategy types
        if hasattr(config.strategy, "value"):
            strategy_name = config.strategy.value
        else:
            strategy_name = str(config.strategy)

        if strategy_name not in cls._evaluators:
            raise ValueError(f"No evaluator registered for strategy: {strategy_name}")

        evaluator_class = cls._evaluators[strategy_name]
        return evaluator_class(config)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Get list of available evaluation strategies."""
        return list(cls._evaluators.keys())


# Utility functions for common evaluation tasks
async def evaluate_agent(
    agent_info: AgentInfo,
    config: EvaluationConfig,
    context: Optional[EvaluationContext] = None,
) -> EvaluationResult:
    """
    Convenience function to evaluate an agent with the specified configuration.

    Args:
        agent_info: Information about the agent to evaluate
        config: Evaluation configuration
        context: Optional evaluation context

    Returns:
        Evaluation results
    """
    evaluator = EvaluatorFactory.create(config)
    return await evaluator.evaluate(agent_info, context)


def create_agent_info(
    name: str, checkpoint_path: Optional[str] = None, **kwargs
) -> AgentInfo:
    """
    Convenience function to create agent info.

    Args:
        name: Agent name
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional agent parameters

    Returns:
        Configured agent info
    """
    return AgentInfo(name=name, checkpoint_path=checkpoint_path, **kwargs)
