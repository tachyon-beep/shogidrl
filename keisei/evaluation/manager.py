from __future__ import annotations

"""EvaluationManager for new evaluation system."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .core import (
    AgentInfo,
    EvaluationContext,
    EvaluationResult,
    EvaluatorFactory,
    EvaluationConfig,
    ModelWeightManager,
    OpponentInfo,
)
from .opponents import OpponentPool


class EvaluationManager:
    """Manage evaluator creation and execution."""

    def __init__(self, config: EvaluationConfig, run_name: str,
                 pool_size: int = 5, elo_registry_path: str | None = None) -> None:
        self.config = config
        self.run_name = run_name
        self.device = "cpu"
        self.policy_mapper = None
        self.model_dir: str | None = None
        self.wandb_active = False
        self.opponent_pool = OpponentPool(pool_size, elo_registry_path)
        
        # Initialize model weight manager for in-memory evaluation
        self.model_weight_manager = ModelWeightManager(
            device=getattr(config, 'temp_agent_device', 'cpu'),
            max_cache_size=getattr(config, 'model_weight_cache_size', 5)
        )
        self.enable_in_memory_eval = getattr(config, 'enable_in_memory_evaluation', True)

    def setup(
        self,
        device: str,
        policy_mapper,
        model_dir: str,
        wandb_active: bool,
    ) -> None:
        """Configure runtime properties."""
        self.device = device
        self.policy_mapper = policy_mapper
        self.model_dir = model_dir
        self.wandb_active = wandb_active

    def evaluate_checkpoint(
        self, agent_checkpoint: str, opponent_checkpoint: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a saved agent checkpoint."""
        agent_info = AgentInfo(
            name="current_agent", 
            checkpoint_path=agent_checkpoint
        )
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )
        evaluator = EvaluatorFactory.create(self.config)
        # Many evaluators are async; run in asyncio event loop
        return asyncio.run(evaluator.evaluate(agent_info, context))

    def evaluate_current_agent(self, agent) -> EvaluationResult:
        """Evaluate an in-memory PPOAgent instance."""

        # Ensure the agent has the expected attributes
        model = getattr(agent, "model", None)
        if model is None:
            raise ValueError("Agent must have a 'model' attribute for evaluation")

        # Switch to eval mode for duration of evaluation
        if hasattr(model, "eval"):
            model.eval()

        agent_info = AgentInfo(
            name=getattr(agent, "name", "current_agent"),
            metadata={"agent_instance": agent},
        )
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )

        evaluator = EvaluatorFactory.create(self.config)
        # Use asyncio.run to handle async evaluator
        result = asyncio.run(evaluator.evaluate(agent_info, context))

        # Restore training mode after evaluation
        if hasattr(model, "train"):
            model.train()

        return result

    async def evaluate_current_agent_async(self, agent) -> EvaluationResult:
        """Async version of evaluate_current_agent for use in async contexts."""

        # Ensure the agent has the expected attributes
        model = getattr(agent, "model", None)
        if model is None:
            raise ValueError("Agent must have a 'model' attribute for evaluation")

        # Switch to eval mode for duration of evaluation
        if hasattr(model, "eval"):
            model.eval()

        agent_info = AgentInfo(
            name=getattr(agent, "name", "current_agent"),
            metadata={"agent_instance": agent},
        )
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )

        evaluator = EvaluatorFactory.create(self.config)
        result = await evaluator.evaluate(agent_info, context)

        # Restore training mode after evaluation
        if hasattr(model, "train"):
            model.train()

        return result

    async def evaluate_current_agent_in_memory(
        self,
        agent,
        opponent_checkpoint: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate agent using in-memory weights without file I/O."""
        if not self.enable_in_memory_eval:
            return self.evaluate_current_agent(agent)  # Fallback to file-based

        try:
            # Extract current agent weights
            agent_weights = self.model_weight_manager.extract_agent_weights(agent)
            agent_info = AgentInfo(
                name=getattr(agent, "name", "current_agent"),
                model_type=type(agent).__name__,
                metadata={"agent_instance": agent, "in_memory_weights": True}
            )

            # Get opponent
            if opponent_checkpoint:
                opponent_path = Path(opponent_checkpoint)
                opponent_weights = self.model_weight_manager.cache_opponent_weights(
                    opponent_path.stem, opponent_path
                )
                opponent_info = OpponentInfo(
                    name=opponent_path.stem,
                    type="ppo",
                    checkpoint_path=str(opponent_path)
                )
            else:
                # Sample from pool
                opponent_path_obj = self.opponent_pool.sample()
                if not opponent_path_obj:
                    raise ValueError("No opponents available in pool")

                opponent_path = Path(opponent_path_obj)
                opponent_weights = self.model_weight_manager.cache_opponent_weights(
                    opponent_path.stem, opponent_path
                )
                opponent_info = OpponentInfo(
                    name=opponent_path.stem,
                    type="ppo",
                    checkpoint_path=str(opponent_path_obj)
                )

            # Create evaluation context
            context = EvaluationContext(
                session_id=f"inmem_{int(time.time())}",
                timestamp=datetime.now(),
                agent_info=agent_info,
                configuration=self.config,
                environment_info={
                    "device": self.device,
                    "evaluation_mode": "in_memory",
                    "opponent_info": opponent_info.__dict__
                }
            )

            # Run in-memory evaluation
            return await self._run_in_memory_evaluation(
                agent_weights, opponent_weights, agent_info, opponent_info, context
            )

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            # Fallback to file-based evaluation on error
            print(f"In-memory evaluation failed, falling back to file-based: {e}")
            return self.evaluate_current_agent(agent)

    async def _run_in_memory_evaluation(
        self,
        agent_weights,
        opponent_weights,
        agent_info,
        opponent_info,
        context
    ) -> EvaluationResult:
        """Run evaluation using in-memory weights."""
        # Get the evaluator
        evaluator = EvaluatorFactory.create(self.config)
        
        # Check if evaluator supports in-memory evaluation
        if hasattr(evaluator, 'evaluate_in_memory'):
            return await evaluator.evaluate_in_memory(
                agent_info,
                context,
                agent_weights=agent_weights,
                opponent_weights=opponent_weights,
                opponent_info=opponent_info
            )
        else:
            # Fallback to regular evaluation
            return await evaluator.evaluate(agent_info, context)

