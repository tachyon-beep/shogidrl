from __future__ import annotations

"""EvaluationManager for new evaluation system."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import uuid4

from .core import (
    AgentInfo,
    EvaluationContext,
    EvaluationResult,
    EvaluatorFactory,
    EvaluationConfig,
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
        self, agent_checkpoint: str, opponent_checkpoint: Optional[str] | None = None
    ) -> EvaluationResult:
        """Evaluate a saved agent checkpoint."""
        agent_info = AgentInfo(name="current_agent", checkpoint_path=agent_checkpoint)
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
        result = asyncio.run(evaluator.evaluate(agent_info, context))

        # Restore training mode after evaluation
        if hasattr(model, "train"):
            model.train()

        return result

