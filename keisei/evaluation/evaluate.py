"""Compatibility wrapper for evaluation entrypoints."""

from .legacy.evaluate import Evaluator, execute_full_evaluation_run
from ..utils.agent_loading import initialize_opponent, load_evaluation_agent
from ..utils import PolicyOutputMapper
from ..utils.utils import EvaluationLogger
from .loop import run_evaluation_loop

__all__ = [
    "Evaluator",
    "execute_full_evaluation_run",
    "load_evaluation_agent",
    "initialize_opponent", 
    "PolicyOutputMapper",
    "EvaluationLogger",
    "run_evaluation_loop",
]
