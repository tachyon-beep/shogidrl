# Thin wrapper for keisei.evaluation.evaluate and core utilities for test patching/mocking
from keisei.evaluation import evaluate as _evaluate_mod
from keisei.utils import PolicyOutputMapper, TrainingLogger, EvaluationLogger, BaseOpponent
from keisei.core.ppo_agent import PPOAgent

# Assign API symbols for patching/mocking in tests
load_evaluation_agent = _evaluate_mod.load_evaluation_agent
EvaluationLogger = _evaluate_mod.EvaluationLogger
Evaluator = _evaluate_mod.Evaluator
run_evaluation_loop = _evaluate_mod.run_evaluation_loop
initialize_opponent = _evaluate_mod.initialize_opponent
execute_full_evaluation_run = _evaluate_mod.execute_full_evaluation_run

# CLI main
main = _evaluate_mod.main