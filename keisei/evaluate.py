# Thin wrapper for keisei.evaluation.evaluate and core utilities for test patching/mocking
from keisei.evaluation import evaluate as _evaluate_mod
from keisei.evaluation.evaluate import main_cli as evaluate_main

# Assign API symbols for patching/mocking in tests
load_evaluation_agent = _evaluate_mod.load_evaluation_agent
EvaluationLogger = _evaluate_mod.EvaluationLogger
Evaluator = _evaluate_mod.Evaluator
run_evaluation_loop = _evaluate_mod.run_evaluation_loop
initialize_opponent = _evaluate_mod.initialize_opponent
execute_full_evaluation_run = _evaluate_mod.execute_full_evaluation_run

# CLI main
evaluate_main_cli = _evaluate_mod.main_cli

if __name__ == "__main__":
    evaluate_main()
