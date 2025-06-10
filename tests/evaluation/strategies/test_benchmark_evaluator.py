import pytest

from keisei.evaluation.core import BenchmarkConfig
from keisei.evaluation.strategies.benchmark import BenchmarkEvaluator


def test_validate_config_invalid_games_per_case():
    cfg = BenchmarkConfig(num_games_per_benchmark_case=0)
    evaluator = BenchmarkEvaluator(cfg)
    assert evaluator.validate_config() is False


def test_validate_config_basic():
    cfg = BenchmarkConfig(
        num_games_per_benchmark_case=2, suite_config=[{"name": "c1", "type": "random"}]
    )
    evaluator = BenchmarkEvaluator(cfg)
    assert evaluator.validate_config() is True
