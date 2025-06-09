import pytest
from unittest.mock import MagicMock

from keisei.evaluation.core import EvaluationContext, LadderConfig
from keisei.evaluation.strategies.ladder import LadderEvaluator


@pytest.mark.asyncio
async def test_initialize_opponent_pool_defaults():
    cfg = LadderConfig()
    evaluator = LadderEvaluator(cfg)
    context = MagicMock(spec=EvaluationContext)
    await evaluator._initialize_opponent_pool(context)
    assert len(evaluator.opponent_pool) > 0


@pytest.mark.asyncio
async def test_initialize_opponent_pool_custom_config():
    cfg = LadderConfig(opponent_pool_config=[{"name": "opp1", "type": "random", "initial_rating": 1600}])
    evaluator = LadderEvaluator(cfg)
    context = MagicMock(spec=EvaluationContext)
    await evaluator._initialize_opponent_pool(context)
    assert len(evaluator.opponent_pool) == 1
    assert evaluator.opponent_pool[0].name == "opp1"
    assert evaluator.elo_tracker.ratings["opp1"] == 1600
