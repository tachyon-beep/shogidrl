import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


def async_test(coro):
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper

from keisei.evaluation.core import AgentInfo, SingleOpponentConfig
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator


@async_test
async def test_load_agent_instance_direct():
    cfg = SingleOpponentConfig(opponent_name="opp")
    evaluator = SingleOpponentEvaluator(cfg)
    dummy_agent = MagicMock()
    info = AgentInfo(name="agent", metadata={"agent_instance": dummy_agent})

    with patch(
        "keisei.evaluation.strategies.single_opponent.load_evaluation_agent",
        new_callable=AsyncMock,
    ) as mock_load:
        loaded = await evaluator._load_evaluation_entity(info, "cpu", 46)
        mock_load.assert_not_called()
        assert loaded is dummy_agent
