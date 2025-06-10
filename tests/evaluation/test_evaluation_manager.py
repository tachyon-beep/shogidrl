from unittest.mock import MagicMock

from keisei.evaluation.core import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationStrategy,
)
from keisei.evaluation.manager import EvaluationManager


class DummyEvaluator:
    def __init__(self, config):
        self.config = config

    async def evaluate(self, agent_info, context):
        return EvaluationResult(context=context, games=[], summary_stats=MagicMock())


def test_evaluate_checkpoint(monkeypatch, tmp_path):
    cfg = EvaluationConfig(strategy=EvaluationStrategy.SINGLE_OPPONENT, num_games=1)
    manager = EvaluationManager(cfg, run_name="test")

    dummy_evaluator = DummyEvaluator(cfg)
    monkeypatch.setattr(
        "keisei.evaluation.manager.EvaluatorFactory.create", lambda cfg: dummy_evaluator
    )

    # Create dummy checkpoint file
    ckpt_path = tmp_path / "agent.ckpt"
    ckpt_path.write_text("dummy")

    result = manager.evaluate_checkpoint(str(ckpt_path))
    assert isinstance(result, EvaluationResult)
    assert result.context.agent_info.checkpoint_path == str(ckpt_path)


def test_evaluate_current_agent(monkeypatch):
    cfg = EvaluationConfig(strategy=EvaluationStrategy.SINGLE_OPPONENT, num_games=1)
    manager = EvaluationManager(cfg, run_name="test")

    dummy_agent = MagicMock()
    dummy_agent.model = MagicMock()

    captured = {}

    class DummyEvaluator:
        def __init__(self, config):
            self.config = config

        async def evaluate(self, agent_info, context):
            captured["agent"] = agent_info.metadata.get("agent_instance")
            return EvaluationResult(
                context=context, games=[], summary_stats=MagicMock()
            )

    monkeypatch.setattr(
        "keisei.evaluation.manager.EvaluatorFactory.create",
        lambda cfg: DummyEvaluator(cfg),
    )

    result = manager.evaluate_current_agent(dummy_agent)
    assert isinstance(result, EvaluationResult)
    assert captured["agent"] is dummy_agent
