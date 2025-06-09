from pathlib import Path
from unittest.mock import MagicMock

from keisei.config_schema import AppConfig
from keisei.evaluation.elo_registry import EloRegistry
from keisei.training.callbacks import EvaluationCallback
from keisei.training.metrics_manager import MetricsManager
from keisei.evaluation.opponents import OpponentPool
from keisei.utils import PolicyOutputMapper
from tests.evaluation.conftest import make_test_config


class DummyAgent:
    def __init__(self):
        self.model = MagicMock()

    def save_model(self, path, *_args):
        Path(path).write_text("dummy")


class DummyTrainer:
    def __init__(self, tmp_path):
        self.config: AppConfig = make_test_config()
        self.config.evaluation.enable_periodic_evaluation = True
        self.config.evaluation.previous_model_pool_size = 2
        self.config.evaluation.elo_registry_path = str(tmp_path / "elo.json")
        self.policy_output_mapper = PolicyOutputMapper()
        self.model_dir = str(tmp_path)
        self.run_name = "test"
        self.global_timestep = 0
        self.total_episodes_completed = 0
        self.agent = DummyAgent()
        self.metrics_manager = MetricsManager()
        self.evaluation_manager = MagicMock()
        self.evaluation_manager.opponent_pool = OpponentPool(
            pool_size=2, elo_registry_path=self.config.evaluation.elo_registry_path
        )
        self.evaluation_manager.evaluate_current_agent = self.fake_eval_current
        self.evaluation_elo_snapshot = None
        self.log_both = lambda *a, **kw: None
        # Add one previous checkpoint
        ck = Path(tmp_path / "old.pth")
        ck.write_text("a")
        self.evaluation_manager.opponent_pool.add_checkpoint(ck)

    def fake_eval_current(self, agent, opponent_checkpoint: str | None = None):
        """Simulate evaluation and update Elo ratings."""
        registry = EloRegistry(Path(self.config.evaluation.elo_registry_path))
        agent_id = self.run_name
        opponent_id = Path(opponent_checkpoint).name if opponent_checkpoint else "opponent"
        registry.update_ratings(agent_id, opponent_id, ["agent_win"])
        registry.save()
        return {"win_rate": 1.0, "loss_rate": 0.0}


def test_evaluation_callback_updates_elo(tmp_path):
    trainer = DummyTrainer(tmp_path)
    cb = EvaluationCallback(trainer.config.evaluation, interval=1)
    cb.on_step_end(trainer)
    reg = EloRegistry(Path(trainer.config.evaluation.elo_registry_path))
    assert trainer.evaluation_elo_snapshot is not None
    assert trainer.evaluation_elo_snapshot["current_rating"] != 1500.0
    assert len(reg.ratings) == 2

