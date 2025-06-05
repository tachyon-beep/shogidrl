from unittest.mock import MagicMock
from pathlib import Path

from keisei.training.callbacks import EvaluationCallback
from keisei.training.previous_model_selector import PreviousModelSelector
from keisei.training.metrics_manager import MetricsManager
from keisei.training.trainer import Trainer
from keisei.config_schema import AppConfig
from keisei.evaluation.elo_registry import EloRegistry
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
        self.execute_full_evaluation_run = self.fake_eval
        self.metrics_manager = MetricsManager()
        self.previous_model_selector = PreviousModelSelector(pool_size=2)
        self.evaluation_elo_snapshot = None
        self.log_both = lambda *a, **kw: None
        # Add one previous checkpoint
        ck = Path(tmp_path / "old.pth")
        ck.write_text("a")
        self.previous_model_selector.add_checkpoint(ck)

    def fake_eval(self, **kwargs):
        # Simulate one win for the agent
        registry = EloRegistry(Path(self.config.evaluation.elo_registry_path))
        registry.update_ratings(
            kwargs["agent_id"], kwargs["opponent_id"], ["agent_win"]
        )
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
