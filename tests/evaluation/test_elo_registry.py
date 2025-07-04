import json
from pathlib import Path

from keisei.evaluation.opponents.elo_registry import EloRegistry


def test_elo_registry_load_update_save(tmp_path):
    path = tmp_path / "elo.json"
    registry = EloRegistry(path)
    assert registry.get_rating("A") == 1500.0
    results = ["agent_win", "agent_win", "draw", "opponent_win"]
    registry.update_ratings("A", "B", results)
    registry.save()
    assert path.exists()
    data = json.loads(path.read_text())
    assert "A" in data["ratings"] and "B" in data["ratings"]
    assert abs(data["ratings"]["A"] - 1500.0) > 1e-6  # A's rating should have changed
