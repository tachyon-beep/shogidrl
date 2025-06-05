from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class EloRegistry:
    """Persistent registry of Elo ratings for agents."""

    path: Path
    default_rating: float = 1500.0
    k_factor: float = 32.0
    ratings: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.load()

    # Persistence helpers
    def load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    self.ratings = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.ratings = {}
        else:
            self.ratings = {}

    def save(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.ratings, f, indent=2)
        except OSError:
            pass

    # Rating helpers
    def get_rating(self, model_id: str) -> float:
        return float(self.ratings.get(model_id, self.default_rating))

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(
        self, model_a_id: str, model_b_id: str, results: List[str]
    ) -> None:
        """Update ratings based on a series of game results."""

        rating_a = self.get_rating(model_a_id)
        rating_b = self.get_rating(model_b_id)

        for res in results:
            if res == "agent_win":
                score_a, score_b = 1.0, 0.0
            elif res == "opponent_win":
                score_a, score_b = 0.0, 1.0
            else:
                score_a = score_b = 0.5

            exp_a = self._expected_score(rating_a, rating_b)
            exp_b = self._expected_score(rating_b, rating_a)

            rating_a += self.k_factor * (score_a - exp_a)
            rating_b += self.k_factor * (score_b - exp_b)

        self.ratings[model_a_id] = rating_a
        self.ratings[model_b_id] = rating_b
