"""
metrics_manager.py: Manages training statistics, metrics tracking, and formatting.
"""

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Deque

from keisei.shogi.shogi_core_definitions import Color
from .elo_rating import EloRatingSystem


@dataclass
class TrainingStats:
    """Container for training statistics."""

    global_timestep: int = 0
    total_episodes_completed: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0


class MetricsHistory:
    """Track historical metrics required for trend visualisation and Elo."""

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self.win_rates_history: List[Dict[str, float]] = []
        self.learning_rates: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.kl_divergences: List[float] = []
        self.entropies: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_rewards: List[float] = []

    def _trim(self, values: List[Any]) -> None:
        while len(values) > self.max_history:
            values.pop(0)

    def add_episode_data(self, win_rates: Dict[str, float]) -> None:
        self.win_rates_history.append(win_rates)
        self._trim(self.win_rates_history)

    def add_ppo_data(self, metrics: Dict[str, float]) -> None:
        if "ppo/learning_rate" in metrics:
            self.learning_rates.append(metrics["ppo/learning_rate"])
            self._trim(self.learning_rates)
        if "ppo/policy_loss" in metrics:
            self.policy_losses.append(metrics["ppo/policy_loss"])
            self._trim(self.policy_losses)
        if "ppo/value_loss" in metrics:
            self.value_losses.append(metrics["ppo/value_loss"])
            self._trim(self.value_losses)
        if "ppo/kl_divergence_approx" in metrics:
            self.kl_divergences.append(metrics["ppo/kl_divergence_approx"])
            self._trim(self.kl_divergences)
        if "ppo/entropy" in metrics:
            self.entropies.append(metrics["ppo/entropy"])
            self._trim(self.entropies)


class MetricsManager:
    """
    Manages training statistics, PPO metrics formatting, and progress tracking.

    Responsibilities:
    - Game outcome statistics (wins/losses/draws)
    - PPO metrics processing and formatting
    - Progress update management
    - Rate calculations and reporting
    """

    def __init__(
        self,
        history_size: int = 1000,
        elo_initial_rating: float = 1500.0,
        elo_k_factor: float = 32.0,
    ) -> None:
        """Initialize metrics manager with zero statistics and helpers."""
        self.stats = TrainingStats()
        self.pending_progress_updates: Dict[str, Any] = {}
        self.history = MetricsHistory(max_history=history_size)
        self.elo_system = EloRatingSystem(
            initial_rating=elo_initial_rating, k_factor=elo_k_factor
        )
        # Enhanced metric storage
        self.moves_per_game: Deque[int] = deque(maxlen=history_size)
        self.turns_per_game: Deque[int] = deque(maxlen=history_size)
        self.games_completed_timestamps: Deque[float] = deque(maxlen=history_size)
        self.win_loss_draw_history: Deque[Tuple[str, float]] = deque(
            maxlen=history_size
        )

    # === Statistics Management ===

    def update_episode_stats(self, winner_color: Optional[Color]) -> Dict[str, float]:
        """
        Update episode statistics based on game outcome.

        Args:
            winner_color: Color of the winner, or None for draw

        Returns:
            Dictionary with updated win rates
        """
        self.stats.total_episodes_completed += 1

        if winner_color == Color.BLACK:
            self.stats.black_wins += 1
        elif winner_color == Color.WHITE:
            self.stats.white_wins += 1
        else:
            self.stats.draws += 1
        win_rates = self.get_win_rates_dict()
        self.history.add_episode_data(win_rates)
        self.elo_system.update_ratings(winner_color)
        return win_rates

    def get_win_rates(self) -> Tuple[float, float, float]:
        """
        Calculate win rates as percentages.

        Returns:
            Tuple of (black_rate, white_rate, draw_rate) as percentages
        """
        total = self.stats.total_episodes_completed
        if total == 0:
            return (0.0, 0.0, 0.0)

        black_rate = (self.stats.black_wins / total) * 100
        white_rate = (self.stats.white_wins / total) * 100
        draw_rate = (self.stats.draws / total) * 100

        return (black_rate, white_rate, draw_rate)

    def get_win_rates_dict(self) -> Dict[str, float]:
        """Get win rates as a dictionary for logging."""
        black_rate, white_rate, draw_rate = self.get_win_rates()
        return {
            "win_rate_black": black_rate,
            "win_rate_white": white_rate,
            "win_rate_draw": draw_rate,
        }

    # === Enhanced Metrics ===

    def log_episode_metrics(
        self,
        moves_made: int,
        turns_count: int,
        result: str,
        episode_reward: float,
    ) -> None:
        """Record metrics for a completed episode."""
        self.moves_per_game.append(moves_made)
        self.turns_per_game.append(turns_count)
        self.history.episode_lengths.append(moves_made)
        self.history.episode_rewards.append(episode_reward)
        now = time.time()
        self.games_completed_timestamps.append(now)
        self.win_loss_draw_history.append((result, now))

    def get_moves_per_game_trend(self, window_size: int = 100) -> List[float]:
        data = list(self.moves_per_game)[-window_size:]
        return data

    def get_games_completion_rate(self, time_window_hours: float = 1.0) -> float:
        cutoff = time.time() - time_window_hours * 3600
        count = sum(ts >= cutoff for ts in self.games_completed_timestamps)
        return count / time_window_hours if time_window_hours > 0 else 0.0

    def get_win_loss_draw_rates(self, window_size: int = 100) -> Dict[str, float]:
        recent = sorted(self.win_loss_draw_history, key=lambda t: t[1])[-window_size:]
        if not recent:
            return {"win": 0.0, "loss": 0.0, "draw": 0.0}
        total = len(recent)
        wins = sum(1 for r, _ in recent if r == "win")
        losses = sum(1 for r, _ in recent if r == "loss")
        draws = sum(1 for r, _ in recent if r == "draw")
        return {"win": wins / total, "loss": losses / total, "draw": draws / total}

    def get_average_turns_trend(self, window_size: int = 100) -> List[float]:
        data = list(self.turns_per_game)[-window_size:]
        return data

    def format_episode_metrics(self, episode_length: int, episode_reward: float) -> str:
        """
        Format episode completion metrics for display.

        Args:
            episode_length: Number of steps in the episode
            episode_reward: Total reward for the episode

        Returns:
            Formatted string with episode metrics
        """
        black_rate, white_rate, draw_rate = self.get_win_rates()

        return (
            f"Ep {self.stats.total_episodes_completed}: "
            f"Len={episode_length}, R={episode_reward:.3f}, "
            f"B={black_rate:.1f}%, W={white_rate:.1f}%, D={draw_rate:.1f}%"
        )

    # === PPO Metrics Management ===

    def format_ppo_metrics(self, learn_metrics: Dict[str, float]) -> str:
        """
        Format PPO learning metrics for display.

        Args:
            learn_metrics: Dictionary of PPO metrics from agent.learn()

        Returns:
            Formatted string with key PPO metrics
        """
        ppo_metrics_parts = []

        if "ppo/learning_rate" in learn_metrics:
            ppo_metrics_parts.append(f"LR:{learn_metrics['ppo/learning_rate']:.2e}")
        if "ppo/kl_divergence_approx" in learn_metrics:
            ppo_metrics_parts.append(
                f"KL:{learn_metrics['ppo/kl_divergence_approx']:.4f}"
            )
        if "ppo/policy_loss" in learn_metrics:
            ppo_metrics_parts.append(f"PolL:{learn_metrics['ppo/policy_loss']:.4f}")
        if "ppo/value_loss" in learn_metrics:
            ppo_metrics_parts.append(f"ValL:{learn_metrics['ppo/value_loss']:.4f}")
        if "ppo/entropy" in learn_metrics:
            ppo_metrics_parts.append(f"Ent:{learn_metrics['ppo/entropy']:.4f}")
        self.history.add_ppo_data(learn_metrics)
        return " ".join(ppo_metrics_parts)

    def format_ppo_metrics_for_logging(self, learn_metrics: Dict[str, float]) -> str:
        """
        Format PPO metrics for detailed logging (JSON format).

        Args:
            learn_metrics: Dictionary of PPO metrics

        Returns:
            JSON-formatted string of metrics
        """
        formatted_metrics = {k: f"{v:.4f}" for k, v in learn_metrics.items()}
        return json.dumps(formatted_metrics)

    # === Progress Updates Management ===

    def update_progress_metrics(self, key: str, value: Any) -> None:
        """
        Store a progress update for later display.

        Args:
            key: Update identifier (e.g., 'ppo_metrics', 'speed')
            value: Update value
        """
        self.pending_progress_updates[key] = value

    def get_progress_updates(self) -> Dict[str, Any]:
        """Get current pending progress updates."""
        return self.pending_progress_updates.copy()

    def clear_progress_updates(self) -> None:
        """Clear pending progress updates after they've been displayed."""
        self.pending_progress_updates.clear()

    # === State Management ===

    def get_final_stats(self) -> Dict[str, int]:
        """
        Get final game statistics for saving with model.

        Returns:
            Dictionary with final statistics
        """
        return {
            "black_wins": self.stats.black_wins,
            "white_wins": self.stats.white_wins,
            "draws": self.stats.draws,
            "total_episodes_completed": self.stats.total_episodes_completed,
            "global_timestep": self.stats.global_timestep,
        }

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """
        Restore statistics from checkpoint data.

        Args:
            checkpoint_data: Checkpoint dictionary with saved statistics
        """
        self.stats.global_timestep = checkpoint_data.get("global_timestep", 0)
        self.stats.total_episodes_completed = checkpoint_data.get(
            "total_episodes_completed", 0
        )
        self.stats.black_wins = checkpoint_data.get("black_wins", 0)
        self.stats.white_wins = checkpoint_data.get("white_wins", 0)
        self.stats.draws = checkpoint_data.get("draws", 0)

    def increment_timestep(self) -> None:
        """Increment the global timestep counter."""
        self.stats.global_timestep += 1

    def increment_timestep_by(self, amount: int) -> None:
        """
        Increment the global timestep counter by a specific amount.

        Args:
            amount: Number of timesteps to add
        """
        if amount < 0:
            raise ValueError("Timestep increment amount must be non-negative")
        self.stats.global_timestep += amount

    # === Properties for Backward Compatibility ===

    @property
    def global_timestep(self) -> int:
        """Current global timestep."""
        return self.stats.global_timestep

    @global_timestep.setter
    def global_timestep(self, value: int) -> None:
        """Set global timestep."""
        self.stats.global_timestep = value

    @property
    def total_episodes_completed(self) -> int:
        """Total episodes completed."""
        return self.stats.total_episodes_completed

    @total_episodes_completed.setter
    def total_episodes_completed(self, value: int) -> None:
        """Set total episodes completed."""
        self.stats.total_episodes_completed = value

    @property
    def black_wins(self) -> int:
        """Number of black wins."""
        return self.stats.black_wins

    @black_wins.setter
    def black_wins(self, value: int) -> None:
        """Set black wins."""
        self.stats.black_wins = value

    @property
    def white_wins(self) -> int:
        """Number of white wins."""
        return self.stats.white_wins

    @white_wins.setter
    def white_wins(self, value: int) -> None:
        """Set white wins."""
        self.stats.white_wins = value

    @property
    def draws(self) -> int:
        """Number of draws."""
        return self.stats.draws

    @draws.setter
    def draws(self, value: int) -> None:
        """Set draws."""
        self.stats.draws = value
