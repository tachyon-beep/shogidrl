"""
metrics_manager.py: Manages training statistics, metrics tracking, and formatting.
"""

import json
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from keisei.shogi.shogi_core_definitions import Color


@dataclass
class TrainingStats:
    """Container for training statistics."""
    global_timestep: int = 0
    total_episodes_completed: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0


class MetricsManager:
    """
    Manages training statistics, PPO metrics formatting, and progress tracking.
    
    Responsibilities:
    - Game outcome statistics (wins/losses/draws)
    - PPO metrics processing and formatting
    - Progress update management
    - Rate calculations and reporting
    """

    def __init__(self):
        """Initialize metrics manager with zero statistics."""
        self.stats = TrainingStats()
        self.pending_progress_updates: Dict[str, Any] = {}
    
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
            
        return self.get_win_rates_dict()
    
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
            "win_rate_draw": draw_rate
        }
    
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
        
        if "ppo/kl_divergence_approx" in learn_metrics:
            ppo_metrics_parts.append(f"KL:{learn_metrics['ppo/kl_divergence_approx']:.4f}")
        if "ppo/policy_loss" in learn_metrics:
            ppo_metrics_parts.append(f"PolL:{learn_metrics['ppo/policy_loss']:.4f}")
        if "ppo/value_loss" in learn_metrics:
            ppo_metrics_parts.append(f"ValL:{learn_metrics['ppo/value_loss']:.4f}")
        if "ppo/entropy" in learn_metrics:
            ppo_metrics_parts.append(f"Ent:{learn_metrics['ppo/entropy']:.4f}")
            
        return " ".join(ppo_metrics_parts)
    
    def format_ppo_metrics_for_logging(self, learn_metrics: Dict[str, float]) -> str:
        """
        Format PPO metrics for detailed logging (JSON format).
        
        Args:
            learn_metrics: Dictionary of PPO metrics
            
        Returns:
            JSON-formatted string of metrics
        """
        formatted_metrics = {k: f'{v:.4f}' for k, v in learn_metrics.items()}
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
            "global_timestep": self.stats.global_timestep
        }
    
    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """
        Restore statistics from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint dictionary with saved statistics
        """
        self.stats.global_timestep = checkpoint_data.get("global_timestep", 0)
        self.stats.total_episodes_completed = checkpoint_data.get("total_episodes_completed", 0)
        self.stats.black_wins = checkpoint_data.get("black_wins", 0)
        self.stats.white_wins = checkpoint_data.get("white_wins", 0)
        self.stats.draws = checkpoint_data.get("draws", 0)
    
    def increment_timestep(self) -> None:
        """Increment the global timestep counter."""
        self.stats.global_timestep += 1
    
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
