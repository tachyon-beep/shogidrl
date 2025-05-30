# keisei/training/training_loop_manager.py
"""
Manages the main training loop execution, previously part of the Trainer class.
"""
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from keisei.config_schema import AppConfig
    from keisei.core.experience_buffer import ExperienceBuffer
    from keisei.core.ppo_agent import PPOAgent
    from keisei.training.callbacks import Callback
    from keisei.training.display import TrainingDisplay
    from keisei.training.step_manager import EpisodeState, StepManager
    from keisei.training.trainer import Trainer  # Forward reference


class TrainingLoopManager:
    """
    Manages the primary iteration logic of the training loop.
    """

    def __init__(
        self,
        trainer: "Trainer",
        # Components are accessed via trainer instance
    ):
        self.trainer = trainer
        self.config = trainer.config  # Convenience access
        self.agent = trainer.agent  # Convenience access
        self.buffer = trainer.experience_buffer  # Convenience access
        self.step_manager = trainer.step_manager  # Convenience access
        self.display = trainer.display  # Convenience access
        self.callbacks = trainer.callbacks  # Convenience access

        self.current_epoch: int = 0
        self.episode_state: Optional["EpisodeState"] = None  # Set by Trainer before run

        # For SPS calculation and display update throttling
        self.last_time_for_sps: float = 0.0
        self.steps_since_last_time_for_sps: int = 0
        self.last_display_update_time: float = 0.0

    def set_initial_episode_state(self, initial_episode_state: "EpisodeState"):
        """Sets the initial episode state, typically provided by the Trainer."""
        self.episode_state = initial_episode_state

    def run(self):
        """
        Executes the main training loop.
        This method contains the core iteration logic.
        """
        log_both = self.trainer.log_both
        if not log_both:
            # This should be set by Trainer.run_training_loop before calling this
            raise RuntimeError(
                "Trainer's log_both callback is not set before running TrainingLoopManager."
            )
        if self.episode_state is None:
            raise RuntimeError(
                "Initial episode state not set in TrainingLoopManager before run."
            )

        self.last_time_for_sps = time.time()
        self.steps_since_last_time_for_sps = 0
        self.last_display_update_time = time.time()

        try:
            while self.trainer.global_timestep < self.config.training.total_timesteps:
                self.current_epoch += 1

                self._run_epoch(log_both)

                if self.trainer.global_timestep >= self.config.training.total_timesteps:
                    log_both(
                        f"Target timesteps ({self.config.training.total_timesteps}) reached during epoch {self.current_epoch}."
                    )
                    break

                if self.episode_state and self.episode_state.current_obs is not None:
                    self.trainer._perform_ppo_update(
                        self.episode_state.current_obs, log_both
                    )
                else:
                    log_both(
                        "[WARNING] Skipping PPO update due to missing current_obs in episode_state. "
                        f"(Timestep: {self.trainer.global_timestep})",
                        also_to_wandb=True,
                    )

                for callback_item in self.callbacks:
                    callback_item.on_step_end(self.trainer)

        except KeyboardInterrupt:
            log_both(
                "Training interrupted by user (KeyboardInterrupt in TrainingLoopManager).",
                also_to_wandb=True,
            )
            raise
        except Exception as e:
            log_message = f"Unhandled exception in TrainingLoopManager.run: {e}"
            if hasattr(self.trainer, "logger") and self.trainer.logger:
                self.trainer.logger.log(log_message)  # Removed exc_info=True
            else:
                print(log_message)
            log_both(f"Unhandled exception in training loop: {e}", also_to_wandb=True)
            raise

    def _run_epoch(self, log_both):
        """
        Runs a single epoch, collecting experiences until the buffer is full or total timesteps are met.
        """
        num_steps_collected_this_epoch = 0
        while num_steps_collected_this_epoch < self.config.training.steps_per_epoch:
            if self.trainer.global_timestep >= self.config.training.total_timesteps:
                break

            if self.episode_state is None:
                log_both(
                    "[ERROR] Episode state is None at the start of a step collection. Resetting.",
                    also_to_wandb=True,
                )
                if self.step_manager is None:
                    raise RuntimeError("StepManager is not available")
                self.episode_state = self.step_manager.reset_episode()
                if self.episode_state is None:
                    raise RuntimeError(
                        "Failed to recover from None episode_state by resetting."
                    )
                continue

            if self.step_manager is None:
                raise RuntimeError("StepManager is not available")
            step_result = self.step_manager.execute_step(
                episode_state=self.episode_state,
                global_timestep=self.trainer.global_timestep,
                logger_func=log_both,
            )

            if not step_result.success:
                log_both(
                    f"[WARNING] Step execution failed at timestep {self.trainer.global_timestep}. Resetting episode.",
                    also_to_wandb=True,
                )
                if self.step_manager is None:
                    raise RuntimeError("StepManager is not available")
                self.episode_state = self.step_manager.reset_episode()
                continue

            if self.step_manager is None:
                raise RuntimeError("StepManager is not available")
            updated_episode_state = self.step_manager.update_episode_state(
                self.episode_state, step_result
            )

            if step_result.done:
                # Use metrics manager to safely update stats
                if step_result.info and "winner" in step_result.info:
                    winner = step_result.info["winner"]
                    if winner == "black":
                        self.trainer.metrics_manager.black_wins += 1
                    elif winner == "white":
                        self.trainer.metrics_manager.white_wins += 1
                    else:
                        self.trainer.metrics_manager.draws += 1
                else:
                    self.trainer.metrics_manager.draws += 1

                game_stats_for_sm = {
                    "black_wins": self.trainer.metrics_manager.black_wins,
                    "white_wins": self.trainer.metrics_manager.white_wins,
                    "draws": self.trainer.metrics_manager.draws,
                }
                if self.step_manager is None:
                    raise RuntimeError("StepManager is not available")
                new_episode_state_after_end = self.step_manager.handle_episode_end(
                    updated_episode_state,
                    step_result,
                    game_stats_for_sm,
                    self.trainer.metrics_manager.total_episodes_completed,
                    log_both,
                )
                self.trainer.metrics_manager.total_episodes_completed += 1

                ep_len = updated_episode_state.episode_length
                ep_rew = updated_episode_state.episode_reward
                ep_metrics_str = f"L:{ep_len} R:{ep_rew:.2f}"

                total_games = (
                    self.trainer.metrics_manager.black_wins
                    + self.trainer.metrics_manager.white_wins
                    + self.trainer.metrics_manager.draws
                )
                bw_rate = (
                    self.trainer.metrics_manager.black_wins / total_games if total_games > 0 else 0.0
                )
                ww_rate = (
                    self.trainer.metrics_manager.white_wins / total_games if total_games > 0 else 0.0
                )
                d_rate = self.trainer.metrics_manager.draws / total_games if total_games > 0 else 0.0

                self.trainer.metrics_manager.pending_progress_updates.update(
                    {
                        "ep_metrics": ep_metrics_str,
                        "black_wins_cum": self.trainer.metrics_manager.black_wins,
                        "white_wins_cum": self.trainer.metrics_manager.white_wins,
                        "draws_cum": self.trainer.metrics_manager.draws,
                        "black_win_rate": bw_rate,
                        "white_win_rate": ww_rate,
                        "draw_rate": d_rate,
                        "total_episodes": self.trainer.metrics_manager.total_episodes_completed,
                    }
                )
                self.episode_state = new_episode_state_after_end
            else:
                self.episode_state = updated_episode_state

            self.trainer.metrics_manager.global_timestep += 1
            num_steps_collected_this_epoch += 1
            self.steps_since_last_time_for_sps += 1

            if (
                self.trainer.global_timestep % self.config.training.render_every_steps
                == 0
            ):
                if hasattr(self.display, "update_log_panel") and callable(
                    self.display.update_log_panel
                ):
                    self.display.update_log_panel(self.trainer)

            current_time = time.time()
            display_update_interval = getattr(
                self.config.training, "rich_display_update_interval_seconds", 0.2
            )

            if (current_time - self.last_display_update_time) > display_update_interval:
                time_delta_sps = current_time - self.last_time_for_sps
                current_speed = (
                    self.steps_since_last_time_for_sps / time_delta_sps
                    if time_delta_sps > 0
                    else 0.0
                )

                self.trainer.metrics_manager.pending_progress_updates.setdefault(
                    "current_epoch", self.current_epoch
                )

                if hasattr(self.display, "update_progress") and callable(
                    self.display.update_progress
                ):
                    self.display.update_progress(
                        self.trainer,
                        current_speed,
                        self.trainer.metrics_manager.pending_progress_updates,
                    )
                self.trainer.metrics_manager.pending_progress_updates.clear()

                self.last_time_for_sps = current_time
                self.steps_since_last_time_for_sps = 0
                self.last_display_update_time = current_time
