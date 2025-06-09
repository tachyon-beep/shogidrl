"""
training/callbacks.py: Periodic task callbacks for the Shogi RL trainer.
"""

import os
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

from keisei.evaluation.elo_registry import EloRegistry

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(ABC):
    def on_step_end(self, trainer: "Trainer"):
        # Base implementation - subclasses override this method
        pass


class CheckpointCallback(Callback):
    def __init__(self, interval: int, model_dir: str):
        self.interval = interval
        self.model_dir = model_dir

    def on_step_end(self, trainer: "Trainer"):
        if (trainer.global_timestep + 1) % self.interval == 0:
            if not trainer.agent:
                if trainer.log_both:
                    trainer.log_both(
                        "[ERROR] CheckpointCallback: Agent not initialized, cannot save checkpoint.",
                        also_to_wandb=True,
                    )
                return

            game_stats = {
                "black_wins": trainer.black_wins,
                "white_wins": trainer.white_wins,
                "draws": trainer.draws,
            }

            # Use the consolidated save_checkpoint method from ModelManager
            success, ckpt_save_path = trainer.model_manager.save_checkpoint(
                agent=trainer.agent,
                model_dir=self.model_dir,  # model_dir is part of CheckpointCallback's state
                timestep=trainer.global_timestep + 1,
                episode_count=trainer.total_episodes_completed,
                stats=game_stats,
                run_name=trainer.run_name,
                is_wandb_active=trainer.is_train_wandb_active,
            )

            if success:
                if trainer.log_both and ckpt_save_path:
                    trainer.log_both(
                        f"Checkpoint saved via ModelManager to {ckpt_save_path}",
                        also_to_wandb=True,
                    )
                    if hasattr(trainer, "evaluation_manager"):
                        trainer.evaluation_manager.opponent_pool.add_checkpoint(
                            ckpt_save_path
                        )
            else:
                if trainer.log_both:
                    trainer.log_both(
                        f"[ERROR] CheckpointCallback: Failed to save checkpoint via ModelManager for timestep {trainer.global_timestep + 1}.",
                        also_to_wandb=True,
                    )


class EvaluationCallback(Callback):
    def __init__(self, eval_cfg, interval: int):
        self.eval_cfg = eval_cfg
        self.interval = interval

    def on_step_end(self, trainer: "Trainer"):
        if not getattr(self.eval_cfg, "enable_periodic_evaluation", False):
            return
        if (trainer.global_timestep + 1) % self.interval == 0:
            if not trainer.agent:
                if trainer.log_both:
                    trainer.log_both(
                        "[ERROR] EvaluationCallback: Agent not initialized, cannot run evaluation.",
                        also_to_wandb=True,
                    )
                return

            # Type narrowing: assert that agent is not None
            assert (
                trainer.agent is not None
            ), "Agent should be initialized at this point"

            # Ensure the model to be evaluated exists and is accessible
            # The Trainer now owns self.model directly.
            # PPOAgent also has a self.model attribute.
            # For evaluation, we typically want the agent's current model state.
            current_model = trainer.agent.model
            if not current_model:
                if trainer.log_both:
                    trainer.log_both(
                        "[ERROR] EvaluationCallback: Agent's model not found.",
                        also_to_wandb=True,
                    )
                return


            opponent_ckpt = None
            if hasattr(trainer, "evaluation_manager"):
                opponent_ckpt = trainer.evaluation_manager.opponent_pool.sample()
            if opponent_ckpt is None:
                if trainer.log_both:
                    trainer.log_both(
                        "[INFO] EvaluationCallback: No previous checkpoint available for Elo evaluation.",
                        also_to_wandb=False,
                    )
                return

            if trainer.log_both is not None:
                trainer.log_both(
                    f"Starting periodic evaluation at timestep {trainer.global_timestep + 1}...",
                    also_to_wandb=True,
                )

            current_model.eval()  # Set the agent's model to eval mode
            eval_results = trainer.evaluation_manager.evaluate_current_agent(
                trainer.agent
            )
            current_model.train()  # Set model back to train mode
            if trainer.log_both is not None:
                trainer.log_both(
                    f"Periodic evaluation finished. Results: {eval_results}",
                    also_to_wandb=True,
                    wandb_data=(
                        dict(eval_results)
                        if isinstance(eval_results, dict)
                        else {"eval_summary": str(eval_results)}
                    ),
                )

                if getattr(self.eval_cfg, "elo_registry_path", None):
                    try:
                        registry = EloRegistry(Path(self.eval_cfg.elo_registry_path))
                        snapshot = {
                            "current_id": trainer.run_name,
                            "current_rating": registry.get_rating(trainer.run_name),
                            "opponent_id": os.path.basename(str(opponent_ckpt)),
                            "opponent_rating": registry.get_rating(
                                os.path.basename(str(opponent_ckpt))
                            ),
                            "last_outcome": (
                                "win"
                                if eval_results
                                and eval_results.get("win_rate", 0)
                                > eval_results.get("loss_rate", 0)
                                else (
                                    "loss"
                                    if eval_results
                                    and eval_results.get("loss_rate", 0)
                                    > eval_results.get("win_rate", 0)
                                    else "draw"
                                )
                            ),
                            "top_ratings": sorted(
                                registry.ratings.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3],
                        }
                        trainer.evaluation_elo_snapshot = snapshot
                    except Exception as e:  # noqa: BLE001
                        if trainer.log_both is not None:
                            trainer.log_both(
                                f"[ERROR] Failed to update Elo registry: {type(e).__name__}: {e}",
                                also_to_wandb=True,
                            )
                        trainer.evaluation_elo_snapshot = None
            # EvaluationManager always handles model mode switching; nothing else to do
