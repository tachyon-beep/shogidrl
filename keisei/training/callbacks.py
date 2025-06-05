"""
training/callbacks.py: Periodic task callbacks for the Shogi RL trainer.
"""

import os
from pathlib import Path

from keisei.evaluation.elo_registry import EloRegistry
from abc import ABC
from typing import TYPE_CHECKING

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
                    if hasattr(trainer, "previous_model_selector"):
                        trainer.previous_model_selector.add_checkpoint(ckpt_save_path)
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

            eval_ckpt_path = os.path.join(
                trainer.model_dir, f"eval_checkpoint_ts{trainer.global_timestep+1}.pth"
            )

            opponent_ckpt = None
            if hasattr(trainer, "previous_model_selector"):
                opponent_ckpt = trainer.previous_model_selector.get_random_checkpoint()
            if opponent_ckpt is None:
                if trainer.log_both:
                    trainer.log_both(
                        "[INFO] EvaluationCallback: No previous checkpoint available for Elo evaluation.",
                        also_to_wandb=False,
                    )
                return

            # Save the current agent state for evaluation
            trainer.agent.save_model(
                eval_ckpt_path,
                trainer.global_timestep + 1,
                trainer.total_episodes_completed,
            )

            if trainer.log_both is not None:
                trainer.log_both(
                    f"Starting periodic evaluation at timestep {trainer.global_timestep + 1}...",
                    also_to_wandb=True,
                )

            current_model.eval()  # Set the agent's model to eval mode
            if trainer.execute_full_evaluation_run is not None:
                eval_results = trainer.execute_full_evaluation_run(
                    agent_checkpoint_path=eval_ckpt_path,
                    opponent_type="ppo",
                    opponent_checkpoint_path=str(opponent_ckpt),
                    num_games=getattr(self.eval_cfg, "num_games", 20),
                    max_moves_per_game=getattr(
                        self.eval_cfg,
                        "max_moves_per_game",
                        trainer.config.env.max_moves_per_game,
                    ),
                    device_str=trainer.config.env.device,
                    log_file_path_eval=getattr(self.eval_cfg, "log_file_path_eval", ""),
                    policy_mapper=trainer.policy_output_mapper,
                    seed=trainer.config.env.seed,
                    wandb_log_eval=getattr(self.eval_cfg, "wandb_log_eval", False),
                    wandb_project_eval=getattr(
                        self.eval_cfg, "wandb_project_eval", None
                    ),
                    wandb_entity_eval=getattr(self.eval_cfg, "wandb_entity_eval", None),
                    wandb_run_name_eval=f"periodic_eval_{trainer.run_name}_ts{trainer.global_timestep+1}",
                    wandb_group=trainer.run_name,
                    wandb_reinit=True,
                    logger_also_stdout=False,
                    elo_registry_path=getattr(self.eval_cfg, "elo_registry_path", None),
                    agent_id=os.path.basename(eval_ckpt_path),
                    opponent_id=os.path.basename(str(opponent_ckpt)),
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
                            "current_id": os.path.basename(eval_ckpt_path),
                            "current_rating": registry.get_rating(
                                os.path.basename(eval_ckpt_path)
                            ),
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
                    except Exception:  # noqa: BLE001
                        trainer.evaluation_elo_snapshot = None
            else:  # if execute_full_evaluation_run is None
                current_model.train()  # Ensure model is set back to train mode
