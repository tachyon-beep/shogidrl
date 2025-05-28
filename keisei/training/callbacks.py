"""
training/callbacks.py: Periodic task callbacks for the Shogi RL trainer.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(ABC):
    def on_step_end(self, trainer: "Trainer"):
        pass


class CheckpointCallback(Callback):
    def __init__(self, interval: int, model_dir: str):
        self.interval = interval
        self.model_dir = model_dir

    def on_step_end(self, trainer: "Trainer"):
        if (trainer.global_timestep + 1) % self.interval == 0:
            ckpt_save_path = os.path.join(
                self.model_dir, f"checkpoint_ts{trainer.global_timestep+1}.pth"
            )
            try:
                trainer.agent.save_model(
                    ckpt_save_path,
                    trainer.global_timestep + 1,
                    trainer.total_episodes_completed,
                    stats_to_save={
                        "black_wins": trainer.black_wins,
                        "white_wins": trainer.white_wins,
                        "draws": trainer.draws,
                    },
                )
                trainer.log_both(
                    f"Checkpoint saved to {ckpt_save_path}", also_to_wandb=True
                )
            except (OSError, RuntimeError) as e:
                trainer.log_both(
                    f"Error saving checkpoint {ckpt_save_path}: {e}",
                    log_level="error",
                    also_to_wandb=True,
                )


import os


class EvaluationCallback(Callback):
    def __init__(self, eval_cfg, interval: int):
        self.eval_cfg = eval_cfg
        self.interval = interval

    def on_step_end(self, trainer: "Trainer"):
        if not getattr(self.eval_cfg, "enable_periodic_evaluation", False):
            return
        if (trainer.global_timestep + 1) % self.interval == 0:
            eval_ckpt_path = os.path.join(
                trainer.model_dir, f"eval_checkpoint_ts{trainer.global_timestep+1}.pth"
            )
            trainer.agent.save_model(
                eval_ckpt_path,
                trainer.global_timestep + 1,
                trainer.total_episodes_completed,
            )
            trainer.log_both(
                f"Starting periodic evaluation at timestep {trainer.global_timestep + 1}...",
                also_to_wandb=True,
            )
            trainer.agent.model.eval()
            eval_results = trainer.execute_full_evaluation_run(
                agent_checkpoint_path=eval_ckpt_path,
                opponent_type=getattr(self.eval_cfg, "opponent_type", "random"),
                opponent_checkpoint_path=getattr(
                    self.eval_cfg, "opponent_checkpoint_path", None
                ),
                num_games=getattr(self.eval_cfg, "num_games", 20),
                max_moves_per_game=getattr(self.eval_cfg, "max_moves_per_game", 256),
                device_str=trainer.config.env.device,
                log_file_path_eval=getattr(self.eval_cfg, "log_file_path_eval", ""),
                policy_mapper=trainer.policy_output_mapper,
                seed=trainer.config.env.seed,
                wandb_log_eval=getattr(self.eval_cfg, "wandb_log_eval", False),
                wandb_project_eval=getattr(self.eval_cfg, "wandb_project_eval", None),
                wandb_entity_eval=getattr(self.eval_cfg, "wandb_entity_eval", None),
                wandb_run_name_eval=f"periodic_eval_{trainer.run_name}_ts{trainer.global_timestep+1}",
                wandb_group=trainer.run_name,
                wandb_reinit=True,
                logger_also_stdout=False,
            )
            trainer.agent.model.train()
            trainer.log_both(
                f"Periodic evaluation finished. Results: {eval_results}",
                also_to_wandb=True,
                wandb_data=(
                    dict(eval_results)
                    if isinstance(eval_results, dict)
                    else {"eval_summary": str(eval_results)}
                ),
            )
