"""
evaluate.py: Main script for evaluating PPO Shogi agents.
"""

import argparse
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv  # type: ignore

import wandb  # Ensure wandb is imported for W&B logging
from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi import shogi_game_io  # For observations
from keisei.shogi.shogi_core_definitions import (  # Added PieceType
    Color,
    MoveTuple,
    PieceType,
)
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import BaseOpponent, EvaluationLogger, PolicyOutputMapper
from keisei.evaluation.loop import ResultsDict

if TYPE_CHECKING:
    pass  # torch already imported above

load_dotenv()  # Load environment variables from .env file


class Evaluator:
    """
    Evaluator class encapsulates the evaluation logic for PPO Shogi agents.
    It manages agent/opponent loading, logging, W&B integration, and runs the evaluation loop.
    """

    def __init__(
        self,
        agent_checkpoint_path: str,
        opponent_type: str,
        opponent_checkpoint_path: Optional[str],
        num_games: int,
        max_moves_per_game: int,
        device_str: str,
        log_file_path_eval: str,
        policy_mapper: PolicyOutputMapper,
        seed: Optional[int] = None,
        wandb_log_eval: bool = False,
        wandb_project_eval: Optional[str] = None,
        wandb_entity_eval: Optional[str] = None,
        wandb_run_name_eval: Optional[str] = None,
        logger_also_stdout: bool = True,
        wandb_extra_config: Optional[dict] = None,
        wandb_reinit: Optional[bool] = None,
        wandb_group: Optional[str] = None,
    ):
        """
        Initialize the Evaluator with all configuration and dependencies.
        """
        self.agent_checkpoint_path = agent_checkpoint_path
        self.opponent_type = opponent_type
        self.opponent_checkpoint_path = opponent_checkpoint_path
        self.num_games = num_games
        self.max_moves_per_game = max_moves_per_game
        self.device_str = device_str
        self.log_file_path_eval = log_file_path_eval
        self.policy_mapper = policy_mapper
        self.seed = seed
        self.wandb_log_eval = wandb_log_eval
        self.wandb_project_eval = wandb_project_eval
        self.wandb_entity_eval = wandb_entity_eval
        self.wandb_run_name_eval = wandb_run_name_eval
        self.logger_also_stdout = logger_also_stdout
        self.wandb_extra_config = wandb_extra_config
        self.wandb_reinit = wandb_reinit
        self.wandb_group = wandb_group
        self._wandb_active: bool = False
        self._wandb_run: Optional[Any] = None
        self._logger: Optional[EvaluationLogger] = None
        self._agent: Optional[PPOAgent] = None
        self._opponent: Optional[Union[PPOAgent, BaseOpponent]] = None

    def _setup(self) -> None:
        """
        Set up seeds, W&B, logger, agent, and opponent.
        Raises RuntimeError if any required component fails to initialize.
        """
        if self.seed is not None:
            try:
                random.seed(self.seed)
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
                if self.device_str == "cuda":
                    torch.cuda.manual_seed_all(self.seed)
                print(f"[Evaluator] Set random seed to: {self.seed}")
            except Exception as e:
                raise RuntimeError("Failed to set random seed") from e

        # W&B Initialization
        if self.wandb_log_eval:
            try:
                wandb_config = {
                    "agent_checkpoint": self.agent_checkpoint_path,
                    "opponent_type": self.opponent_type,
                    "opponent_checkpoint": self.opponent_checkpoint_path,
                    "num_games": self.num_games,
                    "max_moves_per_game": self.max_moves_per_game,
                    "device": self.device_str,
                    "seed": self.seed,
                }
                if self.wandb_extra_config:
                    wandb_config.update(self.wandb_extra_config)
                wandb_kwargs: Dict[str, Any] = {
                    "project": self.wandb_project_eval or "keisei-evaluation-runs",
                    "entity": self.wandb_entity_eval,
                    "name": self.wandb_run_name_eval,
                    "config": wandb_config,
                }
                if self.wandb_reinit is not None:
                    wandb_kwargs["reinit"] = self.wandb_reinit
                if self.wandb_group is not None:
                    wandb_kwargs["group"] = self.wandb_group
                try:
                    self._wandb_run = wandb.init(**wandb_kwargs)  # type: ignore
                    print(
                        f"[Evaluator] W&B logging enabled: {self._wandb_run.name if self._wandb_run else ''}"
                    )
                    self._wandb_active = True
                except (OSError, RuntimeError, ValueError) as e:
                    print(
                        f"[Evaluator] Error initializing W&B: {e}. W&B logging disabled."
                    )
                    self._wandb_active = False
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[Evaluator] Unexpected error during W&B initialization: {e}")
                self._wandb_active = False
            except Exception as e:
                print(f"[Evaluator] Critical error during W&B initialization: {e}")
                self._wandb_active = False

        # Ensure log directory exists
        log_dir_eval = os.path.dirname(self.log_file_path_eval)
        if log_dir_eval and not os.path.exists(log_dir_eval):
            try:
                os.makedirs(log_dir_eval)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create log directory '{log_dir_eval}': {e}"
                ) from e

        # Logger
        try:
            self._logger = EvaluationLogger(
                self.log_file_path_eval, also_stdout=self.logger_also_stdout
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EvaluationLogger: {e}") from e
        # Agent and opponent
        # Load input_channels from config
        from keisei.utils.utils import load_config

        config = load_config()
        input_channels = config.env.input_channels

        from keisei.utils.agent_loading import (
            initialize_opponent,
            load_evaluation_agent,
        )

        try:
            self._agent = load_evaluation_agent(
                self.agent_checkpoint_path,
                self.device_str,
                self.policy_mapper,
                input_channels,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluation agent: {e}") from e
        try:
            self._opponent = initialize_opponent(
                self.opponent_type,
                self.opponent_checkpoint_path,
                self.device_str,
                self.policy_mapper,
                input_channels,
            )
        except ValueError as e:
            print(f"Error initializing opponent {self.opponent_type}: {e}")
            return None
        except FileNotFoundError as e:
            print(f"Error initializing PPO opponent: {e}")
            return None
        except Exception as e:  # pylint: disable=broad-except
            print(f"Unexpected error initializing opponent {self.opponent_type}: {e}")
            return None

        if self._logger is None or self._agent is None or self._opponent is None:
            raise RuntimeError(
                "Evaluator setup failed: logger, agent, or opponent is None."
            )

    def evaluate(self) -> Optional[ResultsDict]:
        """
        Run the evaluation and return the results dictionary.
        Raises RuntimeError if the Evaluator is not properly initialized or if evaluation fails.
        """
        self._setup()
        results_summary = None
        if self._logger is None or self._agent is None or self._opponent is None:
            raise RuntimeError("Evaluator not properly initialized.")
        try:
            with self._logger as logger:
                logger.log("Starting Shogi Agent Evaluation (Evaluator class call).")
                logger.log(
                    f"Parameters: agent_ckpt='{self.agent_checkpoint_path}', opponent='{self.opponent_type}', num_games={self.num_games}"
                )
                from keisei.evaluation.loop import run_evaluation_loop

                results_summary = run_evaluation_loop(
                    self._agent,
                    self._opponent,
                    self.num_games,
                    logger,
                    self.max_moves_per_game,
                )
                logger.log(f"[Evaluator] Evaluation Summary: {results_summary}")
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[Evaluator] Error during evaluation run: {e}")
            results_summary = None
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error during evaluation run: {e}")
            results_summary = None

        if self._wandb_active and results_summary is not None:
            try:
                wandb.log(  # type: ignore
                    {
                        "eval/final_win_rate": results_summary["win_rate"],
                        "eval/final_loss_rate": results_summary["loss_rate"],
                        "eval/final_draw_rate": results_summary["draw_rate"],
                        "eval/final_avg_game_length": results_summary[
                            "avg_game_length"
                        ],
                    }
                )

            except (OSError, RuntimeError, ValueError) as e:
                print(
                    f"[Evaluator] Error logging final W&B metrics: {e}"
                )  # Added print
            except Exception as e:  # pylint: disable=broad-except
                print(
                    f"[Evaluator] Unexpected error logging final W&B metrics: {e}"
                )  # Added print
        if self._wandb_active and self._wandb_run:
            try:
                self._wandb_run.finish()  # type: ignore
                if hasattr(wandb, "finish"):
                    wandb.finish()  # Also call global wandb.finish() for test mocks
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[Evaluator] Error finishing W&B run: {e}")
            except Exception as e:  # pylint: disable=broad-except
                print(f"[Evaluator] Unexpected error finishing W&B run: {e}")
        return results_summary


def execute_full_evaluation_run(
    agent_checkpoint_path: str,
    opponent_type: str,
    opponent_checkpoint_path: Optional[str],
    num_games: int,
    max_moves_per_game: int,
    device_str: str,
    log_file_path_eval: str,
    policy_mapper: PolicyOutputMapper,
    seed: Optional[int] = None,
    wandb_log_eval: bool = False,
    wandb_project_eval: Optional[str] = None,
    wandb_entity_eval: Optional[str] = None,
    wandb_run_name_eval: Optional[str] = None,
    logger_also_stdout: bool = True,
    wandb_extra_config: Optional[dict] = None,
    wandb_reinit: Optional[bool] = None,
    wandb_group: Optional[str] = None,
) -> Optional[ResultsDict]:
    """
    Legacy-compatible wrapper for Evaluator class. Runs a full evaluation and returns the results dict.
    """
    evaluator = Evaluator(
        agent_checkpoint_path=agent_checkpoint_path,
        opponent_type=opponent_type,
        opponent_checkpoint_path=opponent_checkpoint_path,
        num_games=num_games,
        max_moves_per_game=max_moves_per_game,
        device_str=device_str,
        log_file_path_eval=log_file_path_eval,
        policy_mapper=policy_mapper,
        seed=seed,
        wandb_log_eval=wandb_log_eval,
        wandb_project_eval=wandb_project_eval,
        wandb_entity_eval=wandb_entity_eval,
        wandb_run_name_eval=wandb_run_name_eval,
        logger_also_stdout=logger_also_stdout,
        wandb_extra_config=wandb_extra_config,
        wandb_reinit=wandb_reinit,
        wandb_group=wandb_group,
    )
    return evaluator.evaluate()


def main_cli():
    """
    Entry point for CLI evaluation. This should parse arguments and call execute_full_evaluation_run.
    """
    parser = argparse.ArgumentParser(description="Evaluate a PPO Shogi agent.")
    parser.add_argument("--agent_checkpoint_path", required=True)
    parser.add_argument(
        "--opponent_type", required=True, choices=["random", "heuristic", "ppo"]
    )
    parser.add_argument("--opponent_checkpoint_path", default=None)
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--max_moves_per_game", type=int, default=300)
    parser.add_argument("--device_str", default="cpu")
    parser.add_argument("--log_file_path_eval", default="eval.log")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb_log_eval", action="store_true")
    parser.add_argument("--wandb_project_eval", default=None)
    parser.add_argument("--wandb_entity_eval", default=None)
    parser.add_argument("--wandb_run_name_eval", default=None)
    parser.add_argument("--logger_also_stdout", action="store_true")
    parser.add_argument("--wandb_reinit", action="store_true")
    parser.add_argument("--wandb_group", default=None)
    args = parser.parse_args()

    # Minimal PolicyOutputMapper for CLI use
    policy_mapper = PolicyOutputMapper()

    results = execute_full_evaluation_run(
        agent_checkpoint_path=args.agent_checkpoint_path,
        opponent_type=args.opponent_type,
        opponent_checkpoint_path=args.opponent_checkpoint_path,
        num_games=args.num_games,
        max_moves_per_game=args.max_moves_per_game,
        device_str=args.device_str,
        log_file_path_eval=args.log_file_path_eval,
        policy_mapper=policy_mapper,
        seed=args.seed,
        wandb_log_eval=args.wandb_log_eval,
        wandb_project_eval=args.wandb_project_eval,
        wandb_entity_eval=args.wandb_entity_eval,
        wandb_run_name_eval=args.wandb_run_name_eval,
        logger_also_stdout=args.logger_also_stdout,
        wandb_reinit=args.wandb_reinit,
        wandb_group=args.wandb_group,
    )
    print("Evaluation results:", results)
