"""
evaluate.py: Main script for evaluating PPO Shogi agents.
"""

import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv

import wandb
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

if TYPE_CHECKING:
    pass  # torch already imported above

load_dotenv()  # Load environment variables from .env file


class SimpleRandomOpponent(BaseOpponent):
    """An opponent that selects a random legal move."""

    def __init__(self, name: str = "SimpleRandomOpponent"):
        super().__init__(name)

    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        """Selects a random move from the list of legal moves."""
        legal_moves = game_instance.get_legal_moves()  # Removed current_player argument
        if not legal_moves:
            # This case should ideally be handled by the game loop checking for game_over
            raise ValueError(
                "No legal moves available for SimpleRandomOpponent, game should be over."
            )
        return random.choice(legal_moves)


class SimpleHeuristicOpponent(BaseOpponent):
    """An opponent that uses simple heuristics to select a move."""

    def __init__(self, name: str = "SimpleHeuristicOpponent"):
        super().__init__(name)

    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        """Selects a move based on simple heuristics."""
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError(
                "No legal moves available for SimpleHeuristicOpponent, game should be over."
            )

        capturing_moves: List[MoveTuple] = []
        non_promoting_pawn_moves: List[MoveTuple] = []
        other_moves: List[MoveTuple] = []

        for move_tuple in legal_moves:
            is_capture = False
            is_pawn_move_no_promo = False

            # Check if it's a BoardMoveTuple: (int, int, int, int, bool)
            if (
                isinstance(move_tuple[0], int)
                and isinstance(move_tuple[1], int)
                and isinstance(move_tuple[2], int)
                and isinstance(move_tuple[3], int)
                and isinstance(move_tuple[4], bool)
            ):

                from_r: int = move_tuple[0]
                from_c: int = move_tuple[1]
                to_r: int = move_tuple[2]
                to_c: int = move_tuple[3]
                promote: bool = move_tuple[4]

                # Heuristic 1: Check for capturing moves.
                destination_piece = game_instance.board[to_r][to_c]
                if (
                    destination_piece is not None
                    and destination_piece.color != game_instance.current_player
                ):
                    is_capture = True  # MODIFIED: Set is_capture to True

                # Heuristic 2: Check for non-promoting pawn moves (only if not a capture).
                if not is_capture:
                    source_piece = game_instance.board[from_r][
                        from_c
                    ]  # MODIFIED: Get source piece
                    if (
                        source_piece
                        and source_piece.type == PieceType.PAWN
                        and not promote
                    ):  # MODIFIED: Check if pawn and not promoting
                        is_pawn_move_no_promo = (
                            True  # MODIFIED: Set is_pawn_move_no_promo to True
                        )
            # Drop moves (Tuple[None, None, int, int, PieceType]) and other types of moves
            # will not pass the isinstance checks above.

            if is_capture:
                capturing_moves.append(move_tuple)
            if is_pawn_move_no_promo:  # Changed to if
                non_promoting_pawn_moves.append(move_tuple)
            else:
                other_moves.append(move_tuple)

        if capturing_moves:
            return random.choice(capturing_moves)
        if non_promoting_pawn_moves:  # Changed to if
            return random.choice(non_promoting_pawn_moves)
        if other_moves:
            return random.choice(other_moves)

        # Fallback, should ideally not be reached if legal_moves is not empty.
        return random.choice(legal_moves)


def load_evaluation_agent(
    checkpoint_path: str,
    device_str: str,
    policy_mapper: PolicyOutputMapper,
    input_channels: int,
) -> PPOAgent:
    """Loads a PPOAgent from a checkpoint for evaluation. Raises FileNotFoundError if checkpoint does not exist."""
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found.")
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
    # Use minimal config for evaluation
    config = AppConfig(
        env=EnvConfig(
            device=device_str,
            input_channels=input_channels,
            num_actions_total=policy_mapper.get_total_actions(),
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=1,
            learning_rate=1e-3,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
        ),
        evaluation=EvaluationConfig(num_games=1, opponent_type="random"),
        logging=LoggingConfig(log_file="/tmp/eval.log", model_dir="/tmp/"),
        wandb=WandBConfig(enabled=False, project="eval", entity=None),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )
    agent = PPOAgent(config=config, device=torch.device(device_str))
    agent.load_model(checkpoint_path)
    agent.model.eval()  # Set the model to evaluation mode
    print(f"Loaded agent from {checkpoint_path} on device {device_str} for evaluation.")
    return agent


def initialize_opponent(
    opponent_type: str,
    opponent_path: Optional[str],
    device_str: str,
    policy_mapper: PolicyOutputMapper,
    input_channels: int,
) -> Union[PPOAgent, BaseOpponent]:  # Adjusted return type
    """Initializes the opponent based on type."""
    if opponent_type == "random":
        print("Initializing SimpleRandomOpponent.")
        return SimpleRandomOpponent()
    elif opponent_type == "heuristic":
        print("Initializing SimpleHeuristicOpponent.")
        return SimpleHeuristicOpponent()
    elif opponent_type == "ppo":
        if not opponent_path:
            raise ValueError("Opponent path must be provided for PPO opponent type.")
        print(f"Initializing PPO opponent from {opponent_path}.")
        return load_evaluation_agent(
            opponent_path, device_str, policy_mapper, input_channels
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def run_evaluation_loop(
    agent_to_eval: PPOAgent,
    opponent: Union[PPOAgent, BaseOpponent],  # Adjusted opponent type
    num_games: int,
    logger: EvaluationLogger,
    policy_mapper: PolicyOutputMapper,
    max_moves_per_game: int,
    device_str: str,
    wandb_enabled: bool = False,  # Added for W&B
) -> dict:
    """Runs the evaluation loop for a set number of games."""
    wins = 0
    losses = 0
    draws = 0
    total_game_length = 0
    device = torch.device(device_str)

    # Determine opponent name for logging
    current_opponent_name = (
        opponent.name
        if isinstance(opponent, BaseOpponent)
        else opponent.__class__.__name__
    )
    logger.log(  # MODIFIED: Changed to logger.log
        f"Starting evaluation: {agent_to_eval.name} vs {current_opponent_name}"
    )

    for game_num in range(1, num_games + 1):
        game = ShogiGame(max_moves_per_game=max_moves_per_game)
        # Alternate starting player: agent_to_eval is Black (Sente) in odd games, White (Gote) in even games
        agent_is_black = game_num % 2 == 1

        # Determine who is playing which color for this game
        black_player = agent_to_eval if agent_is_black else opponent
        white_player = opponent if agent_is_black else agent_to_eval

        # Corrected log message format
        logger.log(  # MODIFIED: Changed to logger.log
            f"Starting Game {game_num}/{num_games}. "
            f"Agent to eval ({agent_to_eval.name}) is {'Black' if agent_is_black else 'White'}. "
            f"Opponent ({current_opponent_name}) is {'White' if agent_is_black else 'Black'}."
        )

        while not game.game_over:
            # Determine whose turn it is based on game.current_player
            active_agent = (
                black_player if game.current_player == Color.BLACK else white_player
            )

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            selected_move: Optional[MoveTuple] = None
            if isinstance(active_agent, PPOAgent):
                obs_np = shogi_game_io.generate_neural_network_observation(game)
                # obs_tensor = torch.tensor( # This variable was unused.
                #     obs_np, dtype=torch.float32, device=device
                # ).unsqueeze(0)
                legal_mask = policy_mapper.get_legal_mask(legal_moves, device)

                if not legal_mask.any() and legal_moves:
                    logger.log(  # MODIFIED: Changed to logger.log
                        f"Error: Game {game_num}, Move {game.move_count + 1}: "
                        f"Agent {active_agent.name} ({game.current_player.name}) has legal moves, "
                        f"but legal_mask is all False. Legal moves: {legal_moves}. "
                        f"This indicates an issue with PolicyOutputMapper or move generation."
                    )
                    # PPOAgent.select_action should handle this.
                    selected_shogi_move = active_agent.select_action(
                        obs_np, legal_mask, is_training=False
                    )[0]
                    selected_move = (
                        selected_shogi_move  # MODIFIED: Assign to selected_move
                    )
                else:
                    selected_shogi_move = active_agent.select_action(
                        obs_np, legal_mask, is_training=False
                    )[0]
                    selected_move = (
                        selected_shogi_move  # MODIFIED: Assign to selected_move
                    )
            elif isinstance(
                active_agent, BaseOpponent
            ):  # Opponent is a BaseOpponent (Random, Heuristic)
                selected_move = active_agent.select_move(game)
            else:
                # This case should not be reached if opponent types are correctly handled
                logger.log(
                    f"CRITICAL: Unsupported agent type for active_agent: {type(active_agent)}"
                )  # MODIFIED: Added log and changed to raise TypeError
                raise TypeError(
                    f"Unsupported agent type for active_agent: {type(active_agent)}"
                )

            if selected_move is None:
                logger.log(  # MODIFIED: Changed to logger.log
                    f"Error: Game {game_num}, Move {game.move_count + 1}: Active agent {active_agent.name} failed to select a move despite legal moves being available."
                )
                # Decide how to handle this: break, assign loss, etc. For now, break.
                break

            game.make_move(selected_move)
            # logger.log_custom_message(f"Game {game_num}, Move {game.move_count}: {active_agent.name} ({game.current_player.name}) played {selected_move}")

        # Game ended
        game_length = game.move_count
        total_game_length += game_length
        winner = game.winner

        outcome_str = "Draw"
        if winner is not None:
            if (winner == Color.BLACK and agent_is_black) or (
                winner == Color.WHITE and not agent_is_black
            ):
                wins += 1
                outcome_str = f"{agent_to_eval.name} (Agent) wins"
            else:
                losses += 1
                outcome_str = f"{current_opponent_name} (Opponent) wins"
        else:  # Draw
            draws += 1

        # Log main evaluation results
        # MODIFIED: Changed to logger.log and formatted the message
        logger.log(
            f"Game {game_num} Result: Opponent: {current_opponent_name}, "
            f"WinRate(cum): {wins / game_num if game_num > 0 else 0:.2f}, "
            f"AvgGameLen(cum): {(total_game_length / game_num if game_num > 0 else 0):.1f}, "
            f"Outcome: {outcome_str}"
        )
        # Log additional custom metrics for this game
        logger.log(  # MODIFIED: Changed to logger.log
            f"Game {game_num} Details: Length: {game_length}, Outcome: {outcome_str}, Agent Eval Color: {'Black' if agent_is_black else 'White'}"
        )
        logger.log(  # MODIFIED: Changed to logger.log
            f"Game {game_num} ended. Winner: {winner if winner else 'Draw'}"
        )

    avg_game_length = total_game_length / num_games if num_games > 0 else 0
    win_rate = wins / num_games if num_games > 0 else 0
    loss_rate = losses / num_games if num_games > 0 else 0
    draw_rate = draws / num_games if num_games > 0 else 0

    results = {
        "num_games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "avg_game_length": avg_game_length,
        "opponent_name": current_opponent_name,
        "agent_name": agent_to_eval.name,
    }

    logger.log(
        f"Evaluation finished. Results: {results}"
    )  # MODIFIED: Changed to logger.log
    if wandb_enabled:
        wandb.log(
            {
                "eval/total_games": num_games,
                "eval/wins": wins,
                "eval/losses": losses,
                "eval/draws": draws,
                "eval/win_rate": win_rate,
                "eval/loss_rate": loss_rate,
                "eval/draw_rate": draw_rate,
                "eval/avg_game_length": avg_game_length,
                # Log opponent name if needed, though it's in config
            }
        )

    return results


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
                raise RuntimeError(f"Failed to set random seed: {e}")

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
                    self._wandb_run = wandb.init(**wandb_kwargs)
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
                raise RuntimeError(f"Failed to create log directory '{log_dir_eval}': {e}")

        # Logger
        try:
            self._logger = EvaluationLogger(
                self.log_file_path_eval, also_stdout=self.logger_also_stdout
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EvaluationLogger: {e}")
        # Agent and opponent
        # Use the correct input_channels from self.policy_mapper if available, else default to 46
        input_channels = getattr(self.policy_mapper, "input_channels", 46)
        try:
            self._agent = load_evaluation_agent(
                self.agent_checkpoint_path,
                self.device_str,
                self.policy_mapper,
                input_channels,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluation agent: {e}")
        try:
            self._opponent = initialize_opponent(
                self.opponent_type,
                self.opponent_checkpoint_path,
                self.device_str,
                self.policy_mapper,
                input_channels,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize opponent: {e}")
        if self._logger is None or self._agent is None or self._opponent is None:
            raise RuntimeError(
                "Evaluator setup failed: logger, agent, or opponent is None."
            )

    def evaluate(self) -> Optional[dict]:
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
                results_summary = run_evaluation_loop(
                    self._agent,
                    self._opponent,
                    self.num_games,
                    logger,
                    self.policy_mapper,
                    self.max_moves_per_game,
                    self.device_str,
                    wandb_enabled=self._wandb_active,
                )
                logger.log(f"[Evaluator] Evaluation Summary: {results_summary}")
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[Evaluator] Error during evaluation run: {e}")
            results_summary = None
        except Exception as e:
            print(f"[Evaluator] Unhandled error during evaluation run: {e}")
            results_summary = None
        # Final W&B logging
        if self._wandb_active and results_summary is not None:
            try:
                wandb.log(
                    {
                        "eval/final_win_rate": results_summary["win_rate"],
                        "eval/final_loss_rate": results_summary["loss_rate"],
                        "eval/final_draw_rate": results_summary["draw_rate"],
                        "eval/final_avg_game_length": results_summary[
                            "avg_game_length"
                        ],
                    }
                )
                print("[Evaluator] Final W&B metrics logged.")
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[Evaluator] Error logging final metrics to W&B: {e}")
            except Exception as e:
                print(f"[Evaluator] Unhandled error logging to W&B: {e}")
        if self._wandb_active:
            try:
                wandb.finish()
                print("[Evaluator] W&B run finished.")
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[Evaluator] Error finishing W&B run: {e}")
            except Exception as e:
                print(f"[Evaluator] Unhandled error finishing W&B run: {e}")
        return results_summary


# --- Backward-compatible wrapper function ---
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
    _called_from_cli: bool = False,
) -> Optional[dict]:
    """
    Backward-compatible wrapper for programmatic evaluation. Calls the Evaluator class.
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


# --- CLI entry point ---
def main():
    """
    CLI entry point for evaluation. Parses arguments and runs evaluation using Evaluator.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a PPO Shogi agent.")
    parser.add_argument(
        "--agent_checkpoint",
        type=str,
        required=True,
        help="Path to agent checkpoint file.",
    )
    parser.add_argument(
        "--opponent_type",
        type=str,
        required=True,
        choices=["random", "heuristic", "ppo"],
        help="Type of opponent.",
    )
    parser.add_argument(
        "--opponent_checkpoint",
        type=str,
        default=None,
        help="Path to opponent checkpoint (if PPO).",
    )
    parser.add_argument(
        "--num_games", type=int, default=10, help="Number of games to play."
    )
    parser.add_argument(
        "--max_moves_per_game", type=int, default=200, help="Maximum moves per game."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for evaluation (cpu/cuda)."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="eval_log.txt",
        help="Path to evaluation log file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--wandb_log_eval", action="store_true", help="Enable W&B logging."
    )
    parser.add_argument(
        "--wandb_project_eval", type=str, default=None, help="W&B project name."
    )
    parser.add_argument(
        "--wandb_entity_eval", type=str, default=None, help="W&B entity."
    )
    parser.add_argument(
        "--wandb_run_name_eval", type=str, default=None, help="W&B run name."
    )
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group name.")
    parser.add_argument("--wandb_reinit", action="store_true", help="W&B reinit flag.")
    args = parser.parse_args()

    policy_mapper = PolicyOutputMapper()
    evaluator = Evaluator(
        agent_checkpoint_path=args.agent_checkpoint,
        opponent_type=args.opponent_type,
        opponent_checkpoint_path=args.opponent_checkpoint,
        num_games=args.num_games,
        max_moves_per_game=args.max_moves_per_game,
        device_str=args.device,
        log_file_path_eval=args.log_file,
        policy_mapper=policy_mapper,
        seed=args.seed,
        wandb_log_eval=args.wandb_log_eval,
        wandb_project_eval=args.wandb_project_eval,
        wandb_entity_eval=args.wandb_entity_eval,
        wandb_run_name_eval=args.wandb_run_name_eval,
        logger_also_stdout=True,
        wandb_extra_config=None,
        wandb_reinit=args.wandb_reinit,
        wandb_group=args.wandb_group,
    )
    results = evaluator.evaluate()
    print("Evaluation Results:")
    print(results)


if __name__ == "__main__":
    main()
