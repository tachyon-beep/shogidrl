"""
evaluate.py: Main script for evaluating PPO Shogi agents.
"""

import argparse
import random
import sys
import os # Added os import
from typing import Optional, List, TYPE_CHECKING, Union

from dotenv import load_dotenv # Add this import

import numpy as np # Import numpy
import torch # For torch.device and model loading
import wandb # Added for Weights & Biases

from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper, EvaluationLogger, BaseOpponent
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import MoveTuple, Color, PieceType # Added PieceType
from keisei.shogi import shogi_game_io # For observations

# Constants from config or to be defined here
INPUT_CHANNELS = 46  # As per your config.py and shogi_core_definitions.py
NUM_ACTIONS_TOTAL = 13527 # As per PolicyOutputMapper, or could be from config if defined there

if TYPE_CHECKING:
    pass # torch already imported above

load_dotenv() # Load environment variables from .env file

class SimpleRandomOpponent(BaseOpponent):
    """An opponent that selects a random legal move."""

    def __init__(self, name: str = "SimpleRandomOpponent"):
        super().__init__(name)

    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        """Selects a random move from the list of legal moves."""
        legal_moves = game_instance.get_legal_moves() # Removed current_player argument
        if not legal_moves:
            # This case should ideally be handled by the game loop checking for game_over
            raise ValueError("No legal moves available for SimpleRandomOpponent, game should be over.")
        return random.choice(legal_moves)

class SimpleHeuristicOpponent(BaseOpponent):
    """An opponent that uses simple heuristics to select a move."""

    def __init__(self, name: str = "SimpleHeuristicOpponent"):
        super().__init__(name)

    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        """Selects a move based on simple heuristics."""
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available for SimpleHeuristicOpponent, game should be over.")
        
        capturing_moves: List[MoveTuple] = []
        non_promoting_pawn_moves: List[MoveTuple] = []
        other_moves: List[MoveTuple] = []

        for move_tuple in legal_moves:
            is_capture = False
            is_pawn_move_no_promo = False

            # Check if it's a BoardMoveTuple: (int, int, int, int, bool)
            if isinstance(move_tuple[0], int) and isinstance(move_tuple[1], int) and \
               isinstance(move_tuple[2], int) and isinstance(move_tuple[3], int) and \
               isinstance(move_tuple[4], bool):
                
                from_r: int = move_tuple[0]
                from_c: int = move_tuple[1]
                to_r: int = move_tuple[2]
                to_c: int = move_tuple[3]
                promote: bool = move_tuple[4]

                # Heuristic 1: Check for capturing moves.
                destination_piece = game_instance.board[to_r][to_c]
                if destination_piece is not None and destination_piece.color != game_instance.current_player:
                    is_capture = True
                
                # Heuristic 2: Check for non-promoting pawn moves (only if not a capture).
                if not is_capture:
                    current_piece_on_board = game_instance.board[from_r][from_c]
                    if current_piece_on_board is not None and \
                       current_piece_on_board.type == PieceType.PAWN and \
                       not promote:
                        is_pawn_move_no_promo = True
            
            # Drop moves (Tuple[None, None, int, int, PieceType]) and other types of moves
            # will not pass the isinstance checks above.

            if is_capture:
                capturing_moves.append(move_tuple)
            elif is_pawn_move_no_promo:
                non_promoting_pawn_moves.append(move_tuple)
            else:
                other_moves.append(move_tuple)
        
        if capturing_moves:
            return random.choice(capturing_moves)
        if non_promoting_pawn_moves:
            return random.choice(non_promoting_pawn_moves)
        if other_moves: 
            return random.choice(other_moves)
        
        # Fallback, should ideally not be reached if legal_moves is not empty.
        return random.choice(legal_moves)


def load_evaluation_agent(
    checkpoint_path: str,
    device_str: str,
    policy_mapper: PolicyOutputMapper,
    input_channels: int
) -> PPOAgent:
    """Loads a PPOAgent from a checkpoint for evaluation."""
    agent = PPOAgent(
        input_channels=input_channels,
        policy_output_mapper=policy_mapper,
        device=device_str
        # Other PPO params like lr, gamma, etc., are not strictly needed for eval-only model,
        # but PPOAgent constructor might require them. For now, assume defaults are fine.
    )
    agent.load_model(checkpoint_path)
    agent.model.eval()  # Set the model to evaluation mode
    print(f"Loaded agent from {checkpoint_path} on device {device_str} for evaluation.")
    return agent


def initialize_opponent(
    opponent_type: str,
    opponent_path: Optional[str],
    device_str: str,
    policy_mapper: PolicyOutputMapper,
    input_channels: int
) -> Union[PPOAgent, BaseOpponent]: # Adjusted return type
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
        return load_evaluation_agent(opponent_path, device_str, policy_mapper, input_channels)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def run_evaluation_loop(
    agent_to_eval: PPOAgent, 
    opponent: Union[PPOAgent, BaseOpponent], # Adjusted opponent type
    num_games: int, 
    logger: EvaluationLogger, 
    policy_mapper: PolicyOutputMapper, 
    max_moves_per_game: int, 
    device_str: str,
    wandb_enabled: bool = False # Added for W&B
) -> dict:
    """Runs the evaluation loop for a set number of games."""
    wins = 0
    losses = 0
    draws = 0
    total_game_length = 0
    device = torch.device(device_str)

    # Determine opponent name for logging
    current_opponent_name = opponent.name if isinstance(opponent, BaseOpponent) else opponent.__class__.__name__
    logger.log_custom_message(f"Starting evaluation: {agent_to_eval.name} vs {current_opponent_name}")

    for game_num in range(1, num_games + 1):
        game = ShogiGame(max_moves_per_game=max_moves_per_game)
        # Alternate starting player: agent_to_eval is Black (Sente) in odd games, White (Gote) in even games
        agent_is_black = game_num % 2 == 1
        
        # Determine who is playing which color for this game
        black_player = agent_to_eval if agent_is_black else opponent
        white_player = opponent if agent_is_black else agent_to_eval
        
        # Corrected log message format
        logger.log_custom_message(f"Starting Game {game_num}/{num_games}. "
                                  f"Agent to eval ({agent_to_eval.name}) is {'Black' if agent_is_black else 'White'}. "
                                  f"Opponent ({current_opponent_name}) is {'White' if agent_is_black else 'Black'}.")

        while not game.game_over:
            # Determine whose turn it is based on game.current_player
            active_agent = black_player if game.current_player == Color.BLACK else white_player
            
            legal_moves = game.get_legal_moves() 
            if not legal_moves:
                break

            selected_move: Optional[MoveTuple] = None
            if isinstance(active_agent, PPOAgent):
                obs_np = shogi_game_io.generate_neural_network_observation(game) 
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0) 
                legal_mask = policy_mapper.get_legal_mask(legal_moves, device)
                
                if not legal_mask.any() and legal_moves:
                    logger.log_custom_message(f"Warning: Game {game_num}, Move {game.move_count + 1}: All legal moves masked out by PPO agent {active_agent.name}. Legal moves: {len(legal_moves)}")
                    selected_move = random.choice(legal_moves) 
                else:
                    action_tuple = active_agent.select_action(obs_np, legal_moves, legal_mask, is_training=False)
                    selected_move = action_tuple[0]
            elif isinstance(active_agent, BaseOpponent): # Opponent is a BaseOpponent (Random, Heuristic)
                selected_move = active_agent.select_move(game)
            else:
                # This case should not be reached if opponent types are correctly handled
                raise TypeError(f"Unsupported agent type for active_agent: {type(active_agent)}")

            if selected_move is None:
                logger.log_custom_message(f"Error: Game {game_num}, Move {game.move_count + 1}: Active agent {active_agent.name} failed to select a move despite legal moves being available.")
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
            if (winner == Color.BLACK and agent_is_black) or \
               (winner == Color.WHITE and not agent_is_black):
                wins += 1
                outcome_str = f"{agent_to_eval.name} (Agent) wins"
            else:
                losses += 1
                outcome_str = f"{current_opponent_name} (Opponent) wins"
        else: # Draw
            draws += 1
        
        # Log main evaluation results
        logger.log_evaluation_result(
            iteration=game_num, # Using game_num as iteration for per-game logging
            opponent_name=current_opponent_name,
            win_rate=wins / game_num if game_num > 0 else 0, # Cumulative win rate
            avg_game_length=total_game_length / game_num if game_num > 0 else 0, # Cumulative avg length
            num_games=game_num # Pass the current game number as num_games for this specific log entry
        )
        # Log additional custom metrics for this game
        logger.log_custom_message(
            f"Game {game_num} Details: Length: {game_length}, Outcome: {outcome_str}, Agent Eval Color: {"Black" if agent_is_black else "White"}"
        )
        logger.log_custom_message(f"Game {game_num} ended. Winner: {winner if winner else 'Draw'}")

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
        "agent_name": agent_to_eval.name
    }
    
    logger.log_custom_message(f"Evaluation finished. Results: {results}")
    if wandb_enabled:
        wandb.log({
            "eval/total_games": num_games,
            "eval/wins": wins,
            "eval/losses": losses,
            "eval/draws": draws,
            "eval/win_rate": win_rate,
            "eval/loss_rate": loss_rate,
            "eval/draw_rate": draw_rate,
            "eval/avg_game_length": avg_game_length,
            # Log opponent name if needed, though it's in config
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO Shogi agent.")
    parser.add_argument("--agent-checkpoint", type=str, required=True, help="Path to the agent's .pth checkpoint file.")
    parser.add_argument("--opponent-type", type=str, required=True, choices=["random", "heuristic", "ppo"], help="Type of opponent.")
    parser.add_argument("--opponent-checkpoint", type=str, help="Path to opponent's .pth checkpoint file (if opponent-type is 'ppo').")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play.")
    parser.add_argument("--max-moves-per-game", type=int, default=300, help="Maximum moves per game.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--log-file", type=str, default="logs/evaluation_log.txt", help="Path to the evaluation log file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    
    # W&B arguments
    parser.add_argument("--wandb-log", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="shogi-ppo-evaluation", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username or team name).")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Custom name for the W&B run.")


    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed) # Add numpy random seed
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        print(f"Set random seed to: {args.seed}")

    # Initialize W&B if enabled
    if args.wandb_log:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args) # Log all CLI arguments
            )
            print(f"Weights & Biases logging enabled. Project: {args.wandb_project}, Run Name: {wandb.run.name if wandb.run else 'N/A'}")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}. W&B logging will be disabled.", file=sys.stderr)
            args.wandb_log = False # Disable if init fails


    policy_mapper = PolicyOutputMapper() # Assuming this doesn't need device yet

    # Ensure log directory exists (EvaluationLogger might not create it)
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    with EvaluationLogger(args.log_file, also_stdout=True) as logger:
        logger.log_custom_message("Starting Shogi Agent Evaluation Script.")
        logger.log_custom_message(f"Arguments: {vars(args)}")

        try:
            agent_to_eval = load_evaluation_agent(
                args.agent_checkpoint, args.device, policy_mapper, INPUT_CHANNELS
            )
            opponent = initialize_opponent(
                args.opponent_type, args.opponent_checkpoint, args.device, policy_mapper, INPUT_CHANNELS
            )

            results = run_evaluation_loop(
                agent_to_eval, 
                opponent, 
                args.num_games, 
                logger, 
                policy_mapper, 
                args.max_moves_per_game, 
                args.device,
                wandb_enabled=args.wandb_log # Pass W&B status
            )

            print("\\n--- Evaluation Summary ---")
            print(f"Agent: {agent_to_eval.name}")
            print(f"Opponent: {opponent.name if isinstance(opponent, BaseOpponent) else opponent.__class__.__name__}")
            print(f"Number of Games: {results['num_games']}")
            print(f"Wins: {results['wins']} ({results['win_rate']:.2%})")
            print(f"Losses: {results['losses']} ({results['loss_rate']:.2%})")
            print(f"Draws: {results['draws']} ({results['draw_rate']:.2%})")
            print(f"Average Game Length: {results['avg_game_length']:.2f} moves")
            logger.log_custom_message(f"Final Summary: {results}")

        except Exception as e:
            logger.log_custom_message(f"An error occurred during evaluation: {e}") # Removed level="ERROR"
            print(f"An error occurred: {e}", file=sys.stderr)
            if args.wandb_log and wandb.run:
                wandb.log({"eval/error": 1}) # Log an error metric
            # sys.exit(1) # Exit with error code # Commented out to allow finally to run
        finally:
            if args.wandb_log and wandb.run:
                wandb.finish()
                print("Weights & Biases run finished.")


if __name__ == "__main__":
    # Add os import for path manipulation if not already present at the top
    import os 
    # This ensures that the script can find other modules in the 'keisei' package
    # when run directly, especially if 'keisei' is not installed in the environment.
    if __package__ is None:
        # Get the directory of the current script (evaluate.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory (which should be the project root, e.g., /home/john/keisei)
        project_root = os.path.dirname(script_dir)
        # Add the project root to sys.path so Python can find the \'keisei\' package
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            # print(f"Added {project_root} to sys.path for module resolution.")

    main()
