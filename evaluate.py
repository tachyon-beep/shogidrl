"""
evaluate.py: Main script for evaluating PPO Shogi agents.
"""

import argparse
import random
import sys
from typing import Optional, List, TYPE_CHECKING, Union

import torch # For torch.device and model loading

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
    device_str: str
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
        current_player_agent = agent_to_eval if agent_is_black else opponent
        other_agent = opponent if agent_is_black else agent_to_eval
        
        logger.log_custom_message(f"Starting Game {game_num}/{num_games}. Agent to eval is {"Black" if agent_is_black else "White"}.")

        while not game.game_over:
            active_agent = current_player_agent if game.current_player == (Color.BLACK if agent_is_black else Color.WHITE) else other_agent
            
            legal_moves = game.get_legal_moves() # Removed current_player argument
            if not legal_moves:
                # This should be caught by game.game_over, but as a safeguard:
                # print(f"Game {game_num}: No legal moves for {game.current_player}, game should be over.")
                break

            selected_move: Optional[MoveTuple] = None
            if isinstance(active_agent, PPOAgent):
                obs_np = shogi_game_io.generate_neural_network_observation(game) # Removed current_player argument
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0) # pylint: disable=unused-variable
                legal_mask = policy_mapper.get_legal_mask(legal_moves, device)
                
                # Ensure legal_mask is not all False if legal_moves exist
                if not legal_mask.any() and legal_moves:
                    logger.log_custom_message(f"Warning: Game {game_num}, Move {game.move_count + 1}: All legal moves masked out by PPO agent. Legal moves: {len(legal_moves)}")
                    # This indicates a potential issue with policy_mapper or move generation
                    # Fallback: choose a random move to prevent crash, or handle as error
                    selected_move = random.choice(legal_moves) 
                else:
                    # Pass is_training=False for deterministic action selection
                    action_tuple = active_agent.select_action(obs_np, legal_moves, legal_mask, is_training=False)
                    selected_move = action_tuple[0] # (move_tuple, policy_idx, log_prob, value)
            
            elif isinstance(active_agent, BaseOpponent): # Catches SimpleRandomOpponent and SimpleHeuristicOpponent
                selected_move = active_agent.select_move(game)
            
            if selected_move:
                # Log move before making it
                # usi_move = policy_mapper.shogi_move_to_usi(selected_move)
                # logger.log_custom_message(f"Game {game_num}, Ply {game.move_count+1} ({active_agent.name} as {game.current_player.name}): {usi_move}")
                game.make_move(selected_move)
            else:
                # This case should ideally not be reached if legal_moves exist and agents select a move.
                active_agent_name = active_agent.name if isinstance(active_agent, BaseOpponent) else active_agent.__class__.__name__
                logger.log_custom_message(f"Error: Game {game_num}, Move {game.move_count + 1}: No move selected by {active_agent_name}.") # Removed sfen
                # Decide how to handle: break, assign loss, etc.
                game.winner = Color.WHITE if game.current_player == Color.BLACK else Color.BLACK # Assign loss to current player
                game.game_over = True
                game.termination_reason = "Error: No move selected"
                break
        
        # Game finished
        total_game_length += game.move_count
        outcome_message = f"Game {game_num} ended. Winner: {game.winner}, Reason: {game.termination_reason}, Length: {game.move_count}"
        logger.log_custom_message(outcome_message)

        if game.winner is None: # Draw
            draws += 1
        elif (game.winner == Color.BLACK and agent_is_black) or (game.winner == Color.WHITE and not agent_is_black):
            wins +=1
        else:
            losses +=1

    avg_game_length = total_game_length / num_games if num_games > 0 else 0
    win_rate = wins / num_games if num_games > 0 else 0
    loss_rate = losses / num_games if num_games > 0 else 0 # For completeness
    draw_rate = draws / num_games if num_games > 0 else 0 # For completeness

    summary_results = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "avg_game_length": avg_game_length,
        "num_games": num_games
    }
    logger.log_custom_message(f"Evaluation finished. Results: {summary_results}")
    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO Shogi agent.")
    parser.add_argument("--agent-checkpoint", type=str, required=True, help="Path to the agent's .pth checkpoint file.")
    parser.add_argument("--opponent-type", type=str, default="random", choices=["random", "heuristic", "ppo"], help="Type of opponent to play against.")
    parser.add_argument("--opponent-checkpoint", type=str, help="Path to the opponent's .pth checkpoint file (if opponent-type is 'ppo').")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play for evaluation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for evaluation (cpu or cuda).")
    parser.add_argument("--max-moves-per-game", type=int, default=256, help="Maximum moves per game before it's declared a draw.")
    parser.add_argument("--log-file", type=str, default="evaluation_log.txt", help="Path to the evaluation log file.")
    # Add other arguments as needed from the plan, e.g., seed

    args = parser.parse_args()

    print("Starting evaluation with the following configuration:")
    print(f"  Agent Checkpoint: {args.agent_checkpoint}")
    print(f"  Opponent Type: {args.opponent_type}")
    if args.opponent_type == "ppo":
        if not args.opponent_checkpoint:
            print("Error: --opponent-checkpoint must be specified when --opponent-type is 'ppo'.", file=sys.stderr)
            sys.exit(1)
        print(f"  Opponent Checkpoint: {args.opponent_checkpoint}")
    print(f"  Number of Games: {args.num_games}")
    print(f"  Device: {args.device}")
    print(f"  Max Moves Per Game: {args.max_moves_per_game}")
    print(f"  Log File: {args.log_file}")

    # Initialize PolicyOutputMapper (needed for PPOAgent)
    policy_mapper = PolicyOutputMapper()

    # Initialize EvaluationLogger
    with EvaluationLogger(log_file_path=args.log_file, also_stdout=True) as eval_logger:
        eval_logger.log_custom_message(f"Evaluation script started. Args: {vars(args)}")

        # --- Agent Loading ---
        agent_to_eval = load_evaluation_agent(
            args.agent_checkpoint, 
            args.device, 
            policy_mapper, 
            INPUT_CHANNELS
        )

        # --- Opponent Initialization ---
        opponent = initialize_opponent(
            args.opponent_type, 
            args.opponent_checkpoint, 
            args.device, 
            policy_mapper, 
            INPUT_CHANNELS
        )
        
        # Determine opponent name for logging summary (moved after opponent initialization)
        opponent_name_for_summary = opponent.name if isinstance(opponent, BaseOpponent) else opponent.__class__.__name__

        # --- Evaluation Loop ---
        results = run_evaluation_loop(
            agent_to_eval, 
            opponent, 
            args.num_games, 
            eval_logger, 
            policy_mapper, 
            args.max_moves_per_game, 
            args.device
        )

        # --- Log Summary using EvaluationLogger's dedicated method ---
        eval_logger.log_evaluation_result(
            iteration=0, # Or some other relevant iteration number if applicable
            opponent_name=opponent_name_for_summary, # Use stored opponent_name
            win_rate=results["win_rate"],
            avg_game_length=results["avg_game_length"],
            num_games=results["num_games"]
        )
        eval_logger.log_custom_message(f"Detailed results: Wins={results['wins']}, Losses={results['losses']}, Draws={results['draws']}")

    print(f"\nEvaluation complete. Log saved to {args.log_file}")
    print(f"Summary: Win Rate: {results['win_rate']:.2f}, Avg Game Length: {results['avg_game_length']:.2f} over {results['num_games']} games.")


if __name__ == "__main__":
    main()
