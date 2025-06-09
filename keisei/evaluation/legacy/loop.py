"""
loop.py: Core evaluation loop for PPO Shogi agents.
"""

from typing import Optional, TypedDict, Union

from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_core_definitions import MoveTuple
from keisei.utils import BaseOpponent, EvaluationLogger, PolicyOutputMapper


class ResultsDict(TypedDict):
    games_played: int
    agent_wins: int
    opponent_wins: int
    draws: int
    game_results: list[str]
    win_rate: float
    loss_rate: float
    draw_rate: float
    avg_game_length: float


def run_evaluation_loop(
    agent_to_eval: PPOAgent,
    opponent: Union[PPOAgent, BaseOpponent],
    num_games: int,
    logger: EvaluationLogger,
    max_moves_per_game: int,
    policy_mapper: PolicyOutputMapper,
) -> ResultsDict:
    import torch

    results: ResultsDict = {
        "games_played": 0,
        "agent_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "game_results": [],
        "win_rate": 0.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "avg_game_length": 0.0,
    }
    from keisei.shogi.shogi_game import ShogiGame

    total_moves = 0
    for game_idx in range(num_games):
        logger.log(f"Starting evaluation game {game_idx + 1}/{num_games}")
        game = ShogiGame(max_moves_per_game=max_moves_per_game)
        move_count = 0
        while not game.game_over and move_count < max_moves_per_game:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.game_over = True
                break
            # Create proper legal mask using PolicyOutputMapper
            device = agent_to_eval.device  # Use agent's device for consistency
            legal_mask = policy_mapper.get_legal_mask(legal_moves, device)
            move = None  # type: ignore
            if game.current_player == 0:  # Sente (Black) - agent
                move_tuple = agent_to_eval.select_action(
                    game.get_observation(),
                    legal_mask,
                    is_training=False,
                )
                if move_tuple is not None:
                    move = (
                        move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
                    )
            else:  # Gote (White) - opponent
                if isinstance(opponent, BaseOpponent):
                    move = opponent.select_move(game)
                else:
                    move_tuple = opponent.select_action(
                        game.get_observation(),
                        legal_mask,
                        is_training=False,
                    )
                    if move_tuple is not None:
                        move = (
                            move_tuple[0]
                            if isinstance(move_tuple, tuple)
                            else move_tuple
                        )
            if move is None or move not in legal_moves:
                logger.log(f"Illegal move selected: {move}. Skipping.")
                game.game_over = True
                break
            try:
                game.make_move(move)
            except Exception as e:
                logger.log(f"Error making move: {e}")
                game.game_over = True
                break
            move_count += 1
        total_moves += move_count
        # Record result
        results["games_played"] += 1
        winner = game.winner
        if winner == 0:
            results["agent_wins"] += 1
            logger.log(f"Agent wins game {game_idx + 1}")
            results["game_results"].append("agent_win")
        elif winner == 1:
            results["opponent_wins"] += 1
            logger.log(f"Opponent wins game {game_idx + 1}")
            results["game_results"].append("opponent_win")
        else:
            results["draws"] += 1
            logger.log(f"Game {game_idx + 1} is a draw")
            results["game_results"].append("draw")
    # Compute summary statistics
    if results["games_played"] > 0:
        results["win_rate"] = results["agent_wins"] / results["games_played"]
        results["loss_rate"] = results["opponent_wins"] / results["games_played"]
        results["draw_rate"] = results["draws"] / results["games_played"]
        results["avg_game_length"] = total_moves / results["games_played"]
    else:
        results["win_rate"] = 0.0
        results["loss_rate"] = 0.0
        results["draw_rate"] = 0.0
        results["avg_game_length"] = 0.0
    return results
