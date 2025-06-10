"""
Tournament evaluation strategy implementation.
"""

import logging
import time  # Ensure time is imported
import uuid  # Ensure uuid is imported
from typing import Any, Dict, List, Optional, Tuple  # Ensure Tuple is imported

import torch  # Ensure torch is imported

from keisei.shogi.shogi_core_definitions import Color

# Keisei specific imports
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper
from keisei.utils.agent_loading import initialize_opponent, load_evaluation_agent

from ..core import (
    AgentInfo,
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
    GameResult,
    OpponentInfo,
    SummaryStats,
    TournamentConfig,
)

# Explicitly import create_game_result - this path needs to be correct
# Assuming create_game_result is a function in evaluation_result.py, not a class
# from ..core.evaluation_result import create_game_result # Removed, will use GameResult constructor


logger = logging.getLogger(__name__)

# Define constants for termination reasons
TERMINATION_REASON_MAX_MOVES = "Max moves reached"
TERMINATION_REASON_ILLEGAL_MOVE = "Illegal/No move"
TERMINATION_REASON_MOVE_EXECUTION_ERROR = "Move execution error"
TERMINATION_REASON_ACTION_SELECTION_ERROR = "Action selection error"
TERMINATION_REASON_NO_LEGAL_MOVES_UNDETERMINED = (
    "No Legal Moves (Outcome Undetermined by ShogiGame)"
)
TERMINATION_REASON_GAME_ENDED_UNSPECIFIED = (
    "Game ended (Reason not specified by ShogiGame)"
)
TERMINATION_REASON_GAME_ENDED_UNSPECIFIED_FALLBACK = (
    "Game ended (Reason not specified by ShogiGame) - Fallback"
)
TERMINATION_REASON_UNKNOWN_LOOP_TERMINATION = "Unknown (Loop terminated unexpectedly)"

# Constants for duplicate string literals
DEFAULT_DEVICE = "cpu"
DEFAULT_INPUT_CHANNELS = 46
IN_MEMORY_FALLBACK_MSG = "In-memory %s creation not yet implemented, falling back to regular loading"
CACHED_WEIGHTS_ERROR_MSG = "Failed to load %s from cached weights: %s"
TERMINATION_REASON_EVAL_STEP_ERROR = "Tournament evaluate_step error"


class TournamentEvaluator(BaseEvaluator):
    """Round-robin tournament evaluation against multiple opponents."""

    def __init__(self, config: TournamentConfig):  # type: ignore
        super().__init__(config)
        self.config: TournamentConfig = config  # type: ignore
        self.policy_mapper = PolicyOutputMapper()

    # --- Game Playing Helper Methods (adapted from SingleOpponentEvaluator) ---

    async def _game_load_evaluation_entity(
        self,
        entity_info: AgentInfo | OpponentInfo,
        device_str: str,
        input_channels: int,
    ) -> Any:
        """Helper to load an agent or opponent for a game."""
        if isinstance(entity_info, AgentInfo):
            if "agent_instance" in entity_info.metadata:
                return entity_info.metadata["agent_instance"]
            return load_evaluation_agent(
                checkpoint_path=entity_info.checkpoint_path or "",
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        if isinstance(entity_info, OpponentInfo) and entity_info.type == "ppo_agent":
            return load_evaluation_agent(
                checkpoint_path=entity_info.checkpoint_path or "",
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        if isinstance(entity_info, OpponentInfo):
            return initialize_opponent(
                opponent_type=entity_info.type,
                opponent_path=entity_info.checkpoint_path,
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        raise ValueError(f"Unknown entity type for loading: {type(entity_info)}")

    async def _game_get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            logger.error(
                "Player entity of type %s does not have a recognized action selection method.",
                type(player_entity).__name__,
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move

    async def _game_validate_and_make_move(
        self,
        game: ShogiGame,
        move: Any,
        legal_moves: List[Any],
        current_player_color_value: int,
        player_entity_type_name: str,
    ) -> bool:
        """Validates the selected move and makes it in the game. Returns True if successful."""
        if move is None or move not in legal_moves:
            logger.warning(
                "Player %s (%s) made an illegal move (%s) or no move. Legal moves: %s. Game ending.",
                current_player_color_value,
                player_entity_type_name,
                move,
                len(legal_moves) if legal_moves else "None",
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = TERMINATION_REASON_ILLEGAL_MOVE
            return False

        try:
            game.make_move(move)
            return True
        except (
            Exception
        ) as e:  # Broad exception to catch any issue during move execution by ShogiGame
            logger.error(
                "Error making move %s for player %s: %s",
                move,
                current_player_color_value,
                e,
                exc_info=True,
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = (
                f"{TERMINATION_REASON_MOVE_EXECUTION_ERROR}: {str(e)}"
            )
            return False

    async def _handle_no_legal_moves(self, game: ShogiGame) -> None:
        """Handles the situation where a player has no legal moves."""
        # ShogiGame.make_move() internally calls apply_move_to_board,
        # which should detect checkmate or stalemate if no legal moves exist
        # and set game.game_over, game.winner, and game.termination_reason.
        # This method is called if, for the current player's turn, get_legal_moves() is empty.

        game.game_over = (
            True  # Ensure game_over is set, though ShogiGame should manage this.
        )

        if game.winner is not None or game.termination_reason:
            logger.info(
                "Game ended: No legal moves for player %s. ShogiGame state: winner=%s, termination=%s",
                game.current_player.value,
                game.winner.name if game.winner else "None",
                game.termination_reason,
            )
        else:
            # This case should ideally not be reached if ShogiGame correctly identifies all terminal states
            # (checkmate, stalemate) when no legal moves are available.
            # If it's reached, it implies a situation ShogiGame didn't explicitly categorize.
            logger.warning(
                "No legal moves for player %s, and ShogiGame did not set a winner or termination_reason. "
                "This may indicate an unhandled terminal state in ShogiGame. "
                "The game outcome will likely be treated as a draw. Current player: %s",
                game.current_player.value,
                game.current_player.name,
            )
            # Defaulting to no winner (draw) and a generic reason if ShogiGame didn't provide one.
            # game.winner is already None and game.termination_reason is already None/empty here due to the 'else' condition.
            game.termination_reason = TERMINATION_REASON_NO_LEGAL_MOVES_UNDETERMINED

    async def _game_process_one_turn(
        self, game: ShogiGame, player_entity: Any, context: EvaluationContext
    ) -> bool:
        """
        Processes one turn for the current player.
        Returns True if the game continues, False if the game ended this turn.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            await self._handle_no_legal_moves(game)
            return False  # Game ended

        current_player_color_value = game.current_player.value
        player_entity_type_name = type(player_entity).__name__

        device_obj: torch.device
        if hasattr(player_entity, "device") and isinstance(
            getattr(player_entity, "device"), torch.device
        ):
            device_obj = getattr(player_entity, "device")
        else:
            default_device = "cpu"
            if (
                context and context.configuration
            ):  # Ensure context and configuration are not None
                default_device = getattr(context.configuration, "default_device", "cpu")
            device_obj = torch.device(default_device)

        legal_mask = self.policy_mapper.get_legal_mask(legal_moves, device_obj)
        move = None
        try:
            move = await self._game_get_player_action(player_entity, game, legal_mask)
        except (
            Exception
        ) as e:  # Broad exception for safety during diverse agent interactions
            logger.error(
                "Error during action selection for player %s (%s): %s",
                current_player_color_value,
                player_entity_type_name,
                e,
                exc_info=True,
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = (
                f"{TERMINATION_REASON_ACTION_SELECTION_ERROR}: {str(e)}"  # Use constant
            )
            return False  # Game ended

        if not await self._game_validate_and_make_move(
            game, move, legal_moves, current_player_color_value, player_entity_type_name
        ):
            return False  # Game ended due to invalid move

        return True  # Game continues

    async def _game_run_game_loop(
        self, sente_player: Any, gote_player: Any, context: EvaluationContext
    ) -> Dict[str, Any]:
        """Runs the Shogi game loop and returns outcome. Sente is player 0, Gote is player 1."""
        max_moves = 500  # Default
        if context and context.configuration:
            max_moves = getattr(context.configuration, "max_moves_per_game", 500)

        player_map = {0: sente_player, 1: gote_player}
        game = ShogiGame(max_moves_per_game=max_moves)
        move_count = 0

        while not game.game_over and move_count < max_moves:
            current_player_entity = player_map[game.current_player.value]
            if not await self._game_process_one_turn(
                game, current_player_entity, context
            ):
                break
            move_count += 1

        # Determine termination reason if not already set by game logic or turn processing
        if not game.termination_reason:
            if move_count >= max_moves:
                game.termination_reason = TERMINATION_REASON_MAX_MOVES
            elif game.game_over:  # Game ended, but no specific reason was set
                game.termination_reason = TERMINATION_REASON_GAME_ENDED_UNSPECIFIED
            else:  # Loop terminated unexpectedly (should ideally not happen)
                game.termination_reason = TERMINATION_REASON_UNKNOWN_LOOP_TERMINATION
                logger.warning(
                    "_game_run_game_loop: Reached '%s'. Moves: %s, Game Over: %s, ShogiGame Reason: %s",
                    game.termination_reason,
                    move_count,
                    game.game_over,
                    game.termination_reason,
                )

        winner_val = game.winner.value if game.winner is not None else None

        return {
            "winner": winner_val,
            "moves_count": move_count,
            "termination_reason": game.termination_reason,
        }

    # --- End of Game Playing Helper Methods ---

    async def _play_games_against_opponent(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        num_games_to_play: int,
        context: EvaluationContext,
    ) -> Tuple[List[GameResult], List[str]]:
        """Plays a set of games against a single opponent, balancing colors."""
        game_results_for_opponent: List[GameResult] = []
        errors_for_opponent: List[str] = []

        for i in range(num_games_to_play):
            current_opponent_info_for_game = OpponentInfo.from_dict(
                opponent_info.to_dict()
            )
            if opponent_info.metadata:  # Ensure metadata is a mutable copy
                current_opponent_info_for_game.metadata = opponent_info.metadata.copy()
            else:
                current_opponent_info_for_game.metadata = {}

            agent_plays_sente_in_this_game = i < (num_games_to_play + 1) // 2
            current_opponent_info_for_game.metadata[
                "agent_plays_sente_in_eval_step"
            ] = agent_plays_sente_in_this_game

            game_desc = f"game {i+1}/{num_games_to_play} vs {opponent_info.name} (Agent as {'Sente' if agent_plays_sente_in_this_game else 'Gote'})"
            try:
                logger.debug("Starting %s", game_desc)
                game_result = await self.evaluate_step(
                    agent_info, current_opponent_info_for_game, context
                )
                game_results_for_opponent.append(game_result)
            except (
                Exception
            ) as e:  # Broad exception for safety during orchestration of a single game step call
                error_msg = f"Error during game orchestration for {game_desc}: {e}"
                logger.error(
                    "Error orchestrating game: %s. Details: %s",
                    game_desc,
                    e,
                    exc_info=True,
                )
                errors_for_opponent.append(error_msg)

        logger.info(
            "Completed %s games against %s.",
            len(game_results_for_opponent),
            opponent_info.name,
        )
        return game_results_for_opponent, errors_for_opponent

    async def evaluate(
        self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """
        Run tournament evaluation.
        """
        if context is None:
            context = self.setup_context(agent_info)

        self.log_evaluation_start(agent_info, context)

        opponents = (
            await self._load_tournament_opponents()
        )  # context argument already removed
        if not opponents:
            logger.warning(
                "No opponents loaded for the tournament. Evaluation will be empty."
            )
            return EvaluationResult(
                context=context,
                games=[],
                summary_stats=SummaryStats.from_games([]),
                analytics_data={
                    "tournament_specific_analytics": {}
                },  # Ensure key exists for namespacing
                errors=["No opponents loaded for tournament."],
            )

        all_game_results: List[GameResult] = []
        errors: List[str] = []

        num_total_games_config = self.config.num_games
        num_games_per_opponent_pair = self.config.num_games_per_opponent

        if num_games_per_opponent_pair is None:
            if opponents:
                num_games_per_opponent_pair = max(
                    1, num_total_games_config // len(opponents)
                )
            else:
                num_games_per_opponent_pair = 0
            logger.info(
                "num_games_per_opponent not set, dynamically calculated to %s per opponent.",
                num_games_per_opponent_pair,
            )

        actual_total_games_to_play = num_games_per_opponent_pair * len(opponents)
        logger.info(
            "Tournament: Agent %s vs %s opponents. %s games per opponent. Total planned: %s.",
            agent_info.name,
            len(opponents),
            num_games_per_opponent_pair,
            actual_total_games_to_play,
        )

        for opponent_info in opponents:
            if num_games_per_opponent_pair > 0:
                results_one_opp, errors_one_opp = (
                    await self._play_games_against_opponent(
                        agent_info, opponent_info, num_games_per_opponent_pair, context
                    )
                )
                all_game_results.extend(results_one_opp)
                errors.extend(errors_one_opp)
            else:
                logger.info(
                    "Skipping games against %s as num_games_per_opponent_pair is 0.",
                    opponent_info.name,
                )

        summary_stats = SummaryStats.from_games(all_game_results)
        tournament_analytics_data = self._calculate_tournament_standings(
            all_game_results, opponents, agent_info
        )

        # Initialize analytics_data. PerformanceAnalyzer (if used via __post_init__) might add to this.
        current_analytics = {}

        evaluation_result = EvaluationResult(
            context=context,
            games=all_game_results,
            summary_stats=summary_stats,
            analytics_data=current_analytics,
            errors=errors,
        )

        # Namespace tournament-specific analytics.
        # If evaluation_result.analytics_data was populated by __post_init__, this adds to it.
        if (
            evaluation_result.analytics_data is None
        ):  # Should not happen if initialized to {}
            evaluation_result.analytics_data = {}

        evaluation_result.analytics_data["tournament_specific_analytics"] = (
            tournament_analytics_data
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    async def evaluate_in_memory(
        self,
        agent_info: AgentInfo,
        context: Optional[EvaluationContext] = None,
        *,
        agent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_info: Optional[OpponentInfo] = None
    ) -> EvaluationResult:
        """
        Run tournament evaluation using in-memory model weights and parallel execution.
        """
        if context is None:
            context = self.setup_context(agent_info)

        self.log_evaluation_start(agent_info, context)

        # Check if parallel execution is enabled
        enable_parallel = getattr(context.configuration, 'enable_parallel_execution', False)
        
        if enable_parallel:
            return await self._evaluate_tournament_parallel(agent_info, context)
        else:
            return await self._evaluate_tournament_sequential_in_memory(agent_info, context)

    async def _evaluate_tournament_parallel(
        self, agent_info: AgentInfo, context: EvaluationContext
    ) -> EvaluationResult:
        """Execute tournament evaluation with parallel game execution."""
        from ..core import create_parallel_game_tasks, BatchGameExecutor
        
        opponents = await self._load_tournament_opponents()
        if not opponents:
            logger.warning("No opponents loaded for the tournament. Evaluation will be empty.")
            return EvaluationResult(
                context=context,
                games=[],
                summary_stats=SummaryStats.from_games([]),
                analytics_data={"tournament_specific_analytics": {}},
                errors=["No opponents loaded for tournament."],
            )

        num_games_per_opponent = self.config.num_games_per_opponent
        if num_games_per_opponent is None:
            num_games_per_opponent = max(1, self.config.num_games // len(opponents))

        logger.info(
            f"Starting parallel tournament: {agent_info.name} vs {len(opponents)} opponents, "
            f"{num_games_per_opponent} games per opponent"
        )

        # Create parallel game tasks
        tasks = create_parallel_game_tasks(
            agent_info=agent_info,
            opponents=opponents,
            games_per_opponent=num_games_per_opponent,
            context=context,
            game_executor=self.evaluate_step_in_memory if hasattr(self, 'evaluate_step_in_memory') else self.evaluate_step
        )

        # Execute games in parallel batches
        batch_executor = BatchGameExecutor(
            batch_size=getattr(context.configuration, 'parallel_batch_size', 8),
            max_concurrent_games=getattr(context.configuration, 'max_concurrent_games', 4),
        )

        def progress_callback(completed: int, total: int):
            logger.info(f"Tournament progress: {completed}/{total} games completed ({completed/total:.1%})")

        game_results, errors = await batch_executor.execute_games_in_batches(
            tasks, progress_callback=progress_callback
        )

        summary_stats = SummaryStats.from_games(game_results)
        tournament_analytics = self._calculate_tournament_standings(game_results, opponents, agent_info)

        evaluation_result = EvaluationResult(
            context=context,
            games=game_results,
            summary_stats=summary_stats,
            analytics_data={"tournament_specific_analytics": tournament_analytics},
            errors=errors,
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    async def _evaluate_tournament_sequential_in_memory(
        self, agent_info: AgentInfo, context: EvaluationContext
    ) -> EvaluationResult:
        """Execute tournament evaluation sequentially using in-memory weights."""
        opponents = await self._load_tournament_opponents()
        if not opponents:
            logger.warning("No opponents loaded for the tournament. Evaluation will be empty.")
            return EvaluationResult(
                context=context,
                games=[],
                summary_stats=SummaryStats.from_games([]),
                analytics_data={"tournament_specific_analytics": {}},
                errors=["No opponents loaded for tournament."],
            )

        all_game_results: List[GameResult] = []
        errors: List[str] = []

        num_games_per_opponent = self.config.num_games_per_opponent
        if num_games_per_opponent is None:
            num_games_per_opponent = max(1, self.config.num_games // len(opponents))

        logger.info(
            f"Starting sequential in-memory tournament: {agent_info.name} vs {len(opponents)} opponents, "
            f"{num_games_per_opponent} games per opponent"
        )

        for opponent_info in opponents:
            if num_games_per_opponent > 0:
                results_one_opp, errors_one_opp = await self._play_games_against_opponent_in_memory(
                    agent_info, opponent_info, num_games_per_opponent, context
                )
                all_game_results.extend(results_one_opp)
                errors.extend(errors_one_opp)
            else:
                logger.info(f"Skipping games against {opponent_info.name} as num_games_per_opponent is 0.")

        summary_stats = SummaryStats.from_games(all_game_results)
        tournament_analytics = self._calculate_tournament_standings(all_game_results, opponents, agent_info)

        evaluation_result = EvaluationResult(
            context=context,
            games=all_game_results,
            summary_stats=summary_stats,
            analytics_data={"tournament_specific_analytics": tournament_analytics},
            errors=errors,
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    async def _play_games_against_opponent_in_memory(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        num_games_to_play: int,
        context: EvaluationContext,
    ) -> Tuple[List[GameResult], List[str]]:
        """Play games against opponent using in-memory evaluation if available."""
        game_results_for_opponent: List[GameResult] = []
        errors_for_opponent: List[str] = []

        for i in range(num_games_to_play):
            current_opponent_info_for_game = OpponentInfo.from_dict(opponent_info.to_dict())
            if opponent_info.metadata:
                current_opponent_info_for_game.metadata = opponent_info.metadata.copy()
            else:
                current_opponent_info_for_game.metadata = {}

            agent_plays_sente_in_this_game = i < (num_games_to_play + 1) // 2
            current_opponent_info_for_game.metadata["agent_plays_sente_in_eval_step"] = agent_plays_sente_in_this_game

            game_desc = f"game {i+1}/{num_games_to_play} vs {opponent_info.name} (Agent as {'Sente' if agent_plays_sente_in_this_game else 'Gote'})"
            try:
                logger.debug(f"Starting in-memory {game_desc}")
                
                # Try in-memory evaluation first, fallback to regular if not available
                if hasattr(self, 'evaluate_step_in_memory'):
                    game_result = await self.evaluate_step_in_memory(
                        agent_info, current_opponent_info_for_game, context
                    )
                else:
                    game_result = await self.evaluate_step(
                        agent_info, current_opponent_info_for_game, context
                    )
                
                game_results_for_opponent.append(game_result)
            except Exception as e:
                error_msg = f"Error during in-memory game orchestration for {game_desc}: {e}"
                logger.error(f"Error orchestrating in-memory game: {game_desc}. Details: {e}", exc_info=True)
                errors_for_opponent.append(error_msg)

        logger.info(f"Completed {len(game_results_for_opponent)} in-memory games against {opponent_info.name}.")
        return game_results_for_opponent, errors_for_opponent

    async def evaluate_step_in_memory(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """
        Evaluate a single game step using in-memory model weights.
        """
        game_id = f"tourney_mem_{context.session_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        agent_plays_sente = opponent_info.metadata.get("agent_plays_sente_in_eval_step", True)

        try:
            # Try to load entities from memory first
            logical_agent_entity = await self._load_evaluation_entity_in_memory(
                agent_info, context
            )
            logical_opponent_entity = await self._load_evaluation_entity_in_memory(
                opponent_info, context
            )

            sente_player = logical_agent_entity if agent_plays_sente else logical_opponent_entity
            gote_player = logical_opponent_entity if agent_plays_sente else logical_agent_entity

            game_outcome = await self._game_run_game_loop(sente_player, gote_player, context)
            duration = time.time() - start_time

            final_winner_code = None
            if game_outcome["winner"] is not None:
                sente_won = game_outcome["winner"] == 0
                final_winner_code = (
                    0 if (agent_plays_sente and sente_won) or (not agent_plays_sente and not sente_won)
                    else 1
                )

            game_metadata = opponent_info.metadata.copy()
            game_metadata["agent_color"] = "Sente" if agent_plays_sente else "Gote"
            game_metadata["sente_player_name"] = agent_info.name if agent_plays_sente else opponent_info.name
            game_metadata["gote_player_name"] = opponent_info.name if agent_plays_sente else agent_info.name
            game_metadata["termination_reason"] = game_outcome["termination_reason"]
            game_metadata["evaluation_mode"] = "in_memory"
            game_metadata.pop("agent_plays_sente_in_eval_step", None)

            return GameResult(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=final_winner_code,
                moves_count=game_outcome["moves_count"],
                duration_seconds=duration,
                metadata=game_metadata,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Critical error in TournamentEvaluator.evaluate_step_in_memory for game {game_id}: {e}", exc_info=True)
            error_metadata = {
                "error": str(e),
                "agent_plays_sente_in_eval_step": agent_plays_sente,
                "termination_reason": f"{TERMINATION_REASON_EVAL_STEP_ERROR}: {str(e)}",
                "evaluation_mode": "in_memory_failed",
            }
            return GameResult(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=None,
                moves_count=0,
                duration_seconds=duration,
                metadata=error_metadata,
            )

    async def _load_evaluation_entity_in_memory(
        self,
        entity_info: AgentInfo | OpponentInfo,
        context: EvaluationContext,
    ) -> Any:
        """Load an agent or opponent using in-memory weights if available."""
        try:
            # Try to create from cached weights using ModelWeightManager
            from pathlib import Path
            from ..core.model_manager import ModelWeightManager
            
            # Create a local manager instance since we don't have direct access to the evaluation manager here
            manager = ModelWeightManager(device=getattr(context.configuration, "default_device", "cpu"))
            
            # For agents, try to use cached weights
            if isinstance(entity_info, AgentInfo) and entity_info.checkpoint_path:
                try:
                    cached_weights = manager.cache_opponent_weights(
                        entity_info.name or "agent", 
                        Path(entity_info.checkpoint_path)
                    )
                    if cached_weights is not None:
                        agent = manager.create_agent_from_weights(
                            cached_weights, 
                            device=getattr(context.configuration, "default_device", "cpu")
                        )
                        logger.debug("Successfully created agent from cached weights")
                        return agent
                except Exception as e:
                    logger.warning("Failed to load agent from cached weights: %s", e)
            
            # For opponents, try to use cached weights
            elif isinstance(entity_info, OpponentInfo) and entity_info.type == "ppo_agent" and entity_info.checkpoint_path:
                try:
                    cached_weights = manager.cache_opponent_weights(
                        entity_info.name or "opponent", 
                        Path(entity_info.checkpoint_path)
                    )
                    if cached_weights is not None:
                        opponent = manager.create_agent_from_weights(
                            cached_weights,
                            device=getattr(context.configuration, "default_device", "cpu")
                        )
                        logger.debug("Successfully created opponent from cached weights")
                        return opponent
                except Exception as e:
                    logger.warning("Failed to load opponent from cached weights: %s", e)
            
            # Fallback to regular loading
            device_str = getattr(context.configuration, "default_device", "cpu")
            input_channels = getattr(context.configuration, "input_channels", 46)
            
            return await self._game_load_evaluation_entity(entity_info, device_str, input_channels)
            
        except Exception as e:
            logger.error("Error loading evaluation entity in memory: %s", e, exc_info=True)
            # Final fallback to regular loading
            device_str = getattr(context.configuration, "default_device", "cpu") 
            input_channels = getattr(context.configuration, "input_channels", 46)
            return await self._game_load_evaluation_entity(entity_info, device_str, input_channels)

    async def _load_tournament_opponents(
        self,
    ) -> List[OpponentInfo]:  # Removed context: EvaluationContext
        """Load all opponents for the tournament based on configuration."""
        opponent_configs = getattr(self.config, "opponent_pool_config", [])
        loaded_opponents: List[OpponentInfo] = []  # Corrected type hint

        if not opponent_configs:
            logger.warning(
                "No opponent configurations found for tournament in config: %s",
                getattr(self.config, "name", "TournamentConfig"),
            )  # Use generic name or actual name if available
            return loaded_opponents

        for i, opp_config_data in enumerate(opponent_configs):
            try:
                if isinstance(opp_config_data, OpponentInfo):
                    # If it's already an OpponentInfo, ensure its metadata is a mutable copy if we plan to modify it later per game.
                    # However, _load_tournament_opponents returns a list that is then iterated, and copies are made in `evaluate`.
                    # So, using it directly here should be fine.
                    loaded_opponents.append(opp_config_data)
                    continue

                if not isinstance(opp_config_data, dict):
                    logger.warning(
                        "Unsupported opponent config format: %s at index %d. Skipping.",
                        type(opp_config_data).__name__,
                        i,
                    )
                    continue

                name = opp_config_data.get("name", f"tournament_opponent_{i+1}")
                opp_type = opp_config_data.get("type", "random")

                # Ensure metadata is a dictionary
                metadata_source = opp_config_data.get("metadata", {})
                metadata = (
                    metadata_source.copy() if isinstance(metadata_source, dict) else {}
                )
                # If original opp_config_data itself was intended as metadata (excluding specific keys like name, type)
                # this logic might need adjustment based on expected structure of opponent_pool_config items.
                # For now, assume 'metadata' key holds the metadata, or the whole item is metadata if 'metadata' key is absent.
                # The current code: metadata = opp_config_data.get('metadata', opp_config_data).copy() was problematic if opp_config_data was not a dict.
                # The revised logic above is safer.

                opponent_info = OpponentInfo(
                    name=name,
                    type=opp_type,
                    checkpoint_path=opp_config_data.get("checkpoint_path"),
                    difficulty_level=opp_config_data.get("difficulty_level"),
                    version=opp_config_data.get("version"),
                    metadata=metadata,
                )
                loaded_opponents.append(opponent_info)
            except (
                Exception
            ) as e:  # Broad exception due to variability in opponent_pool_config structure
                logger.error(
                    "Failed to load opponent from config data at index %d (%s): %s",
                    i,
                    opp_config_data,
                    e,
                    exc_info=True,
                )

        logger.info("Loaded %s opponents for the tournament.", len(loaded_opponents))
        return loaded_opponents

    def _calculate_tournament_standings(
        self,
        results: List[GameResult],
        opponents: List[OpponentInfo],
        agent_info: AgentInfo,
    ) -> Dict[str, Any]:
        """Calculate tournament standings and statistics."""
        # This method should parse 'results' to create a summary of how the agent
        # performed against each opponent and overall tournament standings.
        # E.g., win/loss/draw record against each, overall tournament win rate, etc.

        standings: Dict[str, Any] = {"per_opponent_results": {}}
        agent_total_wins = 0
        agent_total_losses = 0
        agent_total_draws = 0

        for opponent in opponents:
            opponent_games = [
                g for g in results if g.opponent_info.name == opponent.name
            ]
            if not opponent_games:
                standings["per_opponent_results"][opponent.name] = {
                    "played": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "win_rate": 0,
                }
                continue

            wins = sum(1 for g in opponent_games if g.winner == 0)  # Agent wins
            losses = sum(1 for g in opponent_games if g.winner == 1)  # Opponent wins
            draws = sum(1 for g in opponent_games if g.winner is None)  # Draws

            agent_total_wins += wins
            agent_total_losses += losses
            agent_total_draws += draws

            standings["per_opponent_results"][opponent.name] = {
                "played": len(opponent_games),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / len(opponent_games) if opponent_games else 0,
            }

        total_tournament_games = (
            agent_total_wins + agent_total_losses + agent_total_draws
        )
        standings["overall_tournament_stats"] = {
            "total_games": total_tournament_games,
            "agent_total_wins": agent_total_wins,
            "agent_total_losses": agent_total_losses,
            "agent_total_draws": agent_total_draws,
            "agent_overall_win_rate": (
                agent_total_wins / total_tournament_games
                if total_tournament_games
                else 0
            ),
        }

        logger.info(
            "Tournament standings calculated for agent %s.", agent_info.name
        )  # Use module logger
        return standings

    def validate_config(self) -> bool:
        if not super().validate_config():
            return False
        if not hasattr(self.config, "opponent_pool_config") or not isinstance(
            getattr(self.config, "opponent_pool_config"), list
        ):
            logger.warning(
                "TournamentConfig.opponent_pool_config is missing or not a list in config: %s. Evaluation might not run as expected.",
                getattr(
                    self.config, "name", "TournamentConfig"
                ),  # Use generic name or actual name if available
            )
        return True


# Register this evaluator with the factory
from ..core import EvaluationStrategy, EvaluatorFactory

EvaluatorFactory.register(
    EvaluationStrategy.TOURNAMENT.value, TournamentEvaluator  # type: ignore
)
