"""
step_manager.py: Manages individual training steps and episode lifecycle.

This module encapsulates the logic for executing single training steps,
handling episode boundaries, and managing the interaction between the agent
and the environment during training.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from keisei.config_schema import AppConfig
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi import ShogiGame
from keisei.utils import PolicyOutputMapper, format_move_with_description_enhanced


@dataclass
class EpisodeState:
    """Represents the current state of a training episode."""
    current_obs: np.ndarray
    current_obs_tensor: torch.Tensor
    episode_reward: float
    episode_length: int


@dataclass
class StepResult:
    """Result of executing a single training step."""
    next_obs: np.ndarray
    next_obs_tensor: torch.Tensor
    reward: float
    done: bool
    info: Dict[str, Any]
    selected_move: Optional[Tuple]
    policy_index: int
    log_prob: float
    value_pred: float
    success: bool = True
    error_message: Optional[str] = None


class StepManager:
    """
    Manages individual training step execution and episode lifecycle.
    
    This class encapsulates the logic for:
    - Executing single training steps
    - Handling episode boundaries and resets
    - Managing demo mode interactions
    - Error handling and recovery during steps
    """

    def __init__(
        self,
        config: AppConfig,
        game: ShogiGame,
        agent: PPOAgent,
        policy_mapper: PolicyOutputMapper,
        experience_buffer: ExperienceBuffer,
    ):
        """
        Initialize the StepManager.
        
        Args:
            config: Application configuration
            game: Shogi game environment
            agent: PPO agent for action selection
            policy_mapper: Maps between game moves and policy outputs
            experience_buffer: Buffer for storing training experiences
        """
        self.config = config
        self.game = game
        self.agent = agent
        self.policy_mapper = policy_mapper
        self.experience_buffer = experience_buffer
        self.device = torch.device(config.env.device)
    
    def execute_step(
        self, 
        episode_state: EpisodeState, 
        global_timestep: int,
        logger_func: Callable[[str, bool, Optional[Dict], str], None]
    ) -> StepResult:
        """
        Execute a single training step.
        
        Args:
            episode_state: Current state of the episode
            global_timestep: Current global training timestep
            logger_func: Function for logging messages
            
        Returns:
            StepResult containing the outcome of the step
        """
        try:
            # Get legal moves for current position
            legal_shogi_moves = self.game.get_legal_moves()
            legal_mask_tensor = self.policy_mapper.get_legal_mask(
                legal_shogi_moves, device=self.device
            )

            # Prepare demo mode info if needed
            piece_info_for_demo = None
            if self.config.demo.enable_demo_mode and legal_shogi_moves:
                piece_info_for_demo = self._prepare_demo_info(legal_shogi_moves)

            # Agent action selection
            selected_shogi_move, policy_index, log_prob, value_pred = (
                self.agent.select_action(
                    episode_state.current_obs, legal_mask_tensor, is_training=True
                )
            )

            # Check if agent failed to select a move
            if selected_shogi_move is None:
                error_msg = f"Agent failed to select a move at timestep {global_timestep}"
                logger_func(
                    f"CRITICAL: {error_msg}. Resetting episode.",
                    True,  # also_to_wandb
                    None,  # wandb_data
                    "error"  # log_level
                )
                
                # Reset game and return failure result
                reset_obs = self.game.reset()
                reset_tensor = torch.tensor(
                    reset_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                
                return StepResult(
                    next_obs=reset_obs,
                    next_obs_tensor=reset_tensor,
                    reward=0.0,
                    done=False,
                    info={},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=error_msg
                )

            # Handle demo mode logging and delay
            if self.config.demo.enable_demo_mode:
                self._handle_demo_mode(
                    selected_shogi_move, 
                    episode_state.episode_length, 
                    piece_info_for_demo,
                    logger_func
                )

            # Execute the move in the environment
            move_result = self.game.make_move(selected_shogi_move)
            if not (isinstance(move_result, tuple) and len(move_result) == 4):
                raise ValueError(f"Invalid move result: {type(move_result)}")
            
            next_obs_np, reward, done, info = move_result

            # Add experience to buffer
            self.experience_buffer.add(
                episode_state.current_obs_tensor.squeeze(0),
                policy_index,
                reward,
                log_prob,
                value_pred,
                done,
                legal_mask_tensor,
            )

            # Create next observation tensor
            next_obs_tensor = torch.tensor(
                next_obs_np,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            return StepResult(
                next_obs=next_obs_np,
                next_obs_tensor=next_obs_tensor,
                reward=reward,
                done=done,
                info=info,
                selected_move=selected_shogi_move,
                policy_index=policy_index,
                log_prob=log_prob,
                value_pred=value_pred,
                success=True
            )

        except ValueError as e:
            error_msg = f"Error during training step: {e}"
            logger_func(
                f"CRITICAL: {error_msg}. Resetting episode.",
                True,  # also_to_wandb
                None,  # wandb_data
                "error"  # log_level
            )
            
            # Reset game and return failure result
            try:
                reset_obs = self.game.reset()
                reset_tensor = torch.tensor(
                    reset_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                
                return StepResult(
                    next_obs=reset_obs,
                    next_obs_tensor=reset_tensor,
                    reward=0.0,
                    done=False,
                    info={},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=error_msg
                )
            except Exception as reset_error:
                # If reset also fails, return a minimal failure result
                return StepResult(
                    next_obs=episode_state.current_obs,
                    next_obs_tensor=episode_state.current_obs_tensor,
                    reward=0.0,
                    done=True,  # Force episode end
                    info={},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=f"{error_msg}; Reset also failed: {reset_error}"
                )

    def handle_episode_end(
        self,
        episode_state: EpisodeState,
        step_result: StepResult,
        game_stats: Dict[str, int],  # black_wins, white_wins, draws
        total_episodes_completed: int,
        logger_func: Callable[..., None]
    ) -> EpisodeState:
        """
        Handle the end of an episode and prepare for the next one.
        
        Args:
            episode_state: Current episode state
            step_result: Result from the final step of the episode
            game_stats: Running game statistics (wins/draws)
            total_episodes_completed: Total episodes completed so far
            logger_func: Function for logging messages
            
        Returns:
            New EpisodeState for the next episode
        """
        # Extract game outcome information
        game_outcome_color = None
        game_outcome_reason = "Unknown"
        
        if step_result.info:
            game_outcome_color = step_result.info.get("winner")
            game_outcome_reason = step_result.info.get("reason", "Unknown")

        # Format game outcome message
        if game_outcome_color == "black":
            game_outcome_message = f"Black wins by {game_outcome_reason}."
        elif game_outcome_color == "white":
            game_outcome_message = f"White wins by {game_outcome_reason}."
        elif game_outcome_color is None:
            game_outcome_message = f"Draw by {game_outcome_reason}."
        else:
            game_outcome_message = f"Game ended: {game_outcome_color} by {game_outcome_reason}."

        # Calculate win rates
        total_games = game_stats["black_wins"] + game_stats["white_wins"] + game_stats["draws"]
        current_black_win_rate = (
            game_stats["black_wins"] / total_games
            if total_games > 0
            else 0.0
        )
        current_white_win_rate = (
            game_stats["white_wins"] / total_games
            if total_games > 0
            else 0.0
        )
        current_draw_rate = (
            game_stats["draws"] / total_games
            if total_games > 0
            else 0.0
        )

        # Log episode completion
        logger_func(
            f"Episode {total_episodes_completed + 1} finished. Length: {episode_state.episode_length}, "
            f"Reward: {episode_state.episode_reward:.2f}. {game_outcome_message}",
            also_to_wandb=True,
            wandb_data={
                "episode_reward": episode_state.episode_reward,
                "episode_length": episode_state.episode_length,
                "game_outcome": game_outcome_color,
                "game_reason": game_outcome_reason,
                "black_win_rate": current_black_win_rate,
                "white_win_rate": current_white_win_rate,
                "draw_rate": current_draw_rate,
            },
            log_level="info"
        )

        # Reset game for next episode
        try:
            reset_result = self.game.reset()
            if not isinstance(reset_result, np.ndarray):
                raise RuntimeError("Game reset failed after episode end")

            reset_obs_tensor = torch.tensor(
                reset_result,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            return EpisodeState(
                current_obs=reset_result,
                current_obs_tensor=reset_obs_tensor,
                episode_reward=0.0,
                episode_length=0
            )

        except (RuntimeError, ValueError, OSError) as e:
            # If reset fails, log error and return current state
            logger_func(
                f"CRITICAL: Game reset failed after episode end: {e}",
                True,  # also_to_wandb
                None,  # wandb_data
                "error"  # log_level
            )
            # Return current state to allow caller to handle the error
            return episode_state

    def reset_episode(self) -> EpisodeState:
        """
        Reset the game and create a new episode state.
        
        Returns:
            New EpisodeState for a fresh episode
        """
        reset_obs = self.game.reset()
        reset_tensor = torch.tensor(
            reset_obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return EpisodeState(
            current_obs=reset_obs,
            current_obs_tensor=reset_tensor,
            episode_reward=0.0,
            episode_length=0
        )

    def update_episode_state(
        self,
        episode_state: EpisodeState,
        step_result: StepResult
    ) -> EpisodeState:
        """
        Update episode state with the results of a step.
        
        Args:
            episode_state: Current episode state
            step_result: Result from the executed step
            
        Returns:
            Updated EpisodeState
        """
        return EpisodeState(
            current_obs=step_result.next_obs,
            current_obs_tensor=step_result.next_obs_tensor,
            episode_reward=episode_state.episode_reward + step_result.reward,
            episode_length=episode_state.episode_length + 1
        )

    def _prepare_demo_info(self, legal_shogi_moves) -> Optional[Any]:
        """
        Prepare piece information for demo mode display.
        
        Args:
            legal_shogi_moves: List of legal moves
            
        Returns:
            Piece information for the first legal move, or None if unavailable
        """
        if not legal_shogi_moves or legal_shogi_moves[0] is None:
            return None
            
        try:
            sample_move = legal_shogi_moves[0]
            if (
                len(sample_move) == 5
                and sample_move[0] is not None
                and sample_move[1] is not None
            ):
                from_r, from_c = sample_move[0], sample_move[1]
                return self.game.get_piece(from_r, from_c)
        except (AttributeError, IndexError, ValueError):
            pass  # Silently ignore errors in demo mode preparation
        
        return None

    def _handle_demo_mode(
        self,
        selected_move: Tuple,
        episode_length: int,
        piece_info_for_demo: Optional[Any],
        logger_func: Callable[[str, bool, Optional[Dict], str], None]
    ) -> None:
        """
        Handle demo mode logging and delay.
        
        Args:
            selected_move: The move selected by the agent
            episode_length: Current episode length
            piece_info_for_demo: Piece information for demo display
            logger_func: Function for logging messages
        """
        # Get current player name
        current_player_name = (
            getattr(
                self.game.current_player,
                "name",
                str(self.game.current_player),
            )
            if hasattr(self.game, "current_player")
            else "Unknown"
        )
        
        # Format move description
        move_str = format_move_with_description_enhanced(
            selected_move,
            self.policy_mapper,
            piece_info_for_demo,
        )
        
        # Log the move
        logger_func(
            f"Move {episode_length + 1}: {current_player_name} played {move_str}",
            False,  # also_to_wandb
            None,   # wandb_data
            "info"  # log_level
        )

        # Add delay for easier observation
        demo_delay = self.config.demo.demo_mode_delay
        if demo_delay > 0:
            time.sleep(demo_delay)
