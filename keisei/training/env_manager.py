"""
training/env_manager.py: Environment management for Shogi RL training.

This module handles environment-related concerns including:
- Game environment initialization and configuration
- Policy output mapper setup and validation
- Action space configuration and validation
- Environment seeding
- Observation space setup
"""

import sys
from typing import Any, Tuple

from keisei.config_schema import AppConfig
from keisei.shogi import ShogiGame
from keisei.utils import PolicyOutputMapper


class EnvManager:
    """Manages environment setup and configuration for training runs."""

    def __init__(self, config: AppConfig, logger_func=None):
        """
        Initialize the EnvManager.

        Args:
            config: Application configuration
            logger_func: Optional logging function for status messages
        """
        self.config = config
        self.logger_func = logger_func or (lambda msg: None)

        # Initialize environment components
        self.game: ShogiGame = None
        self.policy_output_mapper: PolicyOutputMapper = None
        self.action_space_size: int = 0
        self.obs_space_shape: Tuple[int, int, int] = None

        # Setup environment
        self._setup_environment()

    def _setup_environment(self):
        """Initialize game environment and related components."""
        try:
            # Initialize the Shogi game
            self.game = ShogiGame()

            # Setup seeding if specified
            if hasattr(self.game, "seed") and self.config.env.seed is not None:
                try:
                    self.game.seed(self.config.env.seed)
                    self.logger_func(f"Environment seeded with: {self.config.env.seed}")
                except Exception as e:
                    self.logger_func(f"Warning: Failed to seed environment: {e}")

            # Setup observation space shape
            self.obs_space_shape = (self.config.env.input_channels, 9, 9)

        except (RuntimeError, ValueError, OSError) as e:
            self.logger_func(f"Error initializing ShogiGame: {e}. Aborting.")
            raise RuntimeError(f"Failed to initialize ShogiGame: {e}") from e

        # Initialize policy output mapper
        try:
            self.policy_output_mapper = PolicyOutputMapper()
            self.action_space_size = self.policy_output_mapper.get_total_actions()

            # Validate action space consistency
            self._validate_action_space()

        except (RuntimeError, ValueError) as e:
            self.logger_func(f"Error initializing PolicyOutputMapper: {e}")
            raise RuntimeError(f"Failed to initialize PolicyOutputMapper: {e}") from e

    def _validate_action_space(self):
        """Validate that action space configuration is consistent."""
        config_num_actions = self.config.env.num_actions_total
        mapper_num_actions = self.policy_output_mapper.get_total_actions()

        if config_num_actions != mapper_num_actions:
            error_msg = (
                f"Action space mismatch: config specifies {config_num_actions} "
                f"actions but PolicyOutputMapper provides {mapper_num_actions} actions"
            )
            self.logger_func(f"CRITICAL: {error_msg}")
            raise ValueError(error_msg)

        self.logger_func(f"Action space validated: {mapper_num_actions} total actions")

    def get_environment_info(self) -> dict:
        """Get information about the current environment configuration."""
        return {
            "game": self.game,
            "policy_mapper": self.policy_output_mapper,
            "action_space_size": self.action_space_size,
            "obs_space_shape": self.obs_space_shape,
            "input_channels": self.config.env.input_channels,
            "num_actions_total": self.config.env.num_actions_total,
            "seed": self.config.env.seed,
            "game_type": type(self.game).__name__,
            "policy_mapper_type": type(self.policy_output_mapper).__name__,
        }

    def reset_game(self):
        """Reset the game environment to initial state."""
        try:
            self.game.reset()
            return True
        except Exception as e:
            self.logger_func(f"Error resetting game: {e}")
            return False

    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly configured and functional.

        Returns:
            bool: True if environment is valid, False otherwise
        """
        try:
            # Check game initialization
            if self.game is None:
                self.logger_func("Environment validation failed: game not initialized")
                return False

            # Check policy mapper
            if self.policy_output_mapper is None:
                self.logger_func(
                    "Environment validation failed: policy mapper not initialized"
                )
                return False

            # Check action space consistency
            if self.action_space_size <= 0:
                self.logger_func(
                    "Environment validation failed: invalid action space size"
                )
                return False

            # Test game reset functionality
            initial_state = self.game.get_board_state_copy()
            if not self.reset_game():
                self.logger_func("Environment validation failed: game reset failed")
                return False

            # Test observation space
            if self.obs_space_shape is None or len(self.obs_space_shape) != 3:
                self.logger_func(
                    "Environment validation failed: invalid observation space shape"
                )
                return False

            self.logger_func("Environment validation passed")
            return True

        except Exception as e:
            self.logger_func(f"Environment validation failed with exception: {e}")
            return False

    def get_legal_moves_count(self) -> int:
        """Get the number of legal moves in the current game state."""
        try:
            legal_moves = self.game.get_legal_moves()
            return len(legal_moves) if legal_moves else 0
        except Exception as e:
            self.logger_func(f"Error getting legal moves count: {e}")
            return 0

    def setup_seeding(self, seed: int = None):
        """
        Setup seeding for the environment.

        Args:
            seed: Optional seed value. If None, uses config seed.
        """
        seed_value = seed if seed is not None else self.config.env.seed

        if seed_value is not None and hasattr(self.game, "seed"):
            try:
                self.game.seed(seed_value)
                self.logger_func(f"Environment re-seeded with: {seed_value}")
                return True
            except Exception as e:
                self.logger_func(f"Error setting environment seed: {e}")
                return False
        return False
