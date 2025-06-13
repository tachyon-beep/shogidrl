"""
Test object factories for evaluation tests.

This module provides factories for creating realistic test objects
to replace excessive mocking in the evaluation test suite.
"""

import tempfile
from datetime import datetime
from typing import List, Optional

import torch

from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.core import (
    AgentInfo,
    EvaluationContext,
    EvaluationStrategy,
    GameResult,
    OpponentInfo,
    create_evaluation_config,
)
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper
from tests.evaluation.conftest import make_test_config


class TestPPOAgent(PPOAgent):
    """Test-specific PPOAgent that always selects valid moves for deterministic testing."""

    def __init__(self, model, config, device, name="TestAgent"):
        super().__init__(model, config, device, name)
        self.policy_mapper = PolicyOutputMapper()

    def select_action(self, obs, legal_mask, *, is_training=True):
        """Override to always return valid MoveTuple objects."""
        # Get first valid action index
        action_idx = self._get_first_valid_action(legal_mask)

        # Convert action index to MoveTuple
        move_tuple = self._convert_action_to_move(action_idx)

        # Return realistic values: MoveTuple, action_idx, log_prob, value
        return move_tuple, action_idx, -1.0, 0.5

    def _get_first_valid_action(self, legal_mask):
        """Get the first valid action from the legal mask."""
        if legal_mask is not None and legal_mask.any():
            legal_indices = legal_mask.nonzero(as_tuple=True)[0]
            return int(legal_indices[0])
        else:
            return 0  # Fallback

    def _convert_action_to_move(self, action_idx):
        """Convert action index to MoveTuple using PolicyOutputMapper."""
        try:
            return self.policy_mapper.policy_index_to_shogi_move(action_idx)
        except Exception:
            # Fallback to a basic move if conversion fails
            return (0, 0, 1, 0, False)  # Simple board move


class EvaluationTestFactory:
    """Factory for creating realistic evaluation test objects."""

    @staticmethod
    def create_test_agent(name: str = "TestAgent", device: str = "cpu") -> PPOAgent:
        """Create real agent with test configuration that always selects valid moves."""
        config = make_test_config()
        policy_mapper = PolicyOutputMapper()

        # Create real neural network model
        model = ActorCritic(
            config.env.input_channels, policy_mapper.get_total_actions()
        )

        # Create TestPPOAgent with deterministic action selection
        agent = TestPPOAgent(
            model=model,
            config=config,
            device=torch.device(device),
            name=name,
        )

        return agent

    @staticmethod
    def create_test_evaluation_config(
        strategy: EvaluationStrategy = EvaluationStrategy.SINGLE_OPPONENT,
        num_games: int = 3,
        **kwargs,
    ):
        """Create realistic evaluation configuration for fast testing."""
        defaults = {
            "wandb_logging": False,
            "timeout_per_game": 30.0,
            "max_concurrent_games": 1,
            "randomize_positions": False,
            "save_games": False,
            "log_level": "INFO",
            "update_elo": False,
            "enable_in_memory_evaluation": True,
            "model_weight_cache_size": 3,
            "enable_parallel_execution": False,
            "process_restart_threshold": 100,
            "temp_agent_device": "cpu",
            "clear_cache_after_evaluation": True,
            "strategy_params": {},
        }

        # Add strategy-specific defaults
        if strategy == EvaluationStrategy.SINGLE_OPPONENT:
            defaults.update(
                {
                    "opponent_name": kwargs.get("opponent_name", "random"),
                    "opponent_params": {},
                    "play_as_both_colors": False,
                    "color_balance_tolerance": 0.1,
                }
            )

        defaults.update(kwargs)

        return create_evaluation_config(
            strategy=strategy, num_games=num_games, **defaults
        )

    @staticmethod
    def create_test_game_state() -> ShogiGame:
        """Create realistic game state for testing."""
        # Return fresh game - move sequences will be implemented later when move format is clarified
        return ShogiGame()

    @staticmethod
    def create_test_agent_info(
        name: str = "TestAgent",
        checkpoint_path: Optional[str] = None,
        agent_type: str = "ppo_agent",
    ) -> AgentInfo:
        """Create realistic agent info for testing."""
        if checkpoint_path is None:
            # Create a temporary checkpoint file
            checkpoint_path = EvaluationTestFactory._create_dummy_checkpoint()

        return AgentInfo(
            name=name,
            checkpoint_path=checkpoint_path,
            model_type=agent_type,
            metadata={"test": True},
        )

    @staticmethod
    def create_test_opponent_info(
        name: str = "TestOpponent",
        opponent_type: str = "random",
        checkpoint_path: Optional[str] = None,
    ) -> OpponentInfo:
        """Create realistic opponent info for testing."""
        return OpponentInfo(
            name=name,
            type=opponent_type,  # Note: OpponentInfo uses 'type' not 'opponent_type'
            checkpoint_path=checkpoint_path,
            metadata={"test_opponent": True},
        )

    @staticmethod
    def create_test_evaluation_context(
        session_id: str = "test_session_123",
        agent_info: Optional[AgentInfo] = None,
        configuration=None,
    ) -> EvaluationContext:
        """Create realistic evaluation context for testing."""
        if agent_info is None:
            agent_info = EvaluationTestFactory.create_test_agent_info()

        if configuration is None:
            configuration = EvaluationTestFactory.create_test_evaluation_config()

        return EvaluationContext(
            session_id=session_id,
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=configuration,
            environment_info={"device": "cpu", "test": True},
            metadata={"test_context": True},
        )

    @staticmethod
    def create_test_game_results(
        count: int = 5,
        agent_info: Optional[AgentInfo] = None,
        opponent_info: Optional[OpponentInfo] = None,
        win_rate: float = 0.5,
    ) -> List[GameResult]:
        """Create realistic game results for testing."""
        if agent_info is None:
            agent_info = EvaluationTestFactory.create_test_agent_info()

        if opponent_info is None:
            opponent_info = EvaluationTestFactory.create_test_opponent_info()

        results = []
        for i in range(count):
            # Determine winner based on win_rate
            if i / count < win_rate:
                winner = 0  # Agent wins
            elif i / count < win_rate + (1 - win_rate) / 2:
                winner = 1  # Opponent wins
            else:
                winner = None  # Draw

            result = GameResult(
                game_id=f"test_game_{i}",
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=winner,
                moves_count=20 + i * 3,  # Realistic move counts
                duration_seconds=30.0 + i * 5,  # Realistic durations
                metadata={"test_game": True, "game_index": i},
            )
            results.append(result)

        return results

    @staticmethod
    def _create_dummy_checkpoint() -> str:
        """Create a temporary checkpoint file for testing."""
        # Create temporary file that persists for the test
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            # Create minimal checkpoint data
            checkpoint_data = {
                "model_state_dict": {
                    "dummy_weight": torch.randn(5, 5),
                    "dummy_bias": torch.randn(5),
                },
                "optimizer_state_dict": {},
                "epoch": 1,
            }
            torch.save(checkpoint_data, tmp.name)
            return tmp.name


class GameStateFactory:
    """Factory for creating realistic game states."""

    @classmethod
    def create_opening_position(cls) -> ShogiGame:
        """Create realistic opening game positions."""
        # Return fresh game - move sequences will be implemented later
        return ShogiGame()

    @classmethod
    def create_midgame_position(cls) -> ShogiGame:
        """Create realistic midgame positions with specified complexity."""
        # Return fresh game - move sequences will be implemented later
        return ShogiGame()

    @classmethod
    def create_endgame_position(cls) -> ShogiGame:
        """Create realistic endgame positions."""
        # Return fresh game - move sequences will be implemented later
        return ShogiGame()


class EvaluationScenarioFactory:
    """Factory for creating complete evaluation scenarios."""

    @classmethod
    def create_balanced_matchup(cls, games_count: int = 10):
        """Create scenario with evenly matched opponents."""
        config = EvaluationTestFactory.create_test_evaluation_config(
            num_games=games_count
        )
        agent_info = EvaluationTestFactory.create_test_agent_info()
        opponent_info = EvaluationTestFactory.create_test_opponent_info()

        return {
            "config": config,
            "agent_info": agent_info,
            "opponent_info": opponent_info,
            "expected_win_rate": 0.5,
        }

    @classmethod
    def create_skill_gap_scenario(cls, elo_difference: int = 400):
        """Create scenario with significant skill difference."""
        strong_agent = EvaluationTestFactory.create_test_agent_info(name="StrongAgent")
        weak_opponent = EvaluationTestFactory.create_test_opponent_info(
            name="WeakOpponent"
        )

        # Expected win rate based on ELO difference
        expected_win_rate = 1 / (1 + 10 ** (-elo_difference / 400))

        return {
            "config": EvaluationTestFactory.create_test_evaluation_config(),
            "agent_info": strong_agent,
            "opponent_info": weak_opponent,
            "expected_win_rate": expected_win_rate,
        }

    @classmethod
    def create_performance_test_scenario(cls, game_count: int = 100):
        """Create large-scale evaluation scenario for performance testing."""
        return {
            "config": EvaluationTestFactory.create_test_evaluation_config(
                num_games=game_count, strategy=EvaluationStrategy.TOURNAMENT
            ),
            "agent_info": EvaluationTestFactory.create_test_agent_info(),
            "opponents": [
                EvaluationTestFactory.create_test_opponent_info(f"Opponent_{i}")
                for i in range(5)
            ],
        }


class ConfigurationTemplates:
    """Templates for common evaluation configurations."""

    @staticmethod
    def quick_evaluation_config():
        """Fast evaluation for unit tests."""
        return create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=3,
            timeout_per_game=30,
            wandb_logging=False,
            opponent_name="random",
        )

    @staticmethod
    def comprehensive_evaluation_config():
        """Thorough evaluation for integration tests."""
        return create_evaluation_config(
            strategy=EvaluationStrategy.TOURNAMENT,
            num_games=20,
            wandb_logging=False,
            # Note: enable_analytics is not a valid field, using strategy_params instead
            strategy_params={"enable_analytics": True},
        )

    @staticmethod
    def performance_evaluation_config():
        """Configuration optimized for performance testing."""
        return create_evaluation_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=50,
            wandb_logging=False,
            enable_in_memory_evaluation=True,
            opponent_name="random",
            strategy_params={"max_moves_per_game": 100},
        )
