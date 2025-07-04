"""Unit tests for StepManager class."""

# pylint: disable=unused-import,unused-argument,protected-access
# flake8: noqa: S1244,S6711,S125

from dataclasses import (  # pylint: disable=unused-import,unused-argument,protected-access
    dataclass,
)
from typing import Any, Dict, Optional  # pylint: disable=unused-import
from unittest.mock import MagicMock, Mock, patch  # pylint: disable=unused-import

import numpy as np
import pytest
import torch

rng = np.random.default_rng(42)  # seeded RNG

# Use numpy.random.Generator for random numbers

from keisei.config_schema import AppConfig
from keisei.shogi.shogi_core_definitions import Color, PieceType
from keisei.training.step_manager import EpisodeState, StepManager, StepResult


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=AppConfig)
    config.env = Mock()
    config.env.device = "cpu"
    config.display = Mock()
    config.display.display_moves = False
    config.display.turn_tick = 0.0
    return config


@pytest.fixture
def mock_components():
    """Create mock components for StepManager."""
    game = Mock()
    agent = Mock()
    policy_mapper = Mock()
    experience_buffer = Mock()

    return {
        "game": game,
        "agent": agent,
        "policy_mapper": policy_mapper,
        "experience_buffer": experience_buffer,
    }


@pytest.fixture
def step_manager(mock_config, mock_components):
    """Create a StepManager instance with mocked dependencies."""
    return StepManager(
        config=mock_config,
        game=mock_components["game"],
        agent=mock_components["agent"],
        policy_mapper=mock_components["policy_mapper"],
        experience_buffer=mock_components["experience_buffer"],
    )


@pytest.fixture
def sample_episode_state():
    """Create a sample episode state for testing."""
    obs = rng.random((10, 10, 20), dtype=np.float32)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return EpisodeState(
        current_obs=obs,
        current_obs_tensor=obs_tensor,
        episode_reward=15.5,
        episode_length=25,
    )


@pytest.fixture
def mock_logger():
    """Create a mock logger function."""
    return Mock()


class TestEpisodeState:
    """Test the EpisodeState dataclass."""

    def test_episode_state_creation(self):
        """Test creating an EpisodeState."""
        obs = rng.random((5, 5, 10), dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        state = EpisodeState(
            current_obs=obs,
            current_obs_tensor=obs_tensor,
            episode_reward=10.0,
            episode_length=5,
        )

        assert np.array_equal(state.current_obs, obs)
        assert torch.equal(state.current_obs_tensor, obs_tensor)
        assert state.episode_reward == pytest.approx(10.0)
        assert state.episode_length == 5


class TestStepResult:
    """Test the StepResult dataclass."""

    def test_step_result_creation(self):
        """Test creating a StepResult."""
        obs = rng.random((5, 5, 10), dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        result = StepResult(
            next_obs=obs,
            next_obs_tensor=obs_tensor,
            reward=2.5,
            done=False,
            info={"test": "value"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
            success=True,
            error_message=None,
        )

        assert np.array_equal(result.next_obs, obs)
        assert torch.equal(result.next_obs_tensor, obs_tensor)
        assert result.reward == pytest.approx(2.5)
        assert result.done is False
        assert result.info == {"test": "value"}
        assert result.selected_move == (1, 2, 3, 4, 5)
        assert result.policy_index == 42
        assert result.log_prob == pytest.approx(-1.5)
        assert result.value_pred == pytest.approx(0.8)
        assert result.success is True
        assert result.error_message is None

    def test_step_result_defaults(self):
        """Test StepResult with default values."""
        obs = rng.random((5, 5, 10), dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        result = StepResult(
            next_obs=obs,
            next_obs_tensor=obs_tensor,
            reward=1.0,
            done=True,
            info={},
            selected_move=None,
            policy_index=0,
            log_prob=0.0,
            value_pred=0.0,
        )

        assert result.success is True  # Default value
        assert result.error_message is None  # Default value


class TestStepManagerInitialization:
    """Test StepManager initialization."""

    def test_initialization(self, mock_config, mock_components):
        """Test successful initialization."""
        manager = StepManager(
            config=mock_config,
            game=mock_components["game"],
            agent=mock_components["agent"],
            policy_mapper=mock_components["policy_mapper"],
            experience_buffer=mock_components["experience_buffer"],
        )

        assert manager.config == mock_config
        assert manager.game == mock_components["game"]
        assert manager.agent == mock_components["agent"]
        assert manager.policy_mapper == mock_components["policy_mapper"]
        assert manager.experience_buffer == mock_components["experience_buffer"]
        assert isinstance(manager.device, torch.device)
        assert str(manager.device) == "cpu"


class TestExecuteStep:
    """Test the execute_step method."""

    def test_successful_step_execution(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test successful execution of a training step."""
        # Setup mocks
        legal_moves = [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)]
        mock_components["game"].get_legal_moves.return_value = legal_moves

        legal_mask = torch.ones(4096, dtype=torch.bool)
        mock_components["policy_mapper"].get_legal_mask.return_value = legal_mask

        selected_move = (1, 2, 3, 4, 5)
        policy_index = 42
        log_prob = -1.5
        value_pred = 0.8
        mock_components["agent"].select_action.return_value = (
            selected_move,
            policy_index,
            log_prob,
            value_pred,
        )

        next_obs = rng.random((10, 10, 20), dtype=np.float32)
        reward = 2.5
        done = False
        info = {"test": "info"}
        mock_components["game"].make_move.return_value = (next_obs, reward, done, info)

        # Execute step
        result = step_manager.execute_step(
            sample_episode_state, global_timestep=100, logger_func=mock_logger
        )

        # Verify results
        assert result.success is True
        assert result.error_message is None
        assert np.array_equal(result.next_obs, next_obs)
        assert result.reward == pytest.approx(reward)
        assert result.done == done
        assert result.info == info
        assert result.selected_move == selected_move
        assert result.policy_index == policy_index
        assert result.log_prob == pytest.approx(log_prob)
        assert result.value_pred == pytest.approx(value_pred)

        # Verify method calls
        mock_components["game"].get_legal_moves.assert_called_once()
        mock_components["policy_mapper"].get_legal_mask.assert_called_once_with(
            legal_moves, device=torch.device("cpu")
        )
        mock_components["agent"].select_action.assert_called_once_with(
            sample_episode_state.current_obs, legal_mask, is_training=True
        )
        mock_components["game"].make_move.assert_called_once_with(selected_move)
        mock_components["experience_buffer"].add.assert_called_once()

    def test_agent_select_action_fails(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test handling when agent fails to select an action."""
        # Setup mocks
        legal_moves = [(1, 2, 3, 4, 5)]
        mock_components["game"].get_legal_moves.return_value = legal_moves

        legal_mask = torch.ones(4096, dtype=torch.bool)
        mock_components["policy_mapper"].get_legal_mask.return_value = legal_mask

        # Agent returns None for selected move
        mock_components["agent"].select_action.return_value = (None, 0, 0.0, 0.0)

        reset_obs = rng.random((10, 10, 20), dtype=np.float32)
        mock_components["game"].reset.return_value = reset_obs

        # Execute step
        result = step_manager.execute_step(
            sample_episode_state, global_timestep=100, logger_func=mock_logger
        )

        # Verify failure result
        assert result.success is False
        assert "Agent failed to select a move" in result.error_message
        assert np.array_equal(result.next_obs, reset_obs)
        assert result.reward == pytest.approx(0.0)
        assert result.done is False
        assert result.selected_move is None

        # Verify error was logged
        mock_logger.assert_called()
        assert "CRITICAL: Agent failed to select a move" in mock_logger.call_args[0][0]

        # Verify game was reset
        mock_components["game"].reset.assert_called_once()

    def test_make_move_raises_exception(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test handling when make_move raises an exception."""
        # Setup mocks
        legal_moves = [(1, 2, 3, 4, 5)]
        mock_components["game"].get_legal_moves.return_value = legal_moves

        legal_mask = torch.ones(4096, dtype=torch.bool)
        mock_components["policy_mapper"].get_legal_mask.return_value = legal_mask

        selected_move = (1, 2, 3, 4, 5)
        mock_components["agent"].select_action.return_value = (
            selected_move,
            42,
            -1.5,
            0.8,
        )

        # make_move raises ValueError
        mock_components["game"].make_move.side_effect = ValueError("Invalid move")

        reset_obs = rng.random((10, 10, 20), dtype=np.float32)
        mock_components["game"].reset.return_value = reset_obs

        # Execute step
        result = step_manager.execute_step(
            sample_episode_state, global_timestep=100, logger_func=mock_logger
        )

        # Verify failure result
        assert result.success is False
        assert "Error during training step: Invalid move" in result.error_message
        assert np.array_equal(result.next_obs, reset_obs)

        # Verify error was logged
        mock_logger.assert_called()
        assert "CRITICAL: Error during training step" in mock_logger.call_args[0][0]

    def test_make_move_invalid_result_format(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test handling when make_move returns invalid format."""
        # Setup mocks
        legal_moves = [(1, 2, 3, 4, 5)]
        mock_components["game"].get_legal_moves.return_value = legal_moves

        legal_mask = torch.ones(4096, dtype=torch.bool)
        mock_components["policy_mapper"].get_legal_mask.return_value = legal_mask

        selected_move = (1, 2, 3, 4, 5)
        mock_components["agent"].select_action.return_value = (
            selected_move,
            42,
            -1.5,
            0.8,
        )

        # make_move returns invalid format (not 4-tuple)
        mock_components["game"].make_move.return_value = (1, 2, 3)  # Only 3 elements

        reset_obs = rng.random((10, 10, 20), dtype=np.float32)
        mock_components["game"].reset.return_value = reset_obs

        # Execute step
        result = step_manager.execute_step(
            sample_episode_state, global_timestep=100, logger_func=mock_logger
        )

        # Verify failure result
        assert result.success is False
        assert "Invalid move result" in result.error_message

    def test_reset_also_fails(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test handling when both move execution and reset fail."""
        # Setup mocks
        legal_moves = [(1, 2, 3, 4, 5)]
        mock_components["game"].get_legal_moves.return_value = legal_moves

        legal_mask = torch.ones(4096, dtype=torch.bool)
        mock_components["policy_mapper"].get_legal_mask.return_value = legal_mask

        selected_move = (1, 2, 3, 4, 5)
        mock_components["agent"].select_action.return_value = (
            selected_move,
            42,
            -1.5,
            0.8,
        )

        # Both make_move and reset fail
        mock_components["game"].make_move.side_effect = ValueError("Move failed")
        mock_components["game"].reset.side_effect = RuntimeError("Reset failed")

        # Execute step
        result = step_manager.execute_step(
            sample_episode_state, global_timestep=100, logger_func=mock_logger
        )

        # Verify failure result with original state
        assert result.success is False
        assert "Reset also failed" in result.error_message
        assert np.array_equal(result.next_obs, sample_episode_state.current_obs)
        assert result.done is True  # Force episode end

    def test_demo_mode_enabled(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test step execution with demo mode enabled."""
        # Enable demo mode
        step_manager.config.display.display_moves = True
        step_manager.config.display.turn_tick = 0.1

        # Setup mocks
        legal_moves = [(1, 2, 3, 4, 5)]
        mock_components["game"].get_legal_moves.return_value = legal_moves
        mock_components["game"].get_piece.return_value = "test_piece"
        mock_components["game"].current_player = Mock()
        mock_components["game"].current_player.name = "TestPlayer"

        legal_mask = torch.ones(4096, dtype=torch.bool)
        mock_components["policy_mapper"].get_legal_mask.return_value = legal_mask

        selected_move = (1, 2, 3, 4, 5)
        mock_components["agent"].select_action.return_value = (
            selected_move,
            42,
            -1.5,
            0.8,
        )

        next_obs = rng.random((10, 10, 20), dtype=np.float32)
        mock_components["game"].make_move.return_value = (next_obs, 1.0, False, {})

        # Mock the format function
        with patch(
            "keisei.training.step_manager.format_move_with_description_enhanced"
        ) as mock_format:
            mock_format.return_value = "formatted_move"

            with patch("time.sleep") as mock_sleep:
                # Execute step
                result = step_manager.execute_step(
                    sample_episode_state, global_timestep=100, logger_func=mock_logger
                )

                # Verify demo mode was handled
                assert result.success is True
                mock_format.assert_called_once()
                mock_sleep.assert_called_once_with(0.1)

                # Demo moves should not be logged to logger
                mock_logger.assert_not_called()

                # Move should be recorded internally for display
                expected_prefix = (
                    f"Move {sample_episode_state.episode_length + 1} (TestPlayer):"
                )
                assert step_manager.move_log[-1].startswith(expected_prefix)


class TestHandleEpisodeEnd:
    """Test the handle_episode_end method."""

    def test_successful_episode_end(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test successful episode end handling."""
        # Create step result with game outcome
        step_result = StepResult(
            next_obs=rng.random((10, 10, 20), dtype=np.float32),
            next_obs_tensor=torch.randn(1, 10, 10, 20),
            reward=5.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        game_stats = {"black_wins": 10, "white_wins": 5, "draws": 2}

        reset_obs = rng.random((10, 10, 20), dtype=np.float32)
        mock_components["game"].reset.return_value = reset_obs

        # Execute episode end handling
        new_state, _ = step_manager.handle_episode_end(
            sample_episode_state, step_result, game_stats, 17, mock_logger
        )

        # Verify new episode state
        assert np.array_equal(new_state.current_obs, reset_obs)
        assert new_state.episode_reward == pytest.approx(0.0)
        assert new_state.episode_length == 0

        # Verify logging was called
        mock_logger.assert_called()
        log_message = mock_logger.call_args[0][0]
        assert "Episode 18 finished" in log_message
        assert "Sente wins by checkmate" in log_message

        # Verify wandb data was logged
        wandb_data = mock_logger.call_args[1]["wandb_data"]
        assert wandb_data["episode_reward"] == sample_episode_state.episode_reward
        assert wandb_data["episode_length"] == sample_episode_state.episode_length
        assert wandb_data["game_outcome"] == "black"
        assert wandb_data["game_reason"] == "checkmate"

    def test_episode_end_white_wins(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test episode end with white victory."""
        step_result = StepResult(
            next_obs=rng.random((10, 10, 20), dtype=np.float32),
            next_obs_tensor=torch.randn(1, 10, 10, 20),
            reward=0.0,
            done=True,
            info={"winner": "white", "reason": "timeout"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        game_stats = {"black_wins": 3, "white_wins": 7, "draws": 0}
        mock_components["game"].reset.return_value = rng.random(
            (10, 10, 20), dtype=np.float32
        )

        # Execute episode end handling
        step_manager.handle_episode_end(
            sample_episode_state, step_result, game_stats, 9, mock_logger
        )

        # Verify white win message
        log_message = mock_logger.call_args[0][0]
        assert "Gote wins by timeout" in log_message

    def test_episode_end_draw(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test episode end with draw."""
        step_result = StepResult(
            next_obs=rng.random((10, 10, 20), dtype=np.float32),
            next_obs_tensor=torch.randn(1, 10, 10, 20),
            reward=0.0,
            done=True,
            info={"winner": None, "reason": "stalemate"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        game_stats = {"black_wins": 2, "white_wins": 2, "draws": 6}
        mock_components["game"].reset.return_value = rng.random(
            (10, 10, 20), dtype=np.float32
        )

        # Execute episode end handling
        step_manager.handle_episode_end(
            sample_episode_state, step_result, game_stats, 9, mock_logger
        )

        # Verify draw message
        log_message = mock_logger.call_args[0][0]
        assert "Draw by stalemate" in log_message

    def test_episode_end_win_rate_calculation(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test win rate calculations in episode end."""
        step_result = StepResult(
            next_obs=rng.random((10, 10, 20), dtype=np.float32),
            next_obs_tensor=torch.randn(1, 10, 10, 20),
            reward=0.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        # Initial stats: 20 black wins, 30 white wins, 50 draws = 100 total
        game_stats = {"black_wins": 20, "white_wins": 30, "draws": 50}
        mock_components["game"].reset.return_value = rng.random(
            (10, 10, 20), dtype=np.float32
        )

        # Execute episode end handling
        step_manager.handle_episode_end(
            sample_episode_state, step_result, game_stats, 99, mock_logger
        )

        # Verify win rates in wandb data (calculated AFTER current game is added)
        # Black wins, so black_wins becomes 21, total games becomes 101
        wandb_data = mock_logger.call_args[1]["wandb_data"]
        assert wandb_data["black_win_rate"] == pytest.approx(21 / 101)
        assert wandb_data["white_win_rate"] == pytest.approx(30 / 101)
        assert wandb_data["draw_rate"] == pytest.approx(50 / 101)
        assert wandb_data["black_wins_total"] == 21
        assert wandb_data["white_wins_total"] == 30
        assert wandb_data["draws_total"] == 50

    def test_episode_end_zero_games(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test episode end with zero total games."""
        step_result = StepResult(
            next_obs=rng.random((10, 10, 20), dtype=np.float32),
            next_obs_tensor=torch.randn(1, 10, 10, 20),
            reward=0.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
        mock_components["game"].reset.return_value = rng.random(
            (10, 10, 20), dtype=np.float32
        )

        # Execute episode end handling
        step_manager.handle_episode_end(
            sample_episode_state, step_result, game_stats, 0, mock_logger
        )

        # Verify win rates are calculated after this game (black wins)
        # black_wins becomes 1, total games becomes 1
        wandb_data = mock_logger.call_args[1]["wandb_data"]
        assert wandb_data["black_win_rate"] == pytest.approx(1.0)  # 1/1
        assert wandb_data["white_win_rate"] == pytest.approx(0.0)  # 0/1
        assert wandb_data["draw_rate"] == pytest.approx(0.0)  # 0/1
        assert wandb_data["black_wins_total"] == 1
        assert wandb_data["white_wins_total"] == 0
        assert wandb_data["draws_total"] == 0

    def test_episode_end_reset_fails(
        self, step_manager, sample_episode_state, mock_logger, mock_components
    ):
        """Test episode end when game reset fails."""
        step_result = StepResult(
            next_obs=rng.random((10, 10, 20), dtype=np.float32),
            next_obs_tensor=torch.randn(1, 10, 10, 20),
            reward=0.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        game_stats = {"black_wins": 1, "white_wins": 0, "draws": 0}

        # Game reset fails
        mock_components["game"].reset.side_effect = RuntimeError("Reset failed")

        # Execute episode end handling
        returned_state, _ = step_manager.handle_episode_end(
            sample_episode_state, step_result, game_stats, 0, mock_logger
        )

        # Should return original episode state when reset fails
        assert returned_state == sample_episode_state

        # Should log error
        error_logged = False
        for call in mock_logger.call_args_list:
            if "CRITICAL: Game reset failed" in call[0][0]:
                error_logged = True
                break
        assert error_logged


class TestResetEpisode:
    """Test the reset_episode method."""

    def test_successful_reset(self, step_manager, mock_components):
        """Test successful episode reset."""
        reset_obs = rng.random((10, 10, 20), dtype=np.float32)
        mock_components["game"].reset.return_value = reset_obs

        new_state = step_manager.reset_episode()

        assert np.array_equal(new_state.current_obs, reset_obs)
        assert new_state.episode_reward == pytest.approx(0.0)
        assert new_state.episode_length == 0
        assert new_state.current_obs_tensor.shape == (1, 10, 10, 20)


class TestUpdateEpisodeState:
    """Test the update_episode_state method."""

    def test_update_episode_state(self, step_manager, sample_episode_state):
        """Test updating episode state with step result."""
        next_obs = rng.random((10, 10, 20), dtype=np.float32)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        step_result = StepResult(
            next_obs=next_obs,
            next_obs_tensor=next_obs_tensor,
            reward=2.5,
            done=False,
            info={},
            selected_move=(1, 2, 3, 4, 5),
            policy_index=42,
            log_prob=-1.5,
            value_pred=0.8,
        )

        updated_state = step_manager.update_episode_state(
            sample_episode_state, step_result
        )

        assert np.array_equal(updated_state.current_obs, next_obs)
        assert torch.equal(updated_state.current_obs_tensor, next_obs_tensor)
        assert updated_state.episode_reward == pytest.approx(
            sample_episode_state.episode_reward + 2.5
        )
        assert updated_state.episode_length == sample_episode_state.episode_length + 1


class TestPrepareAndHandleDemoMode:
    """Test demo mode helper methods."""

    def test_prepare_demo_info_success(self, step_manager, mock_components):
        """Test successful demo info preparation."""
        legal_moves = [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)]
        mock_components["game"].get_piece.return_value = "test_piece"

        piece_info = step_manager._prepare_demo_info(legal_moves)

        assert piece_info == "test_piece"
        mock_components["game"].get_piece.assert_called_once_with(1, 2)

    @pytest.mark.parametrize(
        "legal_moves,expected_result,test_description",
        [
            ([], None, "empty moves"),
            ([None], None, "none move"),
            ([(1, 2, 3)], None, "invalid move format (too short)"),
            ([(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)], "test_piece", "valid moves"),
        ],
        ids=["empty", "none", "invalid_format", "valid"],
    )
    def test_prepare_demo_info_scenarios(
        self,
        step_manager,
        mock_components,
        legal_moves,
        expected_result,
        test_description,
    ):
        """Test demo info preparation with various move scenarios."""
        if expected_result == "test_piece":
            mock_components["game"].get_piece.return_value = "test_piece"

        piece_info = step_manager._prepare_demo_info(legal_moves)

        assert piece_info == expected_result

        # Only check get_piece call for valid moves
        if legal_moves and legal_moves[0] is not None and len(legal_moves[0]) >= 5:
            mock_components["game"].get_piece.assert_called_once_with(1, 2)
        else:
            mock_components["game"].get_piece.assert_not_called()

    def test_prepare_demo_info_exception_handling(self, step_manager, mock_components):
        """Test demo info preparation when get_piece fails."""
        legal_moves = [(1, 2, 3, 4, 5)]
        mock_components["game"].get_piece.side_effect = AttributeError("No such method")

        piece_info = step_manager._prepare_demo_info(legal_moves)
        assert piece_info is None

    def test_handle_demo_mode(self, step_manager, mock_logger, mock_components):
        """Test demo mode handling."""
        selected_move = (1, 2, 3, 4, 5)
        episode_length = 10
        piece_info = "test_piece"

        # Setup current player
        mock_components["game"].current_player = Mock()
        mock_components["game"].current_player.name = "TestPlayer"

        with patch(
            "keisei.training.step_manager.format_move_with_description_enhanced"
        ) as mock_format:
            mock_format.return_value = "formatted_move"

            with patch("time.sleep") as mock_sleep:
                step_manager.config.display.turn_tick = 0.5

                step_manager._handle_demo_mode(
                    selected_move, episode_length, piece_info, mock_logger
                )

                mock_format.assert_called_once_with(
                    selected_move, step_manager.policy_mapper, piece_info
                )

                mock_sleep.assert_called_once_with(0.5)

                mock_logger.assert_not_called()
                assert step_manager.move_history[-1] == selected_move
                assert step_manager.move_log[-1].startswith("Move 11 (TestPlayer):")

    def test_handle_demo_mode_no_current_player(
        self, step_manager, mock_logger, mock_components
    ):
        """Test demo mode handling when current_player is not available."""
        selected_move = (1, 2, 3, 4, 5)
        episode_length = 5
        piece_info = None

        # No current_player attribute
        del mock_components["game"].current_player

        with patch(
            "keisei.training.step_manager.format_move_with_description_enhanced"
        ) as mock_format:
            mock_format.return_value = "formatted_move"

            step_manager._handle_demo_mode(
                selected_move, episode_length, piece_info, mock_logger
            )

            mock_logger.assert_not_called()
            assert "Unknown" in step_manager.move_log[-1]

    def test_handle_demo_mode_no_delay(
        self, step_manager, mock_logger, mock_components
    ):
        """Test demo mode handling with no delay."""
        selected_move = (1, 2, 3, 4, 5)
        episode_length = 0
        piece_info = None

        mock_components["game"].current_player = Mock()
        mock_components["game"].current_player.name = "Player1"

        with patch(
            "keisei.training.step_manager.format_move_with_description_enhanced"
        ) as mock_format:
            mock_format.return_value = "formatted_move"

            with patch("time.sleep") as mock_sleep:
                step_manager.config.display.turn_tick = 0.0

                step_manager._handle_demo_mode(
                    selected_move, episode_length, piece_info, mock_logger
                )

                # Sleep should not be called
                mock_sleep.assert_not_called()


class TestExecuteStepNoLegalMoves:
    """Test StepManager behavior when no legal moves are available."""

    @pytest.fixture
    def step_manager(self, mock_config, mock_components):
        """Create a StepManager instance for testing."""
        return StepManager(
            config=mock_config,
            game=mock_components["game"],
            agent=mock_components["agent"],
            policy_mapper=mock_components["policy_mapper"],
            experience_buffer=mock_components["experience_buffer"],
        )

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()

    def test_execute_step_no_legal_moves_terminal_condition(
        self, step_manager, mock_components, mock_logger
    ):
        """Test that execute_step handles no legal moves as terminal condition."""
        # Arrange
        mock_components["game"].get_legal_moves.return_value = []  # No legal moves
        mock_board_repr = np.zeros((9, 9, 14))
        mock_components["game"].get_board_representation.return_value = mock_board_repr
        mock_components["game"].get_game_state.return_value = Mock()

        # Mock reset method and return value
        mock_reset_obs = np.zeros((9, 9, 14))
        mock_components["game"].reset = Mock(return_value=mock_reset_obs)

        # Create episode state
        mock_obs = np.zeros((9, 9, 14))
        mock_obs_tensor = torch.zeros((1, 14, 9, 9))
        episode_state = EpisodeState(
            current_obs=mock_obs,
            current_obs_tensor=mock_obs_tensor,
            episode_reward=0.0,
            episode_length=0,
        )

        # Act
        result = step_manager.execute_step(episode_state, 100, mock_logger)

        # Assert
        assert isinstance(result, StepResult)
        assert result.done is True
        assert result.info["terminal_reason"] == "no_legal_moves"
        assert result.next_obs is not None
        assert abs(result.reward - 0.0) < 1e-6  # Use epsilon comparison for floats

        # Verify game was reset
        mock_components["game"].reset.assert_called_once()

        # Verify agent.select_action was NOT called (upstream handling)
        mock_components["agent"].select_action.assert_not_called()

        # Verify appropriate logging
        mock_logger.assert_called()
        log_calls = [call[0][0] for call in mock_logger.call_args_list]
        terminal_logs = [
            log
            for log in log_calls
            if "no legal moves" in log.lower() or "terminal" in log.lower()
        ]
        assert len(terminal_logs) > 0, "Expected terminal log for no legal moves"

    def test_execute_step_no_legal_moves_with_episode_state(
        self, step_manager, mock_components, mock_logger
    ):
        """Test no legal moves handling with existing episode state."""
        # Arrange
        mock_obs = np.zeros((9, 9, 14))
        mock_obs_tensor = torch.zeros((1, 14, 9, 9))
        episode_state = EpisodeState(
            current_obs=mock_obs,
            current_obs_tensor=mock_obs_tensor,
            episode_reward=10.0,
            episode_length=5,
        )

        mock_components["game"].get_legal_moves.return_value = []  # No legal moves
        mock_components["game"].get_board_representation.return_value = mock_obs
        mock_components["game"].get_game_state.return_value = Mock()

        # Mock reset method and return value
        mock_reset_obs = np.zeros((9, 9, 14))
        mock_components["game"].reset = Mock(return_value=mock_reset_obs)

        # Act
        result = step_manager.execute_step(episode_state, 50, mock_logger)

        # Assert
        assert result.done is True
        assert result.info["terminal_reason"] == "no_legal_moves"

        # Verify observation data is present in result
        assert result.next_obs is not None

        # Verify game reset
        mock_components["game"].reset.assert_called_once()

    def test_execute_step_normal_flow_with_legal_moves(
        self, step_manager, mock_components, mock_logger
    ):
        """Test that normal flow continues when legal moves are available."""
        # Arrange
        legal_moves = [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)]  # Mock legal moves
        mock_obs = np.zeros((9, 9, 14))
        mock_components["game"].get_legal_moves.return_value = legal_moves
        mock_components["game"].get_board_representation.return_value = mock_obs
        mock_components["game"].get_game_state.return_value = Mock()

        # Mock policy mapper
        mock_legal_mask = torch.tensor([True, False, True])
        mock_components["policy_mapper"].get_legal_mask.return_value = mock_legal_mask

        # Mock agent selection - should return 4 values
        mock_selected_move = legal_moves[0]
        mock_policy_index = 0
        mock_log_prob = 0.5
        mock_value = 0.3
        mock_components["agent"].select_action.return_value = (
            mock_selected_move,
            mock_policy_index,
            mock_log_prob,
            mock_value,
        )

        # Mock game execution - should return 4-tuple (next_obs, reward, done, info)
        mock_next_obs = np.zeros((9, 9, 14))
        mock_reward = 0.5
        mock_done = False
        mock_info = {}
        mock_components["game"].make_move.return_value = (
            mock_next_obs,
            mock_reward,
            mock_done,
            mock_info,
        )

        # Create episode state
        mock_obs_tensor = torch.zeros((1, 14, 9, 9))
        episode_state = EpisodeState(
            current_obs=mock_obs,
            current_obs_tensor=mock_obs_tensor,
            episode_reward=0.0,
            episode_length=0,
        )

        # Act
        result = step_manager.execute_step(episode_state, 10, mock_logger)

        # Assert - should proceed normally, not terminate early
        assert result.done is False  # Not terminal from no legal moves

        # Verify normal flow was followed
        mock_components["game"].get_legal_moves.assert_called_once()
        mock_components["policy_mapper"].get_legal_mask.assert_called_once_with(
            legal_moves, device=step_manager.device
        )
        mock_components["agent"].select_action.assert_called_once()
        mock_components["game"].make_move.assert_called_once()

    def test_execute_step_no_legal_moves_logs_appropriate_level(
        self, step_manager, mock_components, mock_logger
    ):
        """Test that no legal moves condition logs at appropriate level."""
        # Arrange
        mock_obs = np.zeros((9, 9, 14))
        mock_components["game"].get_legal_moves.return_value = []
        mock_components["game"].get_board_representation.return_value = mock_obs
        mock_components["game"].get_game_state.return_value = Mock()

        # Mock reset method and return value
        mock_reset_obs = np.zeros((9, 9, 14))
        mock_components["game"].reset = Mock(return_value=mock_reset_obs)

        # Create episode state
        mock_obs_tensor = torch.zeros((1, 14, 9, 9))
        episode_state = EpisodeState(
            current_obs=mock_obs,
            current_obs_tensor=mock_obs_tensor,
            episode_reward=0.0,
            episode_length=0,
        )

        # Act
        step_manager.execute_step(episode_state, 75, mock_logger)

        # Assert - verify logging was called
        mock_logger.assert_called()

        # Check that logging mentions the terminal condition
        log_calls = mock_logger.call_args_list
        assert len(log_calls) > 0, "Expected at least one log call"

        # Look for terminal-related log messages
        terminal_related_logs = []
        for call in log_calls:
            if len(call[0]) > 0:  # Check if there are positional arguments
                log_message = str(call[0][0]).lower()
                if "no legal moves" in log_message or "terminal" in log_message:
                    terminal_related_logs.append(call)

        assert len(terminal_related_logs) > 0, "Expected terminal-related log message"


class TestBestCaptureTracking:
    """Tests for best capture metrics."""

    def _setup_step(self, step_manager, mock_components, color, captured_type):
        step_manager.game.get_legal_moves.return_value = [(1, 2, 3, 4, False)]
        step_manager.policy_mapper.get_legal_mask.return_value = torch.ones(1)
        step_manager.agent.select_action.return_value = (
            (1, 2, 3, 4, False),
            0,
            0.0,
            0.0,
        )
        next_obs = np.zeros((1,))
        info = {"captured_piece_type": captured_type}
        step_manager.game.current_player = color
        step_manager.game.make_move.return_value = (next_obs, 0.0, False, info)

        dummy_state = EpisodeState(
            current_obs=np.zeros((1,)),
            current_obs_tensor=torch.zeros((1, 1)),
            episode_reward=0.0,
            episode_length=0,
        )
        step_manager.execute_step(dummy_state, 0, Mock())

    def test_best_capture_per_player(self, step_manager, mock_components):
        """Sente and Gote should track captures independently."""
        self._setup_step(step_manager, mock_components, Color.BLACK, "ROOK")

        assert step_manager.sente_best_capture == "Rook"
        assert step_manager.gote_best_capture is None

    def test_move_logs_reset_on_episode_end(
        self, step_manager, sample_episode_state, mock_components, mock_logger
    ):
        """Move history and logs should reset after episode end."""
        step_manager.move_log = ["dummy"]
        step_manager.move_history = [(1, 2, 3, 4, False)]

        step_result = StepResult(
            next_obs=np.zeros((1,)),
            next_obs_tensor=torch.zeros((1, 1)),
            reward=0.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, False),
            policy_index=0,
            log_prob=0.0,
            value_pred=0.0,
        )

        mock_components["game"].reset.return_value = np.zeros((1,))

        step_manager.handle_episode_end(
            sample_episode_state,
            step_result,
            {"black_wins": 0, "white_wins": 0, "draws": 0},
            0,
            mock_logger,
        )

        assert step_manager.move_log == []
        assert step_manager.move_history == []

    def test_best_capture_reset_on_episode_end(
        self, step_manager, sample_episode_state, mock_components, mock_logger
    ):
        """Best capture values should reset after episode end."""
        step_manager.sente_best_capture = "Rook"
        step_manager.gote_best_capture = "Bishop"

        step_result = StepResult(
            next_obs=np.zeros((1,)),
            next_obs_tensor=torch.zeros((1, 1)),
            reward=0.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, False),
            policy_index=0,
            log_prob=0.0,
            value_pred=0.0,
        )

        mock_components["game"].reset.return_value = np.zeros((1,))

        step_manager.handle_episode_end(
            sample_episode_state,
            step_result,
            {"black_wins": 0, "white_wins": 0, "draws": 0},
            0,
            mock_logger,
        )

        assert step_manager.sente_best_capture is None
        assert step_manager.gote_best_capture is None


class TestActivityCounters:
    """Ensure capture, drop, and promotion counts latch and reset."""

    def _exec_move(self, step_manager, mock_components, color, move, info=None):
        mock_components["game"].get_legal_moves.return_value = [move]
        mock_components["policy_mapper"].get_legal_mask.return_value = torch.ones(1)
        mock_components["agent"].select_action.return_value = (move, 0, 0.0, 0.0)
        mock_components["game"].current_player = color
        mock_components["game"].make_move.return_value = (
            np.zeros((1,)),
            0.0,
            False,
            info or {},
        )
        dummy_state = EpisodeState(
            current_obs=np.zeros((1,)),
            current_obs_tensor=torch.zeros((1, 1)),
            episode_reward=0.0,
            episode_length=0,
        )
        step_manager.execute_step(dummy_state, 0, Mock())

    def test_counts_latch_and_reset(
        self, step_manager, mock_components, mock_logger, sample_episode_state
    ):
        self._exec_move(
            step_manager,
            mock_components,
            Color.BLACK,
            (1, 2, 3, 4, False),
            {"captured_piece_type": "PAWN"},
        )
        assert step_manager.sente_capture_count == 1

        self._exec_move(
            step_manager,
            mock_components,
            Color.BLACK,
            (None, None, 4, 4, PieceType.PAWN),
        )
        assert step_manager.sente_capture_count == 1
        assert step_manager.sente_drop_count == 1

        self._exec_move(
            step_manager,
            mock_components,
            Color.BLACK,
            (1, 2, 3, 4, True),
        )
        assert step_manager.sente_promo_count == 1

        step_result = StepResult(
            next_obs=np.zeros((1,)),
            next_obs_tensor=torch.zeros((1, 1)),
            reward=0.0,
            done=True,
            info={"winner": "black", "reason": "checkmate"},
            selected_move=(1, 2, 3, 4, False),
            policy_index=0,
            log_prob=0.0,
            value_pred=0.0,
        )

        mock_components["game"].reset.return_value = np.zeros((1,))

        step_manager.handle_episode_end(
            sample_episode_state,
            step_result,
            {"black_wins": 0, "white_wins": 0, "draws": 0},
            0,
            mock_logger,
        )

        assert step_manager.sente_capture_count == 0
        assert step_manager.sente_drop_count == 0
        assert step_manager.sente_promo_count == 0


if __name__ == "__main__":
    pytest.main([__file__])
