"""
Unit and integration tests for the evaluate.py script.
"""

import sys
import random
from unittest.mock import patch, MagicMock, call
from typing import Optional, Tuple # Ensure Tuple and Optional are imported

import pytest
import torch # Re-add torch import
import numpy as np

# Imports from the project
from keisei.ppo_agent import PPOAgent # Actual PPOAgent for type hints and structure
from keisei.utils import PolicyOutputMapper, EvaluationLogger, BaseOpponent
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import MoveTuple

# Functions and classes to test from evaluate.py
from evaluate import (
    SimpleRandomOpponent,
    SimpleHeuristicOpponent,
    load_evaluation_agent,
    initialize_opponent,
    run_evaluation_loop,
    main as evaluate_main, # Rename to avoid conflict with pytest main
    INPUT_CHANNELS
)

# A mock PPOAgent for testing purposes
# Inherit from PPOAgent to satisfy type hints for run_evaluation_loop, and BaseOpponent for other uses.
class MockPPOAgent(PPOAgent, BaseOpponent): # Inherit from PPOAgent and BaseOpponent
    def __init__(self, input_channels, policy_output_mapper, device_str, name="MockPPOAgentForTest"):
        # Initialize PPOAgent part
        PPOAgent.__init__(self, input_channels=input_channels, policy_output_mapper=policy_output_mapper, device=device_str)
        # Initialize BaseOpponent part
        BaseOpponent.__init__(self, name=name) # Pass the name to BaseOpponent

        self.model = MagicMock()  # Mock the underlying torch model after PPOAgent init
        self._is_ppo_agent_mock = True # Flag to identify this mock
        self.name = name # Revert to direct assignment for name

    def load_model(self, file_path: str) -> dict: # Parameter name changed to file_path, return type to dict
        # print(f"MockPPOAgent: Pretending to load model from {file_path}")
        return {} # Return an empty dict as per PPOAgent

    def select_action(self, obs: np.ndarray, legal_shogi_moves: list[MoveTuple], legal_mask: torch.Tensor, is_training=False) -> Tuple[Optional[MoveTuple], int, float, float]: # Matched all parameter names and types with PPOAgent
        if not legal_shogi_moves:
            # This case should be handled by the game loop checking game_over or no legal_moves
            raise ValueError("MockPPOAgent.select_action: No legal moves provided.")
        selected_move: Optional[MoveTuple] = random.choice(legal_shogi_moves)
        # PPOAgent.select_action returns: (move_tuple, policy_idx, log_prob, value)
        return selected_move, 0, 0.0, 0.0

    def get_value(self, obs_np: np.ndarray) -> float: # obs_np type changed to np.ndarray
        """Mocked get_value method."""
        return 0.0 # Return a dummy float value

    # If used as a BaseOpponent directly (e.g. PPO vs PPO where one is simplified)
    def select_move(self, game_instance: ShogiGame) -> MoveTuple: # Return type changed back to MoveTuple
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError("MockPPOAgent.select_move: No legal moves available.")
        # Simplified for BaseOpponent interface, actual PPO logic is in select_action
        obs_np = MagicMock(spec=np.ndarray) # Dummy observation, spec for type hint if needed
        legal_mask_tensor = MagicMock(spec=torch.Tensor) # Dummy mask, spec for type hint
        action_result = self.select_action(obs_np, legal_moves, legal_mask_tensor, is_training=False)
        selected_move = action_result[0]
        if selected_move is None:
            # This should ideally not happen if legal_moves is not empty.
            # Handle cases where select_action might return None for the move.
            raise ValueError("MockPPOAgent.select_move: select_action returned None for a move despite legal moves being available.")
        return selected_move


@pytest.fixture
def policy_mapper():
    return PolicyOutputMapper()

@pytest.fixture
def eval_logger_setup(tmp_path):
    log_file = tmp_path / "test_eval.log"
    logger = EvaluationLogger(str(log_file), also_stdout=False)
    with logger: # Ensure logger is used as a context manager
        yield logger, str(log_file)
    # logger.close() is handled by the context manager's __exit__

@pytest.fixture
def shogi_game_initial():
    return ShogiGame()

# --- Tests for Opponent Classes ---

def test_simple_random_opponent_select_move(shogi_game_initial: ShogiGame):
    opponent = SimpleRandomOpponent()
    legal_moves = shogi_game_initial.get_legal_moves()
    selected_move = opponent.select_move(shogi_game_initial)
    assert selected_move in legal_moves

def test_simple_heuristic_opponent_select_move(shogi_game_initial: ShogiGame):
    opponent = SimpleHeuristicOpponent()
    legal_moves = shogi_game_initial.get_legal_moves()
    selected_move = opponent.select_move(shogi_game_initial)
    assert selected_move in legal_moves
    # Add more specific tests for heuristics later if needed

# --- Tests for Initialization Functions ---

def test_initialize_opponent_types(policy_mapper):
    opponent_random = initialize_opponent("random", None, "cpu", policy_mapper, INPUT_CHANNELS)
    assert isinstance(opponent_random, SimpleRandomOpponent)

    opponent_heuristic = initialize_opponent("heuristic", None, "cpu", policy_mapper, INPUT_CHANNELS)
    assert isinstance(opponent_heuristic, SimpleHeuristicOpponent)

    with pytest.raises(ValueError, match="Unknown opponent type"):
        initialize_opponent("unknown", None, "cpu", policy_mapper, INPUT_CHANNELS)

@patch('evaluate.load_evaluation_agent') # Mock load_evaluation_agent within evaluate.py
def test_initialize_opponent_ppo(mock_load_agent, policy_mapper):
    mock_ppo_instance = MockPPOAgent(INPUT_CHANNELS, policy_mapper, "cpu", name="MockPPOAgentForTest")
    mock_load_agent.return_value = mock_ppo_instance

    opponent_ppo = initialize_opponent("ppo", "dummy_path.pth", "cpu", policy_mapper, INPUT_CHANNELS)
    mock_load_agent.assert_called_once_with("dummy_path.pth", "cpu", policy_mapper, INPUT_CHANNELS)
    assert opponent_ppo == mock_ppo_instance

    with pytest.raises(ValueError, match="Opponent path must be provided for PPO opponent type."):
        initialize_opponent("ppo", None, "cpu", policy_mapper, INPUT_CHANNELS)


@patch('evaluate.PPOAgent') # Mock PPOAgent class within evaluate.py
def test_load_evaluation_agent_mocked(MockPPOAgentClass, policy_mapper):
    mock_agent_instance = MagicMock(spec=PPOAgent) # Create a MagicMock that mimics PPOAgent
    mock_agent_instance.model = MagicMock() # Ensure the instance has a .model attribute that is also a mock
    MockPPOAgentClass.return_value = mock_agent_instance

    agent = load_evaluation_agent("dummy_checkpoint.pth", "cpu", policy_mapper, INPUT_CHANNELS)

    MockPPOAgentClass.assert_called_once_with(
        input_channels=INPUT_CHANNELS,
        policy_output_mapper=policy_mapper,
        device="cpu"
    )
    mock_agent_instance.load_model.assert_called_once_with("dummy_checkpoint.pth")
    mock_agent_instance.model.eval.assert_called_once()
    assert agent == mock_agent_instance

# --- Test for Core Evaluation Loop ---

def test_run_evaluation_loop_basic(policy_mapper, eval_logger_setup):
    logger, log_file_path = eval_logger_setup

    agent_to_eval = MockPPOAgent(INPUT_CHANNELS, policy_mapper, "cpu", name="AgentToEval")
    opponent = SimpleRandomOpponent(name="TestRandomOpponent")

    num_games = 2
    max_moves = 5 # Keep games short for testing

    results = run_evaluation_loop(
        agent_to_eval, opponent, num_games, logger, policy_mapper, max_moves, "cpu"
    )

    assert results["num_games"] == num_games
    assert "wins" in results
    assert "losses" in results
    assert "draws" in results
    assert results["wins"] + results["losses"] + results["draws"] == num_games
    assert "avg_game_length" in results
    assert results["avg_game_length"] <= max_moves # Can be less if game ends early

    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    assert f"Starting evaluation: {agent_to_eval.name} vs {opponent.name}" in log_content
    assert f"Starting Game 1/{num_games}" in log_content
    assert f"Starting Game 2/{num_games}" in log_content
    assert "Game 1 ended" in log_content
    assert "Game 2 ended" in log_content
    assert "Evaluation finished. Results:" in log_content
    assert "Agent to eval is Black" in log_content # Game 1
    assert "Agent to eval is White" in log_content # Game 2

# --- Test for Main Script Execution ---

@patch('evaluate.PolicyOutputMapper') # Patch the class in the evaluate module
@patch('evaluate.load_evaluation_agent')
@patch('evaluate.initialize_opponent')
@patch('evaluate.run_evaluation_loop')
@patch('evaluate.EvaluationLogger') # Mock the logger class itself
def test_evaluate_main_flow(mock_eval_logger_class, mock_run_loop, 
                            mock_init_opponent, mock_load_agent, MockPolicyOutputMapperClass, tmp_path, monkeypatch): # pylint: disable=too-many-positional-args,too-many-locals
    # Setup mocks
    # Create an instance of the original PolicyOutputMapper to be returned by the mock
    # This ensures that the instance used in the test assertions is the same one used by the code under test.
    actual_policy_mapper_instance = PolicyOutputMapper()
    MockPolicyOutputMapperClass.return_value = actual_policy_mapper_instance

    mock_agent_instance = MockPPOAgent(INPUT_CHANNELS, actual_policy_mapper_instance, "cpu", name="LoadedMainAgent")
    mock_opponent_instance = SimpleRandomOpponent(name="MainRandomOpponent")
    mock_logger_instance = MagicMock(spec=EvaluationLogger)

    mock_load_agent.return_value = mock_agent_instance
    mock_init_opponent.return_value = mock_opponent_instance
    mock_run_loop.return_value = {
        "wins": 1, "losses": 0, "draws": 1, "num_games": 2,
        "win_rate": 0.5, "loss_rate": 0.0, "draw_rate": 0.5,
        "avg_game_length": 10
    }
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    # Simulate command line arguments
    log_file = tmp_path / "main_eval.log"
    test_args = [
        "evaluate.py",
        "--agent-checkpoint", "agent.pth",
        "--opponent-type", "random",
        "--num-games", "2",
        "--device", "cpu",
        "--max-moves-per-game", "50",
        "--log-file", str(log_file)
    ]
    monkeypatch.setattr(sys, 'argv', test_args)

    evaluate_main()

    # Assertions
    MockPolicyOutputMapperClass.assert_called_once() # Ensure it was instantiated
    mock_load_agent.assert_called_once_with("agent.pth", "cpu", actual_policy_mapper_instance, INPUT_CHANNELS)
    mock_init_opponent.assert_called_once_with("random", None, "cpu", actual_policy_mapper_instance, INPUT_CHANNELS)
    mock_run_loop.assert_called_once_with(
        mock_agent_instance, mock_opponent_instance, 2, mock_logger_instance, actual_policy_mapper_instance, 50, "cpu"
    )

    mock_eval_logger_class.assert_called_once_with(log_file_path=str(log_file), also_stdout=True)
    # Using f-string for the log message to avoid issues with str(log_file) representation
    expected_log_message_part = f"'agent_checkpoint': 'agent.pth', 'opponent_type': 'random', 'opponent_checkpoint': None, 'num_games': 2, 'device': 'cpu', 'max_moves_per_game': 50, 'log_file': '{str(log_file)}'"
    # Check if any call to log_custom_message contains the expected part
    found_log_message = False
    for call_args in mock_logger_instance.log_custom_message.call_args_list:
        if expected_log_message_part in call_args[0][0]:
            found_log_message = True
            break
    assert found_log_message, f"Expected log message part not found: {expected_log_message_part}"
    mock_logger_instance.log_evaluation_result.assert_called_once()


@patch('evaluate.PolicyOutputMapper') # Patch the class in the evaluate module
@patch('evaluate.load_evaluation_agent')
@patch('evaluate.initialize_opponent')
@patch('evaluate.run_evaluation_loop')
@patch('evaluate.EvaluationLogger')
def test_evaluate_main_ppo_opponent(mock_eval_logger_class, mock_run_loop, mock_init_opponent, mock_load_agent, MockPolicyOutputMapperClass, tmp_path, monkeypatch):
    actual_policy_mapper_instance = PolicyOutputMapper()
    MockPolicyOutputMapperClass.return_value = actual_policy_mapper_instance

    mock_agent_instance = MockPPOAgent(INPUT_CHANNELS, actual_policy_mapper_instance, "cpu", name="LoadedMainAgentPPO")
    mock_opponent_ppo_instance = MockPPOAgent(INPUT_CHANNELS, actual_policy_mapper_instance, "cpu", name="LoadedOpponentPPO")
    mock_logger_instance = MagicMock(spec=EvaluationLogger)

    mock_load_agent.side_effect = [mock_agent_instance, mock_opponent_ppo_instance]
    mock_init_opponent.return_value = mock_opponent_ppo_instance

    mock_run_loop.return_value = {"wins": 1, "losses": 1, "draws": 0, "num_games": 2, "win_rate": 0.5, "loss_rate":0.5, "draw_rate":0.0, "avg_game_length": 15}
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    log_file = tmp_path / "main_eval_ppo.log"
    test_args = [
        "evaluate.py",
        "--agent-checkpoint", "agent.pth",
        "--opponent-type", "ppo",
        "--opponent-checkpoint", "opponent.pth",
        "--num-games", "2",
        "--log-file", str(log_file)
    ]
    monkeypatch.setattr(sys, 'argv', test_args)

    evaluate_main()

    MockPolicyOutputMapperClass.assert_called_once()
    mock_init_opponent.assert_called_once_with("ppo", "opponent.pth", "cpu", actual_policy_mapper_instance, INPUT_CHANNELS)

    assert mock_load_agent.call_args_list[0] == call("agent.pth", "cpu", actual_policy_mapper_instance, INPUT_CHANNELS)

    mock_run_loop.assert_called_once_with(
        mock_agent_instance, mock_opponent_ppo_instance, 2, mock_logger_instance, actual_policy_mapper_instance, 256, "cpu" # 256 is default max_moves
    )
    mock_eval_logger_class.assert_called_once_with(log_file_path=str(log_file), also_stdout=True)

def test_evaluate_main_ppo_opponent_no_path(capsys, monkeypatch):
    test_args = [
        "evaluate.py",
        "--agent-checkpoint", "agent.pth",
        "--opponent-type", "ppo",
        # No --opponent-checkpoint
        "--num-games", "1"
    ]
    monkeypatch.setattr(sys, 'argv', test_args)
    with pytest.raises(SystemExit) as e:
        evaluate_main()
    assert e.value.code == 1 # argparse exits with 2 for errors, but main catches and exits with 1
    captured = capsys.readouterr()
    assert "Error: --opponent-checkpoint must be specified when --opponent-type is 'ppo'." in captured.err
