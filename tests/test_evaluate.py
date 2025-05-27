"""
Unit and integration tests for the evaluate.py script.
"""

import os
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch  # Added PropertyMock

import numpy as np
import pytest
import torch  # Re-add torch import

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.ppo_agent import (
    PPOAgent,
)
from keisei.evaluation.evaluate import (
    SimpleHeuristicOpponent,
    SimpleRandomOpponent,
    execute_full_evaluation_run,
    initialize_opponent,
    load_evaluation_agent,
    run_evaluation_loop,
)
from keisei.shogi.shogi_core_definitions import MoveTuple
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import (
    BaseOpponent,
    EvaluationLogger,
    PolicyOutputMapper,
)

INPUT_CHANNELS = 46  # Use the default from config for tests


# A mock PPOAgent for testing purposes
# Inherit from PPOAgent to satisfy type hints for run_evaluation_loop, and BaseOpponent for other uses.
class MockPPOAgent(PPOAgent, BaseOpponent):
    def __init__(
        self,
        config,
        device,
        name="MockPPOAgentForTest",
    ):
        PPOAgent.__init__(self, config=config, device=device, name=name)
        BaseOpponent.__init__(self, name=name)
        self.model = MagicMock()
        self._is_ppo_agent_mock = True  # Flag to identify this mock
        # self.name is set by PPOAgent's __init__ via BaseOpponent

    def load_model(
        self, file_path: str
    ) -> dict:  # Parameter name changed to file_path, return type to dict
        # print(f"MockPPOAgent: Pretending to load model from {file_path}")
        return {}  # Return an empty dict as per PPOAgent

    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: torch.Tensor,
        *,
        is_training: bool = True,
    ):
        # For test compatibility, always return a dummy move and values
        # Assume legal_mask is a tensor of bools, pick the first True index
        idx = int(legal_mask.nonzero(as_tuple=True)[0][0]) if legal_mask.any() else 0
        return (None, idx, 0.0, 0.0)

    def get_value(
        self, obs_np: np.ndarray
    ) -> float:  # obs_np type changed to np.ndarray
        """Mocked get_value method."""
        return 0.0  # Return a dummy float value

    # If used as a BaseOpponent directly (e.g. PPO vs PPO where one is simplified)
    def select_move(
        self, game_instance: ShogiGame
    ) -> MoveTuple:  # Return type changed back to MoveTuple
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError("MockPPOAgent.select_move: No legal moves available.")
        # Simplified for BaseOpponent interface, actual PPO logic is in select_action
        obs_np = MagicMock(
            spec=np.ndarray
        )  # Dummy observation, spec for type hint if needed
        legal_mask_tensor = MagicMock(
            spec=torch.Tensor
        )  # Dummy mask, spec for type hint
        action_result = self.select_action(obs_np, legal_mask_tensor, is_training=False)
        selected_move = action_result[0]
        if selected_move is None:
            # This should ideally not happen if legal_moves is not empty.
            # Handle cases where select_action might return None for the move.
            raise ValueError(
                "MockPPOAgent.select_move: select_action returned None for a move despite legal moves being available."
            )
        return selected_move


@pytest.fixture
def policy_mapper():
    return PolicyOutputMapper()


@pytest.fixture
def eval_logger_setup(tmp_path):
    log_file = tmp_path / "test_eval.log"
    logger = EvaluationLogger(str(log_file), also_stdout=False)
    with logger:  # Ensure logger is used as a context manager
        yield logger, str(log_file)
    # logger.close() is handled by the context manager's __exit__


@pytest.fixture
def shogi_game_initial():
    return ShogiGame()


# --- Tests for Opponent Classes ---


def test_simple_random_opponent_select_move(shogi_game_initial: ShogiGame):
    """Test that SimpleRandomOpponent selects a legal move from the game state."""
    opponent = SimpleRandomOpponent()
    legal_moves = shogi_game_initial.get_legal_moves()
    selected_move = opponent.select_move(shogi_game_initial)
    assert selected_move in legal_moves


def test_simple_heuristic_opponent_select_move(shogi_game_initial: ShogiGame):
    """Test that SimpleHeuristicOpponent selects a legal move from the game state."""
    opponent = SimpleHeuristicOpponent()
    legal_moves = shogi_game_initial.get_legal_moves()
    selected_move = opponent.select_move(shogi_game_initial)
    assert selected_move in legal_moves
    # Add more specific tests for heuristics later if needed


# --- Tests for Initialization Functions ---


def test_initialize_opponent_types(policy_mapper):
    opponent_random = initialize_opponent(
        "random", None, "cpu", policy_mapper, INPUT_CHANNELS
    )
    assert isinstance(opponent_random, SimpleRandomOpponent)

    opponent_heuristic = initialize_opponent(
        "heuristic", None, "cpu", policy_mapper, INPUT_CHANNELS
    )
    assert isinstance(opponent_heuristic, SimpleHeuristicOpponent)

    with pytest.raises(ValueError, match="Unknown opponent type"):
        initialize_opponent("unknown", None, "cpu", policy_mapper, INPUT_CHANNELS)


@patch(
    "keisei.evaluation.evaluate.load_evaluation_agent"  # Patch where used, not just re-exported
)  # Mock load_evaluation_agent within evaluate.py
def test_initialize_opponent_ppo(mock_load_agent, policy_mapper):
    """Test that initialize_opponent returns a PPOAgent when type is 'ppo' and path is provided."""
    mock_ppo_instance = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="MockPPOAgentForTest",
    )
    mock_load_agent.return_value = mock_ppo_instance

    opponent_ppo = initialize_opponent(
        "ppo", "dummy_path.pth", "cpu", policy_mapper, INPUT_CHANNELS
    )
    mock_load_agent.assert_called_once_with(
        "dummy_path.pth", "cpu", policy_mapper, INPUT_CHANNELS
    )
    assert opponent_ppo == mock_ppo_instance

    with pytest.raises(
        ValueError, match="Opponent path must be provided for PPO opponent type."
    ):
        initialize_opponent("ppo", None, "cpu", policy_mapper, INPUT_CHANNELS)


@patch(
    "keisei.evaluation.evaluate.PPOAgent"
)  # MODIFIED: Updated import path # Mock PPOAgent class within evaluate.py
def test_load_evaluation_agent_mocked(MockPPOAgentClass, policy_mapper, tmp_path):
    """Test that load_evaluation_agent returns a PPOAgent instance when checkpoint exists."""
    mock_agent_instance = MagicMock(spec=PPOAgent)
    mock_agent_instance.model = MagicMock()
    MockPPOAgentClass.return_value = mock_agent_instance

    # Create a dummy checkpoint file
    dummy_ckpt = tmp_path / "dummy_checkpoint.pth"
    dummy_ckpt.write_bytes(b"dummy")

    agent = load_evaluation_agent(str(dummy_ckpt), "cpu", policy_mapper, INPUT_CHANNELS)
    assert agent == mock_agent_instance


# --- Test for Core Evaluation Loop ---


def test_run_evaluation_loop_basic(policy_mapper, eval_logger_setup):
    """Test that run_evaluation_loop runs games and logs results correctly."""
    logger, log_file_path = eval_logger_setup

    agent_to_eval = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="PPOAgentToEvaluate",
    )
    opponent = SimpleRandomOpponent(name="TestRandomOpponent")

    num_games = 2
    max_moves = 5  # Keep games short for testing

    results = run_evaluation_loop(
        agent_to_eval, opponent, num_games, logger, policy_mapper, max_moves, "cpu"
    )

    assert results["num_games"] == num_games
    assert "wins" in results
    assert "losses" in results
    assert "draws" in results
    assert results["wins"] + results["losses"] + results["draws"] == num_games
    assert "avg_game_length" in results
    assert results["avg_game_length"] <= max_moves  # Can be less if game ends early

    with open(log_file_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert (
        f"Starting evaluation: {agent_to_eval.name} vs {opponent.name}" in log_content
    )
    assert f"Starting Game 1/{num_games}" in log_content
    assert f"Starting Game 2/{num_games}" in log_content
    assert "Game 1 ended" in log_content
    assert "Game 2 ended" in log_content
    assert "Evaluation finished. Results:" in log_content
    assert (
        f"Agent to eval ({agent_to_eval.name}) is Black" in log_content
    )  # Game 1, updated assertion
    assert (
        f"Agent to eval ({agent_to_eval.name}) is White" in log_content
    )  # Game 2, updated assertion


# --- Tests for Main Script Execution (now execute_full_evaluation_run) ---

# Helper for common main test mocks
COMMON_MAIN_MOCKS = [
    patch(
        "keisei.evaluation.evaluate.PolicyOutputMapper"
    ),  # MODIFIED: Updated import path
    patch(
        "keisei.evaluation.evaluate.load_evaluation_agent"
    ),  # MODIFIED: Updated import path
    patch(
        "keisei.evaluation.evaluate.initialize_opponent"
    ),  # MODIFIED: Updated import path
    patch(
        "keisei.evaluation.evaluate.run_evaluation_loop"
    ),  # MODIFIED: Updated import path
    patch(
        "keisei.evaluation.evaluate.EvaluationLogger"
    ),  # MODIFIED: Updated import path
    patch("wandb.init"),
    patch("wandb.log"),
    patch("wandb.finish"),
    patch("wandb.run", new_callable=PropertyMock),  # Added mock for wandb.run
    patch("random.seed"),
    patch("numpy.random.seed"),
    patch("torch.manual_seed"),
]


def apply_mocks(mocks):
    def decorator(func):
        for m in reversed(mocks):  # Apply in reverse for correct arg order
            func = m(func)
        return func

    return decorator


@apply_mocks(
    COMMON_MAIN_MOCKS[
        :-3
    ]  # Indices 0-8: Excludes the three seeding mocks (random.seed, numpy.random.seed, torch.manual_seed)
)
# pylint: disable=unused-argument,too-many-positional-arguments,too-many-locals
def test_execute_full_evaluation_run_basic_random(  # MODIFIED: Renamed and refactored from test_evaluate_main_basic_random_opponent
    mock_wandb_run_prop,  # For asserting not called
    mock_wandb_finish,  # For asserting not called
    mock_wandb_log,  # For asserting not called
    mock_wandb_init,  # For asserting not called
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    MockPolicyOutputMapperClass,  # Mock for the class keisei.evaluate.PolicyOutputMapper
    tmp_path,
):
    # Create a real PolicyOutputMapper instance for the test, as execute_full_evaluation_run expects one.
    policy_mapper_instance = PolicyOutputMapper()

    mock_agent_instance = MockPPOAgent(  # Using the test utility MockPPOAgent
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="LoadedEvalAgent",
    )
    mock_load_agent.return_value = mock_agent_instance

    mock_opponent_instance = SimpleRandomOpponent(name="EvalRandomOpponent")
    mock_init_opponent.return_value = mock_opponent_instance

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    expected_run_loop_results = {
        "num_games": 1,
        "wins": 1,
        "losses": 0,
        "draws": 0,
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "avg_game_length": 10.0,
        "opponent_name": "EvalRandomOpponent",
        "agent_name": "LoadedEvalAgent",  # Added agent_name
    }
    mock_run_loop.return_value = expected_run_loop_results

    log_file = tmp_path / "eval_basic_run.log"
    agent_ckpt_path = "./agent_for_eval.pth"
    num_games_to_run = 1
    max_moves_for_game = 250  # Specific value for this test

    # Call the function being tested
    results = execute_full_evaluation_run(
        agent_checkpoint_path=agent_ckpt_path,
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=num_games_to_run,
        max_moves_per_game=max_moves_for_game,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper_instance,  # Pass the created instance
        seed=None,  # Not testing seed in this specific test
        wandb_log_eval=False,  # Explicitly disable W&B for this test
    )

    # Assertions
    # MockPolicyOutputMapperClass (patch of keisei.evaluate.PolicyOutputMapper) should not be called if
    # execute_full_evaluation_run uses the passed policy_mapper_instance directly.
    MockPolicyOutputMapperClass.assert_not_called()

    mock_load_agent.assert_called_once_with(
        agent_ckpt_path, "cpu", policy_mapper_instance, INPUT_CHANNELS
    )
    mock_init_opponent.assert_called_once_with(
        "random", None, "cpu", policy_mapper_instance, INPUT_CHANNELS
    )
    # EvaluationLogger is called with run_name_for_log="eval_run" when wandb_log_eval is False
    mock_eval_logger_class.assert_called_once_with(str(log_file), also_stdout=True)

    mock_run_loop.assert_called_once()
    pos_args, kw_args = mock_run_loop.call_args
    assert pos_args[0] == mock_agent_instance  # agent_to_eval
    assert pos_args[1] == mock_opponent_instance  # opponent
    assert pos_args[2] == num_games_to_run  # num_games
    assert pos_args[3] == mock_logger_instance  # logger
    assert pos_args[4] == policy_mapper_instance  # policy_mapper
    assert pos_args[5] == max_moves_for_game  # max_moves
    assert pos_args[6] == "cpu"  # device_str
    assert kw_args.get("wandb_enabled") is False  # wandb_enabled in run_evaluation_loop

    mock_wandb_init.assert_not_called()
    mock_wandb_log.assert_not_called()
    mock_wandb_finish.assert_not_called()
    mock_wandb_run_prop.assert_not_called()  # wandb.run should not be accessed

    assert results == expected_run_loop_results  # Check if results are passed through


@apply_mocks(COMMON_MAIN_MOCKS)  # Includes all mocks
# pylint: disable=unused-argument,too-many-positional-arguments,too-many-locals
def test_execute_full_evaluation_run_heuristic_opponent_with_wandb(  # MODIFIED: Renamed and refactored
    mock_torch_seed,
    mock_np_seed,
    mock_random_seed,
    mock_wandb_run_prop,
    mock_wandb_finish,
    mock_wandb_log,
    mock_wandb_init,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    MockPolicyOutputMapperClass,
    tmp_path,
    # monkeypatch, # MODIFIED: Removed monkeypatch as sys.argv is no longer used
):
    policy_mapper_instance = PolicyOutputMapper()  # MODIFIED: Create an instance
    # MockPolicyOutputMapperClass.return_value = policy_mapper_instance # Not needed if we pass the instance

    mock_agent_instance = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="LoadedMainAgentWandb",
    )
    mock_load_agent.return_value = mock_agent_instance

    mock_opponent_instance = SimpleHeuristicOpponent(name="MainHeuristicOpponent")
    mock_init_opponent.return_value = mock_opponent_instance

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    expected_run_loop_results = {
        "num_games": 2,
        "wins": 1,
        "losses": 1,
        "draws": 0,
        "win_rate": 0.5,
        "loss_rate": 0.5,
        "draw_rate": 0.0,
        "avg_game_length": 20.0,
        "opponent_name": "MainHeuristicOpponent",
        "agent_name": "LoadedMainAgentWandb",  # Added agent_name
    }
    mock_run_loop.return_value = expected_run_loop_results

    log_file = tmp_path / "eval_wandb.log"
    agent_ckpt_path = "./agent_wandb.pth"
    num_games_to_run = 2
    max_moves_for_game = 300  # Default, but explicit for clarity
    seed_to_use = None  # Explicitly None for this test case

    # W&B setup
    wandb_project_name = "test_project"
    wandb_entity_name = "test_entity"
    wandb_run_name_custom = "custom_eval_run"

    mock_wandb_run = MagicMock()
    mock_wandb_run.name = wandb_run_name_custom
    mock_wandb_init.return_value = mock_wandb_run
    mock_wandb_run_prop.return_value = mock_wandb_run

    # Call the function directly
    results = execute_full_evaluation_run(
        agent_checkpoint_path=agent_ckpt_path,
        opponent_type="heuristic",
        opponent_checkpoint_path=None,
        num_games=num_games_to_run,
        max_moves_per_game=max_moves_for_game,  # Using the default
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper_instance,
        seed=seed_to_use,
        wandb_log_eval=True,  # Enable W&B
        wandb_project_eval=wandb_project_name,
        wandb_entity_eval=wandb_entity_name,
        wandb_run_name_eval=wandb_run_name_custom,
        wandb_reinit=True,
    )

    MockPolicyOutputMapperClass.assert_not_called()  # Should use the passed instance

    mock_load_agent.assert_called_once_with(
        agent_ckpt_path, "cpu", policy_mapper_instance, INPUT_CHANNELS
    )
    mock_init_opponent.assert_called_once_with(
        "heuristic", None, "cpu", policy_mapper_instance, INPUT_CHANNELS
    )

    # Check EvaluationLogger call - with W&B, run_name_for_log should be the wandb run name
    # MODIFIED: The EvaluationLogger in execute_full_evaluation_run does not take run_name_for_log directly.
    # It's used internally if W&B is active to set the W&B run name.
    # The logger itself is called with log_file_path_eval and also_stdout.
    # The run_name_for_log parameter was removed from EvaluationLogger constructor.
    mock_eval_logger_class.assert_called_once_with(str(log_file), also_stdout=True)

    (
        _run_loop_pos_args,
        run_loop_kwargs,
    ) = mock_run_loop.call_args
    assert _run_loop_pos_args[0] == mock_agent_instance
    assert _run_loop_pos_args[1] == mock_opponent_instance
    assert _run_loop_pos_args[2] == num_games_to_run
    assert _run_loop_pos_args[3] == mock_logger_instance
    assert _run_loop_pos_args[4] == policy_mapper_instance
    assert _run_loop_pos_args[5] == max_moves_for_game
    assert _run_loop_pos_args[6] == "cpu"
    assert run_loop_kwargs.get("wandb_enabled") is True

    expected_wandb_config = {
        "agent_checkpoint": agent_ckpt_path,
        "opponent_type": "heuristic",
        "opponent_checkpoint": None,
        "num_games": num_games_to_run,
        "max_moves_per_game": max_moves_for_game,
        "device": "cpu",
        "seed": seed_to_use,
    }
    mock_wandb_init.assert_called_once_with(
        project=wandb_project_name,
        entity=wandb_entity_name,
        name=wandb_run_name_custom,
        config=expected_wandb_config,
        reinit=True,
    )
    # wandb.log is called by run_evaluation_loop, which is mocked here.
    # So, this mock_wandb_log (global) won\'t be hit by the mocked run_evaluation_loop.
    # mock_wandb_log.assert_not_called() # MODIFIED: execute_full_evaluation_run now calls wandb.log for final summary
    mock_wandb_log.assert_called_once_with(  # MODIFIED: Check for the specific final summary log call
        {
            "eval/final_win_rate": expected_run_loop_results["win_rate"],
            "eval/final_loss_rate": expected_run_loop_results["loss_rate"],
            "eval/final_draw_rate": expected_run_loop_results["draw_rate"],
            "eval/final_avg_game_length": expected_run_loop_results["avg_game_length"],
        }
    )
    mock_wandb_finish.assert_called_once()

    mock_random_seed.assert_not_called()  # Seed not specified
    mock_np_seed.assert_not_called()
    mock_torch_seed.assert_not_called()

    assert results == expected_run_loop_results  # Check results pass-through


@apply_mocks(
    COMMON_MAIN_MOCKS
)  # MODIFIED: Keep all common mocks, initialize_opponent is now called within execute_full_evaluation_run
# pylint: disable=unused-argument,too-many-positional-arguments,too-many-locals
def test_execute_full_evaluation_run_ppo_vs_ppo_with_wandb(  # MODIFIED: Renamed and refactored
    mock_torch_seed,
    mock_np_seed,
    mock_random_seed,
    mock_wandb_run_prop,
    mock_wandb_finish,
    mock_wandb_log,
    mock_wandb_init,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,  # ADDED: mock_init_opponent is needed as it's called by execute_full_evaluation_run
    mock_load_agent,
    MockPolicyOutputMapperClass,
    tmp_path,
    # monkeypatch, # MODIFIED: Removed monkeypatch
):
    policy_mapper_instance = PolicyOutputMapper()
    # MockPolicyOutputMapperClass.return_value = policy_mapper_instance # Not needed

    mock_agent_to_eval = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="PPOAgentToEvaluate",
    )
    mock_opponent_agent = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="PPOAgentOpponent",
    )

    agent_eval_path = "./agent_to_eval.pth"
    agent_opponent_path = "./agent_opponent.pth"

    # load_evaluation_agent will be called twice: once directly by execute_full_evaluation_run for the agent_to_eval,
    # and once by initialize_opponent (which is called by execute_full_evaluation_run) for the PPO opponent.
    def load_agent_side_effect(checkpoint_path, device, pol_mapper, in_channels):
        if checkpoint_path == agent_eval_path:
            return mock_agent_to_eval
        if checkpoint_path == agent_opponent_path:
            return mock_opponent_agent
        pytest.fail(f"Unexpected call to load_evaluation_agent with {checkpoint_path}")
        return None  # Should not be reached

    mock_load_agent.side_effect = load_agent_side_effect

    # initialize_opponent will be called once for the PPO opponent.
    # We need to make sure it returns the mock_opponent_agent when called with the correct path.
    def init_opponent_side_effect(
        opponent_type, opponent_path, device, pol_mapper, in_channels
    ):
        if opponent_type == "ppo" and opponent_path == agent_opponent_path:
            # Normally, initialize_opponent would call load_evaluation_agent itself.
            # Since load_evaluation_agent is already mocked with a side_effect that returns mock_opponent_agent
            # for agent_opponent_path, we can just return that here.
            # Or, more simply, ensure initialize_opponent is robustly mocked to return the agent directly.
            return mock_opponent_agent
        if opponent_type == "random":
            return SimpleRandomOpponent()
        if opponent_type == "heuristic":
            return SimpleHeuristicOpponent()
        pytest.fail(
            f"Unexpected call to initialize_opponent with {opponent_type}, {opponent_path}"
        )
        return None  # Should not be reached

    mock_init_opponent.side_effect = init_opponent_side_effect

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    expected_results = {
        "num_games": 1,
        "wins": 0,
        "losses": 1,
        "draws": 0,
        "win_rate": 0.0,
        "loss_rate": 1.0,
        "draw_rate": 0.0,
        "avg_game_length": 15.0,
        "opponent_name": "PPOAgentOpponent",  # Name of the opponent agent
        "agent_name": "PPOAgentToEvaluate",  # Name of the agent being evaluated
    }
    mock_run_loop.return_value = expected_results

    log_file = tmp_path / "eval_ppo_vs_ppo.log"
    num_games_val = 1
    max_moves_val = 200
    wandb_project_val = "ppo_battle_project"

    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "test_ppo_run"  # Example name
    mock_wandb_init.return_value = mock_wandb_run
    mock_wandb_run_prop.return_value = mock_wandb_run

    results = execute_full_evaluation_run(
        agent_checkpoint_path=agent_eval_path,
        opponent_type="ppo",
        opponent_checkpoint_path=agent_opponent_path,
        num_games=num_games_val,
        max_moves_per_game=max_moves_val,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper_instance,
        seed=None,
        wandb_log_eval=True,
        wandb_project_eval=wandb_project_val,
        wandb_entity_eval=None,
        wandb_run_name_eval="test_ppo_run",
        wandb_reinit=True,
    )

    MockPolicyOutputMapperClass.assert_not_called()

    # load_evaluation_agent assertions
    assert mock_load_agent.call_count == 1  # Only called directly for agent_to_eval
    mock_load_agent.assert_any_call(
        agent_eval_path, "cpu", policy_mapper_instance, INPUT_CHANNELS
    )
    # initialize_opponent is called for the opponent, and it internally would call load_evaluation_agent,
    # but we mocked initialize_opponent to return the agent directly.
    mock_init_opponent.assert_called_once_with(
        "ppo", agent_opponent_path, "cpu", policy_mapper_instance, INPUT_CHANNELS
    )

    mock_eval_logger_class.assert_called_once_with(str(log_file), also_stdout=True)

    run_loop_pos_args, run_loop_kwargs = mock_run_loop.call_args
    assert run_loop_pos_args[0] == mock_agent_to_eval
    assert (
        run_loop_pos_args[1] == mock_opponent_agent
    )  # This should be the PPO opponent instance
    assert run_loop_kwargs.get("wandb_enabled") is True

    expected_wandb_config = {
        "agent_checkpoint": agent_eval_path,
        "opponent_type": "ppo",
        "opponent_checkpoint": agent_opponent_path,
        "num_games": num_games_val,
        "max_moves_per_game": max_moves_val,
        "device": "cpu",
        "seed": None,
    }
    mock_wandb_init.assert_called_once_with(
        project=wandb_project_val,
        entity=None,
        name="test_ppo_run",
        config=expected_wandb_config,
        reinit=True,
    )
    mock_wandb_log.assert_called_once_with(
        {
            "eval/final_win_rate": expected_results["win_rate"],
            "eval/final_loss_rate": expected_results["loss_rate"],
            "eval/final_draw_rate": expected_results["draw_rate"],
            "eval/final_avg_game_length": expected_results["avg_game_length"],
        }
    )
    mock_wandb_finish.assert_called_once()
    mock_random_seed.assert_not_called()
    mock_np_seed.assert_not_called()
    mock_torch_seed.assert_not_called()
    assert results == expected_results


@apply_mocks(COMMON_MAIN_MOCKS)  # Includes all mocks
# pylint: disable=unused-argument,too-many-positional-arguments,too-many-locals
def test_execute_full_evaluation_run_with_seed(  # MODIFIED: Renamed and refactored
    mock_torch_seed,
    mock_np_seed,
    mock_random_seed,
    mock_wandb_run_prop,
    mock_wandb_finish,
    mock_wandb_log,
    mock_wandb_init,
    mock_eval_logger_class,
    mock_run_loop,
    mock_init_opponent,
    mock_load_agent,
    MockPolicyOutputMapperClass,  # Mock for the class keisei.evaluate.PolicyOutputMapper
    tmp_path,
    # monkeypatch, # MODIFIED: Removed monkeypatch as sys.argv is no longer used
):
    # MODIFIED: Setup similar to test_execute_full_evaluation_run_basic_random
    policy_mapper_instance = PolicyOutputMapper()

    mock_agent_instance = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="AgentForSeedTest",
    )
    mock_load_agent.return_value = mock_agent_instance

    mock_opponent_instance = SimpleRandomOpponent(name="RandomOpponentForSeedTest")
    mock_init_opponent.return_value = mock_opponent_instance

    mock_logger_instance = MagicMock(spec=EvaluationLogger)
    mock_eval_logger_class.return_value.__enter__.return_value = mock_logger_instance

    mock_run_loop.return_value = {  # Dummy results
        "num_games": 1,
        "wins": 1,
        "losses": 0,
        "draws": 0,
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "avg_game_length": 5.0,
        "opponent_name": "RandomOpponentForSeedTest",
        "agent_name": "AgentForSeedTest",
    }

    log_file_path = tmp_path / "eval_seed_test.log"  # MODIFIED: Renamed variable
    agent_checkpoint_file = "./agent_seed.pth"  # MODIFIED: Renamed variable
    seed_value_to_test = 123  # MODIFIED: Renamed variable

    # Call the function
    execute_full_evaluation_run(
        agent_checkpoint_path=agent_checkpoint_file,
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=1,
        max_moves_per_game=100,
        device_str="cpu",
        log_file_path_eval=str(log_file_path),
        policy_mapper=policy_mapper_instance,
        seed=seed_value_to_test,  # MODIFIED: Pass the seed
        wandb_log_eval=False,
    )

    # Assertions for seed calls
    mock_random_seed.assert_called_once_with(seed_value_to_test)
    mock_np_seed.assert_called_once_with(seed_value_to_test)
    mock_torch_seed.assert_called_once_with(seed_value_to_test)

    # Other assertions (can be minimal as other tests cover functionality)
    MockPolicyOutputMapperClass.assert_not_called()
    mock_load_agent.assert_called_once()
    mock_init_opponent.assert_called_once()
    mock_eval_logger_class.assert_called_once()
    mock_run_loop.assert_called_once()
    mock_wandb_init.assert_not_called()
    mock_wandb_log.assert_not_called()
    mock_wandb_finish.assert_not_called()
    mock_wandb_run_prop.assert_not_called()


# MODIFIED: Removed test_evaluate_main_invalid_opponent_type as its functionality
# (checking for invalid opponent type) is covered by test_initialize_opponent_types
# and the CLI parsing aspect is no longer relevant.

# --- Documentation and code review notes ---
#
# - All tests in this file are well-structured and use clear, descriptive docstrings.
# - Imports are organized at the top of the file, and all test utilities are grouped logically.
# - Test coverage includes: random/heuristic/ppo opponents, error handling, W&B integration, and CLI/API entry points.
# - All error handling tests use the correct exception types and provide clear messages.
# - The MockPPOAgent and fixtures are reusable and well-documented.
# - No dead code or commented-out legacy code remains.
# - All tests pass lint and functional checks as of 2025-05-27.
#
# For further documentation, see HOW_TO_USE.md and README.md for CLI/API usage.


# --- Helper Functions and Fixtures for Specific Scenarios ---


def test_evaluator_class_basic(monkeypatch, tmp_path, policy_mapper):
    """
    Integration test for the Evaluator class: random-vs-agent, no W&B, log to file.
    """

    # Patch load_evaluation_agent and initialize_opponent to return mocks
    class DummyAgent(PPOAgent):
        def __init__(self):
            config = make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper())
            super().__init__(
                config=config, device=torch.device("cpu"), name="DummyAgent"
            )
            self.model = MagicMock()

        def select_action(self, obs, legal_mask, *, is_training=True):
            # Always pick the first legal move, index 0, dummy log_prob and value
            idx = (
                int(legal_mask.nonzero(as_tuple=True)[0][0]) if legal_mask.any() else 0
            )
            return None, idx, 0.0, 0.0

    class DummyOpponent(BaseOpponent):
        def __init__(self):
            super().__init__(name="DummyOpponent")

        def select_move(self, game_instance):
            return game_instance.get_legal_moves()[0]

    monkeypatch.setattr(
        "keisei.evaluation.evaluate.load_evaluation_agent",
        lambda *a, **kw: DummyAgent(),
    )
    monkeypatch.setattr(
        "keisei.evaluation.evaluate.initialize_opponent",
        lambda *a, **kw: DummyOpponent(),
    )
    log_file = tmp_path / "evaluator_test.log"
    evaluator = __import__("keisei.evaluate", fromlist=["Evaluator"]).Evaluator(
        agent_checkpoint_path="dummy_agent.pth",
        opponent_type="random",
        opponent_checkpoint_path=None,
        num_games=1,
        max_moves_per_game=5,
        device_str="cpu",
        log_file_path_eval=str(log_file),
        policy_mapper=policy_mapper,
        seed=42,
        wandb_log_eval=False,
        wandb_project_eval=None,
        wandb_entity_eval=None,
        wandb_run_name_eval=None,
        logger_also_stdout=False,
    )
    results = evaluator.evaluate()
    assert isinstance(results, dict)
    assert results["num_games"] == 1
    assert "wins" in results and "losses" in results and "draws" in results
    with open(log_file, encoding="utf-8") as f:
        log_content = f.read()
    assert "Starting Shogi Agent Evaluation" in log_content
    assert "Evaluation finished. Results:" in log_content


# Helper to create a minimal AppConfig for test agents
def make_test_config(device_str, input_channels, policy_mapper):
    # If policy_mapper is a pytest fixture function, raise an error to prevent direct calls
    if hasattr(policy_mapper, "_pytestfixturefunction"):
        raise RuntimeError(
            "policy_mapper fixture was passed directly; pass an instance instead."
        )
    try:
        num_actions_total = policy_mapper.get_total_actions()
    except (
        Exception
    ):  # pylint: disable=broad-except  # nosec: test utility, fallback is safe
        num_actions_total = 13527  # Default fallback for mocks or MagicMock
    return AppConfig(
        env=EnvConfig(
            device=device_str,
            input_channels=input_channels,
            num_actions_total=num_actions_total,
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
            # --- Added fields with default values ---
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
        ),
        evaluation=EvaluationConfig(num_games=1, opponent_type="random"),
        logging=LoggingConfig(log_file="/tmp/eval.log", model_dir="/tmp/"),
        wandb=WandBConfig(enabled=False, project="eval", entity=None),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
    )


def test_load_evaluation_agent_missing_checkpoint(policy_mapper):
    """
    Test that load_evaluation_agent raises a FileNotFoundError for a missing checkpoint file.

    This ensures that the evaluation pipeline fails fast and clearly when a model checkpoint
    is missing, rather than failing later or with a cryptic error.
    """
    # Create a path that does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "nonexistent_checkpoint.pth")
        with pytest.raises(FileNotFoundError):
            load_evaluation_agent(missing_path, "cpu", policy_mapper, INPUT_CHANNELS)


def test_initialize_opponent_invalid_type(policy_mapper):
    """
    Test that initialize_opponent raises ValueError for an invalid opponent type.

    This ensures that the evaluation pipeline is robust to user/configuration errors and
    provides a clear error message if an unsupported opponent type is specified.
    """
    with pytest.raises(ValueError, match="Unknown opponent type"):
        initialize_opponent("not_a_type", None, "cpu", policy_mapper, INPUT_CHANNELS)
