# PPO Shogi Agent Evaluation System Plan

**Date:** May 23, 2025
**Version:** 1.3 (Added Weights & Biases integration to plan)

## 1. Introduction and Objectives

This document outlines the plan for implementing an evaluation system for the PPO-based Shogi agent. The primary goal is to systematically measure the performance of trained agents, compare different model versions, and track improvements over time.

**Key Objectives:**

*   **Performance Measurement:** Quantify the playing strength of the agent.
*   **Model Comparison:** Objectively compare different iterations of the agent (e.g., different hyperparameters, network architectures, or checkpoints from a single training run).
*   **Progress Tracking:** Monitor improvements during training by periodically evaluating against a consistent benchmark.
*   **Identify Weaknesses:** (Potentially) Gather data that can help identify systematic weaknesses in the agent's play.
*   **Reproducibility:** Ensure that evaluations can be run consistently.

## 2. Core Components and Functionality

### 2.1. Evaluation Loop

A dedicated mechanism will be implemented to run a series of Shogi games for evaluation purposes.

*   **Agent Loading:**
    *   The system must allow specifying a saved agent checkpoint (`.pth` file) to be evaluated.
    *   The agent will be loaded in evaluation mode (e.g., `model.eval()`, no gradient calculations).
*   **Opponent(s):** The evaluation system should support playing against various types of opponents:
    *   **Fixed Baseline Agent:**
        *   A simple heuristic-based agent (e.g., random valid moves, greedy piece capture). This provides a stable, albeit weak, benchmark.
        *   A previously saved checkpoint of the PPO agent, acting as a fixed "sparring partner."
    *   **Specific Checkpoints:** Ability to pit two specified agent checkpoints against each other.
    *   *(Future)* Potentially an external Shogi engine via USI protocol if feasible.
*   **Game Execution:**
    *   The system will manage the game flow, alternating colors (Sente/Gote) for the agent under evaluation over a set number of games.
    *   Standard Shogi rules will apply, including game termination conditions (checkmate, resignation, max moves).
*   **Deterministic vs. Stochastic Mode:**
    *   The agent under evaluation should primarily use a deterministic policy (i.g., selecting the action with the highest probability) for consistent performance measurement.
    *   Optionally, allow stochastic evaluation to measure performance variability.

### 2.2. Metrics to Track

The following metrics will be recorded for each evaluation session:

*   **Win/Loss/Draw Rates:**
    *   Overall W/L/D against the chosen opponent(s).
    *   Separate W/L/D rates when playing as Sente (Black) and Gote (White).
*   **Average Game Length:** Number of moves per game.
*   **Score/Reward:** (If applicable) Average cumulative reward if the environment provides a clear scoring mechanism beyond W/L/D.
*   *(Future)* **Elo Rating:** Implement or integrate a simple Elo rating system to track strength changes over time, especially when evaluating against a pool of agents or evolving baselines.
*   *(Future)* **Move Consistency/Quality:** If a stronger reference engine is available, compare agent moves to the reference engine's suggestions (e.g., percentage of moves matching top N choices).
*   **Time Metrics:** Average time taken per move by the agent.

### 2.3. Configuration

The evaluation process should be configurable via command-line arguments or a configuration file:

*   **Agent Checkpoint Path:** Path to the `.pth` file of the agent to evaluate.
*   **Opponent Type/Path:** Specification of the opponent (e.g., "random", "heuristic", path to opponent checkpoint).
*   **Number of Games:** Total games to play for the evaluation (e.g., 50, 100).
*   **Device:** `cpu` or `cuda`.
*   **Deterministic Mode:** Boolean flag (default: True).
*   **Max Moves per Game:** Override for evaluation games.
*   **Seed:** For reproducibility of opponent actions if stochastic.

### 2.4. Logging and Reporting

*   **Structured Log Files:**
    *   A dedicated log file for each evaluation run, storing:
        *   Configuration used for the run.
        *   Game-by-game results (winner, loser, draw, game length, final board state or SGF/KIF if possible).
        *   Summary statistics (overall W/L/D, average game length, etc.).
*   **Console Output:** Clear summary of results printed to the console upon completion.
*   **Weights & Biases Integration (Re-implementation):** Send metrics to Weights & Biases for visualization and comparison across multiple evaluations. This was previously implemented and needs to be restored.
*   *(Future)* **Experiment Tracking Integration:** Send metrics to tools like Weights & Biases or TensorBoard for visualization and comparison across multiple evaluations.
*   *(Future)* **Visualization:** Basic plots for win rates over time if evaluations are run periodically.

## 3. Implementation Plan

### 3.1. New Script/Module (e.g., `evaluate.py`)

A new Python script, `/home/john/keisei/evaluate.py`, **has been created and is substantially complete.** This script serves as the main entry point for running evaluations.

**Key Functions within `evaluate.py` (Implemented):**

*   **`main()`:**
    *   Parses command-line arguments (using `argparse`).
    *   Loads the agent to be evaluated using `load_evaluation_agent()`.
    *   Initializes the opponent agent using `initialize_opponent()`.
    *   Sets up an `EvaluationLogger`.
    *   Calls `run_evaluation_loop()`.
    *   Prints summary metrics to console and logs them.
*   **`load_evaluation_agent(checkpoint_path: str, device: str, policy_mapper: PolicyOutputMapper, input_channels: int) -> PPOAgent`:**
    *   Instantiates a `PPOAgent`.
    *   Calls `agent.load_model(checkpoint_path)`.
    *   Sets the agent to evaluation mode (`agent.model.eval()`).
    *   Returns the loaded agent.
*   **`initialize_opponent(opponent_type: str, opponent_path: Optional[str], device: str, policy_mapper: PolicyOutputMapper, input_channels: int) -> PPOAgent | BaseOpponent`:**
    *   If `opponent_type` is "ppo", loads another `PPOAgent` using `load_evaluation_agent`.
    *   If `opponent_type` is "random", initializes `SimpleRandomOpponent`.
    *   If `opponent_type` is "heuristic", initializes `SimpleHeuristicOpponent`.
*   **`run_evaluation_loop(agent_to_eval: PPOAgent, opponent: PPOAgent | BaseOpponent, num_games: int, logger: EvaluationLogger, policy_mapper: PolicyOutputMapper, max_moves_per_game: int, device: str)`:**
    *   Iterates `num_games` times.
    *   For each game:
        *   Alternates starting color for `agent_to_eval`.
        *   Initializes a `ShogiGame` instance.
        *   Plays out the game move by move.
        *   Records game outcome and length.
        *   Logs game details using `EvaluationLogger`.
    *   Calculates and returns aggregate metrics.
    *   **Logging of agent names (e.g., `agent_to_eval.name`) is now correctly implemented.**
*   **`SimpleRandomOpponent(BaseOpponent)` and `SimpleHeuristicOpponent(BaseOpponent)` classes:**
    *   These classes implement a `select_move(self, game_instance: ShogiGame) -> MoveTuple` method (Note: signature updated from original plan's `select_action`).

### 3.2. Modifications to Existing Code

*   **`keisei/ppo_agent.py` (`PPOAgent`):**
    *   **Added `name` attribute:** Initialized in `__init__` (e.g., `self.name = name`) to allow agents to have distinct names for logging and identification. This was crucial for fixing issues in `test_evaluate.py`.
    *   **`select_action()` method:**
        *   Uses `is_training` to toggle deterministic behavior. `self.model.train(is_training)` sets the mode of `ActorCritic`.
        *   `ActorCritic.get_action_and_value()` uses `deterministic` flag.
    *   **`load_model()` method:** Handles optimizer state device placement.
*   **`keisei/neural_network.py` (`ActorCritic`):**
    *   **`get_action_and_value()` method:** `deterministic` flag is used.
*   **`keisei/shogi/shogi_game.py` (`ShogiGame`):**
    *   Methods like `reset()`, `make_move()`, `get_legal_moves()`, `game_over`, `winner` are used as planned.
    *   `max_moves_per_game` parameter is utilized.
*   **`keisei/shogi/shogi_game_io.py`:**
    *   **`generate_neural_network_observation()`:** Used by PPO agents.
    *   *(Future)* `game_to_kif()` for `EvaluationLogger`.
*   **`keisei/utils.py`:**
    *   **`PolicyOutputMapper`:** Methods `get_legal_mask()` and `policy_index_to_shogi_move()` are used.
    *   **New `EvaluationLogger` class:** Implemented and used in `evaluate.py`.
        *   Methods like `log_custom_message`, `log_evaluation_result` are in use.
        *   (Specific methods like `log_evaluation_config`, `log_game_start`, `log_move`, `log_game_end`, `log_summary_metrics` from the original plan are effectively covered by current `EvaluationLogger` capabilities, though perhaps not with those exact names).
    *   **New `BaseOpponent(ABC)` abstract base class:** Implemented and used by `SimpleRandomOpponent` and `SimpleHeuristicOpponent`.
        *   `select_move(self, game_instance: ShogiGame) -> MoveTuple` is the method used by opponents (updated from `select_action` in plan).
*   **`keisei/train.py`:**
    *   No direct changes are immediately required for the initial evaluation system.
    *   *(Future Consideration)*: As mentioned in the plan, a hook could be added to call `evaluate.py` (perhaps as a subprocess or by importing its main evaluation function) periodically. This would involve:
        *   Identifying points in the training loop (e.g., after `agent.save_model()`).
        *   Constructing the command-line arguments for `evaluate.py`, including the path to the just-saved checkpoint and a standard benchmark opponent.
        *   Using `subprocess.run()` to execute the evaluation.
*   **`config.py`:**
    *   No direct changes needed for evaluation script itself, as evaluation parameters will be command-line arguments.
    *   However, if periodic evaluation is integrated into `train.py`, `config.py` might need new settings like:
        *   `EVAL_FREQ_TIMESTEPS` or `EVAL_FREQ_EPISODES`
        *   `EVAL_NUM_GAMES`
        *   `EVAL_OPPONENT_TYPE`
        *   `EVAL_OPPONENT_PATH` (if evaluating against a fixed checkpoint)

### 3.3. Checkpoint Management

*   **DONE:** The evaluation script will take a direct path to a specific agent checkpoint.
*   For periodic evaluation during training, a strategy will be needed (e.g., evaluate the latest saved checkpoint).

## 4. Workflow

1.  **DONE:** **User Invocation:**
    ```bash
    python keisei/evaluate.py --agent-checkpoint <path_to_agent.pth> --opponent-type <type> [--opponent-checkpoint <path_to_opponent.pth>] --num-games <N> --device <cpu/cuda>
    ```
2.  **DONE:** **Setup:** The script loads the agent, initializes the opponent, and sets up logging.
3.  **DONE:** **Evaluation:** The evaluation loop runs the specified number of games, alternating playing colors.
4.  **DONE:** **Results:**
    *   Game-by-game results are logged.
    *   Final summary statistics are printed to the console and saved to the evaluation log file.
    *   (Future) Metrics are pushed to an experiment tracking service.

## 5. Testing Strategy

*   **Unit Tests (new tests in `tests/test_evaluate.py`):**
    *   Test `evaluate.load_evaluation_agent`: **Largely covered by integration tests.**
    *   Test `evaluate.initialize_opponent`: **Largely covered by integration tests.**
    *   Test `SimpleRandomOpponent.select_move` and `SimpleHeuristicOpponent.select_move`: **Implicitly tested via `run_evaluation_loop` tests.**
    *   Test metric calculation logic within `run_evaluation_loop`: **Covered by `test_run_evaluation_loop_basic` and other tests in `test_evaluate.py`.**
    *   Test `EvaluationLogger` methods: **Covered by tests ensuring log output is correct (e.g., `test_run_evaluation_loop_basic` checking log messages).**
*   **Integration Tests (expanding `tests/test_evaluate.py`):**
    *   Test the full `evaluate.main()` flow: **`test_run_evaluation_loop_basic` in `tests/test_evaluate.py` covers the core loop. Full `main()` CLI testing is a good next step.**
        *   Verify command-line argument parsing.
        *   Verify log file creation and content.
        *   Verify console output.
    *   Test evaluation of a saved PPO agent checkpoint against another: **Partially covered by `MockPPOAgent` usage; direct checkpoint testing is a good next step.**
    *   The `test_evaluate.py` script **has been significantly updated and is successfully running tests for `run_evaluation_loop`**, including fixes for `MockPPOAgent` compatibility and logging.

## 6. Future Enhancements

*   **Tournament Mode:** Allow evaluation of multiple agents against each other in a round-robin or other tournament format.
*   **More Sophisticated Baselines:** Integrate stronger heuristic-based agents or simple search-based agents (e.g., Minimax with limited depth).
*   **USI Protocol Integration:** Allow evaluation against external Shogi engines that support the USI protocol.
*   **Elo Rating System:** Implement a local Elo rating system to track relative agent strengths more dynamically.
*   **Advanced Analytics:**
    *   Collect and analyze game trajectories (e.g., common opening lines, end-game performance).
    *   Visualize game play or specific board positions of interest.
*   **Web Interface/Dashboard:** A simple web interface to view evaluation results and compare runs.

This plan provides a roadmap for developing a robust evaluation system. Initial implementation will focus on core functionality (Section 2.1-2.4, 3.1-3.2) with a simple baseline opponent.
