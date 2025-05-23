# PPO Shogi Agent Evaluation System Plan

**Date:** May 23, 2025
**Version:** 1.5

**Changelog:**
*   **v1.5 (May 23, 2025):**
    *   Added "Current Status" section detailing completed implementation tasks.
    *   Reflected completion of W&B integration in `evaluate.py`.
    *   Reflected completion of periodic evaluation integration in `train.py`.
    *   Reflected completion of comprehensive CLI testing for `evaluate.main()` in `tests/test_evaluate.py`.
    *   Reflected completion of tests for periodic evaluation in `tests/test_train.py`.
    *   Updated `config.py` with all necessary parameters for periodic evaluation and W&B.
*   **v1.4 (May 23, 2025):**
    *   Detailed planning for comprehensive CLI testing of `evaluate.main()`.
    *   Added specific testing strategies for PPO checkpoint vs. PPO checkpoint evaluations.
    *   Integrated detailed plan for Weights & Biases (W&B) re-implementation.
    *   Outlined plan for integrating periodic evaluation into `train.py`, including `config.py` changes.
    *   Updated relevant sections (Implementation Plan, Testing Strategy, Logging, Configuration, Workflow) to reflect these next development activities.
*   **v1.3 (May 23, 2025):** Added Weights & Biases integration to plan (as a future item initially). Substantial completion of `evaluate.py` and `test_evaluate.py` acknowledged.

## 1. Introduction and Objectives

This document outlines the plan for implementing an evaluation system for the PPO-based Shogi agent. The primary goal is to systematically measure the performance of trained agents, compare different model versions, and track improvements over time.

**Key Objectives:**

*   **Performance Measurement:** Quantify the playing strength of the agent.
*   **Model Comparison:** Objectively compare different iterations of the agent (e.g., different hyperparameters, network architectures, or checkpoints from a single training run).
*   **Progress Tracking:** Monitor improvements during training by periodically evaluating against a consistent benchmark.
*   **Identify Weaknesses:** (Potentially) Gather data that can help identify systematic weaknesses in the agent's play.
*   **Reproducibility:** Ensure that evaluations can be run consistently.

## 1.A. Current Status (As of May 23, 2025)

**The PPO Shogi Agent Evaluation System, as outlined in this plan, has been substantially implemented.**

**Key Accomplishments:**

*   **`evaluate.py` (Core Evaluation Script):**
    *   **Functionality:** Fully implemented, supporting evaluation of a PPO agent checkpoint against Random, Heuristic, or another PPO agent checkpoint.
    *   **Command-Line Interface:** Robust CLI implemented with `argparse` for all planned configurations (agent/opponent paths, game numbers, device, seeds, etc.).
    *   **W&B Integration:**
        *   Successfully integrated Weights & Biases logging.
        *   CLI arguments (`--wandb-log`, `--wandb-project`, etc.) control W&B initialization.
        *   Configuration and summary metrics are logged to W&B.
        *   `wandb.finish()` is correctly called.
    *   **Logging:** `EvaluationLogger` implemented and functional, creating detailed per-run log files and console summaries.
    *   **Agent Loading:** `load_evaluation_agent` and `initialize_opponent` functions are working as specified.
    *   **Core Loop:** `run_evaluation_loop` correctly manages game execution, metric calculation, and logging.
    *   **Miscellaneous:** Necessary imports (`os`, `numpy`) added, and `numpy.random` is seeded for reproducibility.

*   **`train.py` (Periodic Evaluation Integration):**
    *   **Mechanism:** Successfully integrated periodic evaluation by calling `evaluate.py` as a subprocess using `subprocess.run()`.
    *   **Trigger:** Evaluation is triggered after each `agent.save_model()` call, controlled by `config.EVAL_DURING_TRAINING`.
    *   **Parameterization:** Correctly passes parameters (checkpoint path, opponent config, W&B settings) from `config.py` to the `evaluate.py` subprocess.
    *   **Results Handling:** Captures and logs a summary of evaluation results from the subprocess output.
    *   **Error Handling:** Implemented basic error handling for the evaluation subprocess.

*   **`config.py` (Configuration Settings):**
    *   All planned parameters for periodic evaluation have been added:
        *   `EVAL_DURING_TRAINING`
        *   `EVAL_NUM_GAMES`
        *   `EVAL_OPPONENT_TYPE`
        *   `EVAL_OPPONENT_CHECKPOINT_PATH`
        *   `EVAL_MAX_MOVES_PER_GAME`
        *   `EVAL_DEVICE`
    *   All planned parameters for W&B logging during periodic evaluation have been added:
        *   `EVAL_WANDB_LOG`
        *   `EVAL_WANDB_PROJECT`
        *   `EVAL_WANDB_ENTITY`
        *   `EVAL_WANDB_RUN_NAME_PREFIX`

*   **`tests/test_evaluate.py` (Testing for `evaluate.py`):**
    *   **Comprehensive CLI Testing:**
        *   Extensive tests for `evaluate.main()` covering valid and invalid argument combinations.
        *   Thorough testing of W&B argument handling (using mocks).
        *   Verification of `stdout`/`stderr` and log file content.
    *   **PPO vs. PPO Testing:** Tests specifically verifying evaluation between two PPO agent checkpoints.
    *   **Core Logic:** Tests for `run_evaluation_loop` and associated helper functions are in place.
    *   **Mocking & Refactoring:** Effective use of `unittest.mock` and test refactoring with helper utilities.
    *   **Status:** All functional tests are passing. PYLINT warnings have been addressed (one known false positive related to `wandb.run` PropertyMock remains but is acceptable).

*   **`tests/test_train.py` (Testing for `train.py` Periodic Evaluation):**
    *   **Subprocess Interaction:** Tests verifying correct `subprocess.run` calls to `evaluate.py` with appropriate arguments.
    *   **Configuration Usage:** Tests ensuring `config.py` parameters are correctly utilized for evaluation.
    *   **W&B Parameter Passing:** Verification that W&B related arguments are correctly passed to the evaluation subprocess.
    *   **Error Handling:** Tests for error handling logic if the evaluation subprocess fails.
    *   **Status:** All tests are passing. PYLINT warnings have been addressed (some minor ones ignored as per development decision for this stage).

**Overall, the system is operational and meets the primary objectives outlined for this development phase.** Future enhancements can now be built upon this solid foundation.

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
*   **`--wandb-log`:** (Optional) Boolean flag to enable logging to Weights & Biases. Defaults to False.
*   **`--wandb-project <project_name>`:** (Optional) W&B project name. Defaults to a suitable project name (e.g., "shogi-ppo-evaluation").
*   **`--wandb-entity <entity_name>`:** (Optional) W&B entity (username or team name). Can often be inferred from environment or W&B login.
*   **`--wandb-run-name <run_name>`:** (Optional) Custom name for the W&B run.

### 2.4. Logging and Reporting

*   **Structured Log Files:**
    *   A dedicated log file for each evaluation run, storing:
        *   Configuration used for the run.
        *   Game-by-game results (winner, loser, draw, game length, final board state or SGF/KIF if possible).
        *   Summary statistics (overall W/L/D, average game length, etc.).
*   **Console Output:** Clear summary of results printed to the console upon completion.
*   **Weights & Biases Integration:**
    *   Metrics will be sent to Weights & Biases for visualization and comparison across multiple evaluations.
    *   `wandb.init()` will be called at the start of `evaluate.main()`.
    *   CLI arguments and key configuration details (agent path, opponent type/path, num_games, device, etc.) will be logged to `wandb.config`.
    *   Summary metrics (W/L/D rates, average game length, etc.) will be logged via `wandb.log()`.
    *   `wandb.finish()` will be called at the end of `evaluate.main()`.
*   *(Future)* **Visualization:** Basic plots for win rates over time if evaluations are run periodically.

## 3. Implementation Plan

### 3.1. New Script/Module (e.g., `evaluate.py`)

A new Python script, `/home/john/keisei/evaluate.py`, **has been created and is substantially complete.** This script serves as the main entry point for running evaluations.

**Key Functions within `evaluate.py` (Implemented & Planned Enhancements):**

*   **`main()`:**
    *   Parses command-line arguments (using `argparse`), including new W&B arguments.
    *   **W&B Integration:**
        *   If `--wandb-log` is specified, calls `wandb.init()` with project, entity, run name, and logs CLI arguments to `wandb.config`.
    *   Loads the agent to be evaluated using `load_evaluation_agent()`.
    *   Initializes the opponent agent using `initialize_opponent()`.
    *   Sets up an `EvaluationLogger`.
    *   Calls `run_evaluation_loop()`.
    *   Prints summary metrics to console and logs them.
    *   **W&B Integration:** Logs summary metrics using `wandb.log()` and calls `wandb.finish()` before exiting.
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
    *   Logging of agent names (e.g., `agent_to_eval.name`) is now correctly implemented.
*   **`SimpleRandomOpponent(BaseOpponent)` and `SimpleHeuristicOpponent(BaseOpponent)` classes:**
    *   These classes implement a `select_move(self, game_instance: ShogiGame) -> MoveTuple` method.

### 3.2. Modifications to Existing Code

*   **`keisei/ppo_agent.py` (`PPOAgent`):**
    *   **Added `name` attribute:** Initialized in `__init__`.
    *   **`select_action()` method:** Uses `is_training` to toggle deterministic behavior.
    *   **`load_model()` method:** Handles optimizer state device placement.
*   **`keisei/neural_network.py` (`ActorCritic`):**
    *   **`get_action_and_value()` method:** `deterministic` flag is used.
*   **`keisei/shogi/shogi_game.py` (`ShogiGame`):**
    *   Methods like `reset()`, `make_move()`, `get_legal_moves()`, `game_over`, `winner` are used.
    *   `max_moves_per_game` parameter is utilized.
*   **`keisei/shogi/shogi_game_io.py`:**
    *   **`generate_neural_network_observation()`:** Used by PPO agents.
    *   *(Future)* `game_to_kif()` for `EvaluationLogger`.
*   **`keisei/utils.py`:**
    *   **`PolicyOutputMapper`:** Methods `get_legal_mask()` and `policy_index_to_shogi_move()` are used.
    *   **New `EvaluationLogger` class:** Implemented and used in `evaluate.py`.
    *   **New `BaseOpponent(ABC)` abstract base class:** Implemented and used.
*   **`keisei/train.py` (Periodic Evaluation Integration):**
    *   **Integration Mechanism:** `subprocess.run()` will be used to call `evaluate.py`. This is chosen for simplicity and to decouple the training and evaluation environments/dependencies if they were to diverge. It also allows evaluation to run as a separate process, minimizing impact on training memory.
    *   **Hook Placement:** The evaluation hook will be placed after each `agent.save_model()` call. This ensures that every saved model checkpoint is evaluated.
    *   **Parameter Passing:**
        *   The path to the just-saved checkpoint will be passed as `--agent-checkpoint`.
        *   Opponent configuration (e.g., `--opponent-type`, `--opponent-checkpoint`) will be configurable via new settings in `config.py` (see below) and passed to `evaluate.py`.
        *   Number of evaluation games (`--num-games`), device (`--device`), and other relevant parameters will also be sourced from `config.py`.
        *   W&B parameters for the evaluation run can also be configured in `config.py` (e.g., `EVAL_WANDB_LOG`, `EVAL_WANDB_PROJECT`).
    *   **Results Handling by `train.py`:**
        *   `train.py` will capture the standard output of `evaluate.py` (which includes the summary).
        *   A summary of the evaluation results (e.g., "Evaluation vs <opponent>: Win Rate X%, Avg Game Length Y") will be logged to the main training log and printed to the console.
        *   *(Future Consideration)* Store the primary evaluation metric (e.g., win rate against a standard opponent) and use it to save a separate "best_eval_model.pth" if the current model surpasses the previous best.
    *   **Error Handling:**
        *   `train.py` will check the return code of the `subprocess.run()` call.
        *   If the evaluation subprocess fails, a warning will be logged, but training will continue by default to prevent interruption of long training runs. An option could be added to `config.py` to halt training on evaluation failure if desired.
*   **`config.py` (New Settings for Periodic Evaluation):**
    *   `EVAL_DURING_TRAINING`: Boolean, to enable/disable periodic evaluation.
    *   `EVAL_NUM_GAMES`: Integer, number of games for periodic evaluation.
    *   `EVAL_OPPONENT_TYPE`: String (e.g., "random", "heuristic", "ppo").
    *   `EVAL_OPPONENT_CHECKPOINT_PATH`: String, path to opponent PPO checkpoint if `EVAL_OPPONENT_TYPE` is "ppo".
    *   `EVAL_MAX_MOVES_PER_GAME`: Integer, max moves for evaluation games.
    *   `EVAL_DEVICE`: String ("cpu" or "cuda").
    *   `EVAL_WANDB_LOG`: Boolean, to enable W&B logging for periodic evaluations.
    *   `EVAL_WANDB_PROJECT`: String, W&B project for evaluation runs.
    *   `EVAL_WANDB_ENTITY`: String, W&B entity for evaluation runs.
    *   `EVAL_WANDB_RUN_NAME_PREFIX`: String, a prefix for W&B run names for periodic evaluations (e.g., "eval_run_"). `train.py` can append timestamp or step count.

### 3.3. Checkpoint Management

*   **DONE:** The evaluation script takes a direct path to a specific agent checkpoint.
*   For periodic evaluation during training, the latest saved checkpoint will be used.

## 4. Workflow

1.  **User Invocation (Manual Evaluation):**
    ```bash
    python keisei/evaluate.py --agent-checkpoint <path_to_agent.pth> \
                              --opponent-type <type> \
                              [--opponent-checkpoint <path_to_opponent.pth>] \
                              --num-games <N> \
                              --device <cpu/cuda> \
                              [--max-moves <M>] \
                              [--seed <S>] \
                              [--wandb-log] \
                              [--wandb-project <project>] \
                              [--wandb-entity <entity>] \
                              [--wandb-run-name <name>]
    ```
2.  **Setup:** The script loads the agent, initializes the opponent, sets up logging, and initializes W&B (if enabled).
3.  **Evaluation:** The evaluation loop runs the specified number of games, alternating playing colors.
4.  **Results:**
    *   Game-by-game results are logged to a file.
    *   Final summary statistics are printed to the console and saved to the evaluation log file.
    *   If W&B is enabled, configuration and summary metrics are pushed to W&B.
5.  **Automated Periodic Evaluation Workflow (During Training):**
    *   Training proceeds in `train.py`.
    *   After `agent.save_model()` (at configured intervals):
        *   `train.py` constructs the command for `evaluate.py` using parameters from `config.py` and the path to the newly saved checkpoint.
        *   `train.py` executes `evaluate.py` as a subprocess.
        *   `evaluate.py` runs as described in steps 2-4 above (manual evaluation), potentially logging to W&B under a distinct run name.
        *   `train.py` captures and logs a summary of the evaluation results from the subprocess output. Training continues.

## 5. Testing Strategy

*   **Unit Tests (new tests in `tests/test_evaluate.py`):**
    *   Test `evaluate.load_evaluation_agent`: Largely covered by integration tests.
    *   Test `evaluate.initialize_opponent`: Largely covered by integration tests.
    *   Test `SimpleRandomOpponent.select_move` and `SimpleHeuristicOpponent.select_move`: Implicitly tested via `run_evaluation_loop` tests.
    *   Test metric calculation logic within `run_evaluation_loop`: Covered by `test_run_evaluation_loop_basic` and other tests in `test_evaluate.py`.
    *   Test `EvaluationLogger` methods: Covered by tests ensuring log output is correct.
*   **Integration Tests (expanding `tests/test_evaluate.py`):**
    *   **Comprehensive CLI Testing for `evaluate.main()`:**
        *   **Objective:** Ensure full robustness and correctness of the command-line interface for `evaluate.py`.
        *   **Test Cases:**
            *   Valid combinations:
                *   Agent vs. Random Opponent.
                *   Agent vs. Heuristic Opponent.
                *   Agent vs. PPO Opponent (requiring `--opponent-checkpoint`).
                *   Variations in `--num-games`, `--max-moves-per-game`, `--device`.
                *   With and without `--wandb-log` and associated W&B arguments (mocking W&B calls).
            *   Invalid/Missing arguments:
                *   Missing `--agent-checkpoint`.
                *   `--opponent-type ppo` without `--opponent-checkpoint`.
                *   Invalid values for `num_games` (e.g., zero, negative).
                *   Invalid `device` string.
            *   **Assertions:**
                *   Verify correct exit codes for valid and invalid invocations.
                *   Mock `argparse` and verify that `evaluate.main()` calls downstream functions (like `run_evaluation_loop`, `wandb.init`) with correctly parsed arguments.
                *   Capture `stdout` and `stderr` to verify console output (e.g., summary metrics, error messages).
                *   For successful runs, check the content and structure of the generated evaluation log file (e.g., using `tmp_path` fixture).
    *   **Direct PPO Checkpoint vs. PPO Checkpoint Evaluation Testing:**
        *   **Objective:** Verify the system's ability to correctly load and evaluate two distinct PPO agent checkpoints against each other.
        *   **Checkpoint Strategy:**
            *   For testing, create two distinct mock PPO agent checkpoints. This can be done by:
                1.  Instantiating `MockPPOAgent` (or a minimal `PPOAgent` with a simple `ActorCritic` model).
                2.  Saving its state dict using `torch.save(agent.model.state_dict(), path_to_checkpoint_A)`.
                3.  Optionally, modify the model slightly (e.g., change a weight value if possible without retraining) or use a different seed/initialization for a second agent and save its state dict to `path_to_checkpoint_B`.
                *   Alternatively, if actual minimally trained PPO agents are available, use paths to those.
        *   **Test Cases in `tests/test_evaluate.py`:**
            *   A test function that calls `run_evaluation_loop` (or `evaluate.main` via `subprocess` or by patching `sys.argv`) configured with:
                *   `--agent-checkpoint path_to_checkpoint_A`
                *   `--opponent-type ppo`
                *   `--opponent-checkpoint path_to_checkpoint_B`
        *   **Assertions:**
            *   Verify that both agents are loaded correctly (mock `load_evaluation_agent` to check paths).
            *   Ensure the game loop runs without errors.
            *   Check that game outcomes are recorded and metrics are generated (e.g., win/loss/draw counts sum to `num_games`).
            *   Verify that the `EvaluationLogger` logs appropriate messages for a PPO vs. PPO game.
    *   Test the full `evaluate.main()` flow: `test_run_evaluation_loop_basic` in `tests/test_evaluate.py` covers the core loop. Full `main()` CLI testing is detailed above.
    *   The `test_evaluate.py` script has been significantly updated and is successfully running tests for `run_evaluation_loop`.
*   **Testing W&B Integration:**
    *   Use `unittest.mock.patch` to mock `wandb.init`, `wandb.config.update`, `wandb.log`, and `wandb.finish`.
    *   Verify that these functions are called with the expected arguments when `--wandb-log` is used.
    *   Test that they are NOT called if `--wandb-log` is omitted.
*   **Testing Periodic Evaluation Integration (in `tests/test_train.py` - new file or existing if appropriate):**
    *   Mock `subprocess.run`.
    *   Verify that `train.py` calls `subprocess.run` with the correct command and arguments for `evaluate.py` at the appropriate times (e.g., after `agent.save_model`).
    *   Test parsing of `config.py` settings related to evaluation.
    *   Verify logging of evaluation summaries by `train.py`.
    *   Test error handling (e.g., if `subprocess.run` indicates evaluation failure).

## 6. Future Enhancements

*   **Tournament Mode:** Allow evaluation of multiple agents against each other in a round-robin or other tournament format.
*   **More Sophisticated Baselines:** Integrate stronger heuristic-based agents (e.g., prioritize captures of higher-value pieces, prefer non-losing moves, simple check evasions) or simple search-based agents (e.g., Minimax with limited depth).
*   **USI Protocol Integration:** Allow evaluation against external Shogi engines that support the USI protocol.
*   **Elo Rating System:** Implement a local Elo rating system to track relative agent strengths more dynamically.
*   **Advanced Analytics:**
    *   Collect and analyze game trajectories (e.g., common opening lines, end-game performance).
    *   Visualize game play or specific board positions of interest.
*   **Web Interface/Dashboard:** A simple web interface to view evaluation results and compare runs.
*   **Save "Best Model" based on Evaluation:** During training, if periodic evaluation is enabled, save a copy of the agent checkpoint that achieves the best performance against a standard benchmark opponent.

This plan provides a roadmap for developing a robust evaluation system. Initial implementation will focus on core functionality (Section 2.1-2.4, 3.1-3.2) with a simple baseline opponent.
