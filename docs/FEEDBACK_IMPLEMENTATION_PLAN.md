# Feedback Implementation Plan

This document outlines the plan to address the feedback received on May 21, 2025. The goal is to improve the DRL Shogi client's robustness, maintainability, performance, and clarity.

## 1. Training Loop (`train.py`)

### 1.1. Dead / Redundant Code

*   **Issue**: Unused `sys` import, commented-out `obs_for_agent` logic, and redundant `print()` calls alongside logging.
*   **Actions**:
    1.  Remove the `import sys` statement if it's confirmed to be unused or if its functionality (e.g., `sys.stderr.write`) can be integrated into `TrainingLogger`.
    2.  Delete the commented-out `obs_for_agent` block.
    3.  Modify `TrainingLogger` to optionally echo logs to `stdout`.
    4.  Replace all `print()` calls within the logger's context in `train.py` with logger calls, configuring the logger to also print to console if desired.
*   **Files**: `train.py`, `keisei/utils.py` (for `TrainingLogger` modification).
*   **Priority**: High (quick wins for clarity).

### 1.2. `done` Logic & Move Execution

*   **Issue**: `done` flag not reset per step; potentially adding junk transitions to the experience buffer for terminal states.
*   **Actions**:
    1.  Ensure `done` is explicitly reset to `False` at the beginning of each step/iteration in the main training loop *before* an action is selected.
    2.  Modify the logic to *not* add an entry to the `ExperienceBuffer` if `selected_shogi_move is None` (i.e., no legal move was found, indicating a terminal state or an error).
    3.  Verify that GAE calculation correctly handles the end of trajectories, especially if terminal states are not added.
*   **Files**: `train.py`, `keisei/experience_buffer.py` (if modification for GAE is needed, though likely handled in `ppo_agent.py`).
*   **Priority**: High (correctness of training data).

### 1.3. Reward Handling

*   **Issue**: Ambiguity around sparse rewards and handling of draws (same reward as loss).
*   **Actions**:
    1.  Add a comment in `train.py` near the reward assignment logic to explicitly state that sparse rewards (only at terminal states) are intentional.
    2.  Evaluate if distinguishing draws from losses with different reward signals is beneficial for the agent's learning. If so, implement a different reward for draws (e.g., a small positive or less negative value than a loss). If not, add a comment explaining why `reward = 0.0` is used for both.
*   **Files**: `train.py`.
*   **Priority**: Medium (clarification and potential minor design adjustment).

### 1.4. Checkpoint Auto-Detection

*   **Issue**: Fragile checkpoint filename parsing and potential misidentification of malformed files as "latest."
*   **Actions**:
    1.  In `train.py`, replace the string splitting logic in `find_latest_checkpoint` (and its helper `extract_ts`) with a regular expression to robustly parse episode and timestep numbers (e.g., `r"episode_(\d+).*ts_(\d+)"`).
    2.  Ensure that `find_latest_checkpoint` filters out or raises an error for filenames that do not match the expected pattern, preventing them from being considered.
*   **Files**: `train.py`.
*   **Priority**: Medium (robustness of checkpointing).

### 1.5. Progress Bar & Logging

*   **Issue**: `tqdm` bar may pause during saves/updates; suggestion to decouple.
*   **Actions**:
    1.  For now, accept the current `tqdm` behavior as a minor UX issue.
    2.  Consider long-term refactoring: Introduce a callback system where saving and learning updates can be registered as callbacks triggered at specific frequencies, potentially running in a separate thread or asynchronously to avoid blocking the main loop's progress bar update. (This is a larger architectural change).
*   **Files**: `train.py`.
*   **Priority**: Low (UX improvement, larger refactor for decoupling).

## 2. PPOAgent & ExperienceBuffer

### 2.1. Agent’s `select_action`

*   **Issue**: Agent might sample from uniform distribution if no legal moves, hiding upstream bugs.
*   **Actions**:
    1.  Modify the training loop in `train.py` to explicitly check if there are any legal moves *before* calling `agent.select_action()`.
    2.  If no legal moves exist (terminal state), bypass the agent call and proceed with game termination logic.
*   **Files**: `train.py`, `keisei/ppo_agent.py` (to ensure `select_action` or `ActorCritic.get_action_and_value` handles an all-false mask gracefully if called, though the primary fix is in `train.py`).
*   **Priority**: High (prevents potential errors and improves debuggability).

### 2.2. Buffer Length vs Steps Per Epoch

*   **Issue**: If terminal frames are not added to the buffer, `len(experience_buffer)` might not reach `cfg.STEPS_PER_EPOCH` as expected. Memory usage with list appends.
*   **Actions**:
    1.  After implementing 1.2.2 (not adding terminal frames), re-evaluate the condition for triggering PPO updates. Ensure it's robust to variable episode lengths. One option is to trigger learning after a certain number of *interactions* or *completed episodes* rather than strictly buffer length if that becomes an issue.
    2.  The current buffer clearing (`self.obs_buf.clear()`, etc.) should be sufficient for Python's list GC. Monitor memory usage during long runs; if it becomes an issue, consider pre-allocating NumPy arrays for the buffer and using a circular buffer pattern. For now, assume current list clearing is adequate.
*   **Files**: `train.py`, `keisei/experience_buffer.py`.
*   **Priority**: Medium (ensure learning triggers correctly after other changes).

### 2.3. Learning Loop (`PPOAgent.learn`)

*   **Issue**: GAE computed in Python lists; entropy calculated over all actions, not just legal ones.
*   **Actions**:
    1.  Refactor GAE computation in `PPOAgent.learn()` to use PyTorch tensor operations for potential performance benefits, especially if GPU is used.
    2.  Modify `ActorCritic.evaluate_actions` (or where entropy is calculated) to accept the `legal_actions_mask`.
    3.  When computing the entropy bonus in `PPOAgent.learn()`, use the `legal_actions_mask` to ensure entropy is calculated only over legal actions. This involves masking the logits before the softmax and entropy calculation, or applying the mask to the probabilities.
*   **Files**: `keisei/ppo_agent.py`, `keisei/neural_network.py` (for `ActorCritic`).
*   **Priority**: Medium (correctness of entropy regularization, potential performance gain).

## 3. Shogi Game & Rule Modules

### 3.1. Dead / Duplicitous Code

*   **Issue**: Commented-out example code in move application logic; unused reserved channels in NN observation.
*   **Actions**:
    1.  Review `keisei/shogi/shogi_move_execution.py` (and related files like `shogi_game.py`) and remove all commented-out example code related to move application. Consolidate move logic if necessary.
    2.  In `keisei/shogi/shogi_game_io.py` (`generate_neural_network_observation`):
        *   Add a more detailed comment explaining *why* channels 44 and 45 are reserved (e.g., "Reserved for future features like repetition count or specific game phase indicators").
        *   If there's no concrete plan for them, consider reducing the channel count from 46 to 44 and update the docstring accordingly.
*   **Files**: `keisei/shogi/shogi_move_execution.py`, `keisei/shogi/shogi_game.py`, `keisei/shogi/shogi_game_io.py`.
*   **Priority**: Medium (code clarity and minor optimization).

### 3.2. `is_uchi_fu_zume` & `can_drop_specific_piece` Performance

*   **Issue**: Extremely expensive deep copies and simulations for pawn-drop-mate checks.
*   **Actions**:
    1.  **Short-term**: Profile the `generate_all_legal_moves` function, specifically focusing on calls to `check_for_uchi_fu_zume` and `can_drop_specific_piece`.
    2.  **Medium-term**: Investigate less computationally intensive ways to check for *uchi-fu-zume*. This is a complex Shogi rule. Options might include:
        *   Optimizing the check: Instead of a full deep copy, can a more lightweight board state update/revert be used for the simulation?
        *   Heuristics: Are there common patterns that can quickly rule out *uchi-fu-zume* without simulation?
        *   Caching: Can results of these checks be cached if game states (or relevant parts) repeat? (Less likely to be effective here).
    3.  This is a significant algorithmic challenge. Initial focus should be on confirming the bottleneck, then exploring Shogi programming resources for efficient *uchi-fu-zume* detection algorithms.
*   **Files**: `keisei/shogi/shogi_rules_logic.py`.
*   **Priority**: High (major performance bottleneck).

### 3.3. SFEN Serialization / Deserialization

*   **Issue**: Brittle SFEN parsing logic due to manual string manipulation.
*   **Actions**:
    1.  In `keisei/shogi/shogi_game_io.py` (`sfen_to_move_tuple` and helpers):
        *   Evaluate replacing the current parsing logic with regular expressions for improved robustness and readability. For example, `P*5e` could be `([PLNSGBR])\*([1-9][a-i])` and `7g7f+` could be `([1-9][a-i])([1-9][a-i])(\+)?`.
        *   Alternatively, search for a well-tested Python library for SFEN parsing. If a suitable lightweight library exists, consider adding it as a dependency.
*   **Files**: `keisei/shogi/shogi_game_io.py`.
*   **Priority**: Medium (robustness and maintainability).

## 4. General Code Smell / Maintainability

### 4.1. Single-Responsibility Principle

*   **Issue**: `train.py` has too many responsibilities.
*   **Actions**:
    1.  Refactor `train.py`:
        *   Move CLI argument parsing (`parse_args`, `apply_config_overrides`) into a separate `cli.py` module or a dedicated `CLIConfig` class.
        *   Encapsulate the main training loop and its associated logic (checkpointing, agent interaction, learning triggers) into a `Trainer` class, possibly in a new `trainer.py` module.
        *   `train.py` would then become a thin script that instantiates the config/CLI handler and the `Trainer`, then starts the training.
*   **Files**: `train.py`, create `keisei/cli.py` (or similar), create `keisei/trainer.py` (or similar).
*   **Priority**: Medium (improves structure and testability).

### 4.2. Error Handling

*   **Issue**: Tendency to print-and-continue instead of failing fast.
*   **Actions**:
    1.  Review critical sections of the code (e.g., move validation, agent action selection, PPO updates).
    2.  Replace "print-and-continue" error handling with specific, informative exceptions where appropriate (e.g., `IllegalMoveError`, `AgentPolicyError`).
    3.  Ensure that unrecoverable errors cause the program to terminate rather than continue in an undefined state.
*   **Files**: Across the codebase, particularly `train.py`, `keisei/ppo_agent.py`, `keisei/shogi/`.
*   **Priority**: Medium (robustness).

### 4.3. Type Hints

*   **Issue**: Spotty type hints, use of `Any`.
*   **Actions**:
    1.  Perform a pass over the codebase to improve type hint coverage.
    2.  Replace `Any` with more specific types where possible.
    3.  Use `typing.TYPE_CHECKING` for circular dependencies in type hints.
    4.  Set up `mypy` with a stricter configuration (e.g., `disallow_untyped_defs = True` is already in `pyproject.toml`, ensure it's enforced) and run it regularly as part of development/CI.
*   **Files**: Across the codebase.
*   **Priority**: Medium (improves maintainability and catches bugs).

### 4.4. Comment-Heavy Sections

*   **Issue**: Commented-out example code adds cognitive overhead.
*   **Actions**:
    1.  Remove commented-out blocks of code that are not TODOs or placeholders for immediate future work.
    2.  If commented code serves as an example or historical reference, move it to a separate document (e.g., `docs/code_examples.md` or `docs/archived_logic.md`) or delete if obsolete.
    3.  Ensure remaining comments explain *why* something is done, not *what* is done (if the code is self-explanatory).
*   **Files**: Across the codebase, especially in `keisei/shogi/` modules.
*   **Priority**: Low (improves readability).

## Timeline & Next Steps

1.  **Phase 1 (High Priority - Correctness & Clarity):**
    *   1.1 (Dead Code)
    *   1.2 (`done` Logic & Buffer)
    *   2.1 (Agent `select_action` Guard)
    *   3.2 (Profile `uchi-fu-zume` - initial investigation)
2.  **Phase 2 (Medium Priority - Robustness & Refinements):**
    *   1.3 (Reward Handling Clarification)
    *   1.4 (Checkpoint Robustness)
    *   2.2 (Buffer Trigger Logic)
    *   2.3 (PPO GAE & Entropy)
    *   3.1 (Shogi Dead Code & NN Obs)
    *   3.3 (SFEN Parsing)
    *   4.1 (SRP - `train.py` Refactor)
    *   4.2 (Error Handling)
    *   4.3 (Type Hints)
3.  **Phase 3 (Lower Priority - Performance & UX):**
    *   1.5 (Progress Bar Decoupling - if deemed necessary)
    *   3.2 (Implement `uchi-fu-zume` optimization - this could be long)
    *   4.4 (Comment Cleanup)

This plan will be reviewed and updated as implementation progresses.
