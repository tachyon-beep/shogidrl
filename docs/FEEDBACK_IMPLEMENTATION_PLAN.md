# Feedback Implementation Plan

This document outlines the plan to address the feedback received on May 21, 2025. The goal is to improve the DRL Shogi client\'s robustness, maintainability, performance, and clarity.

**Status Key:**
*   **DONE**: Completed
*   **WIP**: Work in Progress
*   **TODO**: Not yet started

## 1. Training Loop (`train.py`)

### 1.1. Dead / Redundant Code - DONE

*   **Issue**: Unused `sys` import, commented-out `obs_for_agent` logic, and redundant `print()` calls alongside logging.
*   **Actions**:
    1.  **DONE** Remove the `import sys` statement.
    2.  **DONE** Delete the commented-out `obs_for_agent` block.
    3.  **DONE** Modify `TrainingLogger` to optionally echo logs to `stdout`.
    4.  **DONE** Replace all `print()` calls within the logger\'s context in `train.py` with logger calls.
*   **Files**: `train.py`, `keisei/utils.py`.
*   **Priority**: High.

### 1.2. `done` Logic & Move Execution - DONE

*   **Issue**: `done` flag not reset per step; potentially adding junk transitions to the experience buffer for terminal states.
*   **Actions**:
    1.  **DONE** Ensure `done` is explicitly reset to `False` at the beginning of each step/iteration.
    2.  **DONE** Modify the logic to *not* add an entry to the `ExperienceBuffer` if `action_idx == -1` (derived from `selected_shogi_move is None`).
    3.  **DONE** Verified GAE calculation handles ends of trajectories correctly (no specific changes needed for this part after 1.2.2).
*   **Files**: `train.py`.
*   **Priority**: High.

### 1.3. Reward Handling - DONE

*   **Issue**: Ambiguity around sparse rewards and handling of draws (same reward as loss).
*   **Actions**:
    1.  **DONE** Add a comment in `train.py` to explicitly state that sparse rewards are intentional.
    2.  **DONE** Added a comment explaining why `reward = 0.0` is used for draws (treated same as loss for now).
*   **Files**: `train.py`.
*   **Priority**: Medium.

### 1.4. Checkpoint Auto-Detection - DONE

*   **Issue**: Fragile checkpoint filename parsing.
*   **Actions**:
    1.  **DONE** In `train.py`, replaced string splitting in `find_latest_checkpoint` with a regular expression.
    2.  **DONE** `find_latest_checkpoint` now filters out filenames not matching the pattern.
*   **Files**: `train.py`.
*   **Priority**: Medium.

### 1.5. Progress Bar & Logging - TODO

*   **Issue**: `tqdm` bar may pause during saves/updates.
*   **Actions**:
    1.  **TODO** Accept current `tqdm` behavior for now.
    2.  **TODO** Consider long-term refactoring (callback system).
*   **Files**: `train.py`.
*   **Priority**: Low.

## 2. PPOAgent & ExperienceBuffer

### 2.1. Agent’s `select_action` Guard - DONE

*   **Issue**: Agent might sample from uniform distribution if no legal moves.
*   **Actions**:
    1.  **DONE** Modified `train.py` to check for legal moves *before* `agent.select_action()`.
    2.  **DONE** If no legal moves, bypass agent call. (Agent\'s internal fallbacks reviewed and deemed sufficient if called with all-false mask, but pre-check is better).
*   **Files**: `train.py`.
*   **Priority**: High.

### 2.2. Buffer Length vs Steps Per Epoch - DONE

*   **Issue**: `len(experience_buffer)` might not reach `cfg.STEPS_PER_EPOCH`.
*   **Actions**:
    1.  **DONE** Added comment to `train.py` clarifying PPO update trigger is based on buffer length reaching `STEPS_PER_EPOCH` after interactions, which is robust to variable episode lengths. No code change needed for the trigger logic itself.
    2.  **DONE** Current list clearing in buffer is deemed adequate for now.
*   **Files**: `train.py`.
*   **Priority**: Medium.

### 2.3. Learning Loop (`PPOAgent.learn`) - DONE

*   **Issue**: GAE in Python lists; entropy over all actions.
*   **Actions**:
    1.  **DONE** Refactored GAE computation in `ExperienceBuffer` to use PyTorch tensor operations.
    2.  **DONE** Modified `ActorCritic.evaluate_actions` to accept `legal_actions_mask` (comment updated).
    3.  **DONE** `PPOAgent.learn` now uses `legal_masks` from buffer for entropy and KL divergence. `ExperienceBuffer` and `train.py` updated to handle `legal_mask`.
*   **Files**: `keisei/ppo_agent.py`, `keisei/neural_network.py`, `keisei/experience_buffer.py`, `train.py`.
*   **Priority**: Medium.

## 3. Shogi Game & Rule Modules

### 3.1. Dead / Duplicitous Code - DONE

*   **Issue**: Commented-out example code in move application logic; clarity of reserved channels in NN observation.
*   **Actions**:
    1.  **DONE** Review `keisei/shogi/shogi_move_execution.py` (and related files like `shogi_game.py`) and remove all commented-out example code related to move application. Consolidate move logic if necessary. (Fixes to `apply_move_to_board` and `undo_move` logic implemented).
    2.  In `keisei/shogi/shogi_game_io.py` (`generate_neural_network_observation`):
        *   **DONE** Add a more detailed comment explaining *why* channels 44 and 45 are reserved.
        *   **DONE** Kept channel count at 46 as per original spec, updated docstring.
    3.  **DONE** Added docstring to `ShogiGame.get_observation()`.
*   **Files**: `keisei/shogi/shogi_move_execution.py`, `keisei/shogi/shogi_game.py`, `keisei/shogi/shogi_game_io.py`.
*   **Priority**: Medium.

### 3.2. `is_uchi_fu_zume` & `can_drop_specific_piece` Performance - WIP

*   **Issue**: Expensive deep copies for pawn-drop-mate checks.
*   **Actions**:
    1.  **DONE** Profile `generate_all_legal_moves`.
    2.  **WIP** Investigate less computationally intensive checks.
        *   **DONE** Refactored `check_for_uchi_fu_zume` to remove `deepcopy` by simulating pawn drops and manually reverting.
        *   **DONE** Refactored `generate_all_legal_moves` to use a make/undo approach for board moves and drop moves, eliminating `deepcopy` for move validation.
        *   **DONE** Successfully ran profiling script after refactoring, confirming significant reduction in `deepcopy` calls and overall performance improvement for `generate_all_legal_moves`.
*   **Files**: `keisei/shogi/shogi_rules_logic.py`, `keisei/shogi/shogi_game.py`, `keisei/shogi/shogi_move_execution.py`.
*   **Priority**: High.

### 3.3. SFEN Serialization / Deserialization - TODO

*   **Issue**: Brittle SFEN parsing.
*   **Actions**:
    1.  **TODO** Evaluate replacing current parsing with regex or library.
*   **Files**: `keisei/shogi/shogi_game_io.py`.
*   **Priority**: Medium.

## 4. General Code Smell / Maintainability

### 4.1. Single-Responsibility Principle - TODO

*   **Issue**: `train.py` has too many responsibilities.
*   **Actions**:
    1.  **TODO** Refactor `train.py` (CLI, Trainer class).
*   **Files**: `train.py`, create `keisei/cli.py`, `keisei/trainer.py`.
*   **Priority**: Medium.

### 4.2. Error Handling - TODO

*   **Issue**: Print-and-continue instead of fail-fast.
*   **Actions**:
    1.  **TODO** Review critical sections, replace with exceptions.
*   **Files**: Across codebase.
*   **Priority**: Medium.

### 4.3. Type Hints - WIP

*   **Issue**: Spotty type hints, use of `Any`.
*   **Actions**:
    1.  **WIP** Ongoing improvements to type hint coverage as files are touched.
    2.  **WIP** Replacing `Any` where feasible.
    3.  **DONE** Used `TYPE_CHECKING` where necessary (e.g. `shogi_move_execution.py`).
    4.  **TODO** Ensure `mypy` is run with strict config.
*   **Files**: Across the codebase.
*   **Priority**: Medium.

### 4.4. Comment-Heavy Sections - WIP

*   **Issue**: Commented-out example code.
*   **Actions**:
    1.  **WIP** Removing commented-out blocks as files are reviewed (e.g. `shogi_move_execution.py`).
    2.  **TODO** Move relevant historical/example code to docs if needed.
    3.  **WIP** Ensuring comments explain *why*.
*   **Files**: Across the codebase.
*   **Priority**: Low.

## Timeline & Next Steps

1.  **Phase 1 (High Priority - Correctness & Clarity):**
    *   **DONE** 1.1 (Dead Code)
    *   **DONE** 1.2 (`done` Logic & Buffer)
    *   **DONE** 2.1 (Agent `select_action` Guard)
    *   **WIP** 3.2 (Profile `uchi-fu-zume` - initial investigation, refactoring complete, performance confirmed)
2.  **Phase 2 (Medium Priority - Robustness & Refinements):**
    *   **DONE** 1.3 (Reward Handling Clarification)
    *   **DONE** 1.4 (Checkpoint Robustness)
    *   **DONE** 2.2 (Buffer Trigger Logic)
    *   **DONE** 2.3 (PPO GAE & Entropy)
    *   **DONE** 3.1 (Shogi Dead Code & NN Obs)
    *   **TODO** 3.3 (SFEN Parsing)
    *   **TODO** 4.1 (SRP - `train.py` Refactor)
    *   **TODO** 4.2 (Error Handling)
    *   **WIP** 4.3 (Type Hints)
3.  **Phase 3 (Lower Priority - Performance & UX):**
    *   **TODO** 1.5 (Progress Bar Decoupling - if deemed necessary)
    *   **DONE** 3.2 (Implement `uchi-fu-zume` optimization - this could be long)
    *   **WIP** 4.4 (Comment Cleanup)

This plan will be reviewed and updated as implementation progresses.
