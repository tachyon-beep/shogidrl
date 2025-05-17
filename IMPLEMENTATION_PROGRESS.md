# DRL Shogi Client: Implementation Progress Tracker

This file tracks the progress of the implementation plan as described in IMPLEMENTATION_PLAN.md.

## Phase 1: Core Game Mechanics (`shogi_engine.py`)
- [x] Implement `Piece` class
- [x] Unit tests for `Piece`
- [x] Implement `ShogiGame` class: board setup, reset, get/set piece, to_string
- [x] Unit tests for `ShogiGame` init/reset, to_string
- [x] Implement `_is_on_board` method
- [x] Unit tests for `_is_on_board`
- [x] Implement `_get_individual_piece_moves` (basic piece movement logic)
- [x] Unit tests for `_get_individual_piece_moves`
- [x] Implement logic for lance and knight (including promoted) in _get_individual_piece_moves
- [x] Unit tests for lance and knight moves
- [x] Implement logic for silver and gold (including promoted) in _get_individual_piece_moves
- [x] Unit tests for silver and gold moves
- [x] Implement logic for bishop and rook (including promoted) in _get_individual_piece_moves
- [x] Unit tests for bishop and rook moves
- [x] Implement move logic, state management, and other foundational methods (`get_legal_moves`, `make_move`, `undo_move`, `_is_in_check`)
- [x] Unit tests for move logic and state management

## Phase 2: State Representation & Action Mapping
- [x] Implement `get_observation()` in `ShogiGame`
- [x] Unit test for `get_observation()`
- [x] Implement minimal `PolicyOutputMapper` in `utils.py` (single pawn move mapping)
- [x] Unit test for minimal `PolicyOutputMapper`
- [x] Fix and verify all tests for minimal mapping
- [x] Expand `PolicyOutputMapper` to cover more move types and drops
- [x] Add/expand unit tests for new mappings

## Phase 3: RL Loop Structure & Random Agent
- [x] Implement `ActorCritic` in `neural_network.py` (initial dummy forward pass)
- [x] Implement `PPOAgent` in `ppo_agent.py` (random action selection initially)
- [x] Implement `ExperienceBuffer` in `experience_buffer.py`
- [x] Implement `train.py` main loop (random agent, no learning)
- [x] Unit tests for all new classes and methods
- [x] Expand `ActorCritic` in `neural_network.py` to support PPO (policy/value loss, optimizer step)
- [x] Expand `PPOAgent` in `ppo_agent.py` to support PPO (update, GAE, etc.)
- [x] Add/expand unit tests for PPO logic

## Phase 4: Full PPO Implementation
- [x] Complete `ActorCritic` and `PPOAgent` with full PPO logic
- [x] Integrate GAE and learning steps
- [x] Unit tests for RL logic

## Phase 5: Advanced Shogi Rules
- [x] Implement Nifu (pawn drop on file with another unpromoted pawn)
- [x] Implement Uchi Fu Zume (pawn drop mate illegal)
- [ ] Implement Sennichite (fourfold repetition)
- [ ] Add model saving/loading
- [ ] Add logging and evaluation
- [ ] Tune hyperparameters and refine architecture as needed
- [ ] Expand tests for edge cases and advanced rules

---

**Legend:**
- [x] Complete
- [ ] Not started/in progress

This file should be updated after each significant implementation or test addition.
