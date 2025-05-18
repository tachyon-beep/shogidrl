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

## Phase 5: Advanced Rules & Engine Completion
- **I. Piece Drops Implementation (`shogi_engine.py`)**
  - [ ] Manage Pieces in Hand (initialize `self.hands`, update `make_move`/`undo_move` for captures)
  - [ ] Generate Legal Drop Moves (in `get_legal_moves`, handle `(None, None, r_to, c_to, piece_type, is_drop=True)`)
  - [ ] Implement Legality Checks for Drops (Nifu, Illegal Drop Squares for P/L/N, Uchi Fu Zume integration)
  - [ ] Execute Drop Moves (in `make_move`, remove from hand, place on board)
  - [ ] Update `get_observation` to include hand pieces representation
- **II. Promotions Implementation (`shogi_engine.py`)**
  - [ ] Define Promotion Zones
  - [ ] Generate Promotion Moves (in `get_legal_moves`, handle optional and forced promotions)
  - [ ] Execute Promotion Moves (in `make_move`, update piece state if `promote=True`)
  - [ ] Update `undo_move` to correctly revert promotions
- **III. Advanced Legality and Game State (`shogi_engine.py`)**
  - [ ] Implement `is_nifu` detection logic
  - [ ] Integrate `is_nifu` into drop move legality checks
  - [ ] Implement `is_uchi_fu_zume` detection logic
  - [ ] Integrate `is_uchi_fu_zume` into pawn drop move legality checks
  - [ ] Implement `detect_sennichite` logic
  - [ ] Integrate `sennichite` into game over condition and repetition history
  - [ ] Robust Checkmate Detection (`is_checkmate`)
  - [ ] Stalemate Detection (`is_stalemate` - no legal moves, not in check)
- **IV. `PolicyOutputMapper` Expansion (`utils.py`)**
  - [ ] Design and implement a comprehensive mapping for all moves (normal, drops, promotions)
  - [ ] Implement `move_to_index`, `index_to_move`, `get_legal_mask`
  - [ ] Update `NUM_ACTIONS_TOTAL` in `config.py`
- **V. Test Coverage for Engine Completion**
  - [ ] Ensure all existing advanced rule tests in `test_shogi_engine.py` pass
  - [ ] Add new unit tests for drop logic (all piece types, valid/invalid squares, hand updates)
  - [ ] Add new unit tests for promotion logic (optional/forced, all promotable pieces, zone checks)
  - [ ] Add new unit tests for `get_observation` with hand pieces
  - [ ] Add new unit tests for `undo_move` with drops and promotions
- [x] Add model saving/loading
- [x] Add logging and evaluation
- [ ] Tune hyperparameters and refine architecture as needed
- [ ] Expand tests for edge cases and advanced rules (ensure all planned items are covered and pass)

---

**Legend:**
- [x] Complete
- [ ] Not started/in progress

This file should be updated after each significant implementation or test addition.
