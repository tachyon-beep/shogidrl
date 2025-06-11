\
<!-- filepath: /home/john/keisei/docs/SHOGI_GAME_REFACTORING_PLAN.md -->
# ShogiGame (`shogi_game.py`) Refactoring Plan

**Last Updated:** June 11, 2025

## 1. Introduction & Goals

The `keisei/shogi/shogi_game.py` file, primarily containing the `ShogiGame` class, has grown significantly. While it benefits from some delegation, it still handles a broad range of responsibilities. This refactoring aims to enhance modularity, improve maintainability, and adhere to the Single Responsibility Principle (SRP).

**Goals of this Refactoring:**

*   **Enhance Modularity:** Further separate concerns, making each module and class more focused.
*   **Improve Maintainability:** Reduce the complexity of `ShogiGame`, making it easier to understand, modify, and debug.
*   **Adhere to Single Responsibility Principle (SRP):** Ensure `ShogiGame` primarily acts as a game state manager and orchestrator, delegating specialized tasks.
*   **Leverage Existing Modules:** Maximize the use of existing specialized modules within the `keisei/shogi/` package.

## 1.1. Current Status

*   **Phase 1 (SFEN to `shogi_game_io.py`):** The initial part of Phase 1, focusing on moving SFEN (Shogi Forsyth-Edwards Notation) parsing and formatting logic from `ShogiGame` to `shogi_game_io.py`, has been **completed**.
*   **Next Focus:** The immediate next step is to complete the remaining part of Phase 1: relocating the core move execution logic (Section 3.2). Recent work (as of June 11, 2025) has included significant improvements and refactoring of tests for `ShogiGame.get_reward()` and the reward component of `ShogiGame.make_move()` (see `tests/test_shogi_game_rewards.py`), and the `MoveApplicationResult` dataclass (Section 3.2.1) has been defined. These steps ensure key aspects are robust before further refactoring of `make_move`.
*   Following that, Phase 2 (centralized game termination) and Phase 3 (further simplification) will be addressed.

## 2. Guiding Principles

1.  **`ShogiGame` as Orchestrator:** The `ShogiGame` class should primarily hold the game state (board, hands, current player, move history, game status) and orchestrate operations by calling functions in other, more specialized modules.
2.  **Delegate, Don't Do (Everything):** Complex logic for I/O, detailed move application mechanics, and rule validation should reside in their respective modules.
3.  **Clear Interfaces:** Maintain clear and concise interfaces between `ShogiGame` and the helper modules.

## 3. Phase 1: Relocating Functionality to Existing Modules (Revised)

### 3.1. SFEN Serialization/Deserialization to `shogi_game_io.py` (✅ Completed)

**Summary of Completion:**
*   SFEN-related constants (`_SFEN_BOARD_CHARS`, `SFEN_HAND_PIECE_CANONICAL_ORDER`) moved to `shogi_game_io.py`.
*   SFEN helper methods (`_sfen_sq`, `_get_sfen_board_char`, `_get_sfen_drop_char`, `_parse_sfen_board_piece`) moved to `shogi_game_io.py`.
*   `ShogiGame.to_sfen_string()` (aliased as `to_sfen()`) now delegates to `shogi_game_io.convert_game_to_sfen_string()`.
*   `ShogiGame.sfen_encode_move()` now delegates to `shogi_game_io.encode_move_to_sfen_string()`.
*   `ShogiGame.from_sfen()` now uses helper functions from `shogi_game_io.py` (`parse_sfen_string_components`, `populate_board_from_sfen_segment`, `populate_hands_from_sfen_segment`).
*   Unnecessary imports removed from `shogi_game.py`.

This has reduced `ShogiGame`'s size and complexity, isolating SFEN logic appropriately.

### 3.2. Move Execution Logic to `shogi_move_execution.py` (❗ Pending - Current Priority)

**Current State:** The `ShogiGame.make_move()` method (currently extensive, over 400 lines) directly handles all aspects of applying a move to the board, including piece placement, captures, and promotions.

**Target State:** The detailed mechanics of applying a *validated* move to the board state (updating piece positions, handling captures, promotions) will reside in `shogi_move_execution.py`. `ShogiGame.make_move` will orchestrate this, focusing on game state updates (history, turn, count) and termination checks.

**Specific Changes:**

1.  **Define `MoveApplicationResult` Dataclass (✅ Completed):**
    *   This dataclass has been defined in `keisei/shogi/shogi_core_definitions.py`.
    *   It is intended to capture the direct results of applying a move to the board, such as captured pieces or promotion status.
    *   The typical structure includes fields like `captured_piece_type: Optional[PieceType]` and `was_promotion: bool`.
    *   This dataclass is now available for use by `apply_move_to_board`.

2.  **New function in `shogi_move_execution.py`:**
    *   `apply_move_to_board(board: List[List[Optional[Piece]]], hands: Dict[Color, Dict[PieceType, int]], move: MoveTuple, current_player: Color) -> MoveApplicationResult`
        *   This function will **directly mutate** the `board` and `hands` passed to it.
        *   It will handle all logic related to piece movement on the board, processing captures (updating the opponent's hand count for the captured piece type), and applying promotions if `move.promote` is true.
        *   It does **not** update game history, player turn, move count, or check for game termination. These remain responsibilities of `ShogiGame`.

3.  **Refactor `ShogiGame.make_move(self, move_tuple: MoveTuple)`:**
    1.  **Pre-condition:** The `move_tuple` is assumed to be legal. Legal move validation should occur *before* calling `make_move` or as the very first step within it, typically by calling a function from `shogi_rules_logic.py` (e.g., `is_move_legal(self, move_tuple)`).
    2.  Call `move_application_result = shogi_move_execution.apply_move_to_board(self.board, self.hands, move_tuple, self.current_player)`.
    3.  Update `self.move_history.append(move_tuple)`.
    4.  Update `self.board_history.append(self._board_state_hash())` (using the new board state).
    5.  Increment `self.move_count`.
    6.  Switch `self.current_player`.
    7.  Call `self._check_and_update_termination_status()` (detailed in Phase 2).

### 3.3. Ensure Core Definitions are in `shogi_core_definitions.py` (Ongoing Verification)

*   Re-verify that all fundamental Shogi constants and types (`Piece`, `PieceType`, `Color`, `MoveTuple`, `PROMOTED_TYPES_SET`, `BASE_TO_PROMOTED_TYPE`, `SYMBOL_TO_PIECE_TYPE`, etc.) are robustly defined in `shogi_core_definitions.py`.
*   The new `MoveApplicationResult` dataclass should ideally be placed here if it's a shared data structure.
*   Ensure all modules import these definitions directly from `shogi_core_definitions.py`.

## 4. Phase 2: Centralized Game Termination Logic in `ShogiGame` (Planned)

**Target State:** A single, private method `_check_and_update_termination_status(self) -> None` in `ShogiGame` will be responsible for checking all game termination conditions and updating the game state (`self.game_over`, `self.winner`, `self.termination_reason`) accordingly.

**Specific Changes:**

*   **New private method in `ShogiGame`:** `_check_and_update_termination_status(self) -> None`
    *   This method will be called after a move is made (at the end of `make_move`) and after a game state is loaded (at the end of `from_sfen`).
    *   It should check conditions in a logical order, as the first one met typically ends the game.
    *   **Checks to perform (delegating to `shogi_rules_logic.py` where appropriate):**
        1.  **Checkmate:**
            *   Condition: `shogi_rules_logic.is_checkmate(game_context=self, color_to_check=self.current_player)` (assuming `current_player` is the one whose turn it *was*, and we are checking if they checkmated the opponent; or adjust based on when player turn is switched).
            *   Action: Set `self.game_over = True`, `self.winner = opponent_color`, `self.termination_reason = "Checkmate"`.
        2.  **Stalemate:**
            *   Condition: If not checkmate, `shogi_rules_logic.is_stalemate(game_context=self, color_to_check=self.current_player)`. (No legal moves for the current player, and they are not in check).
            *   Action: Set `self.game_over = True`, `self.winner = None`, `self.termination_reason = "Stalemate"`.
        3.  **Max Moves Reached:**
            *   Condition: `if self.move_count >= self._max_moves_this_game:`
            *   Action: Set `self.game_over = True`, `self.winner = None` (or as per specific game rules for max moves), `self.termination_reason = "MaxMovesReached"`.
        4.  **Repetition (Sennichite):**
            *   Condition: `shogi_rules_logic.check_for_sennichite(board_history=self.board_history, ruleset_variant="standard")` (function needs access to board history and potentially whose turn it is to determine if it's a losing/drawing repetition).
            *   Action: Set `self.game_over = True`, `self.termination_reason = "Repetition"`. Winner/draw status depends on specific sennichite rules (e.g., perpetual check variants).
        5.  **Impasse (持将棋 - Jishogi) (Optional - if implemented):**
            *   Condition: `shogi_rules_logic.check_for_impasse(game_context=self)`.
            *   Action: Set `self.game_over = True`, `self.termination_reason = "Impasse"`. Winner/draw based on impasse rules.
        6.  **Try Rule (入玉宣言勝ち - Nyugyoku) (Optional - if implemented):**
            *   Condition: `shogi_rules_logic.check_for_try_rule(game_context=self, declaring_player=self.current_player)`.
            *   Action: Set `self.game_over = True`, `self.winner = declaring_player`, `self.termination_reason = "TryRule"`.
    *   The method ensures that once `self.game_over` is true, no further checks are made and the game status attributes are consistently set.

## 5. Phase 3: Further `ShogiGame` Simplification (Future)

**Goal:** After completing Phase 1 (move execution) and Phase 2 (termination logic), further reduce the complexity within the `ShogiGame` class to solidify its role as a lean orchestrator of game state and game flow.

**Potential Areas for Review and Refactoring:**

*   **`_setup_initial_board(self)`:** If this method contains complex logic beyond simple array initialization, consider if parts can be delegated to a helper function or a configuration loader.
*   **`get_observation(self)`:** Currently delegates to `shogi_game_io.generate_neural_network_observation(self)`. Ensure this remains a clean delegation and no complex feature engineering creeps into `ShogiGame`.
*   **Large Helper Methods:** Review any remaining private helper methods within `ShogiGame`. If they perform distinct, complex tasks not directly related to orchestration, consider if they belong in a more specialized module (e.g., `shogi_rules_logic.py` for rule-related computations, `shogi_features.py` for feature extraction if not already there, or a new utility module).
*   **State Management Granularity:** Assess if `ShogiGame` manages too many distinct aspects of state that could be encapsulated into smaller, dedicated objects (e.g., a `HistoryManager` object if history tracking becomes very complex). This is a more advanced refactoring consideration.
*   **Parameter Objects:** For methods with many parameters (especially those calling out to other modules), consider introducing parameter objects to improve clarity and maintainability.

## 6. Expected State of `ShogiGame` Class Post-Refactoring

*   **Primary Role:** State management (board, hands, current player, history, game status) and orchestration of game operations by delegating to specialized modules for SFEN I/O, move execution, rule validation, and feature generation.
*   **Reduced Size and Complexity:** Significantly more focused and easier to understand.
*   **Key Attributes (illustrative):**
    *   `board: List[List[Optional[Piece]]]`
    *   `hands: Dict[Color, Dict[PieceType, int]]`
    *   `current_player: Color`
    *   `move_count: int`
    *   `game_over: bool`
    *   `winner: Optional[Color]`
    *   `termination_reason: Optional[str]`
    *   `move_history: List[MoveTuple]`
    *   `board_history: List[str]` (hashes of board states)
    *   `_max_moves_this_game: int`
    *   `_initial_board_setup_done: bool`
*   **Key Methods (illustrative, focusing on orchestration):**
    *   `__init__(self, ...)`
    *   `reset(self)`
    *   `make_move(self, move_tuple: MoveTuple)`: Orchestrates validation, execution (via `shogi_move_execution`), history updates, and termination checks.
    *   `get_legal_moves(self) -> List[MoveTuple]`: Delegates to `shogi_rules_logic.generate_all_legal_moves(self)`.
    *   `is_in_check(self, color: Color) -> bool`: Delegates to `shogi_rules_logic.is_in_check(self, color)`.
    *   `get_observation(self) -> np.ndarray`: Delegates to `shogi_game_io.generate_neural_network_observation(self)`.
    *   `to_sfen(self) -> str`: Delegates to `shogi_game_io.convert_game_to_sfen_string(self)`.
    *   `from_sfen(cls, sfen_str: str, ...)`: Orchestrates parsing (via `shogi_game_io`), populates its own state, and calls `_check_and_update_termination_status`.
    *   `sfen_encode_move(self, move_tuple: MoveTuple) -> str`: Delegates to `shogi_game_io.encode_move_to_sfen_string(...)`.
    *   `_check_and_update_termination_status(self)` (private helper for Phase 2).
    *   `_board_state_hash(self) -> str` (private helper for history).

## 7. Implementation Steps (High-Level - Revised Roadmap)

1.  **Preparation (✅ Completed):**
    *   The `MoveApplicationResult` dataclass has been defined in `keisei/shogi/shogi_core_definitions.py`.
    *   The function signature for `apply_move_to_board` in `keisei/shogi/shogi_move_execution.py` is specified in section 3.2.2 and is ready for implementation.

2.  **Phase 1.2 Implementation (Move Execution to `shogi_move_execution.py` - ❗ Current Priority):**
    *   Implement `apply_move_to_board` in `shogi_move_execution.py`. This involves carefully migrating the core board/hands mutation logic from the current `ShogiGame.make_move` method.
    *   Refactor `ShogiGame.make_move` to:
        *   Call `shogi_rules_logic.is_move_legal` (or confirm this responsibility is handled by the caller).
        *   Call the new `shogi_move_execution.apply_move_to_board`.
        *   Handle remaining responsibilities: update move history, board history, move count, and switch current player.
        *   (Defer call to `_check_and_update_termination_status` until Phase 2 is implemented, or add a placeholder).

3.  **Phase 2 Implementation (Centralized Termination Logic):**
    *   Implement the `_check_and_update_termination_status(self)` method in `ShogiGame`, ensuring it correctly delegates complex rule checks (checkmate, stalemate, sennichite, etc.) to functions in `shogi_rules_logic.py`.
    *   Integrate calls to `_check_and_update_termination_status` at the end of the refactored `ShogiGame.make_move` and `ShogiGame.from_sfen`.

4.  **Phase 3 Implementation (Further `ShogiGame` Simplification):**
    *   Systematically review `ShogiGame` against the points outlined in Section 5 (Phase 3).
    *   Implement refactorings to further delegate responsibilities and simplify the `ShogiGame` class.

5.  **Testing (Continuous - Crucial for each step):**
    *   Develop and run thorough unit tests for all new and modified functions, especially for `apply_move_to_board` and `_check_and_update_termination_status`.
    *   Ensure existing integration tests for `ShogiGame` continue to pass after each refactoring step.
    *   Pay special attention to edge cases in move execution (promotions, drops, captures) and all game termination conditions.
    *   Verify that `ShogiGame.make_move` correctly orchestrates all operations and maintains a consistent game state.
    *   **Recent Progress (June 11, 2025):**
        *   The test suite for `ShogiGame.get_reward()` and the reward values returned by `ShogiGame.make_move()` (specifically concerning checkmate scenarios) in `tests/test_shogi_game_rewards.py` has been significantly refactored and strengthened. This involved:
            *   Parametrizing terminal state tests.
            *   Simplifying test setups for `get_reward`.
            *   Correcting board setups for checkmate detection in `make_move` tests.
            *   Ensuring `ValueError` is raised by `get_reward` if `perspective_player_color` is `None`.
        *   Tests for `ShogiGame.seed()` logging were also corrected in `tests/test_seeding.py`.
        *   These improvements ensure greater confidence in the existing behavior of these components before and during further refactoring.

This detailed plan should provide a clear roadmap for the refactoring effort.
