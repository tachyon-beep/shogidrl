\
# ShogiGame (`shogi_game.py`) Refactoring Plan

**Last Updated:** June 11, 2025

## 1. Introduction & Goals

The `keisei/shogi/shogi_game.py` file, primarily containing the `ShogiGame` class, has grown to over 1000 lines. While it benefits from some delegation to other modules in the `keisei/shogi/` directory (like `shogi_rules_logic.py`, `shogi_move_execution.py`, `shogi_game_io.py`, and `features.py`), it still handles a broad range of responsibilities including detailed SFEN (Shogi Forsyth-Edwards Notation) parsing/formatting, direct board manipulation for SFEN, and intricate game state setup from SFEN.

**Goals of this Refactoring:**

*   **Enhance Modularity:** Further separate concerns, making each module and class more focused.
*   **Improve Maintainability:** Reduce the complexity of `ShogiGame`, making it easier to understand, modify, and debug.
*   **Adhere to Single Responsibility Principle (SRP):** Ensure `ShogiGame` primarily acts as a game state manager and orchestrator, delegating specialized tasks.
*   **Leverage Existing Modules:** Maximize the use of existing specialized modules within the `keisei/shogi/` package.

## 2. Guiding Principles

1.  **`ShogiGame` as Orchestrator:** The `ShogiGame` class should primarily hold the game state (board, hands, current player, move history, game status) and orchestrate operations by calling functions in other, more specialized modules.
2.  **Delegate, Don't Do (Everything):** Complex logic for I/O (especially SFEN), detailed move application mechanics, and rule validation should reside in their respective modules.
3.  **Clear Interfaces:** Maintain clear and concise interfaces between `ShogiGame` and the helper modules.

## 3. Phase 1: Relocating Functionality to Existing Modules

### 3.1. SFEN Serialization/Deserialization to `shogi_game_io.py`

**Current State:** `ShogiGame` contains extensive logic for converting game states to/from SFEN strings, including helper methods for character parsing and coordinate conversion.

**Target State:** All direct SFEN string manipulation, parsing, formatting, and associated helper logic will reside in `shogi_game_io.py`. `ShogiGame` will call functions from `shogi_game_io.py` for these operations.

**Specific Changes:**

*   **Constants to Move from `ShogiGame` to `shogi_game_io.py`:**
    *   `_SFEN_BOARD_CHARS: Dict[PieceType, str]`
    *   `SFEN_HAND_PIECE_CANONICAL_ORDER: List[PieceType]` (currently an inline list in `to_sfen_string`)

*   **Helper Methods to Move from `ShogiGame` to `shogi_game_io.py` (as private/internal functions):**
    *   `_sfen_sq(r: int, c: int) -> str`
    *   `_get_sfen_board_char(piece: Piece, sfen_board_chars_map: Dict[PieceType, str]) -> str` (will take the map as an arg)
    *   `_get_sfen_drop_char(piece_type: PieceType) -> str`
    *   `_parse_sfen_board_piece(sfen_char_on_board: str, is_promoted_sfen_token: bool, symbol_to_piece_type_map: Dict[str, PieceType], base_to_promoted_map: Dict[PieceType, PieceType], promoted_types_set: Set[PieceType]) -> Tuple[PieceType, Color]` (will take maps as args)

*   **Refactoring `ShogiGame.to_sfen_string(self) -> str`:**
    *   The core logic will be moved to a new function in `shogi_game_io.py`:
        `convert_game_to_sfen_string(game: ShogiGame) -> str`
    *   `ShogiGame.to_sfen_string` (or a new public method `to_sfen()`) will simply call `shogi_game_io.convert_game_to_sfen_string(self)`.

*   **Refactoring `ShogiGame.sfen_encode_move(self, move_tuple: MoveTuple) -> str`:**
    *   The core logic will be moved to a new function in `shogi_game_io.py`:
        `encode_move_to_sfen_string(move_tuple: MoveTuple, sfen_sq_converter: Callable[[int, int], str], sfen_drop_char_getter: Callable[[PieceType], str]) -> str`
        (The converter and getter functions can be passed from `ShogiGame` or `shogi_game_io` can import its own helpers).
        Alternatively, `encode_move_to_sfen_string(move_tuple: MoveTuple)` could directly use helper functions within `shogi_game_io.py`.
    *   `ShogiGame.sfen_encode_move` will call this new function.

*   **Refactoring `ShogiGame.from_sfen(cls, sfen_str: str, max_moves_for_game_instance: int) -> "ShogiGame"`:**
    1.  **Initial Parsing of SFEN String Parts:**
        *   A new function in `shogi_game_io.py`: `parse_sfen_string_components(sfen_str: str) -> Tuple[str, str, str, str]`
            *   This function will handle the regex matching and splitting of the SFEN string into `board_sfen, turn_sfen, hands_sfen, move_number_sfen`.
            *   `ShogiGame.from_sfen` will call this first.
    2.  **Game Instance Initialization:**
        *   `ShogiGame.from_sfen` will instantiate `game = cls(max_moves_per_game=max_moves_for_game_instance)`.
        *   It will set `game.current_player` based on `turn_sfen` and `game.move_count` from `move_number_sfen`.
        *   Initialize `game.board`, `game.hands`, `game.move_history`, `game.board_history`.
    3.  **Board Population:**
        *   A new function in `shogi_game_io.py`:
            `populate_board_from_sfen_segment(board_array: List[List[Optional[Piece]]], board_sfen_segment: str, symbol_to_piece_type_map: Dict[str, PieceType], base_to_promoted_map: Dict[PieceType, PieceType], promoted_types_set: Set[PieceType], parse_sfen_board_piece_func: Callable[..., Tuple[PieceType, Color]])`
            *   This function will contain the detailed loop for parsing the board segment of the SFEN string and placing pieces on the `board_array`. It will use the `_parse_sfen_board_piece` (now in `shogi_game_io`) logic.
            *   `ShogiGame.from_sfen` will call this with `game.board` and the necessary maps/sets (imported from `shogi_core_definitions.py`).
    4.  **Hands Population:**
        *   A new function in `shogi_game_io.py`:
            `populate_hands_from_sfen_segment(hands_dict: Dict[int, Dict[PieceType, int]], hands_sfen_segment: str, symbol_to_piece_type_map: Dict[str, PieceType])`
            *   This function will parse the hands segment and update the `hands_dict`.
            *   `ShogiGame.from_sfen` will call this with `game.hands`.
    5.  **Finalization:**
        *   `ShogiGame.from_sfen` will set `game._initial_board_setup_done = True`.
        *   It will add the initial `game._board_state_hash()` to `game.board_history`.
        *   Crucially, it will then call `game._check_and_update_termination_status()` (see Phase 2).

### 3.2. Move Execution Logic to `shogi_move_execution.py`

**Current State:** `ShogiGame` would typically have a `make_move` method that directly alters the board, hands, captures pieces, handles promotions, updates player turn, and move count. (This method might not be fully implemented in the provided snippet but is a standard part of such a class).

**Target State:** The detailed mechanics of applying a *validated* move to the board state (updating piece positions, handling captures, promotions) will be in `shogi_move_execution.py`. `ShogiGame.make_move` will orchestrate this.

**Specific Changes:**

*   **New/Enhanced function in `shogi_move_execution.py`:**
    *   `apply_move_to_board(game: ShogiGame, move: MoveTuple) -> MoveApplicationResult`
        *   This function will directly mutate `game.board` and `game.hands`.
        *   It will handle piece movement, captures (adding to opponent's hand), and promotions.
        *   `MoveApplicationResult` (a new dataclass or NamedTuple) could return information like `captured_piece_type: Optional[PieceType]`, `was_promotion: bool`.
*   **Refined `ShogiGame.make_move(self, move_tuple: MoveTuple)` (Conceptual):**
    1.  **Pre-condition:** The `move_tuple` is assumed to be legal (validation should occur before or at the start of `make_move`, possibly using `shogi_rules_logic.is_move_legal(self, move_tuple)`).
    2.  Call `move_result = shogi_move_execution.apply_move_to_board(self, move_tuple)`.
    3.  Update `self.move_history.append(move_tuple)`.
    4.  Update `self.board_history.append(self._board_state_hash())`.
    5.  Increment `self.move_count`.
    6.  Switch `self.current_player`.
    7.  Call `self._check_and_update_termination_status()` (see Phase 2).

### 3.3. Ensure Core Definitions are in `shogi_core_definitions.py`

*   Re-verify that all fundamental Shogi constants and types (`Piece`, `PieceType`, `Color`, `MoveTuple`, `PROMOTED_TYPES_SET`, `BASE_TO_PROMOTED_TYPE`, `SYMBOL_TO_PIECE_TYPE`, `get_unpromoted_types()`, etc.) are robustly defined in `shogi_core_definitions.py`.
*   Ensure all other modules (`shogi_game.py`, `shogi_game_io.py`, `shogi_rules_logic.py`, etc.) import these definitions directly from `shogi_core_definitions.py`.

## 4. Phase 2: New Abstractions and Internal `ShogiGame` Refinements

### 4.1. Centralized Game Termination Logic in `ShogiGame`

**Current State:** Termination logic might be scattered or implicitly handled (e.g., end of `from_sfen`).

**Target State:** A single, private method in `ShogiGame` will be responsible for checking all game termination conditions and updating the game state accordingly.

**Specific Changes:**

*   **New private method in `ShogiGame`:** `_check_and_update_termination_status(self) -> None`
    *   This method will be called after a move is made (`make_move`) or after a game state is loaded (`from_sfen`).
    *   **Checks to perform:**
        1.  **Max Moves:** `if self.move_count >= self._max_moves_this_game:`
            *   Set `self.game_over = True`, `self.termination_reason = "MaxMovesReached"`, `self.winner = None` (or as per game rules for max moves).
        2.  **Checkmate/Stalemate:**
            *   `legal_moves = shogi_rules_logic.generate_all_legal_moves(self)`
            *   `if not legal_moves:`
                *   `is_check = shogi_rules_logic.is_in_check(self, self.current_player)`
                *   If `is_check`: Checkmate (`self.termination_reason = "Tsumi"` (Checkmate), `self.winner = opponent_color`).
                *   Else: Stalemate (`self.termination_reason = "Stalemate"`, `self.winner = None`).
                *   Set `self.game_over = True`.
        3.  **Sennichite (Repetition):**
            *   `if shogi_rules_logic.check_for_sennichite(self):` (This function will use `self.board_history`).
            *   Set `self.game_over = True`, `self.termination_reason = "Sennichite"`, `self.winner = None`.
    *   This method ensures `self.game_over`, `self.winner`, and `self.termination_reason` are consistently updated.

### 4.2. Board Hashing and History (`ShogiGame`)

*   The `_board_state_hash(self) -> str` method, responsible for creating a hashable representation of the board state for repetition checks, is appropriate to keep within `ShogiGame`.
*   The `board_history: List[str]` attribute will continue to be managed by `ShogiGame` (populated in `reset`, `make_move`, and `from_sfen` after board setup).

## 5. Expected State of `ShogiGame` Class Post-Refactoring

*   **Primary Role:** State management (board, hands, current player, history, game status) and orchestration of game operations by delegating to specialized modules.
*   **Reduced Size and Complexity:** Significantly more focused.
*   **Key Attributes (illustrative):**
    *   `board: List[List[Optional[Piece]]]`
    *   `hands: Dict[int, Dict[PieceType, int]]`
    *   `current_player: Color`
    *   `move_count: int`
    *   `game_over: bool`
    *   `winner: Optional[Color]`
    *   `termination_reason: Optional[str]`
    *   `move_history: List[MoveTuple]`
    *   `board_history: List[str]`
    *   `_max_moves_this_game: int`
    *   `_initial_board_setup_done: bool`
*   **Key Methods (illustrative):**
    *   `__init__(self, max_moves_per_game: int)`
    *   `reset(self) -> np.ndarray` (returns initial observation)
    *   `_setup_initial_board(self)`
    *   `get_piece(self, row: int, col: int) -> Optional[Piece]`
    *   `set_piece(self, row: int, col: int, piece: Optional[Piece])` (mainly for setup/testing)
    *   `make_move(self, move_tuple: MoveTuple)`: Orchestrates validation (via `shogi_rules_logic`), execution (via `shogi_move_execution`), history updates, and termination checks.
    *   `get_legal_moves(self) -> List[MoveTuple]`: Delegates to `shogi_rules_logic.generate_all_legal_moves(self)`.
    *   `is_in_check(self, color: Color, ...) -> bool`: Delegates to `shogi_rules_logic.is_in_check(self, color, ...)`.
    *   `get_observation(self) -> np.ndarray`: Delegates to `shogi_game_io.generate_neural_network_observation(self)`.
    *   `to_sfen(self) -> str`: Delegates to `shogi_game_io.convert_game_to_sfen_string(self)`.
    *   `from_sfen(cls, sfen_str: str, ...)`: Orchestrates parsing via `shogi_game_io`, populates its own state, and calls `_check_and_update_termination_status`.
    *   `sfen_encode_move(self, move_tuple: MoveTuple) -> str`: Delegates to `shogi_game_io.encode_move_to_sfen_string(...)`.
    *   `__deepcopy__(self, memo)`
    *   `_check_and_update_termination_status(self)` (private helper)
    *   `_board_state_hash(self) -> str` (private helper)
    *   Properties (e.g., `max_moves_per_game`).

## 6. Implementation Steps (High-Level)

1.  **Preparation:**
    *   Define any new dataclasses/NamedTuples (e.g., `MoveApplicationResult`).
    *   Carefully define the function signatures for new/modified functions in `shogi_game_io.py` and `shogi_move_execution.py`.
2.  **Phase 1 Implementation (SFEN to `shogi_game_io.py`):**
    *   Move SFEN-related constants (`_SFEN_BOARD_CHARS`, etc.) to `shogi_game_io.py`.
    *   Implement SFEN helper functions (e.g., `_sfen_sq`, `_parse_sfen_board_piece`) in `shogi_game_io.py` by migrating logic from `ShogiGame`.
    *   Implement `convert_game_to_sfen_string` in `shogi_game_io.py`; refactor `ShogiGame.to_sfen_string` to call it.
    *   Implement `encode_move_to_sfen_string` in `shogi_game_io.py`; refactor `ShogiGame.sfen_encode_move` to call it.
    *   Implement `parse_sfen_string_components`, `populate_board_from_sfen_segment`, and `populate_hands_from_sfen_segment` in `shogi_game_io.py`.
    *   Refactor `ShogiGame.from_sfen` to use these new `shogi_game_io.py` functions.
3.  **Phase 1 Implementation (Move Execution to `shogi_move_execution.py`):**
    *   Implement `apply_move_to_board` in `shogi_move_execution.py`.
    *   Refactor/Implement `ShogiGame.make_move` to use `apply_move_to_board`.
4.  **Phase 2 Implementation (Internal `ShogiGame` Refinements):**
    *   Implement the `_check_and_update_termination_status(self)` method in `ShogiGame`.
    *   Integrate calls to `_check_and_update_termination_status` at the end of the refactored `ShogiGame.make_move` and `ShogiGame.from_sfen`.
5.  **Testing:**
    *   Thoroughly test each refactored component and the integrated `ShogiGame` class.
    *   Pay special attention to SFEN parsing/generation for edge cases and correctness.
    *   Verify game termination conditions are correctly identified.
    *   Ensure existing unit tests pass and add new ones for the refactored logic.

This detailed plan should provide a clear roadmap for the refactoring effort.
