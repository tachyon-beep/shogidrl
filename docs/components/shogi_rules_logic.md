# Software Documentation Template for Subsystems - shogi_rules_logic

## 📘 shogi_rules_logic.py as of May 2025

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `/home/john/keisei/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `May 2025`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Implements comprehensive Shogi game rules, move generation, and validation logic. Provides the core rule enforcement engine that ensures all game operations comply with official Shogi rules including piece movement patterns, check detection, and legal move generation.

* **Key Responsibilities:**
  - Generate all legal moves for current game position
  - Validate individual moves against Shogi rules
  - Detect check, checkmate, and stalemate conditions
  - Implement piece-specific movement patterns
  - Handle special rules (promotion zones, drop restrictions, etc.)
  - Provide king safety analysis and attack detection

* **Domain Context:**
  Complete implementation of Japanese Chess (Shogi) rules as specified by the Japan Shogi Association, including all standard piece movements, capture rules, promotion mechanics, and game termination conditions.

* **High-Level Architecture / Interaction Summary:**
  
  * Serves as the authoritative rule engine for the Shogi game system
  * Provides validation services to move execution and game state management
  * Generates legal moves for AI training and human play
  * Integrates with move execution to ensure rule compliance

---

### 2. Modules 📦

* **Module Name:** `shogi_rules_logic.py`

  * **Purpose:** Implement complete Shogi rule system with move generation and validation.
  * **Design Patterns Used:** 
    - Strategy pattern for piece-specific movement rules
    - Visitor pattern for board analysis operations
    - Command pattern for move validation
    - Factory pattern for move generation
  * **Key Functions/Classes Provided:** 
    - `get_legal_moves()` - Complete legal move generation
    - `is_valid_move()` - Individual move validation
    - `is_in_check()` - Check detection
    - `find_king()` - King location utilities
    - Piece-specific movement functions for all piece types
    - Drop validation and promotion logic
  * **Configuration Surface:**
    * Movement pattern definitions for each piece type
    * Promotion zone specifications
    * Drop restriction rules
    * Check detection parameters
  * **Dependencies:**
    * **Internal:**
      - `shogi_core_definitions`: Core types (Color, MoveTuple, Piece, PieceType)
    * **External:**
      - `typing`: Type annotations for complex function signatures
  * **External API Contracts:**
    - Provides rule validation for game engine
    - Generates legal moves for AI training
    - Ensures rule compliance across all game operations
  * **Side Effects / Lifecycle Considerations:**
    - No persistent state (stateless rule functions)
    - Temporary game state analysis during move generation
  * **Usage Examples:**
    ```python
    from keisei.shogi.shogi_rules_logic import get_legal_moves, is_valid_move, is_in_check
    
    # Generate all legal moves
    legal_moves = get_legal_moves(game)
    print(f"Player has {len(legal_moves)} legal moves")
    
    # Validate a specific move
    move = (6, 4, 5, 4, False)
    if is_valid_move(game, move):
        game.make_move(move)
    
    # Check for check condition
    if is_in_check(game, game.current_player):
        print("King is in check!")
    ```

---

### 3. Classes 🏛️

**No classes defined in this module.** This module provides comprehensive rule functions organized by piece type and rule category.

---

### 4. Functions 🔧

* **Function Name:** `get_legal_moves`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Generate complete list of all legal moves for the current player.
  * **Parameters:**
    - `game: ShogiGame` - Game instance to analyze
  * **Returns:** `List[MoveTuple]` - All legal moves (board moves and drops)
  * **Algorithmic Note:**
    - Generates all possible moves then filters for legality
    - Combines board moves and drop moves
    - Applies king safety filtering to prevent illegal positions
  * **Performance:** Optimized for frequent calls during AI training
  * **Usage Example:**
    ```python
    legal_moves = get_legal_moves(game)
    for move in legal_moves:
        # Evaluate move for AI decision making
        pass
    ```

* **Function Name:** `is_valid_move`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Validate whether a specific move is legal in the current position.
  * **Parameters:**
    - `game: ShogiGame` - Game context
    - `move: MoveTuple` - Move to validate
  * **Returns:** `bool` - True if move is legal
  * **Algorithmic Note:**
    - Checks piece-specific movement rules
    - Validates destination square availability
    - Ensures move doesn't leave king in check
  * **Usage Example:**
    ```python
    if is_valid_move(game, proposed_move):
        execute_move(proposed_move)
    else:
        print("Illegal move")
    ```

* **Function Name:** `is_in_check`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Determine if the specified player's king is currently in check.
  * **Parameters:**
    - `game: ShogiGame` - Game state to analyze
    - `player_color: Color` - Player whose king to check
    - `debug_recursion: bool = False` - Enable debug output
  * **Returns:** `bool` - True if king is in check
  * **Algorithmic Note:**
    - Locates king position
    - Checks all opponent pieces for attacks on king square
    - Handles edge cases where king is missing (invalid positions)
  * **Usage Example:**
    ```python
    if is_in_check(game, Color.BLACK):
        print("Black king is in check")
    ```

* **Function Name:** `find_king`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Locate the king of the specified color on the board.
  * **Parameters:**
    - `game: ShogiGame` - Game instance
    - `color: Color` - Color of king to find
  * **Returns:** `Optional[Tuple[int, int]]` - King position or None if not found
  * **Usage Example:**
    ```python
    king_pos = find_king(game, Color.WHITE)
    if king_pos:
        row, col = king_pos
        print(f"White king at ({row}, {col})")
    ```

* **Function Name:** `can_piece_move_to`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Check if a piece can legally move to a specific destination.
  * **Parameters:**
    - `game: ShogiGame` - Game context
    - `piece: Piece` - Piece to move
    - `from_row: int, from_col: int` - Source position
    - `to_row: int, to_col: int` - Destination position
  * **Returns:** `bool` - True if move is possible for this piece type
  * **Algorithmic Note:**
    - Delegates to piece-specific movement functions
    - Checks path clearance for sliding pieces
    - Validates destination square occupancy
  * **Usage Example:**
    ```python
    piece = game.board[2][3]
    if can_piece_move_to(game, piece, 2, 3, 4, 5):
        print("Move is possible")
    ```

* **Function Name:** `get_piece_moves`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Generate all possible moves for a specific piece at a given position.
  * **Parameters:**
    - `game: ShogiGame` - Game context
    - `row: int, col: int` - Piece position
  * **Returns:** `List[MoveTuple]` - All moves for this piece
  * **Algorithmic Note:**
    - Calls piece-specific movement generators
    - Handles promotion possibilities
    - Filters out moves that would leave king in check
  * **Usage Example:**
    ```python
    piece_moves = get_piece_moves(game, 6, 4)  # Get moves for piece at (6,4)
    ```

* **Function Name:** `get_drop_moves`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Generate all legal piece drops for the current player.
  * **Parameters:**
    - `game: ShogiGame` - Game state
  * **Returns:** `List[MoveTuple]` - All legal drop moves
  * **Algorithmic Note:**
    - Checks hand inventory for available pieces
    - Validates drop restrictions (pawn column limits, etc.)
    - Ensures drops don't create immediate checkmate (pawn/lance drops)
  * **Usage Example:**
    ```python
    drops = get_drop_moves(game)
    print(f"Can drop pieces in {len(drops)} positions")
    ```

* **Function Name:** `is_promotion_mandatory`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Check if promotion is mandatory for a piece making a specific move.
  * **Parameters:**
    - `piece_type: PieceType` - Type of piece moving
    - `from_row: int, to_row: int` - Source and destination rows
    - `player_color: Color` - Player making the move
  * **Returns:** `bool` - True if promotion is required
  * **Algorithmic Note:**
    - Handles mandatory promotion for pieces that cannot retreat
    - Checks promotion zone boundaries for each player
    - Applies piece-specific promotion rules
  * **Usage Example:**
    ```python
    if is_promotion_mandatory(PieceType.PAWN, 1, 0, Color.BLACK):
        # Must promote pawn reaching far rank
        promote = True
    ```

* **Function Name:** `is_checkmate`

  * **Defined In:** `shogi_rules_logic.py`
  * **Purpose:** Determine if the current player is in checkmate.
  * **Parameters:**
    - `game: ShogiGame` - Game state to analyze
  * **Returns:** `bool` - True if checkmate detected
  * **Algorithmic Note:**
    - First checks if king is in check
    - If in check, tries all legal moves to see if check can be escaped
    - Returns True only if no legal moves can resolve check
  * **Usage Example:**
    ```python
    if is_checkmate(game):
        game.end_game("checkmate")
    ```

---

### 5. Data Structures 📊

* **Structure Name:** `Piece Movement Patterns`
  * **Type:** `Movement direction vectors and rules`
  * **Purpose:** Define how each piece type can move on the board
  * **Format:** Direction vectors, range limits, and special rules
  * **Fields:** 
    - Basic pieces: Single-step and multi-step movements
    - Promoted pieces: Enhanced movement patterns
    - Special cases: Knight L-moves, Pawn restrictions
  * **Used In:** Move generation and validation functions

* **Structure Name:** `Drop Restriction Rules`
  * **Type:** `Validation criteria for piece drops`
  * **Purpose:** Enforce legal drop placement according to Shogi rules
  * **Format:** Position-based and piece-specific restrictions
  * **Fields:**
    - Pawn drop restrictions (no double pawns in columns)
    - Piece placement zones (no pieces in impossible positions)
    - Checkmate restrictions (no immediate mate by drops)
  * **Used In:** Drop move generation and validation

* **Structure Name:** `Promotion Zone Definitions`
  * **Type:** `Board area specifications`
  * **Purpose:** Define promotion zones for each player
  * **Format:** Row ranges for promotion eligibility
  * **Fields:**
    - Player-specific promotion zones (first 3 ranks for opponent territory)
    - Mandatory promotion conditions
    - Optional promotion choices
  * **Used In:** Promotion validation and move generation

---

### 6. Inter-Module Relationships 🔗

* **Dependency Graph:**
  ```
  shogi_rules_logic.py (rule engine)
    ├── imports from → shogi_core_definitions.py (types, constants)
    ├── called by → shogi_game.py (move validation)
    ├── called by → shogi_move_execution.py (termination detection)
    └── used by → training systems (legal move generation)
  ```

* **Integration Points:**
  - ShogiGame delegates all rule checking to this module
  - Move execution uses termination detection functions
  - AI training systems use legal move generation

* **Data Flow:**
  - Game state flows in for analysis
  - Legal moves and validation results flow out
  - Rule compliance decisions propagate through system

---

### 7. Implementation Notes 🔧

#### Code Quality & Best Practices

* **Rule Accuracy:**
  - Complete implementation of official Shogi rules
  - Comprehensive edge case handling
  - Accurate piece movement patterns

* **Performance Optimization:**
  - Efficient move generation algorithms
  - Early termination for impossible moves
  - Optimized king safety checking

#### Performance Considerations

* **Move Generation Efficiency:**
  - Vectorized direction calculations where possible
  - Lazy evaluation of expensive validation checks
  - Caching of frequently computed values

* **Memory Management:**
  - Minimal temporary object creation during generation
  - Efficient list operations for move collection
  - Stack-efficient recursive algorithms

#### Maintainability

* **Modular Design:**
  - Piece-specific functions for easy extension
  - Clear separation between different rule categories
  - Comprehensive function documentation

#### Error Handling

* **Robust Validation:**
  - Defensive programming for edge cases
  - Clear error reporting for rule violations
  - Graceful handling of invalid game states

---

### 8. Testing Strategy 🧪

#### Current Test Coverage
- Legal move generation accuracy
- Move validation correctness
- Check detection reliability
- Promotion rule compliance
- Drop restriction enforcement

#### Recommended Testing Approach
- Unit tests for each piece movement pattern
- Integration tests for complex game positions
- Property-based testing for rule consistency
- Performance tests for move generation speed
- Compliance tests against known game positions

---

### 9. Security Considerations 🔒

* **Rule Integrity:**
  - Immutable rule implementations prevent tampering
  - Comprehensive validation prevents invalid game states
  - Protection against rule exploitation

* **Input Validation:**
  - Bounds checking for all board operations
  - Validation of piece types and positions
  - Error handling for malformed input

---

### 10. Future Enhancements 🚀

#### Potential Improvements
1. **Rule Variants:** Support for different Shogi variants and handicap games
2. **Performance Optimization:** Bitboard representation for faster operations
3. **Analysis Features:** Enhanced position evaluation and tactical detection
4. **International Rules:** Support for different international Shogi rule sets

#### Backward Compatibility
- Current rule implementation is highly stable
- New variants can be added without affecting core rules
- Clear versioning strategy for any rule modifications

---

### 11. Dependencies & Requirements 📋

#### Runtime Dependencies
- **Internal:** shogi_core_definitions for types and constants
- **External:** Python typing module for type annotations

#### Development Dependencies
- Testing frameworks for rule validation
- Performance profiling tools for optimization
- Rule compliance verification tools

---

### 12. Configuration 🛠️

#### Rule Parameters
- Configurable promotion zones for variants
- Adjustable piece movement patterns
- Customizable drop restrictions

#### Performance Tuning
- Configurable move generation depth
- Adjustable validation thoroughness
- Optional caching for expensive operations

This comprehensive rule engine ensures complete compliance with Shogi rules while providing the performance and accuracy required for both human play and AI training scenarios.
