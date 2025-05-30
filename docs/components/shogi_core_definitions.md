# Software Documentation Template for Subsystems - shogi_core_definitions

## 📘 shogi_core_definitions.py as of May 2025

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `/home/john/keisei/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `May 2025`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Defines fundamental types, enums, constants, and data structures for the Shogi game engine. This module provides the core building blocks including piece representations, color definitions, move structures, and observation tensor layouts for neural network training.

* **Key Responsibilities:**
  - Define core enumerations (Color, PieceType, TerminationReason)
  - Provide piece representation and manipulation utilities
  - Establish move tuple structures for board moves and drops
  - Define observation tensor layout constants for neural networks
  - Support game notation systems (KIF, USI, SFEN)

* **Domain Context:**
  Japanese Chess (Shogi) domain focusing on complete piece type definitions including promoted variants, standard Shogi rules and notation, and AI-optimized data structures for deep reinforcement learning.

* **High-Level Architecture / Interaction Summary:**
  
  * Serves as the foundation layer for all other shogi modules
  * Provides type safety and consistency across the game engine
  * Enables neural network integration through structured observation tensors
  * Supports multiple notation standards for interoperability

---

### 2. Modules 📦

* **Module Name:** `shogi_core_definitions.py`

  * **Purpose:** Define fundamental types, constants, and data structures for Shogi game engine.
  * **Design Patterns Used:** Enum pattern for type safety, Factory pattern for piece creation, Constants pattern for observation layout
  * **Key Functions/Classes Provided:** 
    - `Color` enum - Player colors (BLACK/WHITE)
    - `PieceType` enum - All Shogi piece types including promoted variants
    - `TerminationReason` enum - Game ending conditions
    - `Piece` class - Individual piece representation
    - `MoveTuple` types - Move representation system
    - Observation tensor constants and mappings
  * **Configuration Surface:**
    * Observation tensor layout constants (46-plane structure)
    * KIF piece symbol mappings
    * Promotion and hand piece mappings
  * **Dependencies:**
    * **Internal:** None (foundation module)
    * **External:**
      - `enum.Enum`: For type-safe enumerations
      - `typing`: Type hints and generic types
  * **External API Contracts:**
    - Provides core types for entire Shogi system
    - Supports multiple game notation formats
  * **Side Effects / Lifecycle Considerations:**
    - No side effects (pure type definitions)
    - Constants computed at module load time
  * **Usage Examples:**
    ```python
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece, MoveTuple
    
    # Create pieces
    pawn = Piece(PieceType.PAWN, Color.BLACK)
    king = Piece(PieceType.KING, Color.WHITE)
    
    # Check piece properties
    if pawn.can_promote():
        promoted_pawn = pawn.promote()
    
    # Create moves
    board_move = (0, 0, 1, 1, True)  # From (0,0) to (1,1) with promotion
    drop_move = (None, None, 4, 4, PieceType.PAWN)  # Drop pawn at (4,4)
    ```

---

### 3. Classes 🏛️

* **Class Name:** `Color`

  * **Defined In Module:** `shogi_core_definitions.py`
  * **Purpose:** Enumerate player colors in Shogi with standard conventions.
  * **Design Role:** Type-safe enumeration for player identification
  * **Inheritance:**
    * **Extends:** `Enum`
    * **Subclasses:** None
  * **Key Attributes/Properties:**
    - `BLACK = 0`: Sente (先手), traditionally moves first
    - `WHITE = 1`: Gote (後手), second player
  * **Key Methods:** Standard enum methods
  * **Interconnections:**
    * Used throughout game engine for player identification
    * Referenced in move validation and game state tracking
  * **Usage Example:**
    ```python
    current_player = Color.BLACK
    opponent = Color.WHITE
    ```

* **Class Name:** `PieceType`

  * **Defined In Module:** `shogi_core_definitions.py`
  * **Purpose:** Enumerate all Shogi piece types including promoted variants.
  * **Design Role:** Comprehensive type system for Shogi pieces with promotion support
  * **Inheritance:**
    * **Extends:** `Enum`
    * **Subclasses:** None
  * **Key Attributes/Properties:**
    - Basic pieces: PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK, KING
    - Promoted pieces: PROMOTED_PAWN through PROMOTED_ROOK
  * **Key Methods:**
    - `to_usi_char()`: Returns USI notation character for drops
  * **Interconnections:**
    * Used by Piece class for type identification
    * Referenced in promotion and movement logic
  * **Usage Example:**
    ```python
    piece_type = PieceType.PAWN
    promoted = PieceType.PROMOTED_PAWN
    usi_char = piece_type.to_usi_char()  # Returns "P"
    ```

* **Class Name:** `TerminationReason`

  * **Defined In Module:** `shogi_core_definitions.py`
  * **Purpose:** Enumerate possible game termination conditions.
  * **Design Role:** Type-safe representation of game ending scenarios
  * **Inheritance:**
    * **Extends:** `Enum`
    * **Subclasses:** None
  * **Key Attributes/Properties:**
    - Standard endings: CHECKMATE, RESIGNATION
    - Special conditions: REPETITION, IMPASSE, MAX_MOVES_EXCEEDED
    - Error conditions: ILLEGAL_MOVE, TIME_FORFEIT
  * **Usage Example:**
    ```python
    if game.game_over:
        reason = TerminationReason.CHECKMATE
    ```

* **Class Name:** `Piece`

  * **Defined In Module:** `shogi_core_definitions.py`
  * **Purpose:** Represent individual Shogi pieces with type and color.
  * **Design Role:** Core game piece with promotion and validation capabilities
  * **Inheritance:**
    * **Extends:** `object`
    * **Subclasses:** None
  * **Key Attributes/Properties:**
    - `piece_type: PieceType`: The type of piece
    - `color: Color`: The owner color
  * **Key Methods:**
    - `is_promoted()`: Check if piece is in promoted state
    - `can_promote()`: Check if piece can be promoted
    - `promote()`: Return promoted version of piece
    - `to_base_type()`: Return unpromoted base type
    - `to_hand_type()`: Return type when captured to hand
  * **Interconnections:**
    * Used throughout board representation
    * Referenced in move execution and validation
  * **Usage Example:**
    ```python
    pawn = Piece(PieceType.PAWN, Color.BLACK)
    if pawn.can_promote():
        tokin = pawn.promote()
    base_type = tokin.to_base_type()  # Returns PAWN
    ```

---

### 4. Functions 🔧

* **Function Name:** `get_unpromoted_types`

  * **Defined In:** `shogi_core_definitions.py`
  * **Purpose:** Return list of piece types that can be held in hand.
  * **Parameters:** None
  * **Returns:** `List[PieceType]` - Unpromoted piece types excluding King
  * **Usage Example:**
    ```python
    hand_types = get_unpromoted_types()  # [PAWN, LANCE, KNIGHT, ...]
    ```

* **Function Name:** `get_piece_type_from_symbol`

  * **Defined In:** `shogi_core_definitions.py`
  * **Purpose:** Convert piece symbol string to PieceType enum.
  * **Parameters:**
    - `symbol: str` - Piece symbol (e.g., "P", "+R", case-insensitive)
  * **Returns:** `PieceType` - Corresponding piece type
  * **Raises/Exceptions:** `ValueError` for invalid symbols
  * **Usage Example:**
    ```python
    piece_type = get_piece_type_from_symbol("P")  # Returns PieceType.PAWN
    promoted = get_piece_type_from_symbol("+R")  # Returns PieceType.PROMOTED_ROOK
    ```

---

### 5. Data Structures 📊

* **Structure Name:** `MoveTuple`
  * **Type:** `Union[BoardMoveTuple, DropMoveTuple]`
  * **Purpose:** Unified move representation supporting board moves and piece drops
  * **Format:** 
    - `BoardMoveTuple`: `(from_row, from_col, to_row, to_col, promote_flag)`
    - `DropMoveTuple`: `(None, None, to_row, to_col, piece_type_to_drop)`
  * **Validation Constraints:** Coordinates must be 0-8, piece types valid for context
  * **Used In:** Move generation, validation, and execution systems

* **Structure Name:** `Observation Tensor Constants`
  * **Type:** `Integer constants`
  * **Purpose:** Define 46-plane observation tensor layout for neural networks
  * **Format:** Channel indices for different data types
  * **Fields:**
    - Channels 0-27: Board piece positions (current/opponent, promoted/unpromoted)
    - Channels 28-41: Hand piece counts
    - Channels 42-45: Meta information (player, move count, reserved)
  * **Used In:** Neural network feature generation and training

* **Structure Name:** `Piece Mapping Dictionaries`
  * **Type:** `Dict[PieceType, PieceType]`
  * **Purpose:** Support piece promotion, demotion, and hand conversion
  * **Fields:**
    - `BASE_TO_PROMOTED_TYPE`: Maps unpromoted to promoted pieces
    - `PROMOTED_TO_BASE_TYPE`: Maps promoted to base pieces
    - `PIECE_TYPE_TO_HAND_TYPE`: Maps captured pieces to hand types
  * **Used In:** Move execution, piece promotion, and capture handling

---

### 6. Inter-Module Relationships 🔗

* **Dependency Graph:**
  ```
  shogi_core_definitions.py (foundation)
    ├── used by → shogi_game.py
    ├── used by → shogi_move_execution.py
    ├── used by → shogi_rules_logic.py
    ├── used by → shogi_game_io.py
    └── used by → features.py
  ```

* **Export Relationships:**
  - Provides fundamental types to all other shogi modules
  - Exports constants for observation tensor generation
  - Supplies notation conversion utilities

* **Data Flow:**
  - Core types flow to all game engine components
  - Observation constants used in neural network feature extraction
  - Move tuples used throughout move processing pipeline

---

### 7. Implementation Notes 🔧

#### Code Quality & Best Practices

* **Type Safety:**
  - Extensive use of enums for type-safe operations
  - Comprehensive type hints throughout
  - Clear separation between different piece states

* **Constants Organization:**
  - Well-organized observation tensor layout
  - Consistent naming conventions
  - Comprehensive piece mapping dictionaries

#### Performance Considerations

* **Memory Efficiency:**
  - Enum-based piece representation minimizes memory usage
  - Pre-computed mapping dictionaries avoid runtime calculations
  - Efficient tuple-based move representation

#### Maintainability

* **Extensibility:**
  - Easy to add new piece types or game variants
  - Modular constant definitions support different observation formats
  - Clean separation between types and behavior

#### Error Handling

* **Validation:**
  - Comprehensive piece type validation
  - Clear error messages for invalid operations
  - Defensive programming in symbol conversion

---

### 8. Testing Strategy 🧪

#### Current Test Coverage
- Enum value validation and consistency
- Piece creation and manipulation operations
- Move tuple format validation
- Symbol conversion accuracy

#### Recommended Testing Approach
- Test all piece type promotions and demotions
- Validate observation tensor constant consistency
- Verify notation system conversions
- Test edge cases in piece operations

---

### 9. Security Considerations 🔒

* **Input Validation:**
  - Symbol parsing includes validation against malformed input
  - Enum-based design prevents invalid piece states
  - Bounds checking implicit in type system

---

### 10. Future Enhancements 🚀

#### Potential Improvements
1. **Extended Notation Support:** Add support for additional Shogi notation formats
2. **Performance Optimization:** Consider more efficient move representation for high-performance scenarios
3. **Validation Enhancement:** Add more comprehensive piece placement validation
4. **Internationalization:** Support for multiple language piece representations

#### Backward Compatibility
- Current enum-based design is highly stable
- New piece types or constants can be added without breaking existing code
- Move tuple format designed to be extensible

---

### 11. Dependencies & Requirements 📋

#### Runtime Dependencies
- **Python Standard Library:**
  - `enum`: Core enumeration support
  - `typing`: Type hint system
- **External:** None

#### Development Dependencies
- Testing frameworks for type validation
- Documentation tools for API reference

---

### 12. Configuration 🛠️

#### Observation Tensor Layout
- 46-plane structure optimized for neural network training
- Configurable through constant modification
- Supports different feature extraction strategies

#### Notation Systems
- KIF symbol mappings for Japanese notation
- USI character support for universal format
- SFEN compatibility for position notation

This comprehensive module serves as the foundation for the entire Shogi game engine, providing type safety, performance, and extensibility for all game operations.
