# Software Documentation Template for Subsystems - shogi_game

## 📘 shogi_game.py as of May 2025

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `/home/john/keisei/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `May 2025`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Implements the main ShogiGame class that orchestrates complete Shogi gameplay, including board state management, move execution, game rules enforcement, and integration with neural network training systems. Serves as the central game engine controller.

* **Key Responsibilities:**
  - Maintain complete game state (board, hands, turn, history)
  - Coordinate move validation and execution through helper modules
  - Provide game reset, initialization, and termination handling
  - Support SFEN notation for position import/export
  - Generate observations for neural network training
  - Track game history and detect repetitions

* **Domain Context:**
  Complete Japanese Chess (Shogi) game implementation designed for both human play and AI training, with full rule compliance, notation support, and deep reinforcement learning integration.

* **High-Level Architecture / Interaction Summary:**
  
  * Acts as the central orchestrator delegating complex operations to specialized modules
  * Integrates with helper modules: shogi_game_io, shogi_move_execution, shogi_rules_logic
  * Provides unified interface for external systems including training and evaluation
  * Maintains comprehensive game state with history tracking for repetition detection

---

### 2. Modules 📦

* **Module Name:** `shogi_game.py`

  * **Purpose:** Implement the main game controller class with complete Shogi functionality.
  * **Design Patterns Used:** 
    - Facade pattern for game operations
    - Delegation pattern for specialized functionality
    - State pattern for game phases
    - Observer pattern for game state changes
  * **Key Functions/Classes Provided:** 
    - `ShogiGame` - Main game engine class
  * **Configuration Surface:**
    * `max_moves_per_game`: Configurable move limit (default 500)
    * Board setup and initial position configuration
    * History tracking and repetition detection settings
  * **Dependencies:**
    * **Internal:**
      - `shogi_core_definitions`: Core types (Color, Piece, PieceType, MoveTuple)
      - `shogi_game_io`: I/O utilities and notation support
      - `shogi_move_execution`: Move validation and execution
      - `shogi_rules_logic`: Game rules and legal move generation
    * **External:**
      - `numpy`: For observation tensor generation
      - `copy`: For deep copying game states
      - `re`: For SFEN notation parsing
      - `typing`: Type hints and annotations
  * **External API Contracts:**
    - Provides complete game interface for training systems
    - Supports standard Shogi notation formats (SFEN, USI)
    - Generates neural network compatible observations
  * **Side Effects / Lifecycle Considerations:**
    - Maintains game history for repetition detection
    - Tracks board positions for state analysis
    - Manages resource cleanup on game reset
  * **Usage Examples:**
    ```python
    # Basic game setup and play
    game = ShogiGame(max_moves_per_game=300)
    game.reset()
    
    # Get legal moves and make a move
    legal_moves = game.get_legal_moves()
    if legal_moves:
        success = game.make_move(legal_moves[0])
    
    # Check game state
    if game.game_over:
        print(f"Game ended: {game.termination_reason}")
        print(f"Winner: {game.winner}")
    
    # Get observation for RL
    obs = game.get_observation()
    ```

---

### 3. Classes 🏛️

* **Class Name:** `ShogiGame`

  * **Defined In Module:** `shogi_game.py`
  * **Purpose:** Complete Shogi game implementation with state management and rule enforcement.
  * **Design Role:** Central game controller that orchestrates all game operations
  * **Inheritance:**
    * **Extends:** `object`
    * **Subclasses:** None (designed as concrete implementation)
  * **Key Attributes/Properties:**
    - `board: List[List[Optional[Piece]]]` - 9x9 game board
    - `hands: Dict[int, Dict[PieceType, int]]` - Player piece inventories
    - `current_player: Color` - Active player
    - `move_count: int` - Total moves played
    - `game_over: bool` - Game termination flag
    - `winner: Optional[Color]` - Winning player if game ended
    - `termination_reason: Optional[str]` - Reason for game termination
    - `move_history: List[Dict[str, Any]]` - Complete move record
    - `board_history: List[Tuple]` - Board state history for repetition detection
  * **Key Methods:**
    - `reset()` - Initialize/reset game to starting position
    - `make_move(move)` - Execute a move with validation
    - `get_legal_moves()` - Generate all legal moves for current player
    - `get_observation()` - Generate neural network observation tensor
    - `is_check()` - Check if current player is in check
    - `from_sfen(sfen)` - Load position from SFEN notation
    - `to_sfen()` - Export current position to SFEN
    - `clone()` - Create deep copy of game state
  * **Interconnections:**
    * **Internal Module Calls:**
      - `shogi_move_execution`: Move validation and execution
      - `shogi_rules_logic`: Legal move generation and rule checks
      - `shogi_game_io`: Observation generation and notation support
    * **External Systems:** RL training environments, evaluation systems
  * **Lifecycle & State:**
    - Initialization: Sets up empty board and hands
    - Active: Maintains game state through move sequence
    - Terminated: Preserves final state and termination information
  * **Threading/Concurrency:**
    - Not thread-safe by design (intended for single-threaded RL training)
    - Clone operations support parallel game instances
  * **Usage Example:**
    ```python
    game = ShogiGame()
    game.reset()
    
    while not game.game_over:
        moves = game.get_legal_moves()
        if moves:
            move = random.choice(moves)  # Simple random policy
            game.make_move(move)
        else:
            break  # No legal moves (shouldn't happen in valid game)
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `reset`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Initialize or reset the game to the standard starting position.
  * **Parameters:** None
  * **Returns:** None
  * **Side Effects:** 
    - Clears board and sets up initial piece positions
    - Resets hands to empty
    - Initializes game state variables
    - Clears move and board history
  * **Usage Example:**
    ```python
    game = ShogiGame()
    game.reset()  # Game ready for play
    ```

* **Function/Method Name:** `make_move`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Execute a move with full validation and state updates.
  * **Parameters:**
    - `move: MoveTuple` - Move to execute (board move or drop)
  * **Returns:** `bool` - True if move was successful, False if invalid
  * **Side Effects:**
    - Updates board state
    - Modifies player hands for captures/drops
    - Advances turn to next player
    - Records move in history
    - Checks for game termination conditions
  * **Preconditions:** Game not terminated, valid move format
  * **Postconditions:** Game state consistent, history updated
  * **Usage Example:**
    ```python
    move = (6, 4, 5, 4, False)  # Move pawn forward
    success = game.make_move(move)
    if success:
        print("Move executed successfully")
    ```

* **Function/Method Name:** `get_legal_moves`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Generate all legal moves for the current player.
  * **Parameters:** None
  * **Returns:** `List[MoveTuple]` - List of all legal moves
  * **Algorithmic Note:** Delegates to shogi_rules_logic for comprehensive move generation
  * **Usage Example:**
    ```python
    legal_moves = game.get_legal_moves()
    print(f"Player has {len(legal_moves)} legal moves")
    ```

* **Function/Method Name:** `get_observation`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Generate neural network observation tensor from current game state.
  * **Parameters:** None
  * **Returns:** `np.ndarray` - 46-plane observation tensor (46, 9, 9)
  * **Algorithmic Note:** Uses shogi_game_io for feature extraction
  * **Usage Example:**
    ```python
    obs = game.get_observation()
    assert obs.shape == (46, 9, 9)
    ```

* **Function/Method Name:** `from_sfen`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Load game position from SFEN (Shogi Forsyth-Edwards Notation) string.
  * **Parameters:**
    - `sfen: str` - SFEN position string
  * **Returns:** None
  * **Side Effects:** Replaces current game state with SFEN position
  * **Raises/Exceptions:** `ValueError` for malformed SFEN strings
  * **Usage Example:**
    ```python
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    game.from_sfen(sfen)
    ```

* **Function/Method Name:** `to_sfen`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Export current game position to SFEN notation string.
  * **Parameters:** None
  * **Returns:** `str` - SFEN representation of current position
  * **Usage Example:**
    ```python
    sfen = game.to_sfen()
    print(f"Current position: {sfen}")
    ```

* **Function/Method Name:** `clone`

  * **Defined In:** `shogi_game.py`
  * **Belongs To:** `ShogiGame`
  * **Purpose:** Create a deep copy of the current game state.
  * **Parameters:** None
  * **Returns:** `ShogiGame` - Independent copy of the game
  * **Usage Example:**
    ```python
    backup = game.clone()
    # Try speculative moves on backup
    ```

---

### 5. Data Structures 📊

* **Structure Name:** `Game State`
  * **Type:** `Class attributes`
  * **Purpose:** Maintain complete Shogi game state
  * **Format:** Multiple interconnected attributes
  * **Fields:**
    - Board: 9x9 grid of Optional[Piece]
    - Hands: Dictionary mapping player colors to piece inventories
    - Game metadata: current player, move count, game status
    - History: Move records and board state history
  * **Validation Constraints:** Board coordinates 0-8, valid piece placements
  * **Used In:** All game operations and state queries

* **Structure Name:** `Move History`
  * **Type:** `List[Dict[str, Any]]`
  * **Purpose:** Track complete game progression for analysis and repetition detection
  * **Format:** List of move records with metadata
  * **Fields:** Move details, captured pieces, game state snapshots
  * **Used In:** Repetition detection, game analysis, debugging

---

### 6. Inter-Module Relationships 🔗

* **Dependency Graph:**
  ```
  shogi_game.py (orchestrator)
    ├── imports → shogi_core_definitions.py (types)
    ├── delegates to → shogi_game_io.py (I/O operations)
    ├── delegates to → shogi_move_execution.py (move processing)
    └── delegates to → shogi_rules_logic.py (rule enforcement)
  ```

* **External Dependencies:**
  - Training systems use ShogiGame as environment
  - Evaluation modules create game instances for testing
  - Features module accesses game state for observation generation

* **Data Flow:**
  - Game state flows to helper modules for processing
  - Move validation results flow back to update state
  - Observations flow to neural network training systems

---

### 7. Implementation Notes 🔧

#### Code Quality & Best Practices

* **Separation of Concerns:**
  - Game orchestration separated from rule implementation
  - Clear delegation to specialized modules
  - Modular design enables independent testing

* **State Management:**
  - Comprehensive state tracking with history
  - Immutable operations where appropriate
  - Clear state transitions and validation

#### Performance Considerations

* **Memory Management:**
  - Efficient board representation with minimal overhead
  - History pruning for long games
  - Optimized observation generation

* **Computational Efficiency:**
  - Lazy evaluation of expensive operations
  - Caching of frequently computed values
  - Efficient move generation and validation

#### Maintainability

* **Extensibility:**
  - Plugin architecture for different rule variants
  - Configurable game parameters
  - Easy integration of new observation formats

#### Error Handling

* **Robust Validation:**
  - Comprehensive move validation before execution
  - Clear error reporting for invalid operations
  - Graceful handling of edge cases

---

### 8. Testing Strategy 🧪

#### Current Test Coverage
- Game initialization and reset functionality
- Move execution and validation
- Legal move generation accuracy
- SFEN import/export correctness
- Observation tensor generation

#### Recommended Testing Approach
- Unit tests for individual methods
- Integration tests for complete game workflows
- Property-based testing for move generation
- Performance tests for long games
- Compliance tests against standard Shogi rules

---

### 9. Security Considerations 🔒

* **Input Validation:**
  - SFEN parsing includes comprehensive validation
  - Move format validation prevents invalid operations
  - Bounds checking for all board operations

* **State Integrity:**
  - Immutable history tracking
  - Consistent state transitions
  - Protection against invalid state modifications

---

### 10. Future Enhancements 🚀

#### Potential Improvements
1. **Performance Optimization:** Implement incremental move generation and board evaluation
2. **Rule Variants:** Support for different Shogi variants and handicap games
3. **Advanced Analysis:** Enhanced repetition detection and game analysis features
4. **Network Play:** Integration with online Shogi protocols and engines

#### Backward Compatibility
- Current API designed for stability
- New features can be added without breaking existing code
- Clear versioning strategy for major changes

---

### 11. Dependencies & Requirements 📋

#### Runtime Dependencies
- **Internal:** All shogi submodules
- **External:** NumPy for observation tensors, standard library modules

#### Development Dependencies
- Testing frameworks for game validation
- Performance profiling tools
- Rule compliance verification tools

---

### 12. Configuration 🛠️

#### Game Parameters
- `max_moves_per_game`: Move limit for training scenarios
- Board setup: Configurable initial positions
- History tracking: Configurable depth and retention

#### Integration Settings
- Observation format: Support for different neural network architectures
- Notation support: Multiple format compatibility
- Performance tuning: Configurable optimization parameters

This comprehensive game engine serves as the core component of the Keisei Shogi DRL system, providing complete, rule-compliant Shogi gameplay suitable for both human interaction and AI training.
