# Software Documentation Template for Subsystems - shogi_game_io

## 📘 shogi_game_io.py as of May 2025

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `/home/john/keisei/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `May 2025`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Provides input/output utilities for the Shogi game engine including neural network observation generation, game notation support (KIF, USI, SFEN), and feature extraction for deep reinforcement learning training systems.

* **Key Responsibilities:**
  - Generate neural network observation tensors from game state
  - Support multiple Shogi notation formats (KIF, USI, SFEN)
  - Provide board visualization and debugging utilities
  - Handle game import/export operations
  - Convert between different position representations

* **Domain Context:**
  Bridges the gap between Shogi game representation and external systems including neural networks, notation standards, and visualization tools, enabling interoperability and training data generation.

* **High-Level Architecture / Interaction Summary:**
  
  * Serves as the I/O interface layer for the Shogi game engine
  * Provides observation generation for neural network training
  * Supports standard Shogi notation formats for position exchange
  * Enables debugging and analysis through visualization utilities

---

### 2. Modules 📦

* **Module Name:** `shogi_game_io.py`

  * **Purpose:** Provide comprehensive I/O utilities for Shogi game operations and neural network integration.
  * **Design Patterns Used:** 
    - Facade pattern for I/O operations
    - Strategy pattern for different notation formats
    - Factory pattern for observation generation
  * **Key Functions/Classes Provided:** 
    - `generate_neural_network_observation()` - Core observation tensor generation
    - SFEN import/export utilities
    - KIF notation support functions
    - Board visualization utilities
  * **Configuration Surface:**
    * Observation tensor layout (46-plane structure)
    * Notation format preferences
    * Visualization output options
  * **Dependencies:**
    * **Internal:**
      - `shogi_core_definitions`: Constants, types, and observation plane definitions
    * **External:**
      - `numpy`: Observation tensor operations
      - `datetime`: Timestamp generation for notation
      - `re`: Regular expression parsing for notation formats
      - `typing`: Type annotations
  * **External API Contracts:**
    - Provides observation tensors compatible with neural networks
    - Supports standard Shogi notation interchange
    - Enables position analysis and debugging
  * **Side Effects / Lifecycle Considerations:**
    - No persistent state (stateless utility functions)
    - Memory allocation for observation tensors
  * **Usage Examples:**
    ```python
    from keisei.shogi.shogi_game_io import generate_neural_network_observation
    
    # Generate observation for training
    obs = generate_neural_network_observation(game)
    assert obs.shape == (46, 9, 9)
    
    # Use in training loop
    state_tensor = torch.from_numpy(obs).float()
    ```

---

### 3. Classes 🏛️

**No classes defined in this module.** This module provides utility functions for I/O operations.

---

### 4. Functions 🔧

* **Function Name:** `generate_neural_network_observation`

  * **Defined In:** `shogi_game_io.py`
  * **Purpose:** Generate standardized 46-plane observation tensor from game state for neural network training.
  * **Parameters:**
    - `game: ShogiGame` - Game instance to extract features from
  * **Returns:** `np.ndarray` - Observation tensor of shape (46, 9, 9)
  * **Algorithmic Note:**
    - Channels 0-27: Board piece positions (current/opponent, unpromoted/promoted)
    - Channels 28-41: Hand piece counts for both players
    - Channels 42-45: Meta information (current player, move count, reserved)
  * **Performance:** Optimized for frequent calls during training
  * **Usage Example:**
    ```python
    obs = generate_neural_network_observation(game)
    # Use as input to neural network
    policy_logits, value = model(torch.from_numpy(obs))
    ```

* **Function Name:** `move_to_usi_string`

  * **Defined In:** `shogi_game_io.py`
  * **Purpose:** Convert MoveTuple to USI (Universal Shogi Interface) notation string.
  * **Parameters:**
    - `move: MoveTuple` - Move to convert
    - `game: ShogiGame` - Game context for validation
  * **Returns:** `str` - USI formatted move string
  * **Raises/Exceptions:** `ValueError` for invalid moves
  * **Usage Example:**
    ```python
    move = (6, 4, 5, 4, False)  # Board move
    usi_str = move_to_usi_string(move, game)  # "5e5d"
    
    drop_move = (None, None, 4, 4, PieceType.PAWN)
    usi_str = move_to_usi_string(drop_move, game)  # "P*5e"
    ```

* **Function Name:** `board_to_ascii`

  * **Defined In:** `shogi_game_io.py`
  * **Purpose:** Generate ASCII representation of game board for debugging and visualization.
  * **Parameters:**
    - `game: ShogiGame` - Game to visualize
    - `show_coordinates: bool = True` - Whether to show row/column labels
  * **Returns:** `str` - ASCII art representation of board
  * **Usage Example:**
    ```python
    ascii_board = board_to_ascii(game)
    print(ascii_board)  # Pretty-printed board with pieces
    ```

* **Function Name:** `sfen_to_game_state`

  * **Defined In:** `shogi_game_io.py`
  * **Purpose:** Parse SFEN notation string and update game state accordingly.
  * **Parameters:**
    - `game: ShogiGame` - Game instance to update
    - `sfen: str` - SFEN position string
  * **Returns:** None (modifies game in-place)
  * **Raises/Exceptions:** `ValueError` for malformed SFEN strings
  * **Side Effects:** Completely replaces current game state
  * **Usage Example:**
    ```python
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    sfen_to_game_state(game, sfen)
    ```

* **Function Name:** `game_state_to_sfen`

  * **Defined In:** `shogi_game_io.py`
  * **Purpose:** Export current game state to SFEN notation string.
  * **Parameters:**
    - `game: ShogiGame` - Game to export
  * **Returns:** `str` - SFEN representation of current position
  * **Usage Example:**
    ```python
    sfen = game_state_to_sfen(game)
    print(f"Position: {sfen}")
    ```

---

### 5. Data Structures 📊

* **Structure Name:** `Observation Tensor Layout`
  * **Type:** `np.ndarray (46, 9, 9)`
  * **Purpose:** Standardized neural network input format
  * **Format:** 46 feature planes over 9x9 board
  * **Fields:**
    - Board planes (0-27): Piece positions by type and player
    - Hand planes (28-41): Piece counts in hands
    - Meta planes (42-45): Game state information
  * **Validation Constraints:** Float32 values, binary for piece positions, counts for hands
  * **Used In:** Neural network training and inference

* **Structure Name:** `Notation Format Mappings`
  * **Type:** `Dict[str, str]`
  * **Purpose:** Support conversion between different notation systems
  * **Format:** String mappings for piece symbols and move representations
  * **Used In:** Import/export operations and notation conversion

---

### 6. Inter-Module Relationships 🔗

* **Dependency Graph:**
  ```
  shogi_game_io.py (I/O utilities)
    ├── imports from → shogi_core_definitions.py (constants, types)
    ├── used by → shogi_game.py (observation generation)
    └── used by → features.py (feature extraction)
  ```

* **External Usage:**
  - Training systems use observation generation functions
  - Evaluation modules use notation conversion utilities
  - Debugging tools use visualization functions

* **Data Flow:**
  - Game state flows in for observation generation
  - Observation tensors flow out to neural networks
  - Notation strings flow bidirectionally for import/export

---

### 7. Implementation Notes 🔧

#### Code Quality & Best Practices

* **Stateless Design:**
  - All functions are stateless utilities
  - No side effects except where explicitly documented
  - Pure functions enable easy testing and reasoning

* **Performance Optimization:**
  - Efficient NumPy operations for observation generation
  - Minimal memory allocations during tensor creation
  - Optimized for high-frequency training calls

#### Performance Considerations

* **Memory Efficiency:**
  - Pre-allocated observation tensors where possible
  - Efficient string operations for notation conversion
  - Minimal temporary object creation

* **Computational Efficiency:**
  - Vectorized operations using NumPy
  - Cached computation results where appropriate
  - Optimized loops for board scanning

#### Maintainability

* **Modular Functions:**
  - Clear separation of concerns between different I/O operations
  - Easy to add new notation formats or observation layouts
  - Comprehensive type hints and documentation

#### Error Handling

* **Input Validation:**
  - Robust parsing of notation strings
  - Clear error messages for malformed input
  - Graceful handling of edge cases

---

### 8. Testing Strategy 🧪

#### Current Test Coverage
- Observation tensor generation accuracy
- Notation format conversion correctness
- Board visualization output validation
- Error handling for malformed input

#### Recommended Testing Approach
- Unit tests for each utility function
- Property-based testing for notation round-trips
- Performance tests for observation generation
- Integration tests with actual game states

---

### 9. Security Considerations 🔒

* **Input Sanitization:**
  - Validation of notation strings before parsing
  - Bounds checking for array operations
  - Protection against malformed input attacks

---

### 10. Future Enhancements 🚀

#### Potential Improvements
1. **Extended Formats:** Support for additional notation formats (CSA, PSN)
2. **Visualization Enhancement:** Rich text and graphical output options
3. **Performance Optimization:** JIT compilation for observation generation
4. **Compression Support:** Efficient storage formats for large datasets

#### Backward Compatibility
- Current functions designed for API stability
- New formats can be added without breaking existing code
- Observation layout changes require versioning strategy

---

### 11. Dependencies & Requirements 📋

#### Runtime Dependencies
- **NumPy:** Core numerical operations and tensor generation
- **Python Standard Library:** datetime, re, typing modules
- **Internal:** shogi_core_definitions for constants and types

#### Development Dependencies
- Testing frameworks for I/O validation
- Performance profiling tools for optimization
- Documentation tools for function reference

---

### 12. Configuration 🛠️

#### Observation Format
- 46-plane tensor layout optimized for CNN architectures
- Configurable through constant modification
- Support for different neural network input requirements

#### Notation Support
- Multiple format compatibility (USI, SFEN, KIF)
- Extensible design for new notation systems
- Configurable output formatting options

This comprehensive I/O module serves as the bridge between the Shogi game engine and external systems, providing essential utilities for neural network training, position analysis, and interoperability with standard Shogi tools.
