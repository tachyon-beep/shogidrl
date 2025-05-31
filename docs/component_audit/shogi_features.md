# Software Documentation Template for Subsystems - features

## 📘 features.py as of May 2025

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `/home/john/keisei/keisei/shogi/`
**Documentation Version:** `1.0`
**Date:** `May 2025`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Provides a feature specification registry and neural network observation builders for the Keisei Shogi DRL system. Implements configurable feature extraction systems that convert game states into standardized tensor formats optimized for different neural network architectures.

* **Key Responsibilities:**
  - Define and register different feature extraction strategies
  - Implement the core46 observation builder for standard CNN architectures
  - Provide extensible framework for custom feature extraction
  - Support multiple observation formats for different AI models
  - Enable feature experimentation and A/B testing

* **Domain Context:**
  Neural network feature engineering for Shogi deep reinforcement learning, focusing on optimal game state representation for CNN-based architectures while maintaining extensibility for future feature engineering experiments.

* **High-Level Architecture / Interaction Summary:**
  
  * Implements a registry pattern for different feature extraction strategies
  * Provides the standard core46 feature builder used throughout the system
  * Enables easy switching between different observation formats
  * Supports experimental feature extraction for research and optimization

---

### 2. Modules 📦

* **Module Name:** `features.py`

  * **Purpose:** Implement feature extraction registry and core observation builders for neural network training.
  * **Design Patterns Used:** 
    - Registry pattern for feature builder management
    - Strategy pattern for different observation formats
    - Factory pattern for feature specification creation
    - Decorator pattern for feature registration
  * **Key Functions/Classes Provided:** 
    - `FeatureSpec` class - Feature specification container
    - `register_feature()` decorator - Feature registration system
    - `build_core46()` function - Standard 46-plane observation builder
    - `FEATURE_REGISTRY` - Global feature builder registry
  * **Configuration Surface:**
    * Feature builder selection and registration
    * Observation tensor layout configuration
    * Extra feature plane definitions
    * Performance optimization settings
  * **Dependencies:**
    * **Internal:**
      - Uses game state from ShogiGame instances
      - Accesses observation constants from shogi_core_definitions
    * **External:**
      - `numpy`: Tensor operations and array manipulation
      - `typing`: Type annotations and callable specifications
  * **External API Contracts:**
    - Provides standardized observation tensors for neural networks
    - Supports pluggable feature extraction strategies
    - Maintains compatibility with existing CNN architectures
  * **Side Effects / Lifecycle Considerations:**
    - Registry populated at module import time
    - Memory allocation for observation tensors
    - No persistent state between feature extractions
  * **Usage Examples:**
    ```python
    from keisei.shogi.features import FeatureSpec, FEATURE_REGISTRY, build_core46
    
    # Use standard core46 features
    core46_spec = FeatureSpec("core46", build_core46, 46)
    observation = core46_spec.build(game)
    
    # Access registered features
    builder = FEATURE_REGISTRY["core46"]
    obs = builder(game)
    
    # Register custom feature
    @register_feature("custom_features")
    def build_custom(game):
        return np.zeros((64, 9, 9))  # Custom format
    ```

---

### 3. Classes 🏛️

* **Class Name:** `FeatureSpec`

  * **Defined In Module:** `features.py`
  * **Purpose:** Encapsulate feature extraction specifications with metadata and building functionality.
  * **Design Role:** Container class that packages feature builders with their specifications
  * **Inheritance:**
    * **Extends:** `object`
    * **Subclasses:** None (designed as concrete specification container)
  * **Key Attributes/Properties:**
    - `name: str` - Human-readable feature specification name
    - `builder: Callable` - Function that builds observation tensor
    - `num_planes: int` - Number of feature planes in output tensor
  * **Key Methods:**
    - `build(game)` - Generate observation tensor for given game state
  * **Interconnections:**
    * **Internal Module Calls:** Delegates to registered builder functions
    * **External Systems:** Used by training systems to generate observations
  * **Lifecycle & State:**
    - Immutable once created
    - Stateless operation (no instance state maintained)
  * **Usage Example:**
    ```python
    spec = FeatureSpec("core46", build_core46, 46)
    observation = spec.build(game_instance)
    assert observation.shape == (46, 9, 9)
    ```

---

### 4. Functions 🔧

* **Function Name:** `register_feature`

  * **Defined In:** `features.py`
  * **Purpose:** Decorator function for registering feature builders in the global registry.
  * **Parameters:**
    - `name: str` - Name to register the feature builder under
  * **Returns:** `Callable` - Decorator function that registers the wrapped function
  * **Side Effects:** Adds function to FEATURE_REGISTRY dictionary
  * **Usage Example:**
    ```python
    @register_feature("experimental_features")
    def build_experimental(game):
        # Custom feature extraction logic
        return np.zeros((32, 9, 9))
    ```

* **Function Name:** `build_core46`

  * **Defined In:** `features.py`
  * **Purpose:** Build the standard 46-plane observation tensor optimized for CNN architectures.
  * **Parameters:**
    - `game: ShogiGame` - Game instance to extract features from
  * **Returns:** `np.ndarray` - Observation tensor of shape (46, 9, 9)
  * **Algorithmic Note:**
    - Channels 0-27: Board piece positions by player and promotion status
    - Channels 28-41: Hand piece counts for both players
    - Channels 42-45: Meta information (current player, move count, reserved)
  * **Performance:** Optimized for frequent calls during training
  * **Side Effects:** Allocates new numpy array for each call
  * **Usage Example:**
    ```python
    observation = build_core46(game)
    # Feed to neural network
    policy, value = model(torch.from_numpy(observation))
    ```

* **Function Name:** `extract_board_features`

  * **Defined In:** `features.py`
  * **Purpose:** Extract piece position features from game board (internal helper).
  * **Parameters:**
    - `game: ShogiGame` - Game state
    - `obs: np.ndarray` - Observation tensor to populate
  * **Returns:** None (modifies obs in-place)
  * **Algorithmic Note:**
    - Populates board piece planes (channels 0-27)
    - Separates current player and opponent pieces
    - Handles promoted and unpromoted piece categories
  * **Usage Example:**
    ```python
    # Called internally by build_core46
    obs = np.zeros((46, 9, 9))
    extract_board_features(game, obs)
    ```

* **Function Name:** `extract_hand_features`

  * **Defined In:** `features.py`
  * **Purpose:** Extract hand piece count features (internal helper).
  * **Parameters:**
    - `game: ShogiGame` - Game state
    - `obs: np.ndarray` - Observation tensor to populate
  * **Returns:** None (modifies obs in-place)
  * **Algorithmic Note:**
    - Populates hand piece planes (channels 28-41)
    - Creates uniform value planes with piece counts
    - Handles both players' hand inventories
  * **Usage Example:**
    ```python
    # Called internally by build_core46
    extract_hand_features(game, obs)
    ```

* **Function Name:** `extract_meta_features`

  * **Defined In:** `features.py`
  * **Purpose:** Extract game metadata features (internal helper).
  * **Parameters:**
    - `game: ShogiGame` - Game state
    - `obs: np.ndarray` - Observation tensor to populate
  * **Returns:** None (modifies obs in-place)
  * **Algorithmic Note:**
    - Populates meta information planes (channels 42-45)
    - Includes current player indicator and move count
    - Reserves channels for future enhancements
  * **Usage Example:**
    ```python
    # Called internally by build_core46
    extract_meta_features(game, obs)
    ```

---

### 5. Data Structures 📊

* **Structure Name:** `FEATURE_REGISTRY`
  * **Type:** `Dict[str, Callable]`
  * **Purpose:** Global registry mapping feature names to builder functions
  * **Format:** String keys mapping to callable feature builders
  * **Fields:** Feature name strings paired with builder functions
  * **Validation Constraints:** Builder functions must accept game parameter and return numpy array
  * **Used In:** Feature selection and dynamic feature building

* **Structure Name:** `EXTRA_PLANES`
  * **Type:** `Dict[str, int]`
  * **Purpose:** Define indices for potential additional feature planes
  * **Format:** String names mapping to plane indices
  * **Fields:**
    - `"check"`: Check status indication
    - `"repetition"`: Position repetition tracking
    - `"prom_zone"`: Promotion zone highlighting
    - `"last2ply"`: Recent move history
    - `"hand_onehot"`: Alternative hand representation
  * **Used In:** Experimental feature development and enhancement

* **Structure Name:** `Observation Tensor Format`
  * **Type:** `np.ndarray (46, 9, 9)`
  * **Purpose:** Standardized neural network input representation
  * **Format:** 46 feature planes over 9x9 Shogi board
  * **Fields:**
    - Board representation: 28 planes for piece positions
    - Hand representation: 14 planes for piece counts
    - Meta information: 4 planes for game state
  * **Validation Constraints:** Float32 values, normalized ranges
  * **Used In:** All neural network training and inference

---

### 6. Inter-Module Relationships 🔗

* **Dependency Graph:**
  ```
  features.py (feature extraction)
    ├── imports from → numpy (tensor operations)
    ├── accesses → ShogiGame (game state)
    ├── uses → shogi_core_definitions (observation constants)
    └── used by → training systems (observation generation)
  ```

* **Integration Points:**
  - Training systems use registered feature builders
  - ShogiGame provides state data for feature extraction
  - Neural networks consume generated observation tensors

* **Data Flow:**
  - Game state flows in from ShogiGame instances
  - Standardized observation tensors flow out to neural networks
  - Feature specifications flow to training configuration systems

---

### 7. Implementation Notes 🔧

#### Code Quality & Best Practices

* **Extensible Design:**
  - Registry pattern enables easy addition of new feature builders
  - Clean separation between feature specification and implementation
  - Modular helper functions for different feature categories

* **Performance Optimization:**
  - Efficient numpy operations for tensor manipulation
  - In-place modifications to minimize memory allocation
  - Optimized for high-frequency training calls

#### Performance Considerations

* **Memory Efficiency:**
  - Pre-allocated observation tensors where possible
  - Minimal temporary array creation
  - Efficient copying and assignment operations

* **Computational Efficiency:**
  - Vectorized operations using numpy
  - Early termination for invalid game states
  - Optimized loops for board scanning

#### Maintainability

* **Modular Architecture:**
  - Clear separation between different feature categories
  - Easy to add new observation formats
  - Comprehensive documentation and type hints

#### Error Handling

* **Robust Feature Extraction:**
  - Validation of game state before feature extraction
  - Graceful handling of edge cases
  - Clear error messages for invalid configurations

---

### 8. Testing Strategy 🧪

#### Current Test Coverage
- Core46 observation generation accuracy
- Feature registry functionality
- Tensor shape and value validation
- Performance benchmarking

#### Recommended Testing Approach
- Unit tests for individual feature extraction functions
- Integration tests with various game states
- Property-based testing for tensor consistency
- Performance tests for training throughput
- Validation tests against known good observations

---

### 9. Security Considerations 🔒

* **Input Validation:**
  - Validation of game state before feature extraction
  - Bounds checking for array operations
  - Protection against malformed game states

* **Registry Security:**
  - Controlled feature registration process
  - Validation of registered builder functions
  - Protection against malicious feature builders

---

### 10. Future Enhancements 🚀

#### Potential Improvements
1. **Advanced Features:** Implement additional feature planes for enhanced AI performance
2. **Optimization:** GPU-accelerated feature extraction for large-scale training
3. **Experimentation:** A/B testing framework for feature effectiveness
4. **Compression:** Efficient feature representation for memory-constrained environments

#### Backward Compatibility
- Current core46 format is stable and widely used
- New features can be added without breaking existing systems
- Clear versioning strategy for feature format changes

---

### 11. Dependencies & Requirements 📋

#### Runtime Dependencies
- **NumPy:** Core tensor operations and array manipulation
- **Internal:** ShogiGame for state access, core definitions for constants
- **External:** Python typing module for annotations

#### Development Dependencies
- Testing frameworks for feature validation
- Performance profiling tools for optimization
- Visualization tools for feature analysis

---

### 12. Configuration 🛠️

#### Feature Selection
- Registry-based feature builder selection
- Configurable observation tensor layouts
- Support for experimental feature formats

#### Performance Tuning
- Adjustable feature extraction depth
- Optional caching for expensive operations
- Configurable memory allocation strategies

This comprehensive feature extraction module provides the critical bridge between Shogi game states and neural network training, enabling efficient and flexible observation generation for the Keisei DRL system.
