# Software Documentation Template for Subsystems - Training Environment Manager

## 📘 training_env_manager.py as of 2024-12-19

**Project Name:** `Keisei Shogi DRL System`
**Folder Path:** `/keisei/training/env_manager.py`
**Documentation Version:** `1.0`
**Date:** `2024-12-19`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Manages environment setup, configuration, and validation for Shogi RL training including game initialization, action space configuration, and observation space setup.

* **Key Responsibilities:**
  - Initialize and configure ShogiGame environment
  - Setup and validate PolicyOutputMapper for action space management
  - Handle environment seeding for reproducible training
  - Validate environment consistency and functionality
  - Provide environment information and state management

* **Domain Context:**
  Environment management in PPO-based DRL system, specifically handling Shogi game environment integration with reinforcement learning infrastructure.

* **High-Level Architecture / Interaction Summary:**
  
  The EnvManager acts as the bridge between the training infrastructure and the Shogi game environment. It handles initialization of the ShogiGame, sets up the PolicyOutputMapper for action space management, and provides validation and information services. The manager ensures consistent configuration between the game environment and training parameters.

---

### 2. Modules 📦

* **Module Name:** `env_manager.py`

  * **Purpose:** Centralized environment management for Shogi RL training
  * **Design Patterns Used:** Manager pattern for environment lifecycle, Facade pattern for environment complexity
  * **Key Functions/Classes Provided:** 
    - `EnvManager` class for environment orchestration
    - Environment validation and configuration consistency
    - Action space and observation space management
  * **Configuration Surface:** Environment configuration including action space, observation space, and seeding

---

### 3. Classes and Functions 🏗️

#### Class: `EnvManager`

**Purpose:** Central manager for environment setup and configuration during training runs.

**Key Attributes:**
- `config`: Application configuration object (AppConfig)
- `logger_func`: Optional logging function for status messages
- `game`: Optional ShogiGame instance
- `policy_output_mapper`: Optional PolicyOutputMapper instance
- `action_space_size`: Total number of actions in action space
- `obs_space_shape`: Observation space shape tuple (channels, height, width)

**Key Methods:**

##### `__init__(config: AppConfig, logger_func: Optional[Callable] = None)`
- **Purpose:** Initialize the environment manager with configuration
- **Parameters:**
  - `config`: Application configuration object
  - `logger_func`: Optional logging function (defaults to no-op)
- **Return Type:** None
- **Key Behavior:**
  - Stores configuration and logger
  - Initializes environment components to None (lazy initialization)
  - Requires explicit setup_environment() call
- **Usage:** Called during trainer initialization

##### `setup_environment() -> Tuple[ShogiGame, PolicyOutputMapper]`
- **Purpose:** Initialize game environment and policy mapper
- **Parameters:** None
- **Return Type:** Tuple of ShogiGame and PolicyOutputMapper
- **Key Behavior:**
  - Creates ShogiGame instance
  - Sets up environment seeding if specified
  - Initializes PolicyOutputMapper
  - Validates action space consistency
  - Sets observation space shape
- **Exceptions:** Raises RuntimeError on initialization failures
- **Usage:** Called during training setup phase

##### `_validate_action_space() -> None`
- **Purpose:** Validate consistency between config and mapper action counts
- **Parameters:** None
- **Return Type:** None
- **Key Behavior:**
  - Compares config action count with mapper action count
  - Raises ValueError on mismatch
  - Logs validation success
- **Exceptions:** Raises ValueError on action space mismatch
- **Usage:** Called internally during setup_environment()

##### `get_environment_info() -> dict`
- **Purpose:** Retrieve comprehensive environment configuration information
- **Parameters:** None
- **Return Type:** Dictionary with environment details
- **Key Information:**
  - Game and policy mapper instances
  - Action and observation space dimensions
  - Configuration parameters
  - Component type names
- **Usage:** For debugging and environment inspection

##### `reset_game() -> bool`
- **Purpose:** Reset the game environment to initial state
- **Parameters:** None
- **Return Type:** Boolean indicating success
- **Key Behavior:**
  - Calls game.reset() if game is initialized
  - Handles exceptions gracefully
  - Returns success status
- **Usage:** For episode resets during training

##### `initialize_game_state() -> Optional[np.ndarray]`
- **Purpose:** Reset game and return initial observation
- **Parameters:** None
- **Return Type:** Optional numpy array of initial observation
- **Key Behavior:**
  - Calls game.reset() which returns initial observation
  - Logs initialization success
  - Returns None on errors
- **Usage:** For starting new training episodes

##### `validate_environment() -> bool`
- **Purpose:** Comprehensive validation of environment configuration and functionality
- **Parameters:** None
- **Return Type:** Boolean indicating validation success
- **Key Validations:**
  - Game and policy mapper initialization
  - Action space size validity
  - Observation space shape validity
  - Game reset functionality
  - Observation consistency (with warnings for differences)
- **Usage:** Called during training setup for environment verification

##### `get_legal_moves_count() -> int`
- **Purpose:** Get number of legal moves in current game state
- **Parameters:** None
- **Return Type:** Integer count of legal moves
- **Key Behavior:**
  - Calls game.get_legal_moves() and returns count
  - Returns 0 on errors or if no game initialized
- **Usage:** For training statistics and debugging

##### `setup_seeding(seed: Optional[int] = None) -> bool`
- **Purpose:** Setup or update environment seeding
- **Parameters:**
  - `seed`: Optional seed value (uses config seed if None)
- **Return Type:** Boolean indicating success
- **Key Behavior:**
  - Uses provided seed or falls back to config seed
  - Calls game.seed() if method exists
  - Handles missing seed methods gracefully
- **Usage:** For reproducible training runs or re-seeding

---

### 4. Data Structures 📊

#### Environment Information Dictionary

```python
{
    "game": ShogiGame,                    # Game instance
    "policy_mapper": PolicyOutputMapper,  # Action mapper instance
    "action_space_size": int,            # Total action count
    "obs_space_shape": Tuple[int, int, int],  # (channels, height, width)
    "input_channels": int,               # Number of input channels
    "num_actions_total": int,            # Config action count
    "seed": Optional[int],               # Environment seed
    "game_type": str,                    # Game class name
    "policy_mapper_type": str            # Mapper class name
}
```

#### Observation Space Configuration

- **Shape:** `(input_channels, 9, 9)` - Standard Shogi board representation
- **Channels:** Configurable number of feature planes
- **Dimensions:** Fixed 9x9 board size for Shogi

---

### 5. Inter-Module Relationships 🔗

#### Dependencies:
- **`keisei.config_schema`** - AppConfig for configuration management
- **`keisei.shogi`** - ShogiGame for game environment
- **`keisei.utils`** - PolicyOutputMapper for action space management
- **`numpy`** - Array operations for observations

#### Used By:
- **`trainer.py`** - Main training orchestrator for environment setup
- **`training_loop_manager.py`** - Environment state management during training
- **`step_manager.py`** - Game state operations during training steps

#### Provides To:
- **Training Infrastructure** - Configured and validated environment
- **Game Interface** - Abstracted access to Shogi game functionality
- **Action Space Management** - Policy output mapping and validation

---

### 6. Implementation Notes 🔧

#### Lazy Initialization:
- Environment components initialized to None in constructor
- Explicit setup_environment() call required for initialization
- Allows for controlled initialization timing

#### Error Handling Strategy:
- Exceptions caught and converted to RuntimeError with context
- Boolean return values for operations that may fail
- Comprehensive logging for debugging failed operations

#### Validation Approach:
- Multi-level validation including configuration, functionality, and consistency
- Observation comparison with warnings for expected differences
- Graceful degradation for non-critical validation failures

---

### 7. Testing Strategy 🧪

#### Unit Tests:
- Test environment manager initialization with various configurations
- Verify environment setup with valid and invalid configurations
- Test validation logic with different environment states
- Validate error handling for initialization failures

#### Integration Tests:
- Test environment integration with actual ShogiGame
- Verify PolicyOutputMapper consistency with game action space
- Test seeding functionality and reproducibility
- Validate observation space configuration with game output

#### Error Scenarios:
- ShogiGame initialization failures
- PolicyOutputMapper configuration errors
- Action space mismatches between config and implementation
- File system errors during seeding or validation

---

### 8. Performance Considerations ⚡

#### Initialization Overhead:
- Environment setup happens once per training run
- Game and mapper initialization may involve significant computation
- Validation operations add setup time but prevent runtime errors

#### Memory Management:
- Game and mapper instances maintained throughout training
- Observation arrays created frequently during training
- Environment info dictionary created on demand

#### Optimization Strategies:
- Cache environment information to avoid repeated computation
- Validate environment once during setup rather than per episode
- Use efficient numpy operations for observation handling

---

### 9. Security Considerations 🔒

#### Configuration Security:
- Environment configuration may contain sensitive paths or parameters
- No validation of configuration values for security implications
- Seeding values could be used for cryptographic purposes

#### Game Environment Access:
- Full access to game state and methods through manager
- No access controls or sandboxing of game operations
- Environment manager has full system access through game

#### Mitigation Strategies:
- Validate configuration parameters before use
- Implement access controls for sensitive game operations
- Monitor resource usage during environment operations

---

### 10. Error Handling 🚨

#### Initialization Errors:
- ShogiGame creation failures raise RuntimeError with context
- PolicyOutputMapper errors converted to RuntimeError
- Configuration inconsistencies raise ValueError

#### Runtime Errors:
- Game reset failures return False and log errors
- Seeding failures handled gracefully with logging
- Validation errors logged but don't stop operation

#### Recovery Strategies:
- Environment can be re-initialized on failures
- Individual operations can be retried
- Training can continue with degraded environment functionality

---

### 11. Configuration ⚙️

#### Required Configuration:
```python
config.env.input_channels: int         # Number of observation channels
config.env.num_actions_total: int      # Total number of actions
```

#### Optional Configuration:
```python
config.env.seed: Optional[int]         # Environment seed for reproducibility
```

#### Configuration Validation:
- Action space consistency between config and PolicyOutputMapper
- Observation space shape validation (must be 3D)
- Seed value validation when provided

---

### 12. Future Enhancements 🚀

#### Environment Extensions:
- Support for multiple game variants or rule sets
- Dynamic action space configuration
- Advanced observation space transformations
- Environment pooling for parallel training

#### Validation Improvements:
- More comprehensive environment testing
- Performance benchmarking during validation
- Automated configuration optimization
- Environment health monitoring

#### Integration Features:
- Environment versioning and compatibility checking
- Hot-swapping of environment components
- Environment state serialization and restoration
- Distributed environment management

---

### 13. Usage Examples 💡

#### Basic Environment Setup:
```python
# Initialize environment manager
env_manager = EnvManager(config, logger_func=print)

# Setup environment components
game, policy_mapper = env_manager.setup_environment()

# Validate environment
if env_manager.validate_environment():
    print("Environment ready for training")
```

#### Environment Information and Debugging:
```python
# Get comprehensive environment info
env_info = env_manager.get_environment_info()
print(f"Action space size: {env_info['action_space_size']}")
print(f"Observation shape: {env_info['obs_space_shape']}")

# Check legal moves in current state
legal_moves_count = env_manager.get_legal_moves_count()
print(f"Current legal moves: {legal_moves_count}")
```

#### Training Episode Management:
```python
# Initialize new episode
initial_obs = env_manager.initialize_game_state()
if initial_obs is not None:
    # Training logic with initial observation
    pass

# Reset environment during training
if env_manager.reset_game():
    # Continue with reset environment
    pass
```

#### Seeding and Reproducibility:
```python
# Setup initial seeding
env_manager.setup_seeding(seed=42)

# Re-seed during training for reproducibility
env_manager.setup_seeding(seed=123)
```

---

### 14. Maintenance Notes 📝

#### Regular Maintenance:
- Monitor environment initialization times for performance regressions
- Review validation logic when game or mapper implementations change
- Update configuration schema when environment capabilities expand

#### Version Compatibility:
- ShogiGame interface changes require manager updates
- PolicyOutputMapper changes need action space validation updates
- Configuration schema changes require validation logic updates

#### Code Quality:
- Maintain consistent error handling patterns
- Keep validation logic comprehensive but efficient
- Document environment requirements and assumptions

---

### 15. Related Documentation 📚

- **`shogi_game.md`** - ShogiGame implementation and interface
- **`utils_policy_output_mapper.md`** - Action space mapping documentation
- **`training_trainer.md`** - Main trainer integration with environment manager
- **`config_schema.md`** - Configuration schema and validation
- **`training_step_manager.md`** - Environment interaction during training steps
