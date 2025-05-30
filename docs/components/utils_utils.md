# Utils Core Module

## Module Overview

The `utils/utils.py` module serves as the central utilities hub for the Keisei Shogi system, containing essential infrastructure classes and functions. This module provides the PolicyOutputMapper for neural network-game interface translation, logging infrastructure for both training and evaluation, configuration loading with override support, opponent base classes, and utility functions for run name generation. It acts as the backbone for shared functionality across the entire system.

## Dependencies

### External Dependencies
- `datetime`: Timestamp generation and formatting
- `json`: JSON configuration file parsing
- `logging`: Standard Python logging infrastructure
- `os`: File system operations and path manipulation
- `sys`: System-specific parameters and functions
- `abc`: Abstract base class functionality
- `torch`: PyTorch tensors and device management
- `yaml`: YAML configuration file parsing
- `pydantic`: Data validation and parsing (ValidationError)
- `rich.console`: Rich terminal output and formatting
- `rich.text`: Rich text objects for terminal UI

### Internal Dependencies
- `keisei.config_schema`: AppConfig configuration classes
- `keisei.shogi.shogi_core_definitions`: Core Shogi types and enums

## Class Documentation

### BaseOpponent (Abstract Base Class)

Abstract base class defining the interface for all game opponents in the system.

**Purpose:** Provides a consistent interface for different opponent implementations.

**Methods:**
```python
@abstractmethod
def select_move(self, game_instance: "ShogiGame") -> "MoveTuple":
    """
    Selects a move given the current game state.
    
    Args:
        game_instance: The current instance of the ShogiGame.
        
    Returns:
        A MoveTuple representing the selected move.
    """
```

### PolicyOutputMapper

Critical class that maps between Shogi moves and neural network policy output indices, enabling the translation between game logic and neural network representations.

**Purpose:** Bridges the gap between human-readable move representations and neural network action spaces.

**Attributes:**
- `idx_to_move: List[MoveTuple]` - Maps indices to move tuples
- `move_to_idx: Dict[MoveTuple, int]` - Maps move tuples to indices
- `_unrecognized_moves_log_cache: Set[str]` - Cache for logging distinct unrecognized moves
- `_USI_DROP_PIECE_CHARS: Dict[PieceType, str]` - USI character mappings for drops

**Key Methods:**

#### `__init__()`
Initializes the mapper by generating all possible move representations:

**Board Move Generation:**
```python
# Generate all possible board moves: (from_r, from_c, to_r, to_c, promote_flag)
for r_from in range(9):
    for c_from in range(9):
        for r_to in range(9):
            for c_to in range(9):
                if r_from == r_to and c_from == c_to:
                    continue  # Skip null moves
                
                # Non-promotion move
                move_no_promo = (r_from, c_from, r_to, c_to, False)
                # Promotion move
                move_promo = (r_from, c_from, r_to, c_to, True)
```

**Drop Move Generation:**
```python
# Generate all possible drop moves: (None, None, to_r, to_c, piece_type)
for r_to in range(9):
    for c_to in range(9):
        for piece_type_enum in hand_piece_types:
            drop_move = (None, None, r_to, c_to, piece_type_enum)
```

#### `get_total_actions() -> int`
Returns the total number of unique actions in the policy output space.

#### `shogi_move_to_policy_index(move: MoveTuple) -> int`
Converts a Shogi MoveTuple to its corresponding policy network index.

**Error Handling:**
- Attempts heuristic matching for DropMoveTuple enum identity issues
- Provides detailed error messages for unmapped moves
- Critical failure on unmapped moves to prevent corrupted experiments

#### `policy_index_to_shogi_move(idx: int) -> MoveTuple`
Converts a policy index back to its Shogi MoveTuple representation.

#### `get_legal_mask(legal_shogi_moves: List[MoveTuple], device: torch.device) -> torch.Tensor`
Creates a boolean mask indicating which actions in the policy output are legal.

**Implementation:**
```python
mask = torch.zeros(self.get_total_actions(), dtype=torch.bool, device=device)
for move in legal_shogi_moves:
    idx = self.shogi_move_to_policy_index(move)
    mask[idx] = True
return mask
```

#### `shogi_move_to_usi(move_tuple: MoveTuple) -> str`
Converts a Shogi MoveTuple to its USI (Universal Shogi Interface) string representation.

**USI Format Examples:**
- Board move: `"7g7f"` (from 7g to 7f)
- Promotion move: `"2b3a+"` (from 2b to 3a with promotion)
- Drop move: `"P*5e"` (pawn drop to 5e)

#### `usi_to_shogi_move(usi_move_str: str) -> MoveTuple`
Converts a USI string back to its Shogi MoveTuple representation.

### TrainingLogger

Comprehensive logging infrastructure for training processes with support for file logging, rich terminal output, and W&B integration.

**Purpose:** Provides unified logging interface for training processes with multiple output targets.

**Attributes:**
- `log_file_path: str` - Path to the log file
- `log_file: Optional[TextIO]` - File handle for log output
- `rich_console: Optional[Console]` - Rich console for terminal output
- `rich_log_panel: Optional[List[Text]]` - Rich text panel for TUI display
- `also_stdout_if_no_rich: bool` - Fallback to stdout when rich not available

**Key Methods:**

#### `__init__(log_file_path, rich_console=None, rich_log_panel=None, also_stdout=None)`
Initializes the training logger with configurable output targets.

#### Context Manager Support:
```python
def __enter__(self) -> "TrainingLogger":
    self.log_file = open(self.log_file_path, "a", encoding="utf-8")
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    if self.log_file:
        self.log_file.close()
```

#### `log(message: str)`
Logs a message to all configured output targets with timestamp.

**Output Targets:**
1. **File Logging**: Appends to log file with timestamp
2. **Rich Console**: Adds to rich log panel for TUI display
3. **Stdout Fallback**: Prints to stderr when rich not available

### EvaluationLogger

Specialized logging infrastructure for evaluation processes with simplified output handling.

**Purpose:** Provides focused logging for evaluation runs with file and console output.

**Attributes:**
- `log_file_path: str` - Path to the evaluation log file
- `also_stdout: bool` - Whether to also print to stdout
- `log_file: Optional[TextIO]` - File handle for log output

**Key Methods:**

#### `__init__(log_file_path: str, also_stdout: bool = True)`
Initializes the evaluation logger with file and console output options.

#### `log(message: str)`
Logs evaluation messages with timestamp to file and optionally console.

## Function Documentation

### Configuration Loading Functions

#### `load_config(config_path=None, cli_overrides=None) -> AppConfig`

Comprehensive configuration loading with hierarchical override support.

**Loading Strategy:**
1. **Base Configuration**: Loads `default_config.yaml` as foundation
2. **File Overrides**: Merges in config_path overrides if provided
3. **CLI Overrides**: Applies command-line overrides on top
4. **Validation**: Validates final configuration with Pydantic

**Override Mapping:**
```python
FLAT_KEY_TO_NESTED = {
    "SEED": "env.seed",
    "DEVICE": "env.device", 
    "TOTAL_TIMESTEPS": "training.total_timesteps",
    "LEARNING_RATE": "training.learning_rate",
    # ... additional mappings
}
```

**Support for Multiple Formats:**
- **YAML Files**: `.yaml` and `.yml` extensions
- **JSON Files**: `.json` extension
- **Flat Overrides**: Uppercase keys mapped to nested structure
- **Nested Overrides**: Direct nested dictionary structure

#### Helper Functions:

**`_load_yaml_or_json(path: str) -> dict`**
Loads configuration from YAML or JSON files based on extension.

**`_merge_overrides(config_data: dict, overrides: dict)`**
Merges override dictionaries into base configuration using dot notation.

**`_map_flat_overrides(overrides: dict) -> dict`**
Maps flat uppercase keys to nested configuration paths.

### Utility Functions

#### `generate_run_name(config: AppConfig, run_name: Optional[str] = None) -> str`

Generates unique run names for training and evaluation sessions.

**Format:** `{prefix}_{model_type}_{features}_{timestamp}`

**Example:** `"keisei_resnet_feats_core46_20231027_153000"`

## Data Structures

### Move Mapping Structure
```python
policy_mapping = {
    "total_actions": int,                    # Total number of possible actions
    "board_moves": List[BoardMoveTuple],     # All possible board moves
    "drop_moves": List[DropMoveTuple],       # All possible drop moves
    "move_to_index": Dict[MoveTuple, int],   # Move to index mapping
    "index_to_move": Dict[int, MoveTuple]    # Index to move mapping
}
```

### Configuration Override Structure
```python
config_override = {
    "flat_keys": {
        "SEED": 42,
        "DEVICE": "cuda:0",
        "TOTAL_TIMESTEPS": 1000000
    },
    "nested_keys": {
        "env.seed": 42,
        "training.learning_rate": 3e-4,
        "wandb.enabled": True
    }
}
```

### USI Character Mappings
```python
USI_DROP_PIECE_CHARS = {
    PieceType.PAWN: "P",
    PieceType.LANCE: "L", 
    PieceType.KNIGHT: "N",
    PieceType.SILVER: "S",
    PieceType.GOLD: "G",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R"
}
```

## Inter-Module Relationships

### Core System Integration
- **Neural Networks**: PolicyOutputMapper provides critical interface
- **Training System**: TrainingLogger provides logging infrastructure
- **Evaluation System**: EvaluationLogger and opponent classes
- **Configuration Management**: Central configuration loading for all modules

### Game Logic Integration
- **Shogi Game**: Move mapping and USI conversion integration
- **Move Execution**: Policy indices translated to game moves
- **Legal Move Generation**: Legal mask creation for policy networks

### External Tool Integration
- **W&B Integration**: Logging infrastructure supports W&B
- **Rich Terminal**: Advanced terminal UI through rich integration
- **USI Compatibility**: Standard interface for Shogi engines

## Implementation Notes

### Policy Mapping Design
- **Complete Coverage**: Maps all possible moves in Shogi
- **Deterministic Ordering**: Consistent index assignment across runs
- **Error Detection**: Critical failures prevent corrupted experiments
- **Memory Efficient**: Pre-computed mappings avoid runtime computation

### Logging Architecture
- **Multiple Targets**: Simultaneous logging to file, console, and TUI
- **Context Managers**: Proper resource management for file handles
- **Rich Integration**: Advanced terminal formatting and display
- **Fallback Behavior**: Graceful degradation when rich unavailable

### Configuration Management
- **Hierarchical Overrides**: Base → file → CLI override chain
- **Format Flexibility**: Support for YAML, JSON, and flat key formats
- **Validation**: Pydantic-based validation ensures configuration integrity
- **Default Handling**: Robust default configuration loading

## Testing Strategy

### Unit Tests
```python
def test_policy_mapper_completeness():
    """Test that PolicyOutputMapper covers all possible moves."""
    # Test total action count
    # Verify bidirectional mapping consistency
    # Test edge cases and boundary conditions
    pass

def test_usi_conversion():
    """Test USI string conversion accuracy."""
    # Test board move conversion
    # Test drop move conversion
    # Test promotion move conversion
    pass

def test_configuration_loading():
    """Test configuration loading and override mechanisms."""
    # Test base configuration loading
    # Test file override application
    # Test CLI override application
    pass

def test_logging_infrastructure():
    """Test logging functionality."""
    # Test file logging
    # Test rich console integration
    # Test context manager behavior
    pass
```

### Integration Tests
```python
def test_neural_network_integration():
    """Test PolicyOutputMapper with actual neural networks."""
    # Test policy output to move conversion
    # Test legal mask generation
    # Test batch processing
    pass

def test_training_logging_integration():
    """Test logging in actual training context."""
    # Test W&B integration
    # Test rich display integration
    # Test concurrent logging
    pass
```

### Performance Tests
```python
def test_policy_mapping_performance():
    """Test PolicyOutputMapper performance."""
    # Benchmark move to index conversion
    # Benchmark index to move conversion
    # Test legal mask generation speed
    pass
```

## Performance Considerations

### Policy Mapping Optimization
- **Pre-computation**: All mappings generated at initialization
- **Dictionary Lookups**: O(1) average case for move-to-index mapping
- **Memory vs. Speed**: Trade-off between memory usage and lookup speed
- **Batch Operations**: Efficient tensor operations for legal masks

### Logging Performance
- **Buffered I/O**: File writing uses buffered output
- **Lazy Formatting**: String formatting only when needed
- **Rich Integration**: Efficient terminal updates through rich
- **Resource Management**: Proper cleanup and resource management

### Configuration Loading
- **Caching**: Configuration validation cached where possible
- **Lazy Loading**: Components loaded only when needed
- **Override Efficiency**: Efficient dictionary merging operations

## Security Considerations

### Input Validation
- **Configuration Validation**: Pydantic validation prevents invalid configurations
- **Path Security**: Safe handling of file paths and configuration locations
- **Type Safety**: Strong typing prevents type-related security issues

### Resource Management
- **File Handle Management**: Proper cleanup of file resources
- **Memory Bounds**: Bounded memory usage for move mappings
- **Error Isolation**: Errors contained within appropriate boundaries

## Error Handling

### Exception Categories
```python
# Configuration errors
try:
    config = AppConfig.parse_obj(config_data)
except ValidationError as e:
    print("Configuration validation error:")
    print(e)
    raise

# Policy mapping errors
def shogi_move_to_policy_index(self, move):
    idx = self.move_to_idx.get(move)
    if idx is None:
        raise ValueError(f"Move {move} not found in PolicyOutputMapper")
    return idx

# File I/O errors
def __enter__(self):
    try:
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
    except IOError as e:
        # Handle file opening errors
        raise
```

### Recovery Strategies
- **Graceful Degradation**: Fallback behaviors for non-critical failures
- **Error Context**: Detailed error messages with context information
- **Resource Cleanup**: Proper cleanup even in error conditions
- **State Preservation**: Maintain valid state during error recovery

## Configuration

### Policy Mapper Configuration
- No external configuration required - self-initializing
- Move mappings are deterministic and complete
- Compatible with all standard Shogi rule variants

### Logging Configuration
```python
training_logger = TrainingLogger(
    log_file_path="/path/to/training.log",
    rich_console=console,              # Optional rich console
    rich_log_panel=log_panel,         # Optional rich log panel
    also_stdout=True                  # Fallback to stdout
)

evaluation_logger = EvaluationLogger(
    log_file_path="/path/to/evaluation.log",
    also_stdout=True                  # Console output flag
)
```

## Future Enhancements

### Planned Features
- **Policy Mapping Optimization**: More efficient move representation
- **Advanced Logging**: Structured logging with metadata
- **Configuration Validation**: Enhanced validation with custom rules
- **Performance Monitoring**: Built-in performance monitoring

### Extensibility Points
- **Custom Policy Mappings**: Support for alternative move representations
- **Pluggable Loggers**: Custom logger implementations
- **Configuration Backends**: Alternative configuration sources
- **USI Extensions**: Support for USI protocol extensions

## Usage Examples

### Policy Mapping Usage
```python
# Initialize policy mapper
policy_mapper = PolicyOutputMapper()

# Convert move to policy index
move = (6, 6, 5, 6, False)  # 7g to 7f
policy_idx = policy_mapper.shogi_move_to_policy_index(move)

# Convert policy index back to move
recovered_move = policy_mapper.policy_index_to_shogi_move(policy_idx)

# Generate legal mask for neural network
legal_moves = game.get_legal_moves()
legal_mask = policy_mapper.get_legal_mask(legal_moves, device)
```

### Logging Usage
```python
# Training logging
with TrainingLogger("/path/to/training.log") as logger:
    logger.log("Training started")
    logger.log(f"Epoch {epoch}: Loss = {loss:.4f}")

# Evaluation logging
with EvaluationLogger("/path/to/eval.log", also_stdout=True) as logger:
    logger.log(f"Evaluation result: Win rate = {win_rate:.2%}")
```

### Configuration Loading
```python
# Load with overrides
config = load_config(
    config_path="custom_config.yaml",
    cli_overrides={
        "TOTAL_TIMESTEPS": 2000000,
        "DEVICE": "cuda:1",
        "wandb.enabled": True
    }
)

# Generate run name
run_name = generate_run_name(config, "custom_experiment")
```

## Maintenance Notes

### Code Quality
- Maintain comprehensive documentation for all public interfaces
- Follow consistent error handling patterns across all classes
- Keep move mapping logic deterministic and well-tested
- Document any changes to USI conversion logic

### Dependencies
- Monitor PyTorch API changes for tensor operations
- Keep Pydantic compatibility for configuration validation
- Update rich integration with library changes
- Maintain YAML/JSON parsing compatibility

### Testing Requirements
- Unit test coverage for all public methods
- Integration tests with real training and evaluation scenarios
- Performance benchmarks for critical operations
- Regression tests for move mapping consistency
