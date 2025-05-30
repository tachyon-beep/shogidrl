# Training Utils Module

## Module Overview

**File**: `keisei/training/utils.py`  
**Purpose**: Provides core utility functions for the Keisei training system, including checkpoint discovery, configuration serialization, directory setup, random seeding, and W&B integration.

This module centralizes common training utilities used across the training pipeline, offering standardized implementations for filesystem operations, experiment management, and external service integration.

## Dependencies

### Internal Dependencies
- None (standalone utility module)

### External Dependencies
- `wandb`: Weights & Biases experiment tracking
- `json`: Configuration serialization
- `os`: Filesystem operations
- `pathlib.Path`: Path manipulation
- `random`: Random number generation
- `sys`: System operations and stderr
- `torch`: PyTorch random seeding
- `typing`: Type annotations

### Standard Library
- `glob`: Pattern-based file discovery
- `functools.partial`: Function partial application

## Function Documentation

### find_latest_checkpoint

```python
def find_latest_checkpoint(checkpoints_dir: str) -> Optional[str]
```

**Purpose**: Discovers the most recent checkpoint file in a directory based on modification time.

**Parameters**:
- `checkpoints_dir` (str): Directory path to search for checkpoints

**Returns**:
- `Optional[str]`: Path to latest checkpoint file, or None if no checkpoints found

**Implementation Notes**:
- Uses glob pattern "*.pt" to find PyTorch checkpoint files
- Sorts by modification time (most recent first)
- Returns None if directory doesn't exist or contains no checkpoints
- Thread-safe operation

### save_config_to_json

```python
def save_config_to_json(config: Any, filepath: str) -> None
```

**Purpose**: Serializes configuration objects to JSON format with proper error handling.

**Parameters**:
- `config` (Any): Configuration object to serialize (typically dataclass or dict)
- `filepath` (str): Target file path for JSON output

**Implementation Notes**:
- Uses `dataclasses.asdict()` if config has `__dataclass_fields__`
- Falls back to direct JSON serialization for dict-like objects
- Creates parent directories if they don't exist
- Handles serialization errors gracefully
- Uses 2-space indentation for readable output

### ensure_directories_exist

```python
def ensure_directories_exist(*directories: str) -> None
```

**Purpose**: Creates directory structure for multiple paths with proper error handling.

**Parameters**:
- `*directories` (str): Variable number of directory paths to create

**Implementation Notes**:
- Uses `Path.mkdir(parents=True, exist_ok=True)` for safe creation
- Handles multiple directories in single call
- Creates parent directories automatically
- Idempotent operation (safe to call multiple times)

### set_seed

```python
def set_seed(seed: int) -> None
```

**Purpose**: Sets random seeds for reproducible training across all random number generators.

**Parameters**:
- `seed` (int): Seed value for reproducibility

**Implementation Notes**:
- Sets Python built-in `random` module seed
- Sets PyTorch manual seed for CPU operations
- Configures PyTorch CUDA for deterministic operations
- Essential for experiment reproducibility
- Should be called early in training initialization

### initialize_wandb

```python
def initialize_wandb(
    run_name: str,
    run_artifact_dir: str,
    project_name: str = "keisei"
) -> bool
```

**Purpose**: Initializes Weights & Biases experiment tracking with comprehensive error handling.

**Parameters**:
- `run_name` (str): Unique identifier for the training run
- `run_artifact_dir` (str): Directory for storing W&B artifacts
- `project_name` (str, optional): W&B project name (default: "keisei")

**Returns**:
- `bool`: True if W&B initialized successfully, False otherwise

**Implementation Notes**:
- Attempts W&B initialization with resume capability
- Uses run_name as W&B run ID for consistent tracking
- Handles multiple exception types (TypeError, ValueError, OSError)
- Provides fallback behavior when W&B unavailable
- Logs initialization status to stderr
- Safe to call multiple times (idempotent)

## Data Structures

### Configuration Serialization Support
The module supports serialization of various configuration types:
- **Dataclass objects**: Automatically converted using `dataclasses.asdict()`
- **Dictionary objects**: Direct JSON serialization
- **Nested structures**: Recursive serialization support

### Error Handling Patterns
Consistent error handling across utilities:
- **FileNotFoundError**: Graceful handling for missing directories/files
- **JSONEncodeError**: Configuration serialization failures
- **W&B Exceptions**: Network and authentication errors
- **Path Errors**: Invalid directory creation attempts

## Inter-Module Relationships

### Upstream Dependencies
- **Training Scripts**: `train.py`, `train_wandb_sweep.py` use utilities for setup
- **Session Manager**: Uses directory creation and config serialization
- **Trainer**: Uses checkpoint discovery and seeding utilities
- **Callback Manager**: Uses W&B initialization for logging callbacks

### Downstream Effects
- **Reproducibility**: Seeding affects all random operations in training
- **Experiment Tracking**: W&B initialization enables comprehensive logging
- **File Organization**: Directory utilities structure training artifacts
- **Configuration Management**: JSON serialization enables config persistence

## Implementation Notes

### Design Patterns
- **Utility Functions**: Stateless functions for maximum reusability
- **Error Resilience**: Graceful degradation when optional services fail
- **Type Safety**: Comprehensive type annotations for all functions
- **Single Responsibility**: Each function has one clear purpose

### Performance Considerations
- **Lazy Evaluation**: W&B only initialized when explicitly called
- **Minimal Overhead**: Utilities designed for low computational cost
- **Efficient File Operations**: Uses pathlib for optimal filesystem access
- **Caching**: Checkpoint discovery can benefit from directory caching

### Platform Compatibility
- **Cross-platform Paths**: Uses pathlib for Windows/Unix compatibility
- **Environment Variables**: Respects W&B configuration from environment
- **Error Messages**: Clear error reporting across platforms

## Testing Strategy

### Unit Testing Approach
```python
def test_find_latest_checkpoint():
    # Test with existing checkpoints
    # Test with empty directory
    # Test with non-existent directory
    
def test_save_config_to_json():
    # Test dataclass serialization
    # Test dict serialization
    # Test error handling
    
def test_ensure_directories_exist():
    # Test directory creation
    # Test nested directory creation
    # Test existing directory handling
    
def test_set_seed():
    # Test reproducibility
    # Test PyTorch seeding
    # Test random module seeding
    
def test_initialize_wandb():
    # Test successful initialization
    # Test failure handling
    # Test resume functionality
```

### Integration Testing
- **Training Pipeline**: Test utilities within complete training workflow
- **Configuration Round-trip**: Test config save/load cycles
- **W&B Integration**: Test experiment tracking end-to-end
- **Checkpoint Discovery**: Test with real checkpoint files

## Performance Considerations

### Computational Overhead
- **Minimal Impact**: All utilities designed for negligible overhead
- **File I/O Optimization**: Efficient directory traversal and JSON operations
- **Memory Usage**: Low memory footprint for all operations
- **Startup Time**: Fast initialization suitable for training scripts

### Scalability Factors
- **Large Checkpoint Directories**: Efficient glob-based discovery
- **Complex Configurations**: JSON serialization scales with config size
- **Multiple Runs**: W&B initialization supports concurrent experiments
- **Directory Creation**: Handles deep directory hierarchies efficiently

## Security Considerations

### Data Protection
- **Configuration Sanitization**: No automatic credential exposure in JSON
- **Path Validation**: Safe directory creation with proper permissions
- **W&B Security**: Relies on W&B's authentication mechanisms
- **File Permissions**: Respects system umask for created files

### Access Control
- **Directory Permissions**: Creates directories with appropriate permissions
- **File Access**: Uses safe file operations with proper error handling
- **Network Security**: W&B communication uses their security protocols
- **Credential Management**: No credential storage in utility functions

## Error Handling

### Exception Categories
- **FileSystem Errors**: Missing directories, permission issues
- **Serialization Errors**: Invalid configuration objects
- **Network Errors**: W&B connectivity issues
- **Validation Errors**: Invalid parameters or configurations

### Recovery Strategies
- **Graceful Degradation**: W&B failures don't stop training
- **Default Fallbacks**: Safe defaults for optional operations
- **Clear Error Messages**: Informative error reporting
- **Logging Integration**: Errors logged to stderr for visibility

## Configuration

### Environment Variables
- **WANDB_API_KEY**: W&B authentication (optional)
- **WANDB_MODE**: W&B operating mode (online/offline/disabled)
- **WANDB_DIR**: Base directory for W&B artifacts
- **PYTHONHASHSEED**: Additional randomness control

### Default Values
- **Project Name**: "keisei" for W&B experiments
- **JSON Indent**: 2 spaces for readable configuration files
- **Checkpoint Pattern**: "*.pt" for PyTorch checkpoints
- **Directory Permissions**: System default umask

## Future Enhancements

### Potential Improvements
- **Caching Layer**: Cache checkpoint discovery results
- **Configuration Validation**: Schema validation for configurations
- **Multiple Backends**: Support for other experiment tracking tools
- **Compression**: Automatic compression for large configuration files
- **Encryption**: Optional encryption for sensitive configurations

### API Extensions
- **Async Support**: Asynchronous versions of I/O operations
- **Streaming**: Streaming JSON serialization for large configs
- **Monitoring**: Health check utilities for training infrastructure
- **Profiling**: Performance profiling utilities

## Usage Examples

### Basic Setup
```python
from keisei.training.utils import (
    ensure_directories_exist,
    set_seed,
    initialize_wandb,
    save_config_to_json
)

# Initialize training environment
set_seed(42)
ensure_directories_exist("checkpoints", "logs", "configs")
save_config_to_json(config, "configs/training_config.json")
wandb_active = initialize_wandb("run_001", "artifacts/run_001")
```

### Checkpoint Management
```python
from keisei.training.utils import find_latest_checkpoint

# Find and load latest checkpoint
checkpoint_path = find_latest_checkpoint("checkpoints/")
if checkpoint_path:
    print(f"Resuming from: {checkpoint_path}")
else:
    print("Starting fresh training")
```

### Configuration Persistence
```python
from dataclasses import dataclass
from keisei.training.utils import save_config_to_json

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs: int = 100

config = TrainingConfig()
save_config_to_json(config, "experiment_config.json")
```

## Maintenance Notes

### Code Quality
- **Type Annotations**: All functions fully typed
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Robust exception management
- **Testing**: High test coverage recommended

### Dependencies
- **W&B Version**: Monitor W&B API changes
- **PyTorch Compatibility**: Ensure seeding works across PyTorch versions
- **Python Version**: Compatible with Python 3.8+
- **Operating System**: Cross-platform compatibility maintained

### Monitoring Points
- **W&B Initialization**: Monitor success/failure rates
- **File Operations**: Watch for permission issues
- **Configuration Size**: Monitor JSON serialization performance
- **Checkpoint Discovery**: Performance with large checkpoint directories
