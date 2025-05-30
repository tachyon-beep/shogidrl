# Session Manager Documentation

## Module Overview

**File:** `keisei/training/session_manager.py`

**Purpose:** Manages session-level lifecycle for training runs, including directory setup, WandB integration, configuration persistence, and session logging.

**Core Functionality:**
- Run name generation and validation
- Directory structure creation and management
- Configuration serialization and persistence
- Weights & Biases initialization and management
- Session logging and reporting
- Seeding and reproducibility setup

## Dependencies

### Internal Dependencies
- `keisei.config_schema`: AppConfig for configuration management
- `keisei.utils.utils`: Utility functions for setup operations
- `.utils`: Training-specific utilities

### External Dependencies
- `os`: Operating system interface for directory operations
- `sys`: System utilities for error output
- `datetime`: Timestamp generation for session tracking
- `wandb`: Weights & Biases integration for experiment tracking

## Class Documentation

### `SessionManager`

**Purpose:** Manages the complete lifecycle of a training session from initialization to finalization.

**Attributes:**

#### Configuration and Arguments
- `config` (AppConfig): Complete application configuration
- `args` (Any): Command-line arguments

#### Session Identity
- `_run_name` (str): Unique identifier for the training run

#### Directory Paths (Private, accessed via properties)
- `_run_artifact_dir` (Optional[str]): Main directory for run artifacts
- `_model_dir` (Optional[str]): Directory for model checkpoints
- `_log_file_path` (Optional[str]): Path to training log file
- `_eval_log_file_path` (Optional[str]): Path to evaluation log file

#### WandB State
- `_is_wandb_active` (Optional[bool]): Whether Weights & Biases is active

**Methods:**

#### `__init__(self, config: AppConfig, args: Any, run_name: Optional[str] = None)`
**Purpose:** Initialize the SessionManager with run name determination.

**Parameters:**
- `config` (AppConfig): Application configuration
- `args` (Any): Command-line arguments
- `run_name` (Optional[str]): Explicit run name override

**Run Name Resolution Priority:**
1. Explicit `run_name` parameter
2. Command-line argument (`args.run_name`)
3. Configuration setting (`config.logging.run_name`)
4. Auto-generated name using `generate_run_name()`

#### Property Methods

#### `run_name -> str`
**Purpose:** Get the resolved run name for this session.

#### `run_artifact_dir -> str`
**Purpose:** Get the main artifact directory path.
**Raises:** RuntimeError if directories not yet set up.

#### `model_dir -> str`
**Purpose:** Get the model checkpoint directory path.
**Raises:** RuntimeError if directories not yet set up.

#### `log_file_path -> str`
**Purpose:** Get the training log file path.
**Raises:** RuntimeError if directories not yet set up.

#### `eval_log_file_path -> str`
**Purpose:** Get the evaluation log file path.
**Raises:** RuntimeError if directories not yet set up.

#### `is_wandb_active -> bool`
**Purpose:** Check if Weights & Biases tracking is active.
**Raises:** RuntimeError if WandB not yet initialized.

#### Setup Methods

#### `setup_directories(self) -> Dict[str, str]`
**Purpose:** Create and configure directory structure for the training run.

**Returns:** Dictionary containing all directory paths
- `"run_artifact_dir"`: Main run directory
- `"model_dir"`: Model checkpoint directory
- `"log_file_path"`: Training log file path
- `"eval_log_file_path"`: Evaluation log file path

**Raises:** RuntimeError on directory creation failure

**Process:**
1. Delegates to `utils.setup_directories()`
2. Sets internal path attributes
3. Ensures directory permissions and accessibility

#### `setup_wandb(self) -> bool`
**Purpose:** Initialize Weights & Biases experiment tracking.

**Preconditions:** Directories must be set up first

**Returns:** True if WandB is successfully activated, False otherwise

**Error Handling:** Catches all WandB setup exceptions and degrades gracefully

**Process:**
1. Validates directory setup prerequisites
2. Delegates to `utils.setup_wandb()`
3. Sets WandB active state
4. Handles initialization failures gracefully

#### `setup_seeding(self) -> None`
**Purpose:** Configure random seeding for reproducibility.

**Delegates to:** `utils.setup_seeding()` for consistent seeding across the application

#### Configuration Methods

#### `save_effective_config(self) -> None`
**Purpose:** Serialize and save the effective configuration to JSON.

**Preconditions:** Directories must be set up first

**Output:** Creates `effective_config.json` in the run artifact directory

**Process:**
1. Ensures artifact directory exists
2. Serializes configuration using `utils.serialize_config()`
3. Writes JSON file with UTF-8 encoding

**Error Handling:** Raises RuntimeError on serialization or file write failures

#### Logging Methods

#### `log_session_info`
```python
def log_session_info(
    self,
    logger_func: Callable[[str], None],
    agent_info: Optional[Dict[str, Any]] = None,
    resumed_from_checkpoint: Optional[str] = None,
    global_timestep: int = 0,
    total_episodes_completed: int = 0,
) -> None
```
**Purpose:** Log comprehensive session information for debugging and monitoring.

**Parameters:**
- `logger_func`: Function to call for each log message
- `agent_info`: Optional agent details (type, name)
- `resumed_from_checkpoint`: Checkpoint path if resuming
- `global_timestep`: Current training timestep
- `total_episodes_completed`: Total episodes completed

**Logged Information:**
- Run title with optional WandB URL
- Run and configuration directories
- Random seed and device configuration
- Agent information
- Training parameters (timesteps, steps per epoch)
- Resume status and progress

#### `log_session_start(self) -> None`
**Purpose:** Log session start event to the training log file.

**Process:**
1. Appends timestamp and session start marker to log file
2. Handles file I/O errors gracefully

#### Session Lifecycle

#### `finalize_session(self) -> None`
**Purpose:** Clean up and finalize the training session.

**WandB Finalization:**
- Implements timeout mechanism (10 seconds) to prevent hanging
- Uses signal handling for timeout enforcement
- Provides graceful degradation on finalization failures
- Forces finish with exit code on timeout or interruption

**Error Handling:** Comprehensive exception handling for all finalization scenarios

#### `get_session_summary(self) -> Dict[str, Any]`
**Purpose:** Generate a comprehensive session summary for debugging and monitoring.

**Returns:** Dictionary containing:
- `run_name`: Session identifier
- `run_artifact_dir`: Main artifact directory
- `model_dir`: Model checkpoint directory
- `log_file_path`: Training log file path
- `is_wandb_active`: WandB activation status
- `seed`: Random seed (if configured)
- `device`: Computing device (if configured)

## Data Structures

### Session State
```python
SessionState = {
    "directories_setup": bool,
    "wandb_initialized": bool,
    "config_saved": bool,
    "seeding_setup": bool,
    "session_active": bool
}
```

### Directory Structure
```
{run_artifact_dir}/
├── effective_config.json      # Serialized configuration
├── training.log               # Training log file
├── evaluation.log             # Evaluation log file
└── models/                    # Model checkpoint directory
    ├── checkpoint_1000.ckpt
    ├── checkpoint_2000.ckpt
    └── final_model.pt
```

## Inter-Module Relationships

### Dependencies
```
SessionManager
    ├── config_schema (configuration)
    ├── utils.utils (setup utilities)
    ├── training.utils (training utilities)
    └── wandb (experiment tracking)
```

### Usage by Other Components
- **Trainer**: Primary consumer for session lifecycle management
- **ModelManager**: Uses directory paths for model persistence
- **DisplayManager**: Uses log file paths for output management
- **Callbacks**: Access session information for custom behavior

## Implementation Notes

### Design Patterns
- **Lazy Initialization**: Properties validate setup state before access
- **Facade Pattern**: Provides unified interface to session management
- **Template Method**: Consistent setup and teardown procedures
- **Defensive Programming**: Comprehensive error handling and validation

### Error Handling Strategy
- **Graceful Degradation**: WandB failures don't stop training
- **Clear Error Messages**: Detailed error reporting for debugging
- **State Validation**: Properties enforce correct initialization order
- **Timeout Mechanisms**: Prevent hanging on external service failures

### Lifecycle Management
```python
# Typical session lifecycle
1. __init__() - Basic initialization
2. setup_directories() - Directory structure creation
3. setup_wandb() - Optional experiment tracking
4. save_effective_config() - Configuration persistence
5. setup_seeding() - Reproducibility setup
6. log_session_start() - Session logging
7. [Training execution]
8. finalize_session() - Cleanup and finalization
```

## Testing Strategy

### Unit Testing
```python
def test_session_manager_initialization():
    """Test basic session manager initialization."""
    config = create_test_config()
    args = create_test_args()
    session_mgr = SessionManager(config, args)
    
    assert session_mgr.run_name is not None
    assert len(session_mgr.run_name) > 0

def test_directory_setup():
    """Test directory creation and path resolution."""
    session_mgr = create_test_session_manager()
    
    dirs = session_mgr.setup_directories()
    assert os.path.exists(dirs["run_artifact_dir"])
    assert os.path.exists(dirs["model_dir"])
    
    # Test property access
    assert session_mgr.run_artifact_dir == dirs["run_artifact_dir"]
    assert session_mgr.model_dir == dirs["model_dir"]

def test_wandb_setup():
    """Test WandB initialization with mocking."""
    with patch('wandb.init') as mock_init:
        session_mgr = create_test_session_manager()
        session_mgr.setup_directories()
        
        result = session_mgr.setup_wandb()
        assert isinstance(result, bool)

def test_config_serialization():
    """Test configuration saving functionality."""
    session_mgr = create_test_session_manager()
    session_mgr.setup_directories()
    session_mgr.save_effective_config()
    
    config_path = os.path.join(
        session_mgr.run_artifact_dir, 
        "effective_config.json"
    )
    assert os.path.exists(config_path)
    
    # Verify JSON is valid
    with open(config_path) as f:
        json.load(f)  # Should not raise
```

### Integration Testing
- **End-to-End Session**: Test complete session lifecycle
- **Error Recovery**: Test behavior under various failure conditions
- **External Services**: Test WandB integration with real and mock services
- **File System**: Test directory operations under various permissions

## Performance Considerations

### Initialization Performance
- **Lazy Property Access**: Properties computed only when accessed
- **Efficient Directory Operations**: Minimal file system calls
- **Fast Name Generation**: Optimized run name generation

### Memory Efficiency
- **Minimal State**: Only essential state stored in memory
- **String Optimization**: Efficient path string handling
- **Configuration Caching**: Avoids repeated serialization

### I/O Optimization
- **Batch File Operations**: Minimize file system interactions
- **Buffered Logging**: Efficient log file writing
- **Async-Ready**: Design compatible with asynchronous operations

## Security Considerations

### File System Security
- **Path Validation**: Safe handling of directory paths
- **Permission Checking**: Proper file system permission validation
- **Temp File Handling**: Secure temporary file operations

### Configuration Security
- **Sensitive Data**: Careful handling of configuration secrets
- **File Permissions**: Appropriate permissions on configuration files
- **Path Traversal**: Protection against path traversal attacks

### External Service Security
- **API Key Handling**: Secure WandB API key management
- **Network Timeouts**: Protection against hanging network operations
- **Error Information**: Safe handling of error messages

## Error Handling

### Common Error Scenarios
```python
# Directory setup failure
try:
    session_mgr.setup_directories()
except RuntimeError as e:
    logger.error(f"Directory setup failed: {e}")
    # Handle setup failure

# WandB initialization failure
result = session_mgr.setup_wandb()
if not result:
    logger.warning("WandB initialization failed, continuing without tracking")

# Configuration save failure
try:
    session_mgr.save_effective_config()
except RuntimeError as e:
    logger.error(f"Config save failed: {e}")
    # Continue training without saved config
```

### Error Recovery Strategies
- **Continue Without WandB**: Training continues if WandB fails
- **Alternative Directories**: Fallback directory creation strategies
- **Graceful Degradation**: Reduced functionality rather than failure

## Configuration

### Session Configuration
```yaml
logging:
  run_name: "custom_run_name"    # Optional explicit run name
  wandb_enabled: true            # Enable WandB tracking
  log_level: "info"             # Logging verbosity
  
env:
  seed: 42                      # Random seed for reproducibility
  device: "cuda"                # Computing device
  
training:
  total_timesteps: 1000000      # Training duration
  steps_per_epoch: 2048         # Steps per PPO epoch
```

### Environment Variables
- `WANDB_API_KEY`: WandB authentication
- `WANDB_PROJECT`: WandB project name
- `KEISEI_LOG_DIR`: Override log directory
- `KEISEI_MODEL_DIR`: Override model directory

## Future Enhancements

### Session Management Improvements
1. **Session Persistence**: Database backend for session tracking
2. **Multi-Session Management**: Concurrent session support
3. **Session Recovery**: Advanced checkpoint and recovery mechanisms
4. **Resource Monitoring**: Built-in resource usage tracking

### Integration Enhancements
```python
# Future: Enhanced session manager
class EnhancedSessionManager(SessionManager):
    def __init__(self, config, args, plugins=None):
        super().__init__(config, args)
        self.plugin_manager = PluginManager(plugins)
    
    def setup_remote_logging(self, backend="tensorboard"):
        """Setup additional logging backends."""
        
    def enable_distributed_session(self, rank, world_size):
        """Enable distributed training session management."""
        
    def setup_auto_checkpointing(self, interval=1000):
        """Setup automatic checkpoint creation."""
```

### Monitoring and Analytics
- **Session Analytics**: Detailed session performance metrics
- **Resource Tracking**: CPU, GPU, memory usage monitoring
- **Cost Tracking**: Training cost estimation and tracking
- **Performance Baselines**: Historical performance comparison

## Usage Examples

### Basic Session Setup
```python
from keisei.training.session_manager import SessionManager

# Create session manager
config = load_config("config.yaml")
args = parse_arguments()
session_mgr = SessionManager(config, args)

# Setup session infrastructure
session_mgr.setup_directories()
session_mgr.setup_wandb()
session_mgr.save_effective_config()
session_mgr.setup_seeding()

# Access session properties
print(f"Run name: {session_mgr.run_name}")
print(f"Model directory: {session_mgr.model_dir}")
print(f"WandB active: {session_mgr.is_wandb_active}")
```

### Custom Run Name
```python
# Explicit run name
session_mgr = SessionManager(config, args, run_name="custom_experiment")

# Or via command line
# python train.py --run_name custom_experiment
```

### Session Information Logging
```python
def my_logger(message):
    print(f"[LOG] {message}")

# Log comprehensive session info
session_mgr.log_session_info(
    logger_func=my_logger,
    agent_info={"type": "PPOAgent", "name": "ShogiAgent"},
    global_timestep=0,
    total_episodes_completed=0
)
```

### Session Finalization
```python
try:
    # Training loop here
    pass
finally:
    # Always finalize session
    session_mgr.finalize_session()
```

### Session Summary
```python
# Get session summary for monitoring
summary = session_mgr.get_session_summary()
print(f"Session: {summary['run_name']}")
print(f"Directory: {summary['run_artifact_dir']}")
print(f"WandB: {summary['is_wandb_active']}")
```

## Maintenance Notes

### Code Reviews
- Verify proper error handling for all external dependencies
- Check timeout mechanisms for external service calls
- Ensure file system operations are safe and atomic
- Validate path handling for cross-platform compatibility

### Performance Monitoring
- Monitor session setup time and resource usage
- Track WandB initialization performance
- Profile file I/O operations for optimization opportunities
- Monitor memory usage during session lifecycle

### Documentation Maintenance
- Keep configuration examples up to date
- Document any changes to directory structure
- Update error handling documentation for new scenarios
- Maintain examples with current API usage