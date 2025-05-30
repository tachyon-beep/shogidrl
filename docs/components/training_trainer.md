# Trainer Class Documentation

## Module Overview

**File:** `keisei/training/trainer.py`

**Purpose:** Orchestrates the entire Shogi Deep Reinforcement Learning training process, managing component initialization, training loop execution, evaluation, and session lifecycle.

**Core Functionality:**
- Central coordinator for all training activities
- Component initialization and dependency management
- Training loop orchestration via manager delegation
- Session management including logging and artifact handling
- Checkpoint resume and model persistence
- Integration with Weights & Biases for experiment tracking

## Dependencies

### Internal Dependencies
- `keisei.config_schema`: AppConfig for configuration management
- `keisei.core.actor_critic_protocol`: ActorCriticProtocol interface
- `keisei.core.experience_buffer`: ExperienceBuffer for storing transitions
- `keisei.core.ppo_agent`: PPOAgent for RL algorithm implementation
- `keisei.evaluation.evaluate`: execute_full_evaluation_run function
- `keisei.utils`: TrainingLogger for logging utilities
- All manager components from training package

### External Dependencies
- `torch`: PyTorch framework for neural networks and device management
- `wandb`: Weights & Biases for experiment tracking
- `datetime`: Timestamp generation for sessions

## Class Documentation

### `Trainer`

**Purpose:** Main orchestrator class that manages the complete training workflow for PPO-based Shogi agents.

**Inheritance:** Inherits from `CompatibilityMixin` for backward compatibility features.

**Attributes:**

#### Core Components
- `config` (AppConfig): Complete application configuration
- `args` (Any): Parsed command-line arguments
- `device` (torch.device): Computing device (CPU/GPU)

#### Model and Agent Components
- `model` (Optional[ActorCriticProtocol]): Neural network model
- `agent` (Optional[PPOAgent]): PPO reinforcement learning agent
- `experience_buffer` (Optional[ExperienceBuffer]): Experience storage
- `step_manager` (Optional[StepManager]): Individual step management

#### Game Environment
- `game`: Shogi game instance
- `policy_output_mapper`: Action space mapping utility
- `action_space_size` (int): Total number of possible actions
- `obs_space_shape` (tuple): Observation space dimensions

#### Manager Components
- `session_manager` (SessionManager): Session lifecycle management
- `display_manager` (DisplayManager): Training visualization
- `model_manager` (ModelManager): Model operations and persistence
- `env_manager` (EnvManager): Environment lifecycle management
- `metrics_manager` (MetricsManager): Training metrics collection
- `callback_manager` (CallbackManager): Training callback orchestration
- `setup_manager` (SetupManager): Component initialization
- `training_loop_manager` (TrainingLoopManager): Main training loop

#### Session Properties
- `run_name` (str): Unique identifier for training run
- `run_artifact_dir` (str): Directory for run artifacts
- `model_dir` (str): Directory for model checkpoints
- `log_file_path` (str): Path to training log file
- `eval_log_file_path` (str): Path to evaluation log file
- `is_train_wandb_active` (bool): Whether W&B tracking is enabled

#### Utilities
- `logger` (TrainingLogger): Structured logging utility
- `rich_console`: Rich console for formatted output
- `rich_log_messages`: Rich log message panel
- `log_both` (Optional[Callable]): Function for dual logging (file + W&B)
- `execute_full_evaluation_run` (Optional[Callable]): Evaluation function
- `resumed_from_checkpoint` (bool): Whether training resumed from checkpoint

**Methods:**

#### `__init__(self, config: AppConfig, args: Any)`
**Purpose:** Initialize the Trainer with comprehensive component setup.

**Parameters:**
- `config` (AppConfig): Complete application configuration
- `args` (Any): Command-line arguments

**Process:**
1. Initialize core attributes and device
2. Create session manager and setup directories
3. Initialize all manager components
4. Setup game components and training infrastructure
5. Handle checkpoint resume if specified
6. Initialize display and callbacks

#### `_initialize_components(self)`
**Purpose:** Initialize all training components using SetupManager delegation.

**Process:**
1. Setup game components (game, policy mapper, action/observation spaces)
2. Setup training components (model, agent, experience buffer)
3. Setup step manager for individual training steps
4. Handle checkpoint resume if specified

#### `_initialize_game_state(self, log_both) -> EpisodeState`
**Purpose:** Initialize the game state for training using EnvManager.

**Parameters:**
- `log_both` (Callable): Logging function for both file and W&B

**Returns:** EpisodeState object representing initial game state

**Error Handling:** Catches and handles game initialization errors gracefully

#### `_perform_ppo_update(self, current_obs_np, log_both)`
**Purpose:** Execute a PPO policy update using the experience buffer.

**Parameters:**
- `current_obs_np`: Current observation for value estimation
- `log_both`: Logging function

**Process:**
1. Compute GAE advantages and returns in experience buffer
2. Execute PPO learning step on agent
3. Clear experience buffer for next collection phase
4. Format and log PPO metrics

#### `_log_run_info(self, log_both)`
**Purpose:** Log comprehensive run information at training start.

**Delegates to:** SetupManager for consistent information formatting

#### `_finalize_training(self, log_both)`
**Purpose:** Finalize training session with model saving and cleanup.

**Process:**
1. Log training completion status
2. Save final model if training completed successfully
3. Save final checkpoint regardless of completion status
4. Finalize W&B session if active
5. Finalize display and save console output

#### `run_training_loop(self)`
**Purpose:** Execute the main training loop with comprehensive session management.

**Process:**
1. Setup session logging
2. Create dual logging function (file + W&B)
3. Log session start and run information
4. Initialize game state
5. Start Rich display context
6. Delegate to TrainingLoopManager for main execution
7. Handle exceptions and finalization

**Exception Handling:**
- `KeyboardInterrupt`: Graceful shutdown on user interruption
- `RuntimeError, ValueError, AttributeError, ImportError`: General error handling

#### `_handle_checkpoint_resume(self)`
**Purpose:** Handle checkpoint resume for backward compatibility.

**Delegates to:** SetupManager for consistent checkpoint handling

**Returns:** Boolean indicating whether resume was successful

## Data Structures

### Trainer State
```python
TrainerState = {
    "initialization_complete": bool,
    "model_ready": bool,
    "agent_ready": bool,
    "session_active": bool,
    "training_active": bool,
    "wandb_active": bool,
    "resumed_from_checkpoint": bool
}
```

### Component Dependencies
```python
ComponentDependencies = {
    "session_manager": ["config", "args"],
    "display_manager": ["config", "log_file_path"],
    "model_manager": ["config", "args", "device", "logger"],
    "env_manager": ["config", "logger"],
    "metrics_manager": [],
    "callback_manager": ["config", "model_dir"],
    "setup_manager": ["config", "device"]
}
```

## Inter-Module Relationships

### Manager Orchestration
```
Trainer (Central Orchestrator)
    ├── SessionManager (session lifecycle)
    ├── DisplayManager (visualization)
    ├── ModelManager (model operations)
    ├── EnvManager (environment)
    ├── MetricsManager (metrics)
    ├── CallbackManager (callbacks)
    ├── SetupManager (initialization)
    ├── StepManager (individual steps)
    └── TrainingLoopManager (main loop)
```

### Usage Patterns
```python
# Primary usage pattern
trainer = Trainer(config, args)
trainer.run_training_loop()

# Component access pattern
trainer.metrics_manager.get_current_stats()
trainer.model_manager.save_checkpoint()
trainer.session_manager.finalize_session()
```

## Implementation Notes

### Design Patterns
- **Facade Pattern**: Trainer provides unified interface to complex training system
- **Manager Pattern**: Responsibility distributed across specialized managers
- **Delegation Pattern**: Trainer delegates specific concerns to appropriate managers
- **Template Method**: Consistent initialization and finalization patterns

### Architecture Considerations
- **Separation of Concerns**: Each manager handles specific training aspects
- **Dependency Injection**: Configuration and dependencies passed to managers
- **Error Isolation**: Errors handled at appropriate levels with graceful degradation
- **Resource Management**: Proper cleanup and finalization of all resources

### Session Lifecycle
```python
# Training session lifecycle
1. __init__() - Component initialization
2. _initialize_components() - Setup training infrastructure
3. run_training_loop() - Main execution
   a. Session start logging
   b. Game state initialization
   c. Training loop delegation
   d. Exception handling
   e. Training finalization
4. _finalize_training() - Cleanup and artifact saving
```

## Testing Strategy

### Unit Testing
```python
def test_trainer_initialization():
    """Test trainer initialization with valid config."""
    config = create_test_config()
    args = create_test_args()
    trainer = Trainer(config, args)
    
    assert trainer.config == config
    assert trainer.args == args
    assert trainer.device is not None
    assert trainer.session_manager is not None

def test_component_initialization():
    """Test all components are properly initialized."""
    trainer = create_test_trainer()
    
    assert trainer.model is not None
    assert trainer.agent is not None
    assert trainer.experience_buffer is not None
    assert all(manager is not None for manager in [
        trainer.session_manager,
        trainer.display_manager,
        trainer.model_manager,
        trainer.env_manager,
        trainer.metrics_manager,
        trainer.callback_manager,
        trainer.setup_manager
    ])

def test_training_finalization():
    """Test proper cleanup during training finalization."""
    trainer = create_test_trainer()
    
    with patch('wandb.run') as mock_wandb:
        trainer._finalize_training(mock_log_function)
        # Verify finalization steps
```

### Integration Testing
- **End-to-End Training**: Test complete training workflow
- **Manager Integration**: Test interactions between managers
- **Error Recovery**: Test error handling and recovery mechanisms
- **Checkpoint Resume**: Test checkpoint loading and training continuation

## Performance Considerations

### Initialization Overhead
- **Lazy Initialization**: Heavy components loaded only when needed
- **Parallel Setup**: Independent managers can be initialized concurrently
- **Resource Pooling**: Shared resources managed efficiently

### Runtime Performance
- **Manager Delegation**: Minimal overhead in training loop
- **Memory Management**: Proper cleanup prevents memory leaks
- **Logging Efficiency**: Structured logging with minimal performance impact

### Monitoring
- **Component Performance**: Individual manager performance tracking
- **Resource Usage**: Memory and GPU utilization monitoring
- **Training Progress**: Real-time metrics and visualization

## Security Considerations

### Configuration Security
- **Input Validation**: Configuration validation before component initialization
- **Resource Limits**: Proper bounds checking on resource allocation
- **Path Security**: Safe handling of file paths and directories

### Session Security
- **Artifact Handling**: Secure handling of model checkpoints and logs
- **External Integration**: Safe integration with W&B and external services
- **Error Information**: Careful handling of sensitive information in logs

## Error Handling

### Initialization Errors
```python
try:
    trainer = Trainer(config, args)
except (RuntimeError, ValueError) as e:
    logger.error(f"Trainer initialization failed: {e}")
    # Handle initialization failure
```

### Training Errors
```python
try:
    trainer.run_training_loop()
except KeyboardInterrupt:
    logger.info("Training interrupted by user")
    # Graceful shutdown
except Exception as e:
    logger.error(f"Training failed: {e}")
    # Error recovery and cleanup
```

### Component Errors
- **Manager Failures**: Isolated error handling for individual managers
- **Resource Errors**: Graceful handling of resource allocation failures
- **External Service Errors**: Robust handling of W&B and other external service issues

## Configuration

### Training Configuration
```yaml
training:
  total_timesteps: 1000000
  learning_rate: 0.0003
  batch_size: 256
  ppo_epochs: 4
  
model:
  type: "resnet"
  tower_depth: 9
  tower_width: 256
  
logging:
  wandb_enabled: true
  log_level: "info"
  checkpoint_interval: 10000
```

### Environment Variables
- `WANDB_API_KEY`: Weights & Biases API key
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `KEISEI_LOG_LEVEL`: Override log level

## Future Enhancements

### Architecture Improvements
1. **Async Manager Pattern**: Enable asynchronous manager operations
2. **Plugin Architecture**: Support for custom managers and components
3. **Distributed Training**: Multi-node training support
4. **Auto-scaling**: Dynamic resource allocation based on workload

### Feature Enhancements
```python
# Future: Enhanced trainer with plugin support
class EnhancedTrainer(Trainer):
    def __init__(self, config, args, plugins=None):
        super().__init__(config, args)
        self.plugin_manager = PluginManager(plugins)
    
    def register_custom_manager(self, name, manager_class):
        """Register custom manager for specific functionality."""
        
    def enable_distributed_training(self, world_size, rank):
        """Enable distributed training across multiple nodes."""
```

### Monitoring Enhancements
- **Real-time Metrics Dashboard**: Live training monitoring
- **Advanced Visualization**: Enhanced training progress visualization
- **Automated Alerting**: Alerts for training anomalies or failures
- **Performance Profiling**: Detailed performance analysis tools

## Usage Examples

### Basic Training
```python
from keisei.training import Trainer
from keisei.config_schema import load_config

# Load configuration
config = load_config("config.yaml")
args = parse_command_line_args()

# Create and run trainer
trainer = Trainer(config, args)
trainer.run_training_loop()
```

### Training with Custom Configuration
```python
# Custom configuration setup
config = AppConfig(
    training=TrainingConfig(
        total_timesteps=500000,
        learning_rate=0.001,
        batch_size=128
    ),
    model=ModelConfig(
        type="resnet",
        tower_depth=6,
        tower_width=128
    )
)

trainer = Trainer(config, args)
trainer.run_training_loop()
```

### Training with Checkpoint Resume
```python
# Resume from checkpoint
args.resume = "path/to/checkpoint.ckpt"
trainer = Trainer(config, args)

if trainer.resumed_from_checkpoint:
    print("Successfully resumed from checkpoint")
    
trainer.run_training_loop()
```

### Manual Component Access
```python
# Access specific components
trainer = Trainer(config, args)

# Access metrics
current_stats = trainer.metrics_manager.get_current_stats()
print(f"Current timestep: {trainer.metrics_manager.global_timestep}")

# Save checkpoint manually
trainer.model_manager.save_checkpoint(
    agent=trainer.agent,
    timestep=trainer.metrics_manager.global_timestep
)

# Access session information
print(f"Run name: {trainer.run_name}")
print(f"Model directory: {trainer.model_dir}")
```

## Maintenance Notes

### Code Reviews
- Verify proper manager initialization order and dependencies
- Check error handling coverage for all components
- Ensure resource cleanup in all code paths
- Validate configuration parameter usage

### Performance Monitoring
- Monitor initialization time and resource usage
- Track memory usage throughout training lifecycle
- Profile manager interactions for bottlenecks
- Monitor external service integration performance

### Documentation Maintenance
- Keep manager interaction diagrams updated
- Document any changes to initialization order
- Maintain examples with current API usage
- Update error handling documentation for new scenarios
