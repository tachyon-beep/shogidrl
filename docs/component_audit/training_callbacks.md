# Software Documentation Template for Subsystems - Training Callbacks

## 📘 training_callbacks.py as of 2024-12-19

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `keisei/training/callbacks.py`
**Documentation Version:** `1.0`
**Date:** `2024-12-19`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Provides concrete callback implementations for periodic training tasks in the Keisei Shogi RL trainer. Implements the callback interface with specific functionality for checkpointing and evaluation, enabling automated training lifecycle management.

* **Key Responsibilities:**
  - Abstract callback base class definition
  - Checkpoint saving at regular intervals
  - Periodic evaluation execution during training
  - Integration with ModelManager for checkpoint operations
  - Evaluation orchestration with model state management

* **Domain Context:**
  Training lifecycle management in PPO-based deep reinforcement learning system for Shogi gameplay. Provides automated periodic tasks that maintain model persistence and performance monitoring.

* **High-Level Architecture / Interaction Summary:**
  Implements the callback pattern for trainer lifecycle events. Callbacks are managed by CallbackManager and executed at specific training steps. Integrates with ModelManager for checkpointing and evaluation infrastructure for performance assessment.

---

### 2. Module Details 📦

* **Module Name:** `callbacks.py`
  
  * **Purpose:** Concrete callback implementations for training lifecycle management
  * **Design Patterns Used:** 
    - Abstract Base Class pattern for callback interface
    - Template Method pattern for step-based execution
    - Strategy pattern for different callback types
  * **Key Functions/Classes Provided:**
    - `Callback` - Abstract base class for all callbacks
    - `CheckpointCallback` - Automated checkpoint saving
    - `EvaluationCallback` - Periodic evaluation execution
  * **Configuration Surface:**
    - Checkpoint interval configuration
    - Evaluation configuration and timing
    - Model directory specification
    - Evaluation opponent and game parameters
  * **Dependencies:**
    - **Internal:**
      - `keisei.training.trainer.Trainer` - Main trainer class (TYPE_CHECKING)
    - **External:**
      - `os` - File system operations
      - `abc.ABC` - Abstract base class functionality
      - `typing.TYPE_CHECKING` - Type checking imports
  * **External API Contracts:**
    - **Callback Interface:** `on_step_end(trainer)` method signature
    - **Trainer Integration:** Expects specific trainer attributes and methods

---

### 3. Classes 🏗️

#### `Callback` (Abstract Base Class)
**Purpose:** Abstract base class defining the callback interface for training lifecycle events.

**Key Methods:**
- `on_step_end(trainer: "Trainer")` - Called at the end of each training step

**Design Pattern:** Abstract Base Class pattern for polymorphic callback execution

**Usage:**
```python
class CustomCallback(Callback):
    def on_step_end(self, trainer: "Trainer"):
        # Custom implementation
        pass
```

#### `CheckpointCallback`
**Purpose:** Implements automated checkpoint saving at regular training intervals with comprehensive state persistence.

**Initialization Parameters:**
- `interval: int` - Number of timesteps between checkpoint saves
- `model_dir: str` - Directory path for checkpoint storage

**Key Attributes:**
```python
self.interval: int      # Checkpoint frequency
self.model_dir: str     # Checkpoint directory path
```

**Key Methods:**

##### `__init__(self, interval: int, model_dir: str)`
**Purpose:** Initialize checkpoint callback with interval and directory configuration.

##### `on_step_end(self, trainer: "Trainer")`
**Purpose:** Execute checkpoint saving when interval conditions are met.

**Checkpoint Logic:**
1. **Interval Check:** `(trainer.global_timestep + 1) % self.interval == 0`
2. **Agent Validation:** Ensure trainer.agent is initialized
3. **Game Statistics Collection:** Gather black_wins, white_wins, draws
4. **ModelManager Integration:** Delegate to trainer.model_manager.save_checkpoint()
5. **Success Logging:** Log checkpoint save status to both console and W&B

**Checkpoint Data Saved:**
```python
game_stats = {
    "black_wins": trainer.black_wins,
    "white_wins": trainer.white_wins,
    "draws": trainer.draws,
}

# Saved via ModelManager with:
# - agent state
# - timestep count
# - episode count
# - game statistics
# - run name
# - W&B status
```

**Error Handling:**
- Agent not initialized: Log error and return
- Checkpoint save failure: Log error with timestep information

#### `EvaluationCallback`
**Purpose:** Implements periodic evaluation execution during training with comprehensive evaluation orchestration.

**Initialization Parameters:**
- `eval_cfg` - Evaluation configuration object
- `interval: int` - Number of timesteps between evaluations

**Key Attributes:**
```python
self.eval_cfg           # Evaluation configuration
self.interval: int      # Evaluation frequency
```

**Key Methods:**

##### `__init__(self, eval_cfg, interval: int)`
**Purpose:** Initialize evaluation callback with configuration and interval.

##### `on_step_end(self, trainer: "Trainer")`
**Purpose:** Execute periodic evaluation when interval and configuration conditions are met.

**Evaluation Logic:**
1. **Configuration Check:** Verify `eval_cfg.enable_periodic_evaluation`
2. **Interval Check:** `(trainer.global_timestep + 1) % self.interval == 0`
3. **Agent/Model Validation:** Ensure trainer.agent and agent.model exist
4. **Checkpoint Creation:** Save temporary evaluation checkpoint
5. **Model Mode Management:** Set model to eval mode, restore to train mode
6. **Evaluation Execution:** Delegate to trainer.execute_full_evaluation_run()
7. **Result Logging:** Log evaluation results to console and W&B

**Evaluation Configuration Parameters:**
```python
eval_parameters = {
    "opponent_type": getattr(eval_cfg, "opponent_type", "random"),
    "opponent_checkpoint_path": getattr(eval_cfg, "opponent_checkpoint_path", None),
    "num_games": getattr(eval_cfg, "num_games", 20),
    "max_moves_per_game": getattr(eval_cfg, "max_moves_per_game", 256),
    "log_file_path_eval": getattr(eval_cfg, "log_file_path_eval", ""),
    "wandb_log_eval": getattr(eval_cfg, "wandb_log_eval", False),
    "wandb_project_eval": getattr(eval_cfg, "wandb_project_eval", None),
    "wandb_entity_eval": getattr(eval_cfg, "wandb_entity_eval", None),
    "wandb_run_name_eval": f"periodic_eval_{trainer.run_name}_ts{timestep}",
    "wandb_group": trainer.run_name,
    "wandb_reinit": True,
    "logger_also_stdout": False
}
```

**Model State Management:**
```python
# Before evaluation
current_model.eval()  # Set to evaluation mode

# After evaluation
current_model.train()  # Restore to training mode
```

**Error Handling:**
- Evaluation disabled: Return without action
- Agent not initialized: Log error and return
- Model not found: Log error and return
- Evaluation execution failure: Ensure model returns to train mode

---

### 4. Data Structures 🗂️

#### Callback Interface
```python
class CallbackProtocol:
    def on_step_end(self, trainer: "Trainer") -> None:
        """Called at the end of each training step."""
        pass
```

#### Checkpoint Data Structure
```python
checkpoint_data = {
    "agent_state": trainer.agent.state_dict(),
    "timestep": trainer.global_timestep + 1,
    "episode_count": trainer.total_episodes_completed,
    "stats": {
        "black_wins": int,
        "white_wins": int,
        "draws": int
    },
    "run_name": str,
    "metadata": {
        "save_timestamp": str,
        "wandb_active": bool
    }
}
```

#### Evaluation Results Structure
```python
evaluation_results = {
    "win_rate": float,
    "games_played": int,
    "wins": int,
    "losses": int,
    "draws": int,
    "avg_game_length": float,
    "evaluation_timestamp": str,
    "opponent_type": str
}
```

#### Trainer Interface Requirements
```python
# Expected trainer attributes for callbacks
trainer_interface = {
    "global_timestep": int,
    "agent": PPOAgent,
    "black_wins": int,
    "white_wins": int,
    "draws": int,
    "total_episodes_completed": int,
    "model_dir": str,
    "run_name": str,
    "is_train_wandb_active": bool,
    "model_manager": ModelManager,
    "config": AppConfig,
    "log_both": callable,
    "execute_full_evaluation_run": callable,
    "policy_output_mapper": callable
}
```

---

### 5. Inter-Module Relationships 🔗

#### Dependencies
- **`keisei.training.trainer.Trainer`** - Main trainer class with expected interface
- **`keisei.training.model_manager.ModelManager`** - Checkpoint saving operations
- **`keisei.evaluation`** - Evaluation execution infrastructure

#### Integration Points
- **CallbackManager:** Callbacks are registered and executed by CallbackManager
- **ModelManager:** CheckpointCallback delegates to ModelManager for save operations
- **Evaluation System:** EvaluationCallback orchestrates evaluation runs
- **Logging Infrastructure:** Both callbacks integrate with trainer logging system

#### Data Flow
```
Training Step → CallbackManager → Callback.on_step_end() → Specific Action (Checkpoint/Evaluation)
                                                         ↓
                                                    Result Logging → Console + W&B
```

#### Trainer Dependencies
- **ModelManager Integration:** Checkpoint saving through consolidated interface
- **Evaluation Infrastructure:** Evaluation execution through trainer methods
- **Logging System:** Unified logging through trainer.log_both method
- **Configuration Access:** Access to training and evaluation configuration

---

### 6. Implementation Notes 💡

#### Design Decisions
1. **Abstract Base Class:** Provides polymorphic interface for different callback types
2. **Interval-Based Execution:** Both callbacks use timestep-based interval checking
3. **ModelManager Integration:** CheckpointCallback delegates to ModelManager for consistency
4. **Model State Management:** EvaluationCallback properly manages model training/eval modes
5. **Error Tolerance:** Callbacks handle missing components gracefully without crashing training

#### Code Organization
- Clear separation between abstract interface and concrete implementations
- Each callback handles its own interval logic and error checking
- Centralized error logging through trainer interface
- Consistent parameter extraction pattern for evaluation configuration

#### Performance Considerations
- Checkpoint saving only occurs at specified intervals
- Evaluation checkpoint creation for isolated evaluation runs
- Model mode switching minimizes impact on training state
- Efficient interval checking with modulo operation

---

### 7. Testing Strategy 🧪

#### Unit Tests
```python
def test_callback_interface():
    """Test abstract callback interface definition."""
    pass

def test_checkpoint_callback_interval():
    """Test checkpoint saving at correct intervals."""
    pass

def test_checkpoint_callback_error_handling():
    """Test checkpoint callback error handling."""
    pass

def test_evaluation_callback_execution():
    """Test evaluation callback execution logic."""
    pass

def test_evaluation_callback_model_state():
    """Test model state management during evaluation."""
    pass
```

#### Integration Tests
```python
def test_checkpoint_callback_integration():
    """Test checkpoint callback with ModelManager."""
    pass

def test_evaluation_callback_integration():
    """Test evaluation callback with evaluation infrastructure."""
    pass

def test_callback_trainer_integration():
    """Test callbacks with trainer lifecycle."""
    pass
```

#### Mock Testing
```python
def test_callbacks_with_mock_trainer():
    """Test callbacks with mocked trainer interface."""
    pass

def test_checkpoint_callback_mock_model_manager():
    """Test checkpoint callback with mocked ModelManager."""
    pass
```

#### Testing Considerations
- Mock trainer interface for isolated callback testing
- Test interval logic with various timestep combinations
- Validate error handling for missing trainer components
- Test evaluation callback model state transitions

---

### 8. Performance Considerations ⚡

#### Efficiency Factors
- **Interval-Based Execution:** Only execute when interval conditions met
- **Efficient Interval Check:** Simple modulo operation for interval detection
- **Lazy Validation:** Quick validation checks before expensive operations
- **Minimal State:** Callbacks maintain minimal internal state

#### Optimization Opportunities
- **Checkpoint Batching:** Batch multiple checkpoint saves for efficiency
- **Evaluation Scheduling:** Optimize evaluation timing to minimize training interruption
- **Model State Caching:** Cache model states to reduce evaluation setup overhead
- **Result Caching:** Cache evaluation results for duplicate configurations

#### Resource Management
- **Temporary Files:** Evaluation checkpoint cleanup after evaluation
- **Memory Usage:** Model state transitions managed efficiently
- **Disk I/O:** Checkpoint saves delegated to optimized ModelManager
- **Compute Resources:** Evaluation execution isolated from training compute

---

### 9. Security Considerations 🔒

#### Input Validation
- **Interval Values:** Validate positive interval values
- **Path Safety:** Model directory and checkpoint paths validated
- **Configuration Safety:** Evaluation configuration parameter validation

#### Security Measures
- **Path Restrictions:** Checkpoint paths restricted to designated directories
- **Model Access:** Controlled access to trainer model and agent state
- **Evaluation Isolation:** Evaluation runs isolated from training state

#### Potential Vulnerabilities
- **Path Traversal:** Checkpoint and evaluation file paths from configuration
- **Model State Access:** Direct access to model parameters and state
- **Process Injection:** Evaluation execution could be exploited

---

### 10. Error Handling 🚨

#### Exception Management
```python
# CheckpointCallback error handling
if not trainer.agent:
    trainer.log_both(
        "[ERROR] CheckpointCallback: Agent not initialized, cannot save checkpoint.",
        also_to_wandb=True
    )
    return

# EvaluationCallback error handling
if not current_model:
    trainer.log_both(
        "[ERROR] EvaluationCallback: Agent's model not found.",
        also_to_wandb=True
    )
    return
```

#### Error Categories
- **Component Missing:** Agent, model, or manager not available
- **Configuration Errors:** Invalid evaluation configuration or intervals
- **I/O Errors:** Checkpoint save or evaluation file errors
- **State Errors:** Model state management failures

#### Recovery Strategies
- **Graceful Degradation:** Continue training when callbacks fail
- **Error Logging:** Comprehensive error logging to console and W&B
- **State Recovery:** Ensure model returns to training mode after evaluation errors
- **Fallback Behavior:** Skip callback execution on component failures

---

### 11. Configuration 📝

#### CheckpointCallback Configuration
```python
checkpoint_config = {
    "interval": 10000,           # Timesteps between checkpoints
    "model_dir": "/path/to/models"  # Checkpoint storage directory
}
```

#### EvaluationCallback Configuration
```python
evaluation_config = {
    "enable_periodic_evaluation": True,
    "interval": 50000,           # Timesteps between evaluations
    "opponent_type": "random",   # Evaluation opponent type
    "num_games": 20,            # Games per evaluation
    "max_moves_per_game": 256,  # Game length limit
    "wandb_log_eval": True,     # Log to W&B
    "wandb_project_eval": "shogi-eval",
    "wandb_entity_eval": "team"
}
```

#### Trainer Integration Configuration
```python
callback_setup = {
    "checkpoint_interval": 10000,
    "evaluation_interval": 50000,
    "enable_periodic_evaluation": True,
    "model_dir": "/path/to/models"
}
```

---

### 12. Future Enhancements 🚀

#### Planned Improvements
1. **Additional Callback Types:** Learning rate scheduling, early stopping, custom metrics
2. **Async Execution:** Background evaluation execution without blocking training
3. **Callback Chaining:** Sequential callback execution with dependencies
4. **Configuration Validation:** Enhanced validation for callback parameters
5. **Performance Monitoring:** Callback execution time tracking and optimization

#### Extension Points
- **Custom Callback Interface:** Support for user-defined callback implementations
- **Event Types:** Additional trainer events beyond step_end
- **Callback Metadata:** Rich metadata collection and logging
- **Distributed Callbacks:** Multi-process callback execution for large-scale training

#### API Evolution
- Maintain backward compatibility for existing callback interface
- Consider async/await pattern for non-blocking callbacks
- Evaluate callback priority and ordering systems

---

### 13. Usage Examples 📋

#### Basic Callback Setup
```python
# Create callbacks
checkpoint_cb = CheckpointCallback(interval=10000, model_dir="/models")
evaluation_cb = EvaluationCallback(eval_cfg=config.evaluation, interval=50000)

# Register with trainer
trainer.register_callbacks([checkpoint_cb, evaluation_cb])
```

#### Checkpoint Callback Usage
```python
# Initialize with specific interval and directory
checkpoint_callback = CheckpointCallback(
    interval=5000,  # Save every 5K timesteps
    model_dir="/path/to/checkpoints"
)

# Callback executes automatically during training
# Saves agent state, timestep, episode count, and game statistics
```

#### Evaluation Callback Usage
```python
# Configure evaluation
eval_config = EvaluationConfig(
    enable_periodic_evaluation=True,
    opponent_type="random",
    num_games=50,
    wandb_log_eval=True
)

# Initialize evaluation callback
eval_callback = EvaluationCallback(
    eval_cfg=eval_config,
    interval=25000  # Evaluate every 25K timesteps
)

# Automatic evaluation execution during training
# Results logged to console and W&B
```

#### Custom Callback Implementation
```python
class MetricsCallback(Callback):
    def __init__(self, metrics_config):
        self.metrics_config = metrics_config
    
    def on_step_end(self, trainer: "Trainer"):
        # Custom metrics collection
        if trainer.global_timestep % 1000 == 0:
            # Log custom metrics
            trainer.log_both(f"Custom metrics at step {trainer.global_timestep}")
```

---

### 14. Maintenance Notes 🔧

#### Regular Maintenance
- **Trainer Interface:** Monitor trainer attribute changes and update callback dependencies
- **ModelManager Integration:** Keep checkpoint callback aligned with ModelManager API
- **Evaluation Infrastructure:** Update evaluation callback with evaluation system changes
- **Configuration Schema:** Maintain compatibility with configuration structure evolution

#### Monitoring Points
- **Callback Execution Time:** Track overhead introduced by callback execution
- **Checkpoint Success Rate:** Monitor checkpoint save success and failure patterns
- **Evaluation Performance:** Track evaluation execution time and resource usage
- **Error Patterns:** Monitor common callback failure modes and error types

#### Documentation Dependencies
- **`training_callback_manager.md`** - Callback management and registration system
- **`training_trainer.md`** - Main trainer class interface and lifecycle
- **`training_model_manager.md`** - Checkpoint saving operations and ModelManager API
- **`evaluation_evaluate.md`** - Evaluation infrastructure and execution details
- **`config_schema.md`** - Configuration structure for callback parameters
