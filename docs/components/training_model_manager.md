# Software Documentation Template for Subsystems - Training Model Manager

## 📘 training_model_manager.py as of 2024-12-19

**Project Name:** `Keisei Shogi DRL System`
**Folder Path:** `/keisei/training/model_manager.py`
**Documentation Version:** `1.0`
**Date:** `2024-12-19`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Manages the complete model lifecycle for Shogi RL training including model configuration, checkpoint management, mixed precision setup, and WandB artifact creation.

* **Key Responsibilities:**
  - Model configuration and factory instantiation with feature specification
  - Mixed precision training setup for CUDA devices
  - Checkpoint loading, saving, and resuming functionality
  - WandB model artifact creation and management
  - Final model persistence and metadata tracking

* **Domain Context:**
  Model lifecycle management in PPO-based DRL system, specifically handling PyTorch model operations, checkpointing, and integration with experiment tracking.

* **High-Level Architecture / Interaction Summary:**
  
  The ModelManager acts as the central coordinator for all model-related operations during training. It integrates with the model factory for creation, handles PyTorch device management, provides checkpoint persistence with WandB artifact tracking, and manages mixed precision training configurations. The manager bridges the gap between training infrastructure and model storage/retrieval.

---

### 2. Modules 📦

* **Module Name:** `model_manager.py`

  * **Purpose:** Comprehensive model lifecycle management for training runs
  * **Design Patterns Used:** Manager pattern for model lifecycle, Factory pattern integration, Strategy pattern for checkpointing
  * **Key Functions/Classes Provided:** 
    - `ModelManager` class for model orchestration
    - Checkpoint save/load with resume functionality
    - WandB artifact creation and metadata management
    - Mixed precision training setup
  * **Configuration Surface:** Model architecture parameters, training configuration, and checkpoint settings

---

### 3. Classes and Functions 🏗️

#### Class: `ModelManager`

**Purpose:** Central manager for model lifecycle operations during training runs.

**Key Attributes:**
- `config`: Application configuration object (AppConfig)
- `args`: Command-line arguments with potential overrides
- `device`: PyTorch device for model operations
- `logger_func`: Optional logging function for status messages
- `model`: Optional ActorCriticProtocol instance (created via create_model)
- `scaler`: Optional GradScaler for mixed precision training
- `feature_spec`: Feature specification for model input
- `obs_shape`: Observation shape tuple (channels, height, width)
- `resumed_from_checkpoint`: Path of resumed checkpoint (if any)
- `checkpoint_data`: Data from loaded checkpoint

**Model Configuration Attributes:**
- `input_features`: Feature set name (from args or config)
- `model_type`: Model architecture type (from args or config)
- `tower_depth`: ResNet tower depth (from args or config)
- `tower_width`: ResNet tower width (from args or config)
- `se_ratio`: Squeeze-and-Excitation ratio (from args or config)

**Key Methods:**

##### `__init__(config: AppConfig, args: Any, device: torch.device, logger_func=None)`
- **Purpose:** Initialize model manager with configuration and device
- **Parameters:**
  - `config`: Application configuration object
  - `args`: Command-line arguments with potential overrides
  - `device`: PyTorch device for model operations
  - `logger_func`: Optional logging function (defaults to no-op)
- **Return Type:** None
- **Key Behavior:**
  - Extracts model configuration from args with config fallbacks
  - Sets up feature specification and observation shape
  - Configures mixed precision training if enabled and supported
  - Initializes checkpoint state tracking
- **Usage:** Called during trainer initialization

##### `_setup_feature_spec() -> None`
- **Purpose:** Setup feature specification and observation shape from config
- **Parameters:** None
- **Return Type:** None
- **Key Behavior:**
  - Retrieves feature spec from features.FEATURE_SPECS registry
  - Sets observation shape to (num_planes, 9, 9) for Shogi
- **Usage:** Called internally during initialization

##### `_setup_mixed_precision() -> None`
- **Purpose:** Configure mixed precision training if enabled and supported
- **Parameters:** None
- **Return Type:** None
- **Key Behavior:**
  - Enables mixed precision only if configured and CUDA available
  - Creates GradScaler for CUDA mixed precision
  - Logs status and warnings for unsupported configurations
- **Usage:** Called internally during initialization

##### `create_model() -> ActorCriticProtocol`
- **Purpose:** Create model using factory and move to device
- **Parameters:** None
- **Return Type:** ActorCriticProtocol instance
- **Key Behavior:**
  - Calls model_factory with configured parameters
  - Moves model to specified device
  - Stores model in self.model attribute
  - Validates successful model creation
- **Exceptions:** Raises RuntimeError if model creation fails
- **Usage:** Called during training setup to create model

##### `handle_checkpoint_resume(agent: PPOAgent, model_dir: str, resume_path_override: Optional[str] = None) -> bool`
- **Purpose:** Handle resuming from checkpoint if specified or auto-detected
- **Parameters:**
  - `agent`: PPO agent to load checkpoint into
  - `model_dir`: Directory to search for checkpoints
  - `resume_path_override`: Optional path override for resuming
- **Return Type:** Boolean indicating if resumed from checkpoint
- **Key Behavior:**
  - Handles "latest" keyword to find most recent checkpoint
  - Supports specific checkpoint paths
  - Copies parent directory checkpoints if needed
  - Updates resume state and loads checkpoint data
- **Usage:** Called during training setup for checkpoint resuming

##### `_find_latest_checkpoint(model_dir: str) -> Optional[str]`
- **Purpose:** Find the latest checkpoint in model directory or parent
- **Parameters:**
  - `model_dir`: Directory to search for checkpoints
- **Return Type:** Optional path to latest checkpoint
- **Key Behavior:**
  - Searches model directory first
  - Falls back to parent directory if needed
  - Uses utils.find_latest_checkpoint for discovery
- **Usage:** Called internally during latest checkpoint resolution

##### `create_model_artifact(model_path: str, artifact_name: str, run_name: str, is_wandb_active: bool, ...) -> bool`
- **Purpose:** Create and upload WandB artifact for model checkpoint
- **Parameters:**
  - `model_path`: Path to model file
  - `artifact_name`: Name for artifact (without run prefix)
  - `run_name`: Current run name for prefixing
  - `is_wandb_active`: Whether WandB is active
  - `artifact_type`: Type of artifact (default: "model")
  - `description`: Optional description
  - `metadata`: Optional metadata dictionary
  - `aliases`: Optional list of aliases
- **Return Type:** Boolean indicating success
- **Key Behavior:**
  - Creates artifact with run name prefix for uniqueness
  - Adds model file to artifact
  - Uploads with specified aliases and metadata
  - Handles errors gracefully without stopping training
- **Usage:** Called when saving checkpoints and final models

##### `save_final_model(agent: PPOAgent, model_dir: str, global_timestep: int, ...) -> Tuple[bool, Optional[str]]`
- **Purpose:** Save final trained model with comprehensive metadata
- **Parameters:**
  - `agent`: PPO agent to save
  - `model_dir`: Directory to save model in
  - `global_timestep`: Current training timestep
  - `total_episodes_completed`: Total episodes completed
  - `game_stats`: Dictionary with win/loss statistics
  - `run_name`: Current run name
  - `is_wandb_active`: Whether WandB is active
- **Return Type:** Tuple of (success, model_path)
- **Key Behavior:**
  - Saves model as "final_model.pth"
  - Creates comprehensive metadata with training statistics
  - Creates WandB artifact with "latest" and "final" aliases
  - Handles save errors gracefully
- **Usage:** Called at successful training completion

##### `save_checkpoint(agent: PPOAgent, model_dir: str, timestep: int, ...) -> Tuple[bool, Optional[str]]`
- **Purpose:** Save periodic model checkpoint during training
- **Parameters:**
  - `agent`: PPO agent to save
  - `model_dir`: Directory to save checkpoint in
  - `timestep`: Current training timestep
  - `episode_count`: Total episodes completed
  - `stats`: Dictionary with game statistics
  - `run_name`: Current run name
  - `is_wandb_active`: Whether WandB is active
  - `checkpoint_name_prefix`: Prefix for checkpoint filename
- **Return Type:** Tuple of (success, checkpoint_path)
- **Key Behavior:**
  - Saves checkpoint with timestep in filename
  - Avoids duplicate saves for same timestep
  - Creates WandB artifact with timestep-specific alias
  - Ensures model directory exists before saving
- **Usage:** Called periodically during training for checkpoint creation

##### `save_final_checkpoint(agent: PPOAgent, model_dir: str, global_timestep: int, ...) -> Tuple[bool, Optional[str]]`
- **Purpose:** Save final checkpoint with complete training statistics
- **Parameters:** Similar to save_checkpoint but for final state
- **Return Type:** Tuple of (success, checkpoint_path)
- **Key Behavior:**
  - Saves final checkpoint with complete metadata
  - Creates WandB artifact with "latest-checkpoint" alias
  - Includes comprehensive training statistics
- **Usage:** Called at training completion for final checkpoint

##### `get_model_info() -> Dict[str, Any]`
- **Purpose:** Get comprehensive information about model configuration
- **Parameters:** None
- **Return Type:** Dictionary with model configuration details
- **Key Information:**
  - Model architecture parameters
  - Feature specification details
  - Mixed precision settings
  - Device configuration
- **Usage:** For debugging and configuration verification

---

### 4. Data Structures 📊

#### Model Configuration Structure

```python
model_config = {
    "model_type": str,              # Architecture type (e.g., "resnet")
    "input_features": str,          # Feature set name (e.g., "core")
    "tower_depth": int,             # ResNet depth
    "tower_width": int,             # ResNet width
    "se_ratio": float,              # Squeeze-and-Excitation ratio
    "obs_shape": Tuple[int, int, int],  # (channels, height, width)
    "use_mixed_precision": bool,    # Mixed precision enabled
    "device": str                   # PyTorch device string
}
```

#### Checkpoint Metadata Structure

```python
checkpoint_metadata = {
    "training_timesteps": int,      # Current timestep
    "total_episodes": int,          # Episodes completed
    "black_wins": int,              # Black player wins
    "white_wins": int,              # White player wins
    "draws": int,                   # Drawn games
    "checkpoint_type": str,         # "periodic" or "final"
    "model_type": str,              # Model architecture
    "feature_set": str              # Feature set used
}
```

#### WandB Artifact Configuration

- **Naming:** `{run_name}-{artifact_name}` for uniqueness
- **Types:** "model" for models, "checkpoint" for checkpoints
- **Aliases:** ["latest", "final"] for final models, ["ts-{timestep}"] for checkpoints
- **Metadata:** Comprehensive training and model information

---

### 5. Inter-Module Relationships 🔗

#### Dependencies:
- **`keisei.training.models.model_factory`** - Model creation factory
- **`keisei.core.actor_critic_protocol`** - Model interface protocol
- **`keisei.core.ppo_agent`** - PPO agent for checkpoint operations
- **`keisei.shogi.features`** - Feature specification registry
- **`torch`** - PyTorch for model operations and mixed precision
- **`wandb`** - Experiment tracking and artifact management
- **`.utils`** - Checkpoint discovery utilities

#### Used By:
- **`trainer.py`** - Main training orchestrator for model management
- **`session_manager.py`** - Model setup during session initialization
- **`callback_manager.py`** - Checkpoint callbacks during training

#### Provides To:
- **Training Infrastructure** - Model creation and lifecycle management
- **Checkpointing System** - Persistent model storage and retrieval
- **Experiment Tracking** - WandB integration for model artifacts
- **Resume Functionality** - Training continuation from saved states

---

### 6. Implementation Notes 🔧

#### Model Creation Strategy:
- Factory pattern integration for flexible model architectures
- Device management for CPU/CUDA compatibility
- Feature specification validation and configuration

#### Checkpoint Management:
- Filename conventions with timestep inclusion
- Duplicate save prevention for efficiency
- Parent directory search for checkpoint discovery
- Graceful error handling without training interruption

#### WandB Integration:
- Unique artifact naming with run prefixes
- Comprehensive metadata for artifact discoverability
- Alias management for easy artifact retrieval
- Error isolation to prevent training failures

#### Mixed Precision Support:
- CUDA device requirement validation
- GradScaler setup for automatic mixed precision
- Fallback to full precision for unsupported configurations

---

### 7. Testing Strategy 🧪

#### Unit Tests:
- Test model creation with various configurations
- Verify checkpoint save/load functionality
- Test WandB artifact creation with mock services
- Validate mixed precision setup logic

#### Integration Tests:
- Test complete model lifecycle during training
- Verify checkpoint resume functionality
- Test WandB integration with real artifacts
- Validate device management across platforms

#### Error Scenarios:
- Model factory failures and error handling
- File system errors during checkpoint operations
- WandB service unavailability
- Invalid configuration parameters

---

### 8. Performance Considerations ⚡

#### Model Operations:
- Device transfer overhead during model creation
- Memory usage for model storage and checkpoints
- Mixed precision benefits for CUDA training

#### Checkpoint Operations:
- File I/O overhead during save operations
- Checkpoint size impact on storage requirements
- WandB upload time for large models

#### Optimization Strategies:
- Avoid duplicate checkpoint saves
- Use efficient PyTorch serialization
- Implement checkpoint compression if needed
- Optimize WandB artifact upload scheduling

---

### 9. Security Considerations 🔒

#### File System Security:
- Checkpoint files contain model weights and training state
- No access controls on checkpoint directories
- Model files could be modified externally

#### WandB Security:
- Artifacts uploaded to external service
- Metadata may contain sensitive training information
- No encryption of model artifacts

#### Model Security:
- Model weights represent intellectual property
- No validation of loaded checkpoint integrity
- Potential for malicious checkpoint injection

#### Mitigation Strategies:
- Implement checkpoint integrity verification
- Add access controls for sensitive model directories
- Validate WandB artifact sources before loading
- Consider encryption for sensitive model data

---

### 10. Error Handling 🚨

#### Model Creation Errors:
- Factory failures raise RuntimeError with context
- Device transfer failures propagated to caller
- Model validation ensures successful creation

#### Checkpoint Errors:
- File I/O errors logged but don't stop training
- Missing checkpoint directories created automatically
- Resume failures allow training to continue from scratch

#### WandB Errors:
- Artifact creation failures logged as warnings
- Service unavailability doesn't block training
- Upload interruptions handled gracefully

#### Recovery Strategies:
- Training continues even with checkpoint failures
- Model creation failures are fatal (appropriate)
- WandB errors degraded gracefully to local operations

---

### 11. Configuration ⚙️

#### Required Configuration:
```python
config.training.model_type: str           # Model architecture
config.training.input_features: str       # Feature set name
config.env.num_actions_total: int         # Action space size
```

#### Optional Configuration:
```python
config.training.tower_depth: int          # ResNet depth
config.training.tower_width: int          # ResNet width
config.training.se_ratio: float           # SE attention ratio
config.training.mixed_precision: bool     # Enable mixed precision
```

#### Command-line Overrides:
- All configuration can be overridden via command-line arguments
- Args take precedence over config file values
- Resume path can be specified via args.resume

---

### 12. Future Enhancements 🚀

#### Advanced Model Management:
- Model versioning and compatibility checking
- Automatic model optimization and quantization
- Model ensemble management and merging
- Dynamic architecture adaptation during training

#### Enhanced Checkpointing:
- Incremental checkpoint saves for large models
- Checkpoint compression and optimization
- Distributed checkpoint management
- Automatic checkpoint cleanup and rotation

#### Improved Integration:
- Enhanced WandB artifact organization
- Model performance metrics tracking
- Automated model validation after loading
- Integration with model serving infrastructure

---

### 13. Usage Examples 💡

#### Basic Model Manager Setup:
```python
# Initialize model manager
model_manager = ModelManager(config, args, device, logger_func=print)

# Create model
model = model_manager.create_model()
print(f"Model created: {type(model).__name__}")

# Get model information
model_info = model_manager.get_model_info()
print(f"Model configuration: {model_info}")
```

#### Checkpoint Resume Workflow:
```python
# Handle checkpoint resuming
agent = PPOAgent(model, config)
resumed = model_manager.handle_checkpoint_resume(agent, model_dir)

if resumed:
    print(f"Resumed from: {model_manager.resumed_from_checkpoint}")
    # Access checkpoint data if needed
    checkpoint_data = model_manager.checkpoint_data
```

#### Periodic Checkpoint Saving:
```python
# Save periodic checkpoint
success, checkpoint_path = model_manager.save_checkpoint(
    agent=agent,
    model_dir="/path/to/models",
    timestep=10000,
    episode_count=500,
    stats={"black_wins": 250, "white_wins": 200, "draws": 50},
    run_name="experiment_1",
    is_wandb_active=True
)

if success:
    print(f"Checkpoint saved: {checkpoint_path}")
```

#### Final Model Saving:
```python
# Save final model with comprehensive metadata
success, model_path = model_manager.save_final_model(
    agent=agent,
    model_dir="/path/to/models",
    global_timestep=50000,
    total_episodes_completed=2500,
    game_stats={"black_wins": 1250, "white_wins": 1000, "draws": 250},
    run_name="experiment_1",
    is_wandb_active=True
)

if success:
    print(f"Final model saved: {model_path}")
```

---

### 14. Maintenance Notes 📝

#### Regular Maintenance:
- Monitor checkpoint file sizes and storage usage
- Review WandB artifact organization and cleanup
- Update model factory integration as architectures evolve
- Validate mixed precision performance benefits

#### Version Compatibility:
- Model factory interface changes require manager updates
- PyTorch version updates may affect checkpoint format
- WandB API changes need artifact creation updates
- Feature specification changes require validation updates

#### Code Quality:
- Maintain consistent error handling patterns
- Keep checkpoint metadata comprehensive and structured
- Document model configuration parameter interactions
- Ensure graceful degradation for all external dependencies

---

### 15. Related Documentation 📚

- **`training_models_init.md`** - Model factory and creation infrastructure
- **`training_models_resnet_tower.md`** - ResNet architecture implementation
- **`core_ppo_agent.md`** - PPO agent checkpoint operations
- **`training_trainer.md`** - Main trainer integration with model manager
- **`training_utils.md`** - Checkpoint discovery and file utilities
- **`shogi_features.md`** - Feature specification and observation space
