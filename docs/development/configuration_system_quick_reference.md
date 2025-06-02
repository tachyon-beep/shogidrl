# Keisei Configuration System - Quick Reference

**Last Updated:** June 2, 2025  
**Status:** Recently Updated with Validation Fixes

## Configuration Structure Overview

### Configuration Classes (`keisei/config_schema.py`)

#### 1. EnvConfig
**Purpose:** Environment and runtime settings
```python
device: str = "cpu"                    # Device: 'cpu' or 'cuda'
input_channels: int = 46               # Neural network input channels
num_actions_total: int = 13527         # Total possible actions
seed: int = 42                         # Random seed
max_moves_per_game: int = 500          # Max moves before draw
```

#### 2. TrainingConfig  
**Purpose:** Training algorithm parameters
```python
total_timesteps: int = 500_000         # Total training steps
steps_per_epoch: int = 2048            # Steps per PPO buffer
ppo_epochs: int = 10                   # PPO update epochs per buffer
minibatch_size: int = 64               # Minibatch size
learning_rate: float = 3e-4            # Optimizer learning rate ✅ VALIDATED (>0)
gamma: float = 0.99                    # Discount factor
# ... additional training parameters
```

#### 3. EvaluationConfig ⭐ **Recently Enhanced**
**Purpose:** Evaluation system settings
```python
enable_periodic_evaluation: bool = True                    # Enable eval during training
evaluation_interval_timesteps: int = 50000                 # Eval frequency ✅ VALIDATED (>0)
num_games: int = 20                                        # Games per eval ✅ VALIDATED (>0)
opponent_type: str = "random"                              # Opponent type
max_moves_per_game: int = 500                              # Max moves per eval game ✅ VALIDATED (>0)
log_file_path_eval: str = "eval_log.txt"                   # Eval log file path ✅ FIXED
wandb_log_eval: bool = False                               # W&B eval logging
```

#### 4. LoggingConfig
**Purpose:** File logging settings
```python
log_file: str = "logs/training_log.txt"                    # Main training log
model_dir: str = "models/"                                 # Model checkpoint directory
run_name: Optional[str] = None                             # Optional run name override
```

#### 5. WandBConfig
**Purpose:** Weights & Biases integration
```python
enabled: bool = True                                       # Enable W&B logging
project: Optional[str] = "keisei-shogi-rl"                # W&B project name
entity: Optional[str] = None                               # W&B entity
log_model_artifact: bool = False                           # Log model artifacts ✅ FIXED
# ... additional W&B settings
```

#### 6. ParallelConfig
**Purpose:** Parallel training settings
```python
enabled: bool = False                                      # Enable parallel training
num_workers: int = 4                                       # Number of workers ✅ VALIDATED (>0)
batch_size: int = 32                                       # Worker batch size ✅ VALIDATED (>0)
# ... additional parallel settings
```

#### 7. DemoConfig
**Purpose:** Demo mode settings
```python
enable_demo_mode: bool = False                             # Enable demo mode
demo_mode_delay: float = 0.5                               # Delay between moves
```

## Validation Rules Summary

### Current Validators ✅
- **TrainingConfig.learning_rate**: Must be positive (> 0)
- **EvaluationConfig.evaluation_interval_timesteps**: Must be positive (> 0) 🆕
- **EvaluationConfig.num_games**: Must be positive (> 0) 🆕  
- **EvaluationConfig.max_moves_per_game**: Must be positive (> 0) 🆕
- **ParallelConfig.num_workers**: Must be positive (> 0)
- **ParallelConfig.batch_size**: Must be positive (> 0)

### Validation Gaps 🔍
Areas identified for future validation enhancement:
- Range validation for percentages (gamma, clip_epsilon, etc.)
- Device string validation ("cpu", "cuda")
- File path validation
- Memory/resource constraint validation
- Inter-field dependency validation

## Configuration Loading

### Primary Entry Point
```python
from keisei.utils import load_config
config = load_config()  # Loads default_config.yaml
config = load_config("custom_config.yaml")  # Loads custom config
```

### Usage Patterns

#### Direct Access
```python
config.training.learning_rate
config.evaluation.log_file_path_eval
config.wandb.log_model_artifact
```

#### Getattr Pattern (Legacy)
```python
log_file_path_eval = getattr(config.evaluation, "log_file_path_eval", "")
```

#### Component Initialization
```python
trainer = Trainer(config.training)
evaluator = Evaluator(config.evaluation)
```

## Recent Fixes & Improvements 🔧

### 1. Schema Completeness (✅ Fixed)
- Added missing `evaluation.log_file_path_eval` field
- Added missing `wandb.log_model_artifact` field
- Fixed silent field dropping during Pydantic validation

### 2. Validation Enhancement (✅ Fixed)  
- Added positive value validation to EvaluationConfig
- Enhanced error messages for invalid configurations
- Improved test coverage for validation scenarios

### 3. Test Coverage (✅ Fixed)
- Fixed failing configuration integration tests
- Added comprehensive validation test scenarios
- Ensured backward compatibility

## Configuration Files

### Default Configuration
- **File**: `default_config.yaml`
- **Purpose**: Production defaults and schema reference
- **Status**: ✅ In sync with schema

### Test Configuration  
- **File**: `test_config.yaml`
- **Purpose**: Override values for testing
- **Status**: ✅ Compatible with schema

## Common Usage Locations

### Training System
- `keisei/training/trainer.py` - Main training loop
- `keisei/training/callbacks.py` - Training callbacks
- `keisei/training/utils.py` - Training utilities

### Evaluation System  
- `keisei/evaluation/evaluate.py` - Evaluation logic
- `keisei/training/callbacks.py` - Periodic evaluation

### Core Components
- `keisei/core/neural_network.py` - Model configuration
- `train.py` - Main entry point

## Best Practices 📋

### For Developers
1. **Schema First**: Always update schema before adding config fields
2. **Validation**: Add appropriate validators for new fields
3. **Testing**: Test both valid and invalid configuration scenarios
4. **Documentation**: Update field descriptions in schema

### For Users
1. **YAML Validation**: Ensure YAML syntax is correct
2. **Type Consistency**: Use correct types (int, float, bool, str)
3. **Value Ranges**: Respect validation constraints (positive values, etc.)
4. **Testing**: Test configuration changes before production use

## Troubleshooting 🔧

### Common Issues
1. **AttributeError on config access**: Field missing from schema - check `config_schema.py`
2. **ValidationError on startup**: Invalid configuration values - check validation rules
3. **Silent config drops**: Field in YAML but not in schema - add to appropriate config class
4. **Type errors**: Incorrect YAML types - ensure values match schema types

### Debug Commands
```bash
# Test configuration loading
python3 -c "from keisei.utils import load_config; print(load_config())"

# Validate specific config section
python3 -c "from keisei.config_schema import EvaluationConfig; print(EvaluationConfig())"

# Check for validation errors
python3 -c "from keisei.config_schema import EvaluationConfig; EvaluationConfig(num_games=-1)"
```

---

**Next Steps:** Execute full configuration system mapping task as outlined in `configuration_system_mapping_task.md`
