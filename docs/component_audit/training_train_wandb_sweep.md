# Software Documentation Template for Subsystems - Training W&B Sweep Script

## 📘 training_train_wandb_sweep.py as of 2024-12-19

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `keisei/training/train_wandb_sweep.py`
**Documentation Version:** `1.0`
**Date:** `2024-12-19`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Specialized training script optimized for Weights & Biases (W&B) hyperparameter sweeps. Provides a streamlined interface for automated hyperparameter optimization with dedicated sweep parameter mapping and simplified CLI arguments. Serves as the entry point for W&B sweep agents.

* **Key Responsibilities:**
  - W&B sweep context detection and configuration extraction
  - Automated parameter mapping from sweep config to application config
  - Simplified CLI interface optimized for sweep usage
  - Multiprocessing setup for CUDA compatibility
  - Trainer initialization with sweep-optimized parameters

* **Domain Context:**
  Hyperparameter optimization layer in PPO-based deep reinforcement learning system for Shogi gameplay. Specialized for automated parameter exploration through W&B sweep infrastructure.

* **High-Level Architecture / Interaction Summary:**
  Lightweight training entry point that detects W&B sweep context, extracts sweep parameters, maps them to application configuration, and delegates training execution to the Trainer class. Optimized for batch execution in sweep environments.

---

### 2. Module Details 📦

* **Module Name:** `train_wandb_sweep.py`
  
  * **Purpose:** W&B sweep-optimized training script with automated parameter mapping
  * **Design Patterns Used:** 
    - Strategy pattern for sweep parameter detection
    - Adapter pattern for parameter mapping
    - Template method for sweep workflow
  * **Key Functions/Classes Provided:**
    - `apply_wandb_sweep_config()` - Sweep parameter extraction and mapping
    - `main()` - Sweep-optimized entry point
  * **Configuration Surface:**
    - Simplified CLI arguments (5 core parameters)
    - Automatic W&B sweep parameter detection
    - Comprehensive parameter mapping to config paths
    - Default configuration file specification
  * **Dependencies:**
    - **Internal:**
      - `keisei.config_schema.AppConfig` - Configuration data structures
      - `keisei.training.trainer.Trainer` - Main training orchestrator
      - `keisei.utils.load_config` - Configuration loading utilities
    - **External:**
      - `wandb` - Weights & Biases experiment tracking and sweep management
      - `argparse` - Command-line argument parsing
      - `multiprocessing` - Process management for CUDA safety
  * **External API Contracts:**
    - **W&B Sweep Integration:** Automatic sweep context detection and parameter extraction
    - **CLI Interface:** Simplified argument specification for sweep usage

---

### 3. Functions 🛠️

#### `apply_wandb_sweep_config()`
**Purpose:** Detects active W&B sweep context and extracts sweep parameters, mapping them to application configuration paths.

**Parameters:** None

**Returns:** 
- `Dict[str, Any]` - Dictionary of configuration overrides extracted from sweep parameters

**Key Functionality:**
1. **Sweep Detection:** Checks for active W&B run context
2. **Parameter Extraction:** Retrieves sweep configuration from wandb.config
3. **Parameter Mapping:** Maps sweep parameters to application config paths
4. **Override Generation:** Creates configuration override dictionary

**Sweep Parameter Mapping:**
```python
sweep_param_mapping = {
    "learning_rate": "training.learning_rate",
    "gamma": "training.gamma",
    "clip_epsilon": "training.clip_epsilon", 
    "ppo_epochs": "training.ppo_epochs",
    "minibatch_size": "training.minibatch_size",
    "value_loss_coeff": "training.value_loss_coeff",
    "entropy_coef": "training.entropy_coef",
    "tower_depth": "training.tower_depth",
    "tower_width": "training.tower_width",
    "se_ratio": "training.se_ratio",
    "steps_per_epoch": "training.steps_per_epoch",
    "gradient_clip_max_norm": "training.gradient_clip_max_norm",
    "lambda_gae": "training.lambda_gae"
}
```

**Return Value:**
```python
# Example return for active sweep
{
    "wandb.enabled": True,
    "training.learning_rate": 3e-4,
    "training.gamma": 0.99,
    "training.clip_epsilon": 0.2,
    "training.ppo_epochs": 4
}

# Return for non-sweep execution
{}
```

#### `main()`
**Purpose:** Main entry point for W&B sweep-enabled training with simplified argument processing and automatic sweep integration.

**Parameters:** None (reads from sys.argv via argparse)

**Returns:** None (exits after training completion)

**Key Functionality:**
1. **Simplified Argument Parsing:**
   - Configuration file path (default: "default_config.yaml")
   - Resume checkpoint path
   - Random seed
   - Device specification
   - Total timesteps override

2. **Sweep Integration:**
   - Automatic sweep parameter extraction
   - Configuration override merging
   - CLI argument priority handling

3. **Training Execution:**
   - Configuration loading with all overrides
   - Trainer initialization and execution

**CLI Arguments (Simplified Set):**

| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--config` | str | "default_config.yaml" | Configuration file path |
| `--resume` | str | None | Checkpoint path or 'latest' |
| `--seed` | int | None | Random seed |
| `--device` | str | None | Training device |
| `--total-timesteps` | int | None | Training duration override |

**Configuration Priority:**
1. CLI arguments (highest priority)
2. W&B sweep parameters
3. Configuration file values (lowest priority)

---

### 4. Data Structures 🗂️

#### Sweep Override Structure
```python
sweep_overrides = {
    "wandb.enabled": bool,           # Always True for sweeps
    "training.learning_rate": float, # Learning rate from sweep
    "training.gamma": float,         # Discount factor from sweep
    "training.clip_epsilon": float,  # PPO clip parameter from sweep
    "training.ppo_epochs": int,      # PPO update epochs from sweep
    "training.minibatch_size": int,  # Minibatch size from sweep
    "training.value_loss_coeff": float, # Value loss coefficient from sweep
    "training.entropy_coef": float,  # Entropy coefficient from sweep
    "training.tower_depth": int,     # ResNet tower depth from sweep
    "training.tower_width": int,     # ResNet tower width from sweep
    "training.se_ratio": float,      # SE block ratio from sweep
    "training.steps_per_epoch": int, # Steps per epoch from sweep
    "training.gradient_clip_max_norm": float, # Gradient clipping from sweep
    "training.lambda_gae": float     # GAE lambda from sweep
}
```

#### CLI Override Structure
```python
cli_overrides = {
    "env.seed": int,                 # Random seed
    "env.device": str,               # Training device
    "training.total_timesteps": int  # Training duration
}
```

#### Final Override Structure
```python
final_overrides = {
    **sweep_overrides,  # W&B sweep parameters
    **cli_overrides     # CLI arguments (take precedence)
}
```

---

### 5. Inter-Module Relationships 🔗

#### Dependencies
- **`keisei.config_schema.AppConfig`** - Configuration data structures and validation
- **`keisei.training.trainer.Trainer`** - Main training orchestrator and execution
- **`keisei.utils.load_config`** - Configuration file loading and override processing

#### Integration Points
- **W&B Sweep System:** Automatic integration with W&B sweep infrastructure
- **Training Infrastructure:** Delegates execution to Trainer with sweep-optimized configuration
- **Configuration System:** Loads and processes configuration with sweep parameter overrides
- **Process Management:** Configures multiprocessing for CUDA compatibility

#### Data Flow
```
W&B Sweep Agent → Sweep Detection → Parameter Extraction → Config Mapping → Override Merging → Trainer Init → Training Execution
```

#### Comparison with `train.py`
- **Simplified CLI:** Fewer arguments focused on sweep essentials
- **Automatic Detection:** Built-in W&B sweep context detection
- **Parameter Focus:** Optimized for hyperparameter optimization workflow
- **Default Config:** Explicit default configuration file specification

---

### 6. Implementation Notes 💡

#### Design Decisions
1. **Simplified Interface:** Reduced CLI complexity for sweep-focused usage
2. **Automatic Detection:** No manual sweep enable flags - automatic context detection
3. **Comprehensive Mapping:** All major hyperparameters mapped for sweep optimization
4. **Default Configuration:** Explicit default config file for consistency
5. **Parameter Priority:** CLI arguments override sweep parameters for manual control

#### Code Organization
- Clear separation between sweep detection and parameter processing
- Centralized parameter mapping for maintainability
- Consistent multiprocessing setup with main training script
- Minimal argument processing focused on sweep essentials

#### Error Handling
- Graceful handling of non-sweep execution (returns empty overrides)
- Multiprocessing setup error handling with fallback
- Configuration validation delegated to load_config utility

---

### 7. Testing Strategy 🧪

#### Unit Tests
```python
def test_sweep_detection():
    """Test W&B sweep context detection."""
    pass

def test_parameter_mapping():
    """Test sweep parameter to config path mapping."""
    pass

def test_override_merging():
    """Test CLI and sweep parameter override merging."""
    pass

def test_non_sweep_execution():
    """Test execution without active W&B sweep."""
    pass
```

#### Integration Tests
```python
def test_wandb_sweep_execution():
    """Test complete training execution within W&B sweep."""
    pass

def test_parameter_override_priority():
    """Test CLI parameter priority over sweep parameters."""
    pass

def test_config_file_integration():
    """Test configuration file loading with sweep overrides."""
    pass
```

#### Testing Considerations
- Mock W&B sweep context for controlled testing
- Test parameter mapping completeness and accuracy
- Validate override priority handling
- Test graceful non-sweep execution

---

### 8. Performance Considerations ⚡

#### Efficiency Factors
- **Minimal Overhead:** Simplified argument processing reduces startup time
- **Direct Mapping:** Efficient parameter mapping without complex transformations
- **Early Detection:** Fast W&B context detection prevents unnecessary processing
- **Optimized CLI:** Reduced argument validation overhead

#### Optimization Opportunities
- **Parameter Caching:** Cache parameter mappings for repeated sweep runs
- **Lazy Validation:** Defer configuration validation to first use
- **Batch Processing:** Optimize for multiple sweep runs in sequence

#### Resource Management
- Minimal memory footprint for sweep parameter processing
- Quick startup time optimized for sweep agent execution
- Efficient parameter extraction and mapping

---

### 9. Security Considerations 🔒

#### Input Validation
- **CLI Arguments:** Basic type validation through argparse
- **Sweep Parameters:** W&B platform validates sweep configuration
- **File Paths:** Configuration file paths validated by load_config

#### Security Measures
- **Limited CLI Surface:** Reduced attack surface through simplified arguments
- **W&B Integration:** Security delegated to W&B platform
- **Configuration Validation:** Delegated to AppConfig Pydantic validation

#### Potential Vulnerabilities
- **Parameter Injection:** W&B sweep parameters could potentially inject malicious values
- **File Path Control:** Configuration file path specified via CLI
- **Process Spawning:** Multiprocessing setup in containerized environments

---

### 10. Error Handling 🚨

#### Exception Management
```python
# Multiprocessing setup error handling
try:
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError as e:
    print(f"Warning: Could not set multiprocessing start method: {e}")
except OSError as e:
    print(f"Error setting multiprocessing start_method: {e}")
```

#### Error Categories
- **Sweep Detection Errors:** Handled gracefully with empty override return
- **Configuration Errors:** Delegated to load_config and AppConfig validation
- **Runtime Errors:** Delegated to Trainer execution phase
- **System Errors:** Multiprocessing setup failures with graceful fallback

#### Recovery Strategies
- **Graceful Non-Sweep:** Execute normally when no sweep detected
- **Parameter Fallback:** Use configuration file values when sweep parameters missing
- **Process Fallback:** Continue with default multiprocessing on setup failure

---

### 11. Configuration 📝

#### Default Configuration
- **Default Config File:** "default_config.yaml" (explicit default)
- **W&B Integration:** Automatic enablement for sweep runs
- **Parameter Mapping:** Comprehensive hyperparameter coverage

#### Sweep Configuration Format
```yaml
# Example W&B sweep configuration
program: train_wandb_sweep.py
method: bayes
metric:
  name: eval/win_rate
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  gamma:
    values: [0.95, 0.99, 0.995]
  clip_epsilon:
    distribution: uniform
    min: 0.1
    max: 0.3
```

#### Runtime Parameters
- **Automatic W&B Config:** Extracted from wandb.config during sweep runs
- **CLI Overrides:** Manual parameter control during sweep execution
- **Configuration File:** Base configuration with sweep parameter overrides

---

### 12. Future Enhancements 🚀

#### Planned Improvements
1. **Dynamic Parameter Mapping:** Runtime-configurable parameter mapping
2. **Sweep Validation:** Pre-sweep parameter validation and conflict detection
3. **Multi-Objective Optimization:** Support for multiple optimization metrics
4. **Sweep Resumption:** Resume interrupted sweep runs from checkpoints
5. **Custom Sweep Strategies:** Support for custom sweep algorithms

#### Extension Points
- **Parameter Sources:** Additional hyperparameter sources beyond W&B
- **Mapping Configuration:** External configuration for parameter mapping
- **Sweep Callbacks:** Custom callbacks for sweep event handling
- **Result Aggregation:** Multi-run result aggregation and analysis

#### API Evolution
- Maintain compatibility with W&B sweep infrastructure
- Consider integration with other hyperparameter optimization platforms
- Evaluate parameter mapping configuration externalization

---

### 13. Usage Examples 📋

#### W&B Sweep Setup
```bash
# Create sweep configuration
wandb sweep sweep_config.yaml

# Run sweep agent (uses train_wandb_sweep.py)
wandb agent <sweep_id>
```

#### Direct Execution
```bash
# Run with default configuration
python -m keisei.training.train_wandb_sweep

# Run with custom configuration
python -m keisei.training.train_wandb_sweep --config configs/sweep.yaml

# Run with device override
python -m keisei.training.train_wandb_sweep --device cuda --seed 42
```

#### Sweep Configuration Example
```yaml
# sweep_config.yaml
program: keisei/training/train_wandb_sweep.py
method: random
metric:
  name: eval/win_rate
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  gamma:
    values: [0.95, 0.99, 0.995]
  ppo_epochs:
    values: [3, 4, 5]
  tower_depth:
    values: [10, 15, 20]
```

#### Manual Sweep Parameter Testing
```bash
# Test sweep parameter mapping manually
python -c "
import wandb
wandb.init()
wandb.config.learning_rate = 1e-4
wandb.config.gamma = 0.99
# Run training script
"
```

---

### 14. Maintenance Notes 🔧

#### Regular Maintenance
- **Parameter Mapping Updates:** Keep sweep parameter mapping current with configuration schema
- **W&B API Compatibility:** Monitor W&B library updates and API changes
- **Sweep Best Practices:** Update based on W&B platform evolution
- **Performance Monitoring:** Track sweep execution efficiency and startup time

#### Monitoring Points
- **Sweep Success Rate:** Monitor successful sweep run completion
- **Parameter Coverage:** Ensure all critical hyperparameters included in mapping
- **Resource Usage:** Track memory and compute overhead during sweep runs
- **Error Patterns:** Monitor common sweep execution failures

#### Documentation Dependencies
- **`training_train.md`** - Main training script comparison and feature differences
- **`training_trainer.md`** - Trainer class integration and execution details
- **`wandb_integration.md`** - Comprehensive W&B integration documentation
- **`config_schema.md`** - Configuration structure and parameter definitions
- **`hyperparameter_optimization.md`** - Sweep strategies and best practices
