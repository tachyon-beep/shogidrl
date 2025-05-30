# Training Models Package Documentation

## Module Overview

**File:** `keisei/training/models/__init__.py`

**Purpose:** Provides model factory functionality and initialization for neural network architectures used in the Keisei Shogi DRL training system.

**Core Functionality:**
- Model factory pattern for creating neural network instances
- Centralized model instantiation with configuration support
- Support for multiple model architectures and testing variants
- Integration with ActorCriticProtocol interface

## Dependencies

### Internal Dependencies
- `keisei.core.actor_critic_protocol`: ActorCriticProtocol interface
- `resnet_tower`: ActorCriticResTower implementation

### External Dependencies
None directly imported at module level

## Class Documentation

This module contains no class definitions but provides factory functions for model creation.

## Function Documentation

### `model_factory`

**Signature:**
```python
def model_factory(
    model_type: str,
    obs_shape: tuple,
    num_actions: int,
    tower_depth: int,
    tower_width: int,
    se_ratio: float,
    **kwargs
) -> ActorCriticProtocol
```

**Purpose:** Factory function that creates and returns neural network models based on configuration parameters.

**Parameters:**
- `model_type` (str): Type of model to create ("resnet", "dummy", "testmodel", "resumemodel")
- `obs_shape` (tuple): Shape of observation space, where `obs_shape[0]` is input channels
- `num_actions` (int): Total number of possible actions in the action space
- `tower_depth` (int): Depth of the neural network tower (number of residual blocks)
- `tower_width` (int): Width of the neural network tower (number of channels)
- `se_ratio` (float): Squeeze-and-Excitation ratio for attention mechanisms
- `**kwargs`: Additional keyword arguments passed to model constructors

**Returns:**
- `ActorCriticProtocol`: An instance implementing the ActorCritic interface

**Raises:**
- `ValueError`: When unknown model_type is provided

**Implementation Details:**

#### Production Models
- **"resnet"**: Creates `ActorCriticResTower` with full configuration parameters
  - Uses all provided configuration parameters
  - Suitable for production training and evaluation

#### Test/Development Models
- **"dummy", "testmodel", "resumemodel"**: Creates minimal `ActorCriticResTower` instances
  - Fixed minimal parameters for testing: `tower_depth=1`, `tower_width=16`, `se_ratio=None`
  - Preserves input/output compatibility while reducing computational requirements
  - Suitable for unit testing, integration testing, and development workflows

## Data Structures

### Model Configuration Schema
```python
ModelConfig = {
    "model_type": str,           # Architecture identifier
    "obs_shape": Tuple[int, ...], # Observation space dimensions
    "num_actions": int,          # Action space size
    "tower_depth": int,          # Network depth
    "tower_width": int,          # Network width
    "se_ratio": Optional[float], # Attention mechanism ratio
    "additional_params": dict    # Model-specific parameters
}
```

### Factory Return Interface
All models returned by the factory implement `ActorCriticProtocol`:
```python
class ActorCriticProtocol:
    def forward(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value_estimate)"""
    
    def get_action_and_value(self, obs, legal_mask=None, deterministic=False):
        """Returns (action, log_prob, value)"""
```

## Inter-Module Relationships

### Dependencies
```
models/__init__.py
    ├── core.actor_critic_protocol (imports interface)
    └── resnet_tower (imports implementation)
```

### Usage Patterns
```python
# Factory usage in training
from keisei.training.models import model_factory

model = model_factory(
    model_type="resnet",
    obs_shape=(46, 9, 9),  # Shogi observation shape
    num_actions=3781,      # Shogi action space
    tower_depth=9,
    tower_width=256,
    se_ratio=0.25
)

# Test model creation
test_model = model_factory(
    model_type="dummy",
    obs_shape=(46, 9, 9),
    num_actions=3781,
    tower_depth=1,  # Ignored for test models
    tower_width=16, # Ignored for test models
    se_ratio=None   # Ignored for test models
)
```

## Implementation Notes

### Design Patterns
- **Factory Pattern**: Centralized model creation with configuration-based instantiation
- **Strategy Pattern**: Different model types implement the same interface
- **Template Method**: Common parameter handling with type-specific customization

### Architecture Considerations
- **Interface Compliance**: All models implement `ActorCriticProtocol`
- **Configuration Flexibility**: Supports arbitrary additional parameters via `**kwargs`
- **Testing Support**: Dedicated lightweight models for testing scenarios

### Model Type Strategy
```python
if model_type == "resnet":
    # Production model with full configuration
    return ActorCriticResTower(input_channels=obs_shape[0], ...)
elif model_type in ["dummy", "testmodel", "resumemodel"]:
    # Lightweight model with fixed minimal parameters
    return ActorCriticResTower(input_channels=obs_shape[0], tower_depth=1, ...)
else:
    # Unknown model type
    raise ValueError(f"Unknown model_type: {model_type}")
```

## Testing Strategy

### Unit Testing
```python
def test_model_factory_resnet():
    """Test ResNet model creation."""
    model = model_factory(
        model_type="resnet",
        obs_shape=(46, 9, 9),
        num_actions=3781,
        tower_depth=9,
        tower_width=256,
        se_ratio=0.25
    )
    assert isinstance(model, ActorCriticResTower)
    assert hasattr(model, 'forward')
    assert hasattr(model, 'get_action_and_value')

def test_model_factory_test_models():
    """Test creation of test/dummy models."""
    for model_type in ["dummy", "testmodel", "resumemodel"]:
        model = model_factory(
            model_type=model_type,
            obs_shape=(46, 9, 9),
            num_actions=3781,
            tower_depth=99,  # Should be ignored
            tower_width=512, # Should be ignored
            se_ratio=0.5     # Should be ignored
        )
        assert isinstance(model, ActorCriticResTower)

def test_model_factory_unknown_type():
    """Test error handling for unknown model types."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        model_factory(
            model_type="nonexistent",
            obs_shape=(46, 9, 9),
            num_actions=3781,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25
        )
```

### Integration Testing
- **Protocol Compliance**: Verify all created models implement required interface
- **Configuration Handling**: Test parameter passing and model behavior
- **Training Integration**: Test models work correctly in training loops

## Performance Considerations

### Model Creation Overhead
- **Lightweight Factory**: Minimal overhead in model instantiation
- **Parameter Validation**: Quick parameter checking and routing
- **Memory Efficiency**: Models created on-demand, not cached

### Test Model Optimization
- **Minimal Parameters**: Test models use reduced parameters for faster execution
- **Preserved Interface**: Same API as production models
- **Development Speed**: Fast iteration for testing and debugging

## Security Considerations

### Model Instantiation Safety
- **Parameter Validation**: Factory validates model_type before instantiation
- **Controlled Creation**: Only known model types can be created
- **Safe Defaults**: Test models use safe minimal parameters

### Configuration Security
- **Type Safety**: All parameters validated before model creation
- **Error Handling**: Clear error messages for invalid configurations

## Error Handling

### Factory Error Management
```python
try:
    model = model_factory(model_type="unknown", ...)
except ValueError as e:
    logger.error(f"Model creation failed: {e}")
    # Handle model creation failure
```

### Common Error Scenarios
- **Unknown Model Type**: Clear error message with supported types
- **Invalid Parameters**: Parameter validation in individual model constructors
- **Configuration Mismatches**: Type and shape validation

## Configuration

### Model Type Configuration
```yaml
# Configuration for ResNet model
model:
  type: "resnet"
  tower_depth: 9
  tower_width: 256
  se_ratio: 0.25

# Configuration for test model
model:
  type: "dummy"
  # Other parameters ignored for dummy models
```

### Factory Configuration
- **Flexible Parameters**: Supports arbitrary additional parameters
- **Type-Specific Handling**: Different configuration strategies per model type
- **Validation**: Parameter validation at factory level

## Future Enhancements

### Model Architecture Extensions
1. **New Architectures**: Add support for Transformer, EfficientNet, etc.
2. **Hybrid Models**: Support for ensemble or multi-tower architectures
3. **Quantized Models**: Support for quantized or pruned model variants
4. **Custom Models**: Plugin system for user-defined architectures

### Factory Improvements
```python
# Enhanced factory with validation and registration
class ModelRegistry:
    def register_model(self, name: str, model_class: type):
        """Register new model type."""
    
    def create_model(self, config: ModelConfig) -> ActorCriticProtocol:
        """Create model with enhanced validation."""

# Future factory interface
def model_factory_v2(config: ModelConfig) -> ActorCriticProtocol:
    """Enhanced factory with configuration objects."""
```

### Configuration Enhancements
- **Model Validation**: Schema validation for model configurations
- **Parameter Optimization**: Automatic parameter tuning for different model types
- **Performance Profiles**: Pre-configured model variants for different use cases

## Usage Examples

### Production Model Creation
```python
from keisei.training.models import model_factory

# Create full ResNet model for training
model = model_factory(
    model_type="resnet",
    obs_shape=(46, 9, 9),     # Shogi feature planes
    num_actions=3781,         # Shogi action space
    tower_depth=9,            # Deep network
    tower_width=256,          # Wide network
    se_ratio=0.25,           # Squeeze-Excitation attention
    dropout_rate=0.1         # Additional parameter via kwargs
)
```

### Test Model Creation
```python
# Create lightweight model for testing
test_model = model_factory(
    model_type="dummy",
    obs_shape=(46, 9, 9),
    num_actions=3781,
    tower_depth=999,  # Ignored
    tower_width=999,  # Ignored
    se_ratio=0.999    # Ignored
)

# Model has same interface but minimal computational requirements
policy_logits, value = test_model.forward(observation)
```

### Custom Model Integration
```python
# Future: Custom model registration
from keisei.training.models import ModelRegistry

registry = ModelRegistry()
registry.register_model("custom_transformer", CustomTransformer)

model = model_factory(
    model_type="custom_transformer",
    obs_shape=(46, 9, 9),
    num_actions=3781,
    attention_heads=8,
    transformer_layers=6
)
```

## Maintenance Notes

### Code Reviews
- Verify new model types implement `ActorCriticProtocol`
- Check parameter handling for consistency
- Ensure test model variants remain lightweight

### Model Registry Management
- Document all supported model types and their parameters
- Maintain consistency in parameter naming conventions
- Keep test model parameters minimal and fixed

### Performance Monitoring
- Monitor model creation overhead
- Track memory usage of different model types
- Optimize factory performance for frequently created models
