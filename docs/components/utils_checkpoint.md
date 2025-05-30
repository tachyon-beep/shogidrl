# Checkpoint Migration Module

## Module Overview

The `utils/checkpoint.py` module provides model checkpoint migration and compatibility utilities for the Keisei Shogi system. This module addresses the challenge of loading older model checkpoints when the model architecture has evolved, particularly when the number of input channels has changed. It implements intelligent padding and truncation strategies to maintain backward compatibility while enabling forward migration of trained models.

## Dependencies

### External Dependencies
- `torch`: PyTorch tensors and neural network modules
- `torch.nn`: Neural network module base classes
- `typing`: Type hints for dictionaries and any types

### Internal Dependencies
None - this is a standalone utility module.

## Function Documentation

### load_checkpoint_with_padding

Loads a checkpoint into a model with automatic handling of input channel dimension mismatches through zero-padding or truncation.

**Signature:**
```python
def load_checkpoint_with_padding(
    model: nn.Module, 
    checkpoint: Dict[str, Any], 
    input_channels: int
) -> None
```

**Parameters:**
- `model: nn.Module` - The target model instance (must have a stem Conv2d layer)
- `checkpoint: Dict[str, Any]` - The loaded checkpoint data (from torch.load)
- `input_channels: int` - Expected number of input channels for the current model

**Returns:**
- `None` - Modifies the model in-place

**Functionality:**

#### 1. Checkpoint Structure Handling
```python
state_dict = (
    checkpoint["model_state_dict"] 
    if "model_state_dict" in checkpoint 
    else checkpoint
)
```
Handles both raw state dictionaries and wrapped checkpoint formats.

#### 2. Stem Layer Identification
```python
stem_key = None
for k in model_state:
    if k.endswith("stem.weight"):
        stem_key = k
        break
```
Automatically locates the first convolutional layer (stem) that handles input channels.

#### 3. Channel Dimension Adaptation

**Zero-Padding (Channel Increase):**
When the current model expects more input channels than the checkpoint:
```python
if old_weight.shape[1] < new_weight.shape[1]:
    # Zero-pad input channels
    pad = torch.zeros(
        (old_weight.shape[0], 
         new_weight.shape[1] - old_weight.shape[1], 
         *old_weight.shape[2:]),
        dtype=old_weight.dtype,
        device=old_weight.device
    )
    padded_weight = torch.cat([old_weight, pad], dim=1)
    state_dict[stem_key] = padded_weight
```

**Truncation (Channel Decrease):**
When the current model expects fewer input channels:
```python
elif old_weight.shape[1] > new_weight.shape[1]:
    # Truncate input channels
    state_dict[stem_key] = old_weight[:, :new_weight.shape[1], :, :]
```

#### 4. Model Loading
```python
model.load_state_dict(state_dict, strict=False)
```
Uses non-strict loading to handle any remaining parameter mismatches gracefully.

## Data Structures

### Checkpoint Format Support
The function supports multiple checkpoint formats:

```python
# Raw state dictionary
checkpoint = {
    "stem.weight": torch.Tensor,  # Shape: [out_channels, in_channels, height, width]
    "layer1.weight": torch.Tensor,
    # ... other model parameters
}

# Wrapped checkpoint
checkpoint = {
    "model_state_dict": {
        "stem.weight": torch.Tensor,
        # ... model parameters
    },
    "optimizer_state_dict": {...},
    "epoch": int,
    "loss": float,
    # ... other metadata
}
```

### Weight Tensor Shapes
The function operates on convolutional weight tensors with the following structure:
```python
weight_shape = [
    out_channels,    # Number of output feature maps
    in_channels,     # Number of input channels (modified by this function)
    kernel_height,   # Convolution kernel height
    kernel_width     # Convolution kernel width
]
```

## Inter-Module Relationships

### Model Integration
- **Neural Networks**: Direct integration with PyTorch models
- **Training System**: Enables loading of older checkpoints during continued training
- **Evaluation System**: Allows evaluation of older models with new feature sets

### Checkpoint Management
- **Model Manager**: Could be integrated with centralized checkpoint management
- **Version Control**: Supports model versioning and migration workflows
- **Compatibility Layer**: Acts as compatibility layer between model versions

## Implementation Notes

### Padding Strategy
The zero-padding approach for additional input channels:
- **Preserves Learned Features**: Existing channels retain their learned weights
- **Neutral Initialization**: New channels start with zero weights (neutral contribution)
- **Gradual Learning**: New channels can be trained incrementally

### Truncation Strategy
The truncation approach for reduced input channels:
- **Feature Selection**: Keeps the first N channels (assumes importance ordering)
- **Information Loss**: Discards later channels (potential information loss)
- **Compatibility**: Maintains model functionality with reduced input

### Error Handling
- **Graceful Degradation**: Uses `strict=False` for partial parameter loading
- **Shape Preservation**: Maintains all dimensions except input channels
- **Device Consistency**: Preserves tensor device and dtype properties

## Testing Strategy

### Unit Tests
```python
def test_load_checkpoint_with_padding_increase():
    """Test loading checkpoint with increased input channels."""
    # Create model with more input channels than checkpoint
    # Verify zero-padding is applied correctly
    # Check that existing weights are preserved
    pass

def test_load_checkpoint_with_padding_decrease():
    """Test loading checkpoint with decreased input channels."""
    # Create model with fewer input channels than checkpoint
    # Verify truncation is applied correctly
    # Check that remaining weights are preserved
    pass

def test_load_checkpoint_same_channels():
    """Test loading checkpoint with same input channels."""
    # Verify normal loading without modification
    pass

def test_checkpoint_format_support():
    """Test both raw and wrapped checkpoint formats."""
    # Test raw state_dict format
    # Test wrapped checkpoint format
    pass
```

### Integration Tests
```python
def test_model_functionality_after_migration():
    """Test that migrated models function correctly."""
    # Load checkpoint with channel modification
    # Run inference and verify outputs
    # Compare with original model behavior where possible
    pass

def test_training_continuation():
    """Test continued training after checkpoint migration."""
    # Load migrated checkpoint
    # Continue training for several steps
    # Verify gradient flow and parameter updates
    pass
```

## Performance Considerations

### Memory Efficiency
- **In-Place Operations**: Modifies state dictionary in-place where possible
- **Temporary Tensors**: Minimizes creation of temporary tensors during padding
- **Device Management**: Preserves original device placement to avoid unnecessary transfers

### Computational Overhead
- **Shape Analysis**: Efficient tensor shape comparison
- **Padding Operations**: Uses efficient torch.cat for concatenation
- **Loading Time**: Minimal overhead added to standard checkpoint loading

## Security Considerations

### Data Integrity
- **Shape Validation**: Implicit validation through tensor operations
- **Type Preservation**: Maintains original dtype and device properties
- **Parameter Integrity**: Preserves existing learned parameters exactly

### Error Isolation
- **Non-Strict Loading**: Isolates parameter loading errors to specific layers
- **Graceful Handling**: Continues loading even with some parameter mismatches
- **State Preservation**: Maintains model in valid state after failed loading attempts

## Error Handling

### Exception Management
The function handles errors through:
- **Non-Strict Loading**: `strict=False` allows partial parameter loading
- **Shape Compatibility**: Automatic adaptation of incompatible shapes
- **Graceful Degradation**: Model remains functional even with some parameter mismatches

### Error Scenarios
```python
# Missing stem layer - function continues without modification
if stem_key is None:
    # No stem layer found, proceed with normal loading
    model.load_state_dict(state_dict, strict=False)

# Device mismatch - preserves original device
pad = torch.zeros(..., device=old_weight.device)  # Same device as original

# Dimension mismatch - handles through padding/truncation
if old_weight.shape[1] != new_weight.shape[1]:
    # Apply appropriate transformation
```

## Configuration

### Model Requirements
- Models must have a convolutional layer with a name ending in "stem.weight"
- The stem layer must be the first layer processing input channels
- Models should follow standard PyTorch nn.Module conventions

### Checkpoint Requirements
- Checkpoints must be valid PyTorch state dictionaries
- Parameter names must be consistent with model architecture
- Tensor shapes must be compatible (except for input channel dimension)

## Future Enhancements

### Planned Features
- **Smart Channel Selection**: Intelligent selection of which channels to keep during truncation
- **Gradual Migration**: Gradual channel adaptation over multiple loading steps
- **Migration Validation**: Validation of migration success and model performance
- **Version Metadata**: Checkpoint versioning to track migration history

### Advanced Migration Strategies
- **Feature Importance**: Use feature importance for intelligent channel selection
- **Progressive Loading**: Gradual introduction of new channels during training
- **Migration Logging**: Detailed logging of migration operations
- **Rollback Support**: Ability to reverse migration operations

## Usage Examples

### Basic Migration
```python
import torch
from keisei.utils.checkpoint import load_checkpoint_with_padding

# Load model and checkpoint
model = MyModel(input_channels=46)  # Current model expects 46 channels
checkpoint = torch.load("old_model.pth")  # Checkpoint has 40 channels

# Migrate checkpoint to new model
load_checkpoint_with_padding(model, checkpoint, input_channels=46)

# Model is now ready with migrated weights
model.eval()
```

### Training Continuation
```python
# Load checkpoint for continued training
model = ResNetModel(input_channels=50)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("checkpoint_epoch_100.pth")

# Migrate model weights
load_checkpoint_with_padding(model, checkpoint, input_channels=50)

# Continue training with migrated model
for epoch in range(101, 200):
    train_epoch(model, optimizer, train_loader)
```

### Evaluation Migration
```python
# Migrate model for evaluation with new features
eval_model = create_evaluation_model(input_channels=48)
old_checkpoint = torch.load("best_model_v1.pth")

# Migrate to new feature set
load_checkpoint_with_padding(eval_model, old_checkpoint, input_channels=48)

# Evaluate with migrated model
results = evaluate_model(eval_model, test_dataset)
```

## Maintenance Notes

### Code Quality
- Maintain clear documentation of migration strategies
- Follow consistent error handling patterns
- Keep migration logic simple and predictable
- Document any changes to migration behavior

### Dependencies
- Monitor PyTorch API changes for tensor operations
- Keep tensor manipulation code up to date
- Maintain compatibility with different PyTorch versions
- Update type hints with Python version changes

### Testing Requirements
- Comprehensive unit tests for all migration scenarios
- Integration tests with real model architectures
- Performance tests for large model migrations
- Regression tests to catch migration behavior changes
