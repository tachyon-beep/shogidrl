# ResNet Tower Model Documentation

## Module Overview

**File:** `keisei/training/models/resnet_tower.py`

**Purpose:** Implements a ResNet-based Actor-Critic neural network architecture with Squeeze-and-Excitation attention for Shogi Deep Reinforcement Learning.

**Core Functionality:**
- ResNet tower architecture with residual connections
- Squeeze-and-Excitation attention mechanisms
- Dual-head design for policy and value estimation
- Legal action masking support
- Batch processing capabilities

## Dependencies

### Internal Dependencies
None (standalone implementation)

### External Dependencies
- `torch`: PyTorch framework for neural network implementation
- `torch.nn`: Neural network modules and layers
- `torch.nn.functional`: Functional operations
- `sys`: System utilities for error output

## Class Documentation

### `SqueezeExcitation`

**Purpose:** Implements Squeeze-and-Excitation attention mechanism for enhancing feature representations.

**Attributes:**
- `fc1` (nn.Conv2d): First fully connected layer for dimensionality reduction
- `fc2` (nn.Conv2d): Second fully connected layer for dimensionality restoration

**Methods:**

#### `__init__(self, channels: int, se_ratio: float = 0.25)`
- **Purpose:** Initialize the SE block with channel attention
- **Parameters:**
  - `channels` (int): Number of input channels
  - `se_ratio` (float): Reduction ratio for bottleneck layer (default: 0.25)

#### `forward(self, x: torch.Tensor) -> torch.Tensor`
- **Purpose:** Apply squeeze-and-excitation attention to input features
- **Process:**
  1. Global average pooling to squeeze spatial dimensions
  2. Two-layer MLP with ReLU and Sigmoid activation
  3. Channel-wise multiplication with original features
- **Returns:** Attention-weighted feature tensor

### `ResidualBlock`

**Purpose:** Implements a residual block with optional Squeeze-and-Excitation attention.

**Attributes:**
- `conv1` (nn.Conv2d): First convolutional layer (3x3)
- `bn1` (nn.BatchNorm2d): First batch normalization layer
- `conv2` (nn.Conv2d): Second convolutional layer (3x3)
- `bn2` (nn.BatchNorm2d): Second batch normalization layer
- `se` (Optional[SqueezeExcitation]): Optional SE attention block

**Methods:**

#### `__init__(self, channels: int, se_ratio: Optional[float] = None)`
- **Purpose:** Initialize residual block with optional attention
- **Parameters:**
  - `channels` (int): Number of channels throughout the block
  - `se_ratio` (Optional[float]): SE ratio if attention is desired

#### `forward(self, x: torch.Tensor) -> torch.Tensor`
- **Purpose:** Apply residual transformation with skip connection
- **Process:**
  1. Conv → BatchNorm → ReLU
  2. Conv → BatchNorm
  3. Optional SE attention
  4. Add skip connection → ReLU
- **Returns:** Transformed feature tensor with residual connection

### `ActorCriticResTower`

**Purpose:** Main Actor-Critic network with ResNet tower and dual heads for policy and value estimation.

**Attributes:**
- `stem` (nn.Conv2d): Initial convolution layer
- `bn_stem` (nn.BatchNorm2d): Stem batch normalization
- `res_blocks` (nn.Sequential): Stack of residual blocks
- `policy_head` (nn.Sequential): Policy estimation head
- `value_head` (nn.Sequential): Value estimation head

**Methods:**

#### `__init__`
```python
def __init__(
    self,
    input_channels: int,
    num_actions_total: int,
    tower_depth: int = 9,
    tower_width: int = 256,
    se_ratio: Optional[float] = None
)
```
- **Purpose:** Initialize the complete Actor-Critic network
- **Parameters:**
  - `input_channels` (int): Number of input feature channels (e.g., 46 for Shogi)
  - `num_actions_total` (int): Total number of possible actions (e.g., 3781 for Shogi)
  - `tower_depth` (int): Number of residual blocks (default: 9)
  - `tower_width` (int): Number of channels in the tower (default: 256)
  - `se_ratio` (Optional[float]): Squeeze-Excitation ratio for all blocks

#### `forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
- **Purpose:** Forward pass through the network
- **Process:**
  1. Stem convolution → BatchNorm → ReLU
  2. ResNet tower (residual blocks)
  3. Parallel policy and value heads
- **Returns:** Tuple of (policy_logits, value_estimate)

#### `get_action_and_value`
```python
def get_action_and_value(
    self,
    obs: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
- **Purpose:** Sample actions and compute values for RL interaction
- **Parameters:**
  - `obs` (torch.Tensor): Input observations
  - `legal_mask` (Optional[torch.Tensor]): Boolean mask for legal actions
  - `deterministic` (bool): Use argmax instead of sampling
- **Process:**
  1. Forward pass to get policy logits and values
  2. Apply legal action masking if provided
  3. Create categorical distribution
  4. Sample or choose deterministic action
- **Returns:** Tuple of (action, log_probability, value)
- **Error Handling:** Handles NaN probabilities with uniform fallback

#### `evaluate_actions`
```python
def evaluate_actions(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
- **Purpose:** Evaluate log probabilities and entropy for given observations and actions
- **Parameters:**
  - `obs` (torch.Tensor): Batch of observations
  - `actions` (torch.Tensor): Batch of taken actions
  - `legal_mask` (Optional[torch.Tensor]): Legal action masks for the batch
- **Process:**
  1. Forward pass for policy logits and values
  2. Apply legal action masking if provided
  3. Compute categorical distribution
  4. Calculate log probabilities and entropy
- **Returns:** Tuple of (log_probabilities, entropy, values)
- **Used In:** PPO training for policy gradient computation

## Data Structures

### Network Architecture
```
Input (batch_size, 46, 9, 9) - Shogi board features
    ↓
Stem Conv2d(46 → tower_width, 3x3) + BatchNorm + ReLU
    ↓
ResNet Tower: tower_depth × ResidualBlock(tower_width)
    ├── Conv2d(3x3) → BatchNorm → ReLU
    ├── Conv2d(3x3) → BatchNorm
    ├── Optional SE Attention
    └── Skip Connection + ReLU
    ↓
Policy Head: Conv2d(tower_width → 2, 1x1) → BatchNorm → ReLU → Flatten → Linear(2×9×9 → num_actions)
Value Head:  Conv2d(tower_width → 2, 1x1) → BatchNorm → ReLU → Flatten → Linear(2×9×9 → 1)
    ↓
Output: (policy_logits, value_estimate)
```

### Tensor Shapes
- **Input:** `(batch_size, input_channels, 9, 9)` - Shogi board representation
- **Policy Output:** `(batch_size, num_actions_total)` - Action probability logits
- **Value Output:** `(batch_size,)` - State value estimates
- **Legal Mask:** `(batch_size, num_actions_total)` - Boolean legal action indicators

## Inter-Module Relationships

### Dependencies
```
resnet_tower.py
    ├── torch (neural network framework)
    ├── torch.nn (network modules)
    └── torch.nn.functional (functional operations)
```

### Usage by Other Modules
- **model_factory**: Creates instances via factory pattern
- **PPOAgent**: Uses for policy and value estimation
- **Training Loop**: Calls get_action_and_value for interaction
- **Experience Buffer**: Receives outputs for storage

## Implementation Notes

### Design Patterns
- **Composite Pattern**: Network composed of reusable building blocks
- **Template Method**: Consistent forward pass structure
- **Strategy Pattern**: Optional SE attention based on configuration

### Architecture Considerations
- **Slim Heads**: Policy and value heads use only 2 channels to reduce parameters
- **Shared Trunk**: Common feature extraction with specialized heads
- **Residual Learning**: Skip connections for deep network training
- **Attention Mechanism**: Optional SE blocks for enhanced feature learning

### Legal Action Masking
```python
# Masking illegal actions
masked_logits = torch.where(
    legal_mask,
    policy_logits,
    torch.tensor(float("-inf"), device=policy_logits.device)
)
```

### Error Handling Strategy
- **NaN Detection**: Monitors for NaN probabilities from masking
- **Fallback Mechanism**: Uses uniform distribution when all actions masked
- **Warning System**: Logs warnings to stderr for debugging

## Testing Strategy

### Unit Testing
```python
def test_squeeze_excitation():
    """Test SE block functionality."""
    se = SqueezeExcitation(channels=64, se_ratio=0.25)
    x = torch.randn(2, 64, 9, 9)
    out = se(x)
    assert out.shape == x.shape

def test_residual_block():
    """Test residual block with and without SE."""
    # Without SE
    block = ResidualBlock(channels=128)
    x = torch.randn(2, 128, 9, 9)
    out = block(x)
    assert out.shape == x.shape
    
    # With SE
    block_se = ResidualBlock(channels=128, se_ratio=0.25)
    out_se = block_se(x)
    assert out_se.shape == x.shape

def test_actor_critic_res_tower():
    """Test complete network functionality."""
    model = ActorCriticResTower(
        input_channels=46,
        num_actions_total=3781,
        tower_depth=3,
        tower_width=64,
        se_ratio=0.25
    )
    
    # Test forward pass
    obs = torch.randn(4, 46, 9, 9)
    policy, value = model(obs)
    assert policy.shape == (4, 3781)
    assert value.shape == (4,)
    
    # Test action sampling
    action, log_prob, value = model.get_action_and_value(obs[:1])
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)
    
    # Test action evaluation
    actions = torch.randint(0, 3781, (4,))
    log_probs, entropy, values = model.evaluate_actions(obs, actions)
    assert log_probs.shape == (4,)
    assert entropy.shape == (4,)
    assert values.shape == (4,)
```

### Integration Testing
- **PPO Integration**: Test compatibility with PPO agent
- **Legal Masking**: Verify correct behavior with legal action masks
- **Batch Processing**: Test various batch sizes and input shapes

## Performance Considerations

### Computational Efficiency
- **Slim Heads**: Reduced parameters in policy/value heads
- **Efficient SE**: Lightweight attention mechanism
- **Optimized Convolutions**: Standard 3x3 kernels for hardware acceleration

### Memory Optimization
- **In-place Operations**: ReLU activations where possible
- **Gradient Checkpointing**: Can be added for memory-constrained training
- **Batch Processing**: Efficient vectorized operations

### Training Stability
- **Batch Normalization**: Stabilizes training across residual blocks
- **Skip Connections**: Enables deep network training
- **Proper Initialization**: PyTorch default initialization

## Security Considerations

### Input Validation
- **Tensor Shapes**: Implicit validation through PyTorch operations
- **Device Consistency**: Ensures tensors are on the same device
- **NaN Handling**: Robust handling of numerical instabilities

### Numerical Stability
- **Masked Softmax**: Safe handling of -inf logits
- **Distribution Creation**: Validation of probability distributions
- **Fallback Mechanisms**: Graceful degradation for edge cases

## Error Handling

### Common Error Scenarios
```python
# Shape mismatch
try:
    policy, value = model(invalid_shape_tensor)
except RuntimeError as e:
    logger.error(f"Shape mismatch: {e}")

# NaN handling
if torch.isnan(probs).any():
    print("Warning: NaNs detected, using uniform distribution")
    probs = torch.ones_like(probs) / probs.shape[-1]

# All actions masked
if not torch.any(legal_mask):
    logger.warning("No legal actions available")
    # Continue with uniform distribution
```

## Configuration

### Model Configuration
```yaml
model:
  type: "resnet"
  input_channels: 46      # Shogi feature planes
  num_actions_total: 3781 # Shogi action space
  tower_depth: 9          # Number of residual blocks
  tower_width: 256        # Channel width
  se_ratio: 0.25         # Squeeze-Excitation ratio
```

### Architecture Variants
- **Lightweight**: `tower_depth=3, tower_width=64`
- **Standard**: `tower_depth=9, tower_width=256`
- **Heavy**: `tower_depth=15, tower_width=512`

## Future Enhancements

### Architecture Improvements
1. **Multi-Scale Features**: Incorporate feature pyramid networks
2. **Efficient Attention**: Replace SE with more efficient attention mechanisms
3. **Mixed Precision**: Support for FP16 training
4. **Dynamic Architectures**: Adaptive depth/width based on complexity

### Implementation Enhancements
```python
# Future: Mixed precision support
@torch.cuda.amp.autocast()
def forward(self, x):
    # Forward pass with automatic mixed precision

# Future: Dynamic attention
class AdaptiveAttention(nn.Module):
    def __init__(self, channels, attention_types=['se', 'cbam', 'eca']):
        # Multi-type attention selection

# Future: Efficient blocks
class EfficientResBlock(nn.Module):
    """More efficient residual block with grouped convolutions."""
```

### Training Optimizations
- **Gradient Checkpointing**: Memory-efficient deep network training
- **Knowledge Distillation**: Teacher-student training paradigms
- **Progressive Training**: Start with shallow networks, gradually increase depth

## Usage Examples

### Basic Model Creation
```python
from keisei.training.models.resnet_tower import ActorCriticResTower

# Create standard model
model = ActorCriticResTower(
    input_channels=46,
    num_actions_total=3781,
    tower_depth=9,
    tower_width=256,
    se_ratio=0.25
)

# Move to GPU
model = model.cuda()
```

### Forward Pass
```python
import torch

# Prepare input
obs = torch.randn(32, 46, 9, 9).cuda()  # Batch of observations

# Forward pass
policy_logits, values = model(obs)
print(f"Policy shape: {policy_logits.shape}")  # [32, 3781]
print(f"Value shape: {values.shape}")          # [32]
```

### Action Sampling with Legal Masks
```python
# Single observation
obs = torch.randn(1, 46, 9, 9).cuda()
legal_mask = torch.randint(0, 2, (1, 3781), dtype=torch.bool).cuda()

# Sample action
action, log_prob, value = model.get_action_and_value(
    obs, 
    legal_mask=legal_mask, 
    deterministic=False
)

# Deterministic action selection
action_det, log_prob_det, value_det = model.get_action_and_value(
    obs, 
    legal_mask=legal_mask, 
    deterministic=True
)
```

### Batch Action Evaluation
```python
# Batch evaluation for PPO training
obs_batch = torch.randn(64, 46, 9, 9).cuda()
actions_batch = torch.randint(0, 3781, (64,)).cuda()

log_probs, entropy, values = model.evaluate_actions(
    obs_batch, 
    actions_batch
)

print(f"Log probs: {log_probs.shape}")  # [64]
print(f"Entropy: {entropy.shape}")      # [64]
print(f"Values: {values.shape}")        # [64]
```

## Maintenance Notes

### Code Reviews
- Verify tensor shape consistency throughout forward passes
- Check device placement for all operations
- Ensure proper error handling for edge cases

### Performance Monitoring
- Monitor training stability with different SE ratios
- Track memory usage with varying batch sizes
- Profile forward/backward pass times

### Architecture Evolution
- Keep architecture compatible with ActorCriticProtocol interface
- Maintain backward compatibility for model checkpoints
- Document any changes to default hyperparameters
