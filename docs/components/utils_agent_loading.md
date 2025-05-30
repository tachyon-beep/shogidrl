# Agent Loading Module

## Module Overview

The `utils/agent_loading.py` module provides utilities for loading trained PPO agents and initializing various types of opponents for evaluation purposes. This module serves as a bridge between saved model checkpoints and the evaluation system, handling the complexities of model restoration, configuration setup, and opponent instantiation. It enables flexible evaluation scenarios by supporting different opponent types including random players, heuristic players, and trained PPO agents.

## Dependencies

### External Dependencies
- `os`: File system operations for checkpoint validation
- `torch`: PyTorch tensors and device management
- `typing`: Type hints and annotations

### Internal Dependencies
- `keisei.config_schema`: Configuration classes for agent setup
- `keisei.core.ppo_agent`: PPO agent implementation
- `keisei.utils.opponents`: Opponent implementations (BaseOpponent, SimpleHeuristicOpponent, SimpleRandomOpponent)

## Function Documentation

### load_evaluation_agent

Loads a trained PPO agent from a checkpoint file for evaluation purposes.

**Signature:**
```python
def load_evaluation_agent(
    checkpoint_path: str,
    device_str: str,
    policy_mapper,
    input_channels: int,
    input_features: Optional[str] = "core46"
) -> PPOAgent
```

**Parameters:**
- `checkpoint_path: str` - Path to the model checkpoint file
- `device_str: str` - Device string (e.g., "cuda:0", "cpu")
- `policy_mapper` - PolicyOutputMapper instance for action mapping
- `input_channels: int` - Number of input channels for the neural network
- `input_features: Optional[str]` - Feature set identifier (default: "core46")

**Returns:**
- `PPOAgent` - Loaded and initialized PPO agent in evaluation mode

**Functionality:**
1. **Checkpoint Validation**: Verifies checkpoint file exists
2. **Configuration Creation**: Builds minimal AppConfig for agent initialization
3. **Agent Instantiation**: Creates PPOAgent with proper configuration
4. **Model Loading**: Loads model weights from checkpoint
5. **Evaluation Mode**: Sets model to evaluation mode for inference

**Configuration Setup:**
The function creates a minimal configuration with dummy values for required fields:
```python
config = AppConfig(
    env=EnvConfig(
        device=device_str,
        input_channels=input_channels,
        num_actions_total=policy_mapper.get_total_actions(),
        seed=42
    ),
    training=TrainingConfig(
        # Minimal required training parameters
        total_timesteps=1,
        steps_per_epoch=1,
        # ... other required fields with dummy values
    ),
    # ... other required configuration sections
)
```

### initialize_opponent

Creates and returns an opponent instance based on the specified type and configuration.

**Signature:**
```python
def initialize_opponent(
    opponent_type: str,
    opponent_path: Optional[str],
    device_str: str,
    policy_mapper,
    input_channels: int
) -> Any
```

**Parameters:**
- `opponent_type: str` - Type of opponent ("random", "heuristic", "ppo")
- `opponent_path: Optional[str]` - Path to opponent checkpoint (required for "ppo")
- `device_str: str` - Device string for PPO opponents
- `policy_mapper` - PolicyOutputMapper instance
- `input_channels: int` - Number of input channels for PPO opponents

**Returns:**
- `Any` - Opponent instance (specific type depends on opponent_type)

**Supported Opponent Types:**
1. **"random"**: Returns `SimpleRandomOpponent` instance
2. **"heuristic"**: Returns `SimpleHeuristicOpponent` instance  
3. **"ppo"**: Returns trained PPO agent loaded from checkpoint

**Error Handling:**
- Raises `ValueError` for unknown opponent types
- Raises `ValueError` if opponent_path missing for PPO opponents
- Propagates checkpoint loading errors from `load_evaluation_agent`

## Data Structures

### Agent Loading Result
The loaded agent has the following key properties:
```python
loaded_agent = {
    "model": torch.nn.Module,        # Neural network in eval mode
    "config": AppConfig,             # Minimal configuration
    "device": torch.device,          # Computation device
    "policy_mapper": PolicyOutputMapper  # Action mapping
}
```

### Opponent Types
```python
opponent_types = {
    "random": SimpleRandomOpponent,      # Random move selection
    "heuristic": SimpleHeuristicOpponent, # Simple heuristic strategy
    "ppo": PPOAgent                      # Trained neural network
}
```

## Inter-Module Relationships

### Core Integration
- **PPO Agent**: Direct instantiation and loading of trained agents
- **Configuration**: Uses AppConfig structure for agent setup
- **Policy Mapping**: Integrates with PolicyOutputMapper for action spaces

### Evaluation Integration
- **Evaluation System**: Primary consumer for loaded agents and opponents
- **Game Environment**: Loaded agents interact with ShogiGame
- **Results Analysis**: Enables comparative evaluation between different agent types

### Opponent Integration
- **Base Classes**: Uses BaseOpponent hierarchy
- **Simple Opponents**: Direct integration with RandomOpponent and HeuristicOpponent
- **Agent Opponents**: Treats loaded PPO agents as opponents

## Implementation Notes

### Configuration Management
The module creates minimal configurations sufficient for agent operation during evaluation:
- **Required Fields**: Only essential parameters are set
- **Dummy Values**: Non-essential parameters use safe defaults
- **Device Management**: Proper device configuration for GPU/CPU evaluation

### Model State Handling
- **Evaluation Mode**: All loaded models set to eval mode for inference
- **Weight Loading**: Checkpoint weights loaded with proper error handling
- **Device Placement**: Models moved to specified devices

### Opponent Abstraction
The function provides a unified interface for different opponent types:
- **Type-based Dispatch**: Simple string-based opponent selection
- **Consistent Interface**: All opponents follow the same usage pattern
- **Error Handling**: Clear error messages for configuration issues

## Testing Strategy

### Unit Tests
```python
def test_load_evaluation_agent():
    """Test agent loading from checkpoint."""
    # Test successful loading
    # Test missing checkpoint file
    # Test invalid checkpoint format
    pass

def test_initialize_opponent_types():
    """Test all opponent type initialization."""
    # Test random opponent
    # Test heuristic opponent  
    # Test PPO opponent with valid checkpoint
    pass

def test_opponent_type_validation():
    """Test opponent type validation."""
    # Test unknown opponent types
    # Test missing PPO checkpoint path
    pass
```

### Integration Tests
```python
def test_agent_evaluation_integration():
    """Test loaded agents in evaluation context."""
    pass

def test_opponent_game_integration():
    """Test opponents with actual game instances."""
    pass
```

### Mock Testing
```python
def test_checkpoint_loading_mocked():
    """Test checkpoint loading with mocked file system."""
    pass
```

## Performance Considerations

### Loading Optimization
- **Lazy Loading**: Agents loaded only when needed
- **Device Optimization**: Efficient device placement and memory management
- **Configuration Caching**: Minimal configuration object creation

### Memory Management
- **Model Cleanup**: Proper cleanup of loaded models
- **Device Memory**: Efficient GPU memory usage for loaded agents
- **Configuration Overhead**: Minimal memory footprint for dummy configurations

## Security Considerations

### File System Security
- **Path Validation**: Checkpoint paths validated before loading
- **File Existence**: Verification of checkpoint file existence
- **Access Control**: Relies on file system permissions

### Model Security
- **Checkpoint Integrity**: Basic validation of checkpoint format
- **Device Security**: Safe device specification and placement
- **Memory Safety**: Proper tensor device management

## Error Handling

### Exception Categories
- **File Errors**: Missing or invalid checkpoint files
- **Configuration Errors**: Invalid parameters or device specifications
- **Model Errors**: Checkpoint loading or model instantiation failures
- **Type Errors**: Unknown opponent types or missing parameters

### Error Recovery
```python
# Checkpoint validation
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

# Opponent type validation  
if opponent_type == "ppo" and not opponent_path:
    raise ValueError("Opponent path must be provided for PPO opponent type.")

# Unknown type handling
else:
    raise ValueError(f"Unknown opponent type: {opponent_type}")
```

## Configuration

### Checkpoint Requirements
- Checkpoint files must contain valid PyTorch state dictionaries
- Models must be compatible with current PPOAgent implementation
- Device specifications must be valid PyTorch device strings

### Feature Set Compatibility
- Input features must match the training configuration
- Input channels must be consistent with model architecture
- Policy mapping must be compatible with saved action spaces

## Future Enhancements

### Planned Features
- **Checkpoint Validation**: Enhanced validation of checkpoint integrity
- **Multi-Model Loading**: Support for ensemble models
- **Streaming Loading**: Efficient loading of large models
- **Model Conversion**: Support for different checkpoint formats

### Extensibility Points
- **Custom Opponents**: Plugin architecture for new opponent types
- **Advanced Configuration**: More flexible configuration management
- **Model Optimization**: Post-loading model optimization (quantization, pruning)
- **Distributed Loading**: Support for loading models across multiple devices

## Usage Examples

### Basic Agent Loading
```python
# Load a trained agent for evaluation
agent = load_evaluation_agent(
    checkpoint_path="/path/to/model.pth",
    device_str="cuda:0",
    policy_mapper=policy_mapper,
    input_channels=46
)

# Agent is ready for evaluation
move = agent.select_action(observation)
```

### Opponent Initialization
```python
# Initialize different opponent types
random_opponent = initialize_opponent(
    opponent_type="random",
    opponent_path=None,
    device_str="cpu",
    policy_mapper=policy_mapper,
    input_channels=46
)

ppo_opponent = initialize_opponent(
    opponent_type="ppo", 
    opponent_path="/path/to/opponent.pth",
    device_str="cuda:0",
    policy_mapper=policy_mapper,
    input_channels=46
)
```

### Evaluation Setup
```python
# Set up evaluation with different opponents
def setup_evaluation(opponent_config):
    opponent = initialize_opponent(
        opponent_type=opponent_config["type"],
        opponent_path=opponent_config.get("path"),
        device_str=opponent_config["device"],
        policy_mapper=policy_mapper,
        input_channels=46
    )
    
    return opponent
```

## Maintenance Notes

### Code Quality
- Maintain consistent error handling patterns
- Document configuration requirements clearly
- Keep opponent type registration up to date
- Follow consistent naming conventions for loaded models

### Dependencies
- Monitor PPOAgent API changes
- Keep configuration schema compatibility
- Update opponent class imports as needed
- Maintain PyTorch version compatibility

### Testing Requirements
- Unit test coverage for all loading scenarios
- Integration tests with real checkpoint files
- Error condition testing for all failure modes
- Performance tests for loading large models
