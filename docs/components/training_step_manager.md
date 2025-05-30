# Training Step Manager Module

## Module Overview

The `training/step_manager.py` module provides the `StepManager` class, which handles the execution of individual training steps within the PPO training loop. This module manages the interaction between the agent and environment, executes moves, collects experiences, and handles episode transitions. It serves as a critical component that orchestrates the step-by-step progression of training episodes.

## Dependencies

### External Dependencies
- `torch`: PyTorch tensors and device management
- `numpy`: Array operations for observations and rewards
- `time`: Demo mode timing delays
- `typing`: Type hints and annotations

### Internal Dependencies
- `keisei.config_schema`: Configuration classes (`AppConfig`)
- `keisei.core.experience_buffer`: Experience storage (`ExperienceBuffer`)
- `keisei.core.ppo_agent`: Agent interface (`PPOAgent`)
- `keisei.shogi.shogi_game`: Game environment (`ShogiGame`)
- `keisei.shogi.shogi_game_io`: Move formatting utilities
- `keisei.training.compatibility_mixin`: Backward compatibility support

## Class Documentation

### StepManager

Main class responsible for executing individual training steps and managing episode state transitions.

**Attributes:**
- `config: AppConfig` - Training configuration settings
- `agent: PPOAgent` - The reinforcement learning agent
- `experience_buffer: ExperienceBuffer` - Buffer for storing training experiences
- `game: ShogiGame` - The Shogi game environment
- `device: torch.device` - PyTorch device for tensor operations
- `policy_mapper: Any` - Maps between action spaces (optional)

**Key Methods:**

#### `__init__(config, agent, experience_buffer, game, device, policy_mapper=None)`
Initializes the step manager with required components for step execution.

#### `execute_step(episode_state, global_timestep, logger_func) -> StepResult`
Executes a single training step, including:
- Agent action selection
- Environment step execution
- Experience collection
- State transitions
- Demo mode handling

#### `handle_episode_end(episode_state, step_result, game_stats, total_episodes_completed, logger_func) -> EpisodeState`
Manages episode completion, including:
- Game outcome logging
- Statistics updates
- Win rate calculations
- Game reset for next episode

#### `reset_episode() -> EpisodeState`
Resets the game environment and creates a new episode state.

#### `update_episode_state(episode_state, step_result) -> EpisodeState`
Updates episode state with results from the latest step.

## Data Structures

### EpisodeState

Represents the current state of a training episode.

```python
@dataclass
class EpisodeState:
    current_obs: np.ndarray           # Current game observation
    current_obs_tensor: torch.Tensor  # Tensor version of observation
    episode_reward: float             # Cumulative episode reward
    episode_length: int               # Number of steps in episode
```

### StepResult

Contains the results of executing a single step.

```python
@dataclass
class StepResult:
    action: int                       # Selected action index
    next_obs: np.ndarray             # Observation after action
    next_obs_tensor: torch.Tensor    # Tensor version of next observation
    reward: float                    # Step reward
    done: bool                       # Episode termination flag
    info: Dict[str, Any]             # Additional step information
    success: bool                    # Step execution success flag
```

## Inter-Module Relationships

### Core Integration
- **PPO Agent**: Interfaces with agent for action selection and value estimation
- **Experience Buffer**: Stores transitions for batch training
- **Neural Network**: Processes observations through the actor-critic network

### Training Integration
- **Training Loop Manager**: Called by the main training loop for step execution
- **Display Manager**: Provides episode statistics and progress updates
- **Metrics Manager**: Receives game outcome statistics

### Game Integration
- **Shogi Game**: Executes moves and manages game state
- **Game I/O**: Formats moves for demo mode display
- **Rules Logic**: Validates moves and determines game outcomes

## Implementation Notes

### Step Execution Flow
1. **Action Selection**: Agent selects action from current observation
2. **Environment Step**: Game executes the selected move
3. **Experience Storage**: Transition stored in experience buffer
4. **State Update**: Episode state updated with new information
5. **Terminal Handling**: Episode end logic for game completion

### Demo Mode Support
- Configurable step delays for human observation
- Enhanced move formatting with piece information
- Player name display and move descriptions
- Optional logging to terminal without W&B

### Error Handling
- Graceful handling of invalid actions
- Automatic episode reset on step failures
- Comprehensive logging of error conditions
- Fallback states for recovery scenarios

## Testing Strategy

### Unit Tests
- Test individual step execution with mock environments
- Validate episode state transitions and updates
- Test error handling and recovery mechanisms
- Verify experience buffer integration

### Integration Tests
- Test step manager with real Shogi game environment
- Validate agent interaction and action selection
- Test episode completion and reset cycles
- Verify statistics collection accuracy

### Stress Tests
- Test long episode handling and memory usage
- Validate performance under high step rates
- Test error recovery under adverse conditions

## Performance Considerations

### Optimization Areas
- **Tensor Operations**: Efficient device management and tensor conversions
- **Memory Usage**: Minimal copying of observations and state data
- **Step Latency**: Fast action selection and environment stepping
- **Buffer Management**: Efficient experience storage and retrieval

### Monitoring Metrics
- Steps per second (SPS) calculation
- Episode length distributions
- Memory usage during long training runs
- Action selection latency

## Security Considerations

### Input Validation
- Validate agent actions before environment execution
- Sanitize episode state data
- Verify tensor shapes and types

### Resource Management
- Monitor memory usage during experience collection
- Prevent unbounded episode length growth
- Manage device memory allocation

## Error Handling

### Exception Categories
- **Step Execution Errors**: Invalid actions, environment failures
- **Episode Transition Errors**: Reset failures, state corruption
- **Resource Errors**: Memory exhaustion, device unavailability
- **Configuration Errors**: Invalid parameters, missing components

### Recovery Strategies
- Automatic episode reset on step failures
- Fallback to CPU device if GPU unavailable
- Graceful degradation in demo mode
- Comprehensive error logging and reporting

## Configuration

### Key Settings
```python
# Demo mode configuration
demo_mode_delay: float = 0.0          # Delay between steps in demo mode

# Training parameters  
render_every_steps: int = 100         # Display update frequency
total_timesteps: int = 1000000        # Maximum training steps
steps_per_epoch: int = 2048           # Steps per training epoch
```

## Future Enhancements

### Planned Features
- **Multi-Environment Support**: Parallel environment execution
- **Advanced Metrics**: More detailed episode statistics
- **Custom Rewards**: Configurable reward shaping
- **Step Profiling**: Detailed performance analysis

### Extensibility Points
- **Custom Step Hooks**: Pre/post step callbacks
- **Alternative Experience Storage**: Different buffer implementations
- **Enhanced Demo Modes**: Interactive step-through capabilities
- **Adaptive Parameters**: Dynamic configuration updates

## Usage Examples

### Basic Step Execution
```python
step_manager = StepManager(
    config=config,
    agent=ppo_agent,
    experience_buffer=buffer,
    game=shogi_game,
    device=torch.device("cuda")
)

# Execute training step
step_result = step_manager.execute_step(
    episode_state=current_state,
    global_timestep=timestep,
    logger_func=log_function
)

# Handle episode completion
if step_result.done:
    new_state = step_manager.handle_episode_end(
        episode_state=updated_state,
        step_result=step_result,
        game_stats=stats,
        total_episodes_completed=episode_count,
        logger_func=log_function
    )
```

### Demo Mode Setup
```python
# Configure demo mode in config
config.demo.demo_mode_delay = 1.0  # 1 second delay between moves

# Step manager automatically handles demo mode display
step_result = step_manager.execute_step(
    episode_state=state,
    global_timestep=timestep,
    logger_func=demo_logger
)
```

## Maintenance Notes

### Code Quality
- Follow typing annotations for all public interfaces
- Maintain comprehensive error handling
- Document complex state transition logic
- Regular profiling for performance optimization

### Dependencies
- Keep PyTorch version compatibility
- Monitor NumPy array operations efficiency
- Update type hints with Python version upgrades
- Maintain backward compatibility with configuration changes

### Testing Requirements
- Unit test coverage for all public methods
- Integration tests with real game environments
- Performance regression tests
- Error condition validation tests