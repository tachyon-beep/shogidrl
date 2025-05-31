# Training Loop Manager Module

## Module Overview

The `training/training_loop_manager.py` module provides the `TrainingLoopManager` class, which manages the primary iteration logic of the PPO training loop. This module was extracted from the main `Trainer` class to improve modularity and separation of concerns. It orchestrates the collection of experiences across multiple episodes, handles epoch management, and coordinates with various training components to execute the core training loop.

## Dependencies

### External Dependencies
- `time`: Timing calculations for steps-per-second (SPS) metrics
- `typing`: Type hints and forward references

### Internal Dependencies
- `keisei.config_schema`: Configuration classes (`AppConfig`)
- `keisei.core.experience_buffer`: Experience storage (`ExperienceBuffer`)
- `keisei.core.ppo_agent`: Agent interface (`PPOAgent`)
- `keisei.training.callbacks`: Callback interface (`Callback`)
- `keisei.training.display`: Training display (`TrainingDisplay`)
- `keisei.training.step_manager`: Step execution (`EpisodeState`, `StepManager`)
- `keisei.training.trainer`: Main trainer class (forward reference)

## Class Documentation

### TrainingLoopManager

Main class responsible for executing the primary training loop iteration logic.

**Attributes:**
- `trainer: Trainer` - Reference to the main trainer instance
- `config: AppConfig` - Training configuration (convenience access)
- `agent: PPOAgent` - The PPO agent (convenience access)
- `buffer: ExperienceBuffer` - Experience buffer (convenience access)
- `step_manager: StepManager` - Step execution manager (convenience access)
- `display: TrainingDisplay` - Display manager (convenience access)
- `callbacks: List[Callback]` - Training callbacks (convenience access)
- `current_epoch: int` - Current training epoch number
- `episode_state: Optional[EpisodeState]` - Current episode state
- `last_time_for_sps: float` - Timestamp for SPS calculation
- `steps_since_last_time_for_sps: int` - Step counter for SPS
- `last_display_update_time: float` - Last display update timestamp

**Key Methods:**

#### `__init__(trainer)`
Initializes the training loop manager with references to trainer components.

#### `set_initial_episode_state(initial_episode_state)`
Sets the initial episode state before starting the training loop.

#### `run()`
Executes the main training loop with the following structure:
- Epoch-based training iteration
- Experience collection until buffer full
- PPO updates between epochs
- Callback execution
- Progress tracking and display updates

#### `_run_epoch(log_both)`
Runs a single epoch, collecting experiences until the configured steps per epoch or total timesteps are reached.

## Data Structures

### Training Loop State

The training loop maintains several pieces of state:

```python
# Progress tracking
current_epoch: int                    # Current epoch number
episode_state: Optional[EpisodeState] # Active episode state

# Performance metrics
last_time_for_sps: float             # SPS calculation timestamp
steps_since_last_time_for_sps: int   # Steps for SPS calculation
last_display_update_time: float      # Display throttling timestamp
```

### Epoch Statistics

Statistics tracked during epoch execution:

```python
# Per-epoch metrics
num_steps_collected_this_epoch: int   # Steps collected in current epoch
black_wins_cum: int                   # Cumulative black wins
white_wins_cum: int                   # Cumulative white wins  
draws_cum: int                        # Cumulative draws
win_rates: Dict[str, float]           # Win rate percentages
```

## Inter-Module Relationships

### Core Training Integration
- **Trainer**: Main coordinator that delegates loop execution
- **Step Manager**: Executes individual training steps
- **PPO Agent**: Performs policy updates between epochs
- **Experience Buffer**: Accumulates training data during epochs

### Support Component Integration
- **Display Manager**: Updates progress visualization
- **Metrics Manager**: Tracks and updates training statistics
- **Callback Manager**: Executes registered training callbacks
- **Session Manager**: Manages training session state

### Configuration Integration
- **Training Config**: Steps per epoch, total timesteps, display intervals
- **Demo Config**: Demo mode settings for step delays
- **Logging Config**: W&B integration and log levels

## Implementation Notes

### Training Loop Architecture
1. **Epoch Loop**: Outer loop that manages training epochs
2. **Step Collection**: Inner loop that collects experiences
3. **PPO Updates**: Policy updates performed between epochs
4. **Callback Execution**: Registered callbacks called at appropriate points
5. **Progress Tracking**: Continuous monitoring of training progress

### Performance Optimization
- **Display Throttling**: Updates limited by configurable intervals
- **SPS Calculation**: Efficient steps-per-second metrics
- **Memory Management**: Minimal state copying and allocation
- **Early Termination**: Stops when target timesteps reached

### Error Handling Strategy
- **Graceful Degradation**: Continue training despite minor errors
- **Episode Recovery**: Reset episodes on step failures
- **State Validation**: Check for None states and handle appropriately
- **Exception Propagation**: Critical errors bubble up to trainer

## Testing Strategy

### Unit Tests
- Test epoch execution logic with mock components
- Validate progress tracking and SPS calculations
- Test error handling and recovery mechanisms
- Verify callback integration and timing

### Integration Tests
- Test complete training loops with real components
- Validate episode transitions and state management
- Test display updates and progress reporting
- Verify PPO update integration

### Performance Tests
- Measure training loop overhead and efficiency
- Test display update throttling effectiveness
- Validate memory usage during long training runs
- Benchmark steps-per-second calculations

## Performance Considerations

### Optimization Areas
- **Loop Overhead**: Minimize per-step computation
- **Display Updates**: Throttle expensive UI operations
- **Memory Allocation**: Reuse objects where possible
- **Metrics Collection**: Efficient statistic aggregation

### Monitoring Metrics
- Steps per second (SPS) during training
- Epoch completion times
- Display update frequencies
- Memory usage patterns

### Scalability Factors
- Episode length distributions
- Buffer size requirements
- Display update intervals
- Callback execution overhead

## Security Considerations

### Input Validation
- Validate episode state integrity
- Check configuration parameter bounds
- Verify trainer component availability

### Resource Management
- Monitor memory usage during long training runs
- Prevent infinite loops in epoch collection
- Manage display update resource consumption

### Error Isolation
- Contain step-level errors within epochs
- Prevent callback failures from stopping training
- Isolate display errors from core training logic

## Error Handling

### Exception Categories
- **Configuration Errors**: Missing or invalid settings
- **Component Errors**: Unavailable trainer components
- **Step Execution Errors**: Failed step collection
- **Update Errors**: PPO update failures
- **Display Errors**: UI update problems

### Recovery Strategies
- **Episode Reset**: Restart episodes on step failures
- **Component Fallbacks**: Continue with reduced functionality
- **Graceful Shutdown**: Clean termination on critical errors
- **State Recovery**: Restore valid states after errors

## Configuration

### Key Settings
```python
# Training loop parameters
total_timesteps: int = 1000000        # Maximum training steps
steps_per_epoch: int = 2048           # Experience collection per epoch
render_every_steps: int = 100         # Display update frequency

# Performance settings
rich_display_update_interval_seconds: float = 0.2  # Display throttling
demo_mode_delay: float = 0.0          # Step delay in demo mode

# PPO update settings
learning_rate: float = 3e-4           # Optimizer learning rate
gamma: float = 0.99                   # Discount factor
```

## Future Enhancements

### Planned Features
- **Adaptive Epochs**: Dynamic epoch sizing based on performance
- **Parallel Collection**: Multi-environment experience collection
- **Advanced Metrics**: More detailed training analytics
- **Checkpoint Recovery**: Resume training from intermediate states

### Extensibility Points
- **Custom Loop Logic**: Pluggable epoch and step collection strategies
- **Enhanced Callbacks**: More granular callback timing options
- **Flexible Scheduling**: Configurable update frequencies
- **Alternative Displays**: Different visualization backends

## Usage Examples

### Basic Training Loop
```python
# Initialize training loop manager
loop_manager = TrainingLoopManager(trainer)

# Set initial episode state
initial_state = step_manager.reset_episode()
loop_manager.set_initial_episode_state(initial_state)

# Execute training loop
try:
    loop_manager.run()
except KeyboardInterrupt:
    print("Training interrupted by user")
```

### Custom Epoch Handling
```python
# Access epoch information during training
def custom_callback(trainer):
    loop_manager = trainer.training_loop_manager
    print(f"Completed epoch {loop_manager.current_epoch}")
    
    # Access performance metrics
    current_sps = loop_manager.steps_since_last_time_for_sps / (
        time.time() - loop_manager.last_time_for_sps
    )
    print(f"Current SPS: {current_sps:.2f}")
```

### Progress Monitoring
```python
# Monitor training progress
while trainer.global_timestep < config.training.total_timesteps:
    # Training loop manager handles the actual execution
    # Progress can be monitored through trainer state
    progress = trainer.global_timestep / config.training.total_timesteps
    print(f"Training progress: {progress:.1%}")
```

## Maintenance Notes

### Code Quality
- Maintain clear separation between loop logic and component management
- Document complex epoch and step collection interactions
- Follow consistent error handling patterns
- Regular profiling for performance optimization

### Dependencies
- Monitor trainer component interface stability
- Keep callback interface backward compatible
- Maintain configuration schema consistency
- Update type hints with Python version changes

### Testing Requirements
- Comprehensive unit test coverage for all loop logic
- Integration tests with real training components
- Performance regression tests for SPS metrics
- Error handling validation across all failure modes