# Performance Profiling Workflow

This document describes how to use the performance profiling tools provided in the Keisei Shogi DRL project for development and optimization.

## Overview

The profiling utilities in `keisei/utils/profiling.py` provide a comprehensive set of tools for monitoring and analyzing performance during development. These tools are designed to help identify bottlenecks and optimize critical paths in the training and game execution pipelines.

## Quick Start

### Basic Usage

```python
from keisei.utils.profiling import perf_monitor, profile_function, profile_code_block

# Time a code block
with perf_monitor.time_operation("model_inference"):
    prediction = model(input_data)

# Use as a decorator
@profile_function
def expensive_computation():
    # Your code here
    pass

# Profile a specific section
with profile_code_block("data_preprocessing"):
    processed_data = preprocess(raw_data)
```

### Training Integration

```python
from keisei.utils.profiling import profile_training_step, perf_monitor

@profile_training_step
def training_step(batch):
    # Your training step logic
    loss = compute_loss(batch)
    optimizer.step()
    return loss

# After training, get statistics
perf_monitor.print_summary()
```

## Available Tools

### 1. PerformanceMonitor Class

The main class for collecting and analyzing performance metrics.

#### Key Methods:
- `time_operation(name)`: Context manager for timing operations
- `increment_counter(name, value=1)`: Increment named counters
- `get_stats()`: Return performance statistics dictionary
- `print_summary()`: Display formatted performance summary
- `reset()`: Clear all collected data

#### Example:
```python
from keisei.utils.profiling import perf_monitor

# Time multiple operations
with perf_monitor.time_operation("move_generation"):
    moves = generate_legal_moves()

with perf_monitor.time_operation("position_evaluation"):
    score = evaluate_position()

# Track custom metrics
perf_monitor.increment_counter("positions_evaluated")
perf_monitor.increment_counter("cache_hits", 5)

# View results
stats = perf_monitor.get_stats()
print(f"Average move generation time: {stats['move_generation_avg']:.4f}s")

# Or print full summary
perf_monitor.print_summary()
```

### 2. Function Decorators

#### `@profile_function`
Automatically times function execution:

```python
@profile_function
def neural_network_forward_pass(input_tensor):
    return model(input_tensor)
```

#### `@profile_training_step`
Specialized decorator for training steps:

```python
@profile_training_step
def train_one_batch(batch_data):
    loss = compute_loss(batch_data)
    optimizer.step()
    return loss
```

#### `@profile_game_operation(operation_name)`
For game-specific operations with custom naming:

```python
@profile_game_operation("mcts_simulation")
def run_simulation(game_state):
    return mcts.simulate(game_state)
```

### 3. Code Profiling

#### `run_profiler(func, *args, **kwargs)`
Run detailed cProfile analysis:

```python
from keisei.utils.profiling import run_profiler

def complex_function():
    # Complex computation
    return result

result, profile_report = run_profiler(complex_function)
print(profile_report)  # Detailed line-by-line profiling
```

### 4. Memory Monitoring

```python
from keisei.utils.profiling import memory_usage_mb

initial_memory = memory_usage_mb()
# ... run some operations ...
final_memory = memory_usage_mb()
print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
```

## Integration Points

### 1. Training Pipeline

Add profiling to your training loop:

```python
from keisei.utils.profiling import perf_monitor, profile_training_step

@profile_training_step
def training_step(batch):
    with perf_monitor.time_operation("forward_pass"):
        predictions = model(batch.inputs)
    
    with perf_monitor.time_operation("loss_computation"):
        loss = criterion(predictions, batch.targets)
    
    with perf_monitor.time_operation("backward_pass"):
        loss.backward()
        optimizer.step()
    
    return loss.item()

# In your training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = training_step(batch)
    
    # Print statistics every epoch
    if epoch % 10 == 0:
        perf_monitor.print_summary()
        perf_monitor.reset()  # Reset for next measurement period
```

### 2. Game Operations

Profile critical game operations:

```python
from keisei.utils.profiling import profile_game_operation, perf_monitor

class ShogiGame:
    @profile_game_operation("move_generation")
    def get_legal_moves(self):
        # Move generation logic
        return moves
    
    @profile_game_operation("position_evaluation")
    def evaluate_position(self):
        # Position evaluation logic
        return score
    
    @profile_game_operation("move_execution")
    def make_move(self, move):
        # Move execution logic
        pass
```

### 3. MCTS Integration

```python
from keisei.utils.profiling import profile_code_block, perf_monitor

class MCTSNode:
    def simulate(self):
        with profile_code_block("mcts_simulation"):
            # Simulation logic
            pass
    
    def expand(self):
        with profile_code_block("mcts_expansion"):
            # Node expansion logic
            pass
    
    def select(self):
        with profile_code_block("mcts_selection"):
            # Selection logic
            pass
```

## Performance Analysis Workflow

### 1. Development Phase

1. **Add Basic Profiling**: Start with `@profile_function` decorators on key functions
2. **Identify Hotspots**: Use `perf_monitor.print_summary()` to see which operations take the most time
3. **Deep Dive**: Use `run_profiler()` on specific functions that show up as bottlenecks

### 2. Optimization Phase

1. **Baseline Measurement**: Record current performance with full profiling enabled
2. **Make Changes**: Implement optimizations
3. **Compare Results**: Use the same profiling setup to measure improvements
4. **Iterate**: Repeat until performance targets are met

### 3. Production Monitoring

For production or final testing:
- Use lighter profiling (avoid `run_profiler()` in production)
- Focus on high-level metrics (training steps per second, inference time)
- Monitor memory usage trends

## Example: Complete Training Session

```python
from keisei.utils.profiling import perf_monitor, profile_training_step, memory_usage_mb

# Setup
initial_memory = memory_usage_mb()

@profile_training_step
def train_batch(batch):
    # Your training logic here
    pass

# Training loop
print("Starting training with profiling...")
for epoch in range(100):
    for batch_idx, batch in enumerate(dataloader):
        loss = train_batch(batch)
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            current_memory = memory_usage_mb()
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, Memory: {current_memory:.1f}MB")
    
    # Print detailed stats every 10 epochs
    if epoch % 10 == 0:
        print(f"\n--- Epoch {epoch} Performance Summary ---")
        perf_monitor.print_summary()
        print(f"Memory usage: {memory_usage_mb() - initial_memory:.1f}MB increase")
        perf_monitor.reset()

print("Training completed. Final performance summary:")
perf_monitor.print_summary()
```

## Best Practices

### 1. Granular Profiling
- Profile at multiple levels: high-level operations and detailed sub-operations
- Use descriptive names for profiled operations
- Group related operations with consistent naming

### 2. Conditional Profiling
```python
import os

# Enable profiling only when needed
PROFILING_ENABLED = os.getenv('ENABLE_PROFILING', 'false').lower() == 'true'

if PROFILING_ENABLED:
    @profile_function
    def my_function():
        pass
else:
    def my_function():
        pass
```

### 3. Regular Monitoring
- Reset statistics periodically to avoid memory buildup
- Archive performance data for trend analysis
- Set up automated performance regression tests

### 4. Memory Considerations
- Profiling adds overhead - disable in production
- The `run_profiler()` function can consume significant memory for long operations
- Monitor memory usage when profiling memory-intensive operations

## Troubleshooting

### Common Issues

1. **High Profiling Overhead**: If profiling significantly slows down execution, reduce granularity or use sampling
2. **Memory Growth**: Reset `perf_monitor` periodically or use more targeted profiling
3. **Missing Statistics**: Ensure operations are actually being called and timing contexts are properly closed

### Performance Tips

1. **Use Context Managers**: Prefer `with perf_monitor.time_operation()` over manual timing
2. **Batch Analysis**: Collect data over multiple runs before analyzing
3. **Focus on Hotspots**: Profile the 20% of code that takes 80% of the time

## Integration with External Tools

The profiling utilities can be combined with external tools:

- **Weights & Biases**: Log performance metrics to W&B dashboards
- **TensorBoard**: Include timing data in TensorBoard logs
- **pytest**: Use in performance regression tests

Example W&B integration:
```python
import wandb
from keisei.utils.profiling import perf_monitor

# After training step
stats = perf_monitor.get_stats()
wandb.log({
    "training_step_avg_time": stats.get("training_step_avg", 0),
    "forward_pass_avg_time": stats.get("forward_pass_avg", 0),
    "memory_usage_mb": memory_usage_mb()
})
```

This profiling system provides comprehensive performance monitoring capabilities while being lightweight enough for development use and detailed enough for serious optimization work.
