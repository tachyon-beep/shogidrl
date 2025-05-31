# Parallel Experience Collection System

This module implements Task 4.2 of the Keisei Shogi training pipeline - a parallel experience collection system that uses multiple worker processes to run self-play games independently and collect experiences for the main training process.

## Overview

The parallel system enables efficient training by:
- Running multiple self-play games simultaneously in separate processes
- Collecting experiences from workers via multiprocessing queues
- Synchronizing model weights between main process and workers
- Managing worker lifecycle (start, stop, pause, reset)

## Architecture

### Core Components

1. **ParallelManager** (`parallel_manager.py`)
   - Main coordinator for managing multiple worker processes
   - Handles worker lifecycle and communication orchestration
   - Provides high-level interface for training loop integration

2. **SelfPlayWorker** (`self_play_worker.py`)
   - Individual worker process implementation
   - Runs self-play games using Shogi environment
   - Collects and batches experiences for transmission

3. **WorkerCommunicator** (`communication.py`)
   - Queue-based communication system between processes
   - Handles experience collection, model updates, and control commands
   - Provides timeout management and error handling

4. **ModelSynchronizer** (`model_sync.py`)
   - Efficient model weight synchronization
   - Optional compression for reduced transmission overhead
   - Tracks synchronization intervals and versioning

## Usage

### Basic Setup

```python
from keisei.training.parallel import ParallelManager
from keisei.core.neural_network import ActorCritic

# Configure the system
env_config = {'board_size': 9, 'max_moves': 200}
model_config = {'input_dim': 81*14, 'hidden_dim': 512, 'action_dim': 2187}
parallel_config = {
    'num_workers': 4,
    'games_per_worker': 10,
    'max_game_length': 200,
    'experience_batch_size': 64,
    'model_sync_interval': 100,
    'timeout': 30.0
}

# Create parallel manager
manager = ParallelManager(env_config, model_config, parallel_config)
```

### Training Loop Integration

```python
# Start worker processes
manager.start_workers()

try:
    for step in range(max_training_steps):
        # Collect experiences from workers
        experiences = manager.collect_experiences()
        
        if experiences:
            # Process experiences and update model
            loss = train_model(experiences)
            
            # Sync updated model to workers
            if step % model_sync_interval == 0:
                manager.sync_model_to_workers()
        
        # Optional: Reset workers periodically
        if step % worker_reset_interval == 0:
            manager.reset_workers()

finally:
    # Clean shutdown
    manager.stop_workers()
```

### Advanced Usage

```python
# Monitor system status
status = manager.get_status()
print(f"Active workers: {status['active_workers']}")
print(f"Queue sizes: {status['queue_info']}")

# Control specific workers
manager.pause_workers([0, 1])  # Pause workers 0 and 1
manager.resume_workers([0, 1])  # Resume workers 0 and 1

# Collect experiences with timeout
experiences = manager.collect_experiences(timeout=5.0)
```

## Configuration

### Parallel Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_workers` | int | 4 | Number of worker processes |
| `games_per_worker` | int | 10 | Games per worker before reset |
| `max_game_length` | int | 200 | Maximum moves per game |
| `experience_batch_size` | int | 64 | Experiences per batch |
| `model_sync_interval` | int | 100 | Steps between model syncs |
| `worker_reset_interval` | int | 1000 | Steps between worker resets |
| `timeout` | float | 30.0 | Queue operation timeout (seconds) |
| `compression_enabled` | bool | True | Enable model weight compression |

### Environment Configuration

```python
env_config = {
    'board_size': 9,           # Shogi board size
    'max_moves': 200,          # Maximum moves per game
    'reward_structure': {...}, # Custom reward configuration
}
```

### Model Configuration

```python
model_config = {
    'input_dim': 81 * 14,      # Board representation size
    'hidden_dim': 512,         # Hidden layer dimensions
    'action_dim': 2187,        # Number of possible actions
    'num_layers': 2,           # Network depth
}
```

## Performance Considerations

### Memory Usage
- Each worker process maintains its own model copy
- Experience buffers are limited by `experience_batch_size`
- Model synchronization uses shared memory when possible

### CPU Utilization
- Optimal `num_workers` typically equals CPU cores - 1
- Workers run independently without GIL contention
- Main process handles coordination and model training

### Queue Management
- Queues have configurable size limits to prevent memory overflow
- Timeout handling prevents deadlocks during communication
- Queue monitoring provides system health visibility

## Error Handling

The system includes comprehensive error handling for:
- Worker process crashes and automatic restart
- Queue communication timeouts
- Model synchronization failures
- Resource cleanup on shutdown

### Common Issues

1. **Worker Timeouts**: Increase `timeout` parameter if workers are slow
2. **Memory Issues**: Reduce `num_workers` or `experience_batch_size`
3. **Queue Overflow**: Increase queue sizes or collection frequency
4. **Model Sync Failures**: Check GPU memory and model size

## Testing

Run the test suite to verify system functionality:

```bash
cd /home/john/keisei
python -m pytest tests/test_parallel_system.py -v
```

Or test individual components:

```bash
python -m pytest tests/test_parallel_system.py::TestParallelSystem::test_worker_communicator_init -v
```

## Integration with Main Training

The parallel system is designed to integrate seamlessly with the existing training pipeline:

1. **Replace Sequential Self-Play**: Use `ParallelManager` instead of single-threaded game execution
2. **Experience Buffer Integration**: Collected experiences match existing `Experience` format
3. **Model Updates**: Compatible with existing `ActorCritic` model architecture
4. **Configuration**: Extends existing configuration system

## Example Scripts

- **Basic Usage**: `/home/john/keisei/examples/parallel_training_example.py`
- **Advanced Configuration**: See `parallel_config` in main training scripts
- **Performance Benchmarking**: Compare with sequential training baseline

## Monitoring and Debugging

### Logging
All components use Python's logging system with configurable levels:

```python
import logging
logging.getLogger('keisei.training.parallel').setLevel(logging.DEBUG)
```

### System Status
Monitor system health during training:

```python
status = manager.get_status()
queue_info = manager.communicator.get_queue_info()
```

### Profiling
Use Python profilers to identify bottlenecks:

```python
import cProfile
cProfile.run('manager.collect_experiences()')
```

## Future Enhancements

Potential improvements for the parallel system:
- Dynamic worker scaling based on load
- GPU-accelerated model inference in workers
- Distributed training across multiple machines
- Advanced experience prioritization
- Real-time performance metrics dashboard

## Dependencies

- `multiprocessing`: Core parallel processing
- `torch`: Model operations and serialization
- `numpy`: Efficient array operations
- `queue`: Thread-safe communication
- `keisei.shogi`: Shogi game environment
- `keisei.core`: Neural network and experience buffer
