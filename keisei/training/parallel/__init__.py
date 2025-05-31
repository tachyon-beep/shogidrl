"""
Parallel training system for Keisei Shogi.

This package provides utilities for parallel experience collection using
multiprocessing workers that run self-play games independently.

Main Components:
- ParallelManager: Coordinates multiple worker processes
- SelfPlayWorker: Individual worker process for self-play games
- WorkerCommunicator: Queue-based communication between processes
- ModelSynchronizer: Efficient model weight synchronization

Example Usage:
    from keisei.training.parallel import ParallelManager
    from keisei.core.neural_network import ActorCritic

    model = ActorCritic(input_dim, hidden_dim, action_dim)
    config = {'num_workers': 4, 'games_per_worker': 10, ...}

    manager = ParallelManager(model, config)
    manager.start_workers()

    # Training loop
    experiences = manager.collect_experiences()
    # ... process experiences and update model ...
    manager.sync_model_to_workers()

    manager.shutdown()
"""

from .communication import WorkerCommunicator
from .model_sync import ModelSynchronizer
from .parallel_manager import ParallelManager
from .self_play_worker import SelfPlayWorker

__all__ = [
    "ParallelManager",
    "SelfPlayWorker",
    "ModelSynchronizer",
    "WorkerCommunicator",
]

__version__ = "1.0.0"
