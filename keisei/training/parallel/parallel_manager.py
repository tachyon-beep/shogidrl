"""
Parallel training manager for coordinating multiple self-play workers.

This module manages the parallel experience collection system, including
worker processes, communication, and model synchronization.
"""

import logging
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.training.parallel.communication import WorkerCommunicator
from keisei.training.parallel.model_sync import ModelSynchronizer
from keisei.training.parallel.self_play_worker import SelfPlayWorker

logger = logging.getLogger(__name__)


class ParallelManager:
    """
    Manages parallel experience collection using multiple worker processes.

    Coordinates worker processes, handles communication, and synchronizes
    model weights between the main training process and workers.
    """

    def __init__(
        self,
        env_config: Dict,
        model_config: Dict,
        parallel_config: Dict,
        device: str = "cuda",
    ):
        """
        Initialize parallel training manager.

        Args:
            env_config: Environment configuration
            model_config: Model configuration
            parallel_config: Parallel training configuration
            device: Device for main process (typically GPU)
        """
        self.env_config = env_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device = torch.device(device)

        # Parallel configuration
        self.num_workers = parallel_config["num_workers"]
        self.batch_size = parallel_config["batch_size"]
        self.enabled = parallel_config["enabled"]

        # Initialize components
        self.communicator = WorkerCommunicator(
            num_workers=self.num_workers,
            max_queue_size=parallel_config["max_queue_size"],
            timeout=parallel_config["timeout_seconds"],
        )

        self.model_sync = ModelSynchronizer(
            sync_interval=parallel_config["sync_interval"],
            compression_enabled=parallel_config["compression_enabled"],
        )

        # Worker processes
        self.workers: List[SelfPlayWorker] = []
        self.worker_stats: Dict[int, Dict] = {}

        # State tracking
        self.total_steps_collected = 0
        self.total_batches_received = 0
        self.last_sync_time = time.time()
        self.is_running = False

        logger.info("ParallelManager initialized with %d workers", self.num_workers)

    def start_workers(self, initial_model: nn.Module) -> bool:
        """
        Start all worker processes.

        Args:
            initial_model: Initial model to distribute to workers

        Returns:
            True if all workers started successfully
        """
        if not self.enabled:
            logger.info("Parallel collection disabled, skipping worker startup")
            return True

        try:
            # Create and start worker processes
            for worker_id in range(self.num_workers):
                worker = SelfPlayWorker(
                    worker_id=worker_id,
                    env_config=self.env_config,
                    model_config=self.model_config,
                    parallel_config=self.parallel_config,
                    experience_queue=self.communicator.experience_queues[worker_id],
                    model_queue=self.communicator.model_queues[worker_id],
                    control_queue=self.communicator.control_queues[worker_id],
                    seed_offset=self.parallel_config["worker_seed_offset"],
                )

                worker.start()
                self.workers.append(worker)

                logger.info("Started worker %d (PID: %d)", worker_id, worker.pid)

            # Send initial model to all workers
            self._sync_model_to_workers(initial_model)

            self.is_running = True
            logger.info("All %d workers started successfully", self.num_workers)
            return True

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Failed to start workers: %s", str(e))
            self.stop_workers()
            return False

    def collect_experiences(self, experience_buffer: ExperienceBuffer) -> int:
        """
        Collect experiences from all workers and add to main buffer.

        Args:
            experience_buffer: Main experience buffer to fill

        Returns:
            Number of experiences collected
        """
        if not self.enabled or not self.is_running:
            return 0

        experiences_collected = 0

        try:
            # Collect from all workers
            worker_batches = self.communicator.collect_experiences()

            for worker_id, batch_data in worker_batches:
                # batch_data is a dictionary with experiences and metadata
                worker_experiences = batch_data["experiences"]
                experience_buffer.add_from_worker_batch(worker_experiences)

                batch_size = batch_data.get("batch_size", 0)
                experiences_collected += batch_size
                self.total_batches_received += 1

                # Update worker stats
                self.worker_stats[worker_id] = {
                    "steps_collected": batch_data.get("steps_collected", 0),
                    "games_played": batch_data.get("games_played", 0),
                    "last_batch_time": batch_data.get("timestamp", time.time()),
                    "last_batch_size": batch_size,
                }

            self.total_steps_collected += experiences_collected

            if experiences_collected > 0:
                logger.debug(
                    "Collected %d experiences from workers", experiences_collected
                )

            return experiences_collected

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to collect experiences: %s", str(e))
            return 0

    def sync_model_if_needed(self, model: nn.Module, current_step: int) -> bool:
        """
        Synchronize model with workers if needed.

        Args:
            model: Current model to synchronize
            current_step: Current training step

        Returns:
            True if synchronization occurred
        """
        if not self.enabled or not self.is_running:
            return False

        if self.model_sync.should_sync(current_step):
            return self._sync_model_to_workers(model, current_step)

        return False

    def _sync_model_to_workers(self, model: nn.Module, current_step: int = 0) -> bool:
        """
        Synchronize model weights to all workers.

        Args:
            model: Model to synchronize
            current_step: Current training step

        Returns:
            True if synchronization succeeded
        """
        try:
            # Send model weights to workers
            self.communicator.send_model_weights(
                model.state_dict(),
                compression_enabled=self.parallel_config["compression_enabled"],
            )

            # Mark sync completed
            self.model_sync.mark_sync_completed(current_step)
            self.last_sync_time = time.time()

            logger.debug("Model synchronized to workers at step %d", current_step)
            return True

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Model synchronization failed: %s", str(e))
            return False

    def stop_workers(self) -> None:
        """Stop all worker processes gracefully."""
        if not self.workers:
            return

        logger.info("Stopping parallel workers...")

        try:
            # Send stop command to all workers
            self.communicator.send_control_command("stop")

            # Wait for workers to finish
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)  # 5 second timeout

                    if worker.is_alive():
                        logger.warning("Force terminating worker %d", worker.pid)
                        worker.terminate()
                        worker.join(timeout=2.0)

            # Clean up communication
            self.communicator.cleanup()

            self.workers.clear()
            self.is_running = False

            logger.info("All workers stopped")

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error stopping workers: %s", str(e))

    def get_parallel_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive parallel training statistics.

        Returns:
            Dictionary with parallel training stats
        """
        queue_info = self.communicator.get_queue_info() if self.is_running else {}
        sync_stats = self.model_sync.get_sync_stats()

        return {
            "enabled": self.enabled,
            "running": self.is_running,
            "num_workers": self.num_workers,
            "total_steps_collected": self.total_steps_collected,
            "total_batches_received": self.total_batches_received,
            "last_sync_time": self.last_sync_time,
            "worker_stats": self.worker_stats.copy(),
            "queue_info": queue_info,
            "sync_stats": sync_stats,
            "collection_rate": self._calculate_collection_rate(),
        }

    def _calculate_collection_rate(self) -> float:
        """Calculate current experience collection rate."""
        if not self.worker_stats:
            return 0.0

        # Simple rate calculation based on recent activity
        current_time = time.time()
        recent_steps = 0

        for stats in self.worker_stats.values():
            last_batch_time = stats.get("last_batch_time", 0)
            if current_time - last_batch_time < 60:  # Last minute
                recent_steps += stats.get("last_batch_size", 0)

        return recent_steps / 60.0  # Steps per second

    def reset_workers(self) -> None:
        """Reset all worker environments."""
        if self.is_running:
            self.communicator.send_control_command("reset")
            logger.info("Reset command sent to all workers")

    def pause_workers(self, duration: float = 1.0) -> None:
        """
        Pause all workers for specified duration.

        Args:
            duration: Pause duration in seconds
        """
        if self.is_running:
            self.communicator.send_control_command("pause", data={"duration": duration})
            logger.info("Pause command sent to all workers (duration=%fs)", duration)

    def is_healthy(self) -> bool:
        """
        Check if parallel system is healthy.

        Returns:
            True if system appears to be functioning normally
        """
        if not self.enabled:
            return True  # Disabled system is considered healthy

        if not self.is_running:
            return False

        # Check if workers are alive
        alive_workers = sum(1 for worker in self.workers if worker.is_alive())
        if alive_workers < self.num_workers:
            logger.warning("Only %d/%d workers alive", alive_workers, self.num_workers)
            return False

        # Check recent activity
        current_time = time.time()
        recent_activity = any(
            current_time - stats.get("last_batch_time", 0) < 30
            for stats in self.worker_stats.values()
        )

        return recent_activity

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.stop_workers()
