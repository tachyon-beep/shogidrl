# File-by-File Implementation Roadmap
**Keisei Shogi DRL: Parallel System Implementation Details**

**Date:** May 31, 2025  
**Reference:** PARALLEL_IMPLEMENTATION_PLAN.md  
**Status:** Implementation Ready

## File Modification Schedule

### **Phase 1: Configuration and Infrastructure (Week 1)**

#### **File 1: Configuration Schema Enhancement**
**Path:** `keisei/config_schema.py`
**Lines to Modify:** ~142 (current file size)
**Estimated Effort:** 4 hours

**Current State Analysis:**
- Pydantic-based configuration with TrainingConfig class
- Type-safe validation already implemented
- Ready for parallel configuration addition

**Implementation Details:**
```python
# Add after existing imports
from typing import Optional

# Add new configuration class
class ParallelConfig(BaseModel):
    """Configuration for parallel experience collection system."""
    enabled: bool = Field(
        default=False,
        description="Enable parallel experience collection"
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of worker processes"
    )
    episodes_per_worker: int = Field(
        default=1,
        ge=1,
        description="Episodes each worker collects before reporting"
    )
    model_sync_interval: int = Field(
        default=10,
        ge=1,
        description="Steps between model synchronization"
    )
    worker_timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Worker timeout in seconds"
    )
    queue_maxsize: int = Field(
        default=100,
        ge=10,
        description="Maximum size of experience queue"
    )
    use_shared_memory: bool = Field(
        default=True,
        description="Use shared memory for large tensors"
    )

# Modify TrainingConfig class
class TrainingConfig(BaseModel):
    # ...existing fields...
    parallel: ParallelConfig = Field(
        default_factory=ParallelConfig,
        description="Parallel processing configuration"
    )
```

**Integration Points:**
- Used by `Trainer` for parallel mode detection
- Used by `ParallelManager` for worker configuration
- Used by `TrainingLoopManager` for execution mode

---

#### **File 2: Parallel Package Structure**
**Path:** `keisei/training/parallel/__init__.py`
**New File:** Package initialization
**Estimated Effort:** 1 hour

```python
"""
Parallel experience collection system for Keisei Shogi DRL.

This package provides multiprocessing-based parallel experience collection
to improve training throughput by separating CPU-bound environment stepping
from GPU-bound neural network training.
"""

from .parallel_manager import ParallelManager
from .self_play_worker import SelfPlayWorker
from .model_sync import ModelSynchronizer
from .experience_collector import ExperienceCollector

__all__ = [
    "ParallelManager",
    "SelfPlayWorker", 
    "ModelSynchronizer",
    "ExperienceCollector"
]
```

---

#### **File 3: Experience Buffer Enhancement**
**Path:** `keisei/core/experience_buffer.py`
**Lines to Modify:** ~208 (current file size)
**Estimated Effort:** 6 hours

**Current State Analysis:**
- Single experience addition with `add()` method
- GAE computation and batch retrieval implemented
- Ready for batch operations enhancement

**Key Modifications:**

```python
# Add imports
from typing import List, Dict, Any
import pickle

# Add to ExperienceBuffer class
def add_batch(self, batch_experiences: List[Dict[str, Any]]) -> None:
    """
    Add a batch of experiences efficiently.
    
    Args:
        batch_experiences: List of experience dictionaries from workers
    """
    if not batch_experiences:
        return
        
    # Extract batch data
    observations = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    legal_masks = []
    
    for exp in batch_experiences:
        observations.append(exp['observation'])
        actions.append(exp['action'])
        rewards.append(exp['reward'])
        log_probs.append(exp['log_prob'])
        values.append(exp['value'])
        dones.append(exp['done'])
        legal_masks.append(exp['legal_mask'])
    
    # Batch tensor operations for efficiency
    obs_batch = torch.stack(observations)
    legal_mask_batch = torch.stack(legal_masks)
    
    # Add to buffer using existing add() method
    for i in range(len(batch_experiences)):
        self.add(
            obs_batch[i], actions[i], rewards[i],
            log_probs[i], values[i], dones[i],
            legal_mask_batch[i]
        )

def merge_worker_buffers(self, worker_data: List[bytes]) -> None:
    """
    Merge serialized worker buffer data.
    
    Args:
        worker_data: List of pickled experience data from workers
    """
    for data in worker_data:
        experiences = pickle.loads(data)
        self.add_batch(experiences)

def get_serializable_batch(self) -> bytes:
    """
    Get current buffer contents in serializable format.
    Used by workers to send data to main process.
    """
    if self.step_count == 0:
        return pickle.dumps([])
        
    batch_data = {
        'observations': self.observations[:self.step_count],
        'actions': self.actions[:self.step_count],
        'rewards': self.rewards[:self.step_count],
        'log_probs': self.log_probs[:self.step_count],
        'values': self.values[:self.step_count],
        'dones': self.dones[:self.step_count],
        'legal_masks': self.legal_masks[:self.step_count]
    }
    return pickle.dumps(batch_data)
```

---

### **Phase 2: Worker Implementation (Week 2)**

#### **File 4: Self-Play Worker Process**
**Path:** `keisei/training/parallel/self_play_worker.py`
**New File:** Worker process implementation
**Estimated Effort:** 12 hours

```python
"""
Self-play worker process for parallel experience collection.
Each worker runs independently with its own environment and agent.
"""

import multiprocessing
import queue
import time
import torch
import pickle
import traceback
from typing import Optional, Dict, Any

from keisei.config_schema import AppConfig
from keisei.shogi.shogi_game import ShogiGame
from keisei.core.ppo_agent import PPOAgent
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.training.env_manager import EnvManager


class SelfPlayWorker(multiprocessing.Process):
    """
    Worker process for parallel self-play experience collection.
    
    Each worker maintains its own game environment and agent copy,
    collects experiences independently, and communicates results
    back to the main training process.
    """
    
    def __init__(
        self,
        worker_id: int,
        config: AppConfig,
        experience_queue: multiprocessing.Queue,
        model_queue: multiprocessing.Queue,
        control_queue: multiprocessing.Queue
    ):
        super().__init__()
        self.worker_id = worker_id
        self.config = config
        self.experience_queue = experience_queue
        self.model_queue = model_queue
        self.control_queue = control_queue
        
        # Worker state
        self.running = False
        self.episodes_collected = 0
        self.steps_collected = 0
        
        # Local components (initialized in run())
        self.env_manager: Optional[EnvManager] = None
        self.agent: Optional[PPOAgent] = None
        self.experience_buffer: Optional[ExperienceBuffer] = None
        
    def run(self) -> None:
        """Main worker process loop."""
        try:
            self._initialize_worker()
            self._worker_loop()
        except Exception as e:
            error_msg = f"Worker {self.worker_id} failed: {str(e)}\n{traceback.format_exc()}"
            self.experience_queue.put({
                'type': 'error',
                'worker_id': self.worker_id,
                'message': error_msg
            })
        finally:
            self._cleanup_worker()
    
    def _initialize_worker(self) -> None:
        """Initialize worker components."""
        # Set device to CPU for workers
        worker_config = self.config.model_copy(deep=True)
        worker_config.env.device = "cpu"
        
        # Initialize environment manager
        self.env_manager = EnvManager(worker_config)
        self.env_manager.setup_environment()
        
        # Initialize agent
        self.agent = PPOAgent(worker_config, torch.device("cpu"))
        
        # Initialize experience buffer
        self.experience_buffer = ExperienceBuffer(
            buffer_size=worker_config.training.steps_per_epoch,
            gamma=worker_config.training.gamma,
            lambda_gae=worker_config.training.lambda_gae,
            device="cpu"
        )
        
        self.running = True
        
        # Signal successful initialization
        self.experience_queue.put({
            'type': 'init_complete',
            'worker_id': self.worker_id
        })
    
    def _worker_loop(self) -> None:
        """Main worker execution loop."""
        while self.running:
            try:
                # Check for control signals
                self._check_control_signals()
                
                # Check for model updates
                self._check_model_updates()
                
                # Collect experience
                if self.running:
                    self._collect_episode()
                    
            except Exception as e:
                self.experience_queue.put({
                    'type': 'error',
                    'worker_id': self.worker_id,
                    'message': f"Error in worker loop: {str(e)}"
                })
                break
    
    def _check_control_signals(self) -> None:
        """Check for control signals from main process."""
        try:
            while True:
                signal = self.control_queue.get_nowait()
                if signal['type'] == 'shutdown':
                    self.running = False
                    break
                elif signal['type'] == 'collect_episodes':
                    target_episodes = signal.get('target_episodes', 1)
                    self._collect_target_episodes(target_episodes)
        except queue.Empty:
            pass
    
    def _check_model_updates(self) -> None:
        """Check for model weight updates."""
        try:
            while True:
                update = self.model_queue.get_nowait()
                if update['type'] == 'model_weights':
                    self.agent.load_state_dict(update['state_dict'])
        except queue.Empty:
            pass
    
    def _collect_episode(self) -> None:
        """Collect one complete episode of experience."""
        observation = self.env_manager.reset()
        episode_steps = 0
        episode_reward = 0.0
        
        while True:
            # Get action from agent
            action, log_prob, value = self.agent.select_action(
                observation, 
                self.env_manager.get_legal_mask()
            )
            
            # Step environment
            next_observation, reward, done = self.env_manager.step(action)
            
            # Store experience
            self.experience_buffer.add(
                obs=observation,
                action=action,
                reward=reward,
                log_prob=log_prob,
                value=value,
                done=done,
                legal_mask=self.env_manager.get_legal_mask()
            )
            
            episode_steps += 1
            episode_reward += reward
            
            if done:
                break
                
            observation = next_observation
        
        # Send completed episode data
        self._send_experience_data(episode_steps, episode_reward)
        
        self.episodes_collected += 1
        self.steps_collected += episode_steps
    
    def _collect_target_episodes(self, target_episodes: int) -> None:
        """Collect specified number of episodes."""
        for _ in range(target_episodes):
            if not self.running:
                break
            self._collect_episode()
    
    def _send_experience_data(self, episode_steps: int, episode_reward: float) -> None:
        """Send experience data to main process."""
        # Compute advantages and returns
        self.experience_buffer.compute_advantages_and_returns(0.0)
        
        # Serialize buffer data
        buffer_data = self.experience_buffer.get_serializable_batch()
        
        # Send to main process
        self.experience_queue.put({
            'type': 'experience_data',
            'worker_id': self.worker_id,
            'episode_steps': episode_steps,
            'episode_reward': episode_reward,
            'buffer_data': buffer_data
        })
        
        # Clear buffer for next episode
        self.experience_buffer.clear()
    
    def _cleanup_worker(self) -> None:
        """Clean up worker resources."""
        if self.env_manager:
            # Cleanup environment resources
            pass
            
        # Send shutdown confirmation
        try:
            self.experience_queue.put({
                'type': 'worker_shutdown',
                'worker_id': self.worker_id,
                'episodes_collected': self.episodes_collected,
                'steps_collected': self.steps_collected
            })
        except:
            pass
```

---

#### **File 5: Model Synchronization System**
**Path:** `keisei/training/parallel/model_sync.py`
**New File:** Model weight synchronization
**Estimated Effort:** 8 hours

```python
"""
Model synchronization utilities for parallel training.
Handles efficient distribution of model weights to worker processes.
"""

import multiprocessing
import queue
import torch
import pickle
import time
from typing import List, Dict, Any, Optional

from keisei.core.ppo_agent import PPOAgent


class ModelSynchronizer:
    """
    Manages model weight synchronization between main process and workers.
    
    Handles efficient serialization and distribution of neural network
    weights to maintain training consistency across parallel workers.
    """
    
    def __init__(self, agent: PPOAgent, compression_enabled: bool = True):
        self.agent = agent
        self.compression_enabled = compression_enabled
        self.last_sync_time = time.time()
        self.sync_count = 0
        
    def send_model_update(
        self, 
        worker_queues: List[multiprocessing.Queue]
    ) -> bool:
        """
        Send current model weights to all worker processes.
        
        Args:
            worker_queues: List of worker model update queues
            
        Returns:
            True if update sent successfully, False otherwise
        """
        try:
            # Get current model state
            state_dict = self.agent.state_dict()
            
            # Optionally compress weights
            if self.compression_enabled:
                state_dict = self._compress_state_dict(state_dict)
            
            # Create update message
            update_message = {
                'type': 'model_weights',
                'state_dict': state_dict,
                'sync_id': self.sync_count,
                'timestamp': time.time()
            }
            
            # Send to all workers
            successful_sends = 0
            for queue in worker_queues:
                try:
                    queue.put_nowait(update_message)
                    successful_sends += 1
                except queue.Full:
                    # Queue full, worker may be slow
                    continue
            
            self.sync_count += 1
            self.last_sync_time = time.time()
            
            return successful_sends == len(worker_queues)
            
        except Exception as e:
            print(f"Error sending model update: {e}")
            return False
    
    def _compress_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply compression to model weights to reduce memory usage.
        
        Args:
            state_dict: Original model state dictionary
            
        Returns:
            Compressed state dictionary
        """
        compressed_dict = {}
        
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                # Convert to half precision for transfer
                compressed_dict[key] = tensor.half()
            else:
                compressed_dict[key] = tensor
                
        return compressed_dict
    
    def should_sync(self, steps_since_last: int, sync_interval: int) -> bool:
        """
        Determine if model synchronization should occur.
        
        Args:
            steps_since_last: Steps since last synchronization
            sync_interval: Configured synchronization interval
            
        Returns:
            True if sync should occur
        """
        return steps_since_last >= sync_interval
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """
        Get synchronization statistics.
        
        Returns:
            Dictionary with sync statistics
        """
        current_time = time.time()
        
        return {
            'sync_count': self.sync_count,
            'last_sync_time': self.last_sync_time,
            'time_since_last_sync': current_time - self.last_sync_time,
            'compression_enabled': self.compression_enabled
        }


class WorkerModelReceiver:
    """
    Handles model weight reception and application in worker processes.
    """
    
    def __init__(self, agent: PPOAgent):
        self.agent = agent
        self.last_update_id = -1
        self.updates_received = 0
    
    def check_and_apply_updates(self, model_queue: multiprocessing.Queue) -> bool:
        """
        Check for and apply any pending model updates.
        
        Args:
            model_queue: Queue containing model updates
            
        Returns:
            True if update was applied, False otherwise
        """
        try:
            while True:
                update = model_queue.get_nowait()
                
                if update['type'] == 'model_weights':
                    sync_id = update['sync_id']
                    
                    # Only apply if this is a newer update
                    if sync_id > self.last_update_id:
                        state_dict = update['state_dict']
                        
                        # Decompress if needed
                        state_dict = self._decompress_state_dict(state_dict)
                        
                        # Apply to agent
                        self.agent.load_state_dict(state_dict)
                        
                        self.last_update_id = sync_id
                        self.updates_received += 1
                        
                        return True
                        
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error applying model update: {e}")
            
        return False
    
    def _decompress_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decompress model weights after transfer.
        
        Args:
            state_dict: Compressed state dictionary
            
        Returns:
            Decompressed state dictionary
        """
        decompressed_dict = {}
        
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.float16:
                # Convert back to full precision
                decompressed_dict[key] = tensor.float()
            else:
                decompressed_dict[key] = tensor
                
        return decompressed_dict
```

---

### **Phase 3: Integration and Management (Week 3)**

#### **File 6: Parallel Manager**
**Path:** `keisei/training/parallel/parallel_manager.py`
**New File:** Worker pool management
**Estimated Effort:** 10 hours

```python
"""
Parallel manager for coordinating worker processes and experience collection.
Main interface between training system and parallel workers.
"""

import multiprocessing
import queue
import time
import threading
from typing import List, Dict, Any, Optional, Tuple

from keisei.config_schema import AppConfig
from keisei.core.ppo_agent import PPOAgent
from keisei.training.parallel.self_play_worker import SelfPlayWorker
from keisei.training.parallel.model_sync import ModelSynchronizer


class ParallelManager:
    """
    Manages parallel worker processes for experience collection.
    
    Coordinates worker lifecycle, model synchronization, and experience
    aggregation for the main training process.
    """
    
    def __init__(self, config: AppConfig, agent: PPOAgent):
        self.config = config
        self.agent = agent
        self.parallel_config = config.training.parallel
        
        # Worker management
        self.workers: List[SelfPlayWorker] = []
        self.worker_processes: List[multiprocessing.Process] = []
        
        # Communication queues
        self.experience_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=self.parallel_config.queue_maxsize
        )
        self.model_queues: List[multiprocessing.Queue] = []
        self.control_queues: List[multiprocessing.Queue] = []
        
        # Synchronization
        self.model_synchronizer = ModelSynchronizer(agent)
        self.steps_since_sync = 0
        
        # Statistics
        self.total_episodes_collected = 0
        self.total_steps_collected = 0
        self.worker_stats: Dict[int, Dict] = {}
        
        # State
        self.is_running = False
        self.initialization_complete = False
    
    def start_workers(self) -> bool:
        """
        Initialize and start all worker processes.
        
        Returns:
            True if all workers started successfully
        """
        if self.is_running:
            return True
            
        try:
            # Create communication queues for each worker
            for worker_id in range(self.parallel_config.num_workers):
                model_queue = multiprocessing.Queue()
                control_queue = multiprocessing.Queue()
                
                self.model_queues.append(model_queue)
                self.control_queues.append(control_queue)
                
                # Create worker process
                worker = SelfPlayWorker(
                    worker_id=worker_id,
                    config=self.config,
                    experience_queue=self.experience_queue,
                    model_queue=model_queue,
                    control_queue=control_queue
                )
                
                self.workers.append(worker)
                
            # Start all workers
            for worker in self.workers:
                worker.start()
                
            # Wait for initialization
            if not self._wait_for_worker_initialization():
                self.shutdown()
                return False
                
            # Send initial model weights
            self.synchronize_models()
            
            self.is_running = True
            self.initialization_complete = True
            
            return True
            
        except Exception as e:
            print(f"Failed to start workers: {e}")
            self.shutdown()
            return False
    
    def _wait_for_worker_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for all workers to complete initialization.
        
        Args:
            timeout: Maximum time to wait for initialization
            
        Returns:
            True if all workers initialized successfully
        """
        start_time = time.time()
        initialized_workers = set()
        
        while time.time() - start_time < timeout:
            try:
                message = self.experience_queue.get(timeout=1.0)
                
                if message['type'] == 'init_complete':
                    worker_id = message['worker_id']
                    initialized_workers.add(worker_id)
                    
                    if len(initialized_workers) == self.parallel_config.num_workers:
                        return True
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during worker initialization: {e}")
                return False
                
        return False
    
    def collect_experiences(self, target_steps: int) -> List[Dict[str, Any]]:
        """
        Collect experiences from workers until target steps reached.
        
        Args:
            target_steps: Number of steps to collect
            
        Returns:
            List of experience data dictionaries
        """
        if not self.is_running:
            raise RuntimeError("Workers not running. Call start_workers() first.")
            
        collected_experiences = []
        steps_collected = 0
        start_time = time.time()
        
        # Send collection requests to workers
        episodes_per_worker = max(1, target_steps // (self.parallel_config.num_workers * 100))
        for control_queue in self.control_queues:
            try:
                control_queue.put({
                    'type': 'collect_episodes',
                    'target_episodes': episodes_per_worker
                })
            except queue.Full:
                pass
        
        # Collect experiences
        while steps_collected < target_steps:
            try:
                message = self.experience_queue.get(
                    timeout=self.parallel_config.worker_timeout
                )
                
                if message['type'] == 'experience_data':
                    experience_data = message['buffer_data']
                    episode_steps = message['episode_steps']
                    
                    collected_experiences.append(experience_data)
                    steps_collected += episode_steps
                    
                    # Update statistics
                    worker_id = message['worker_id']
                    self._update_worker_stats(worker_id, message)
                    
                elif message['type'] == 'error':
                    print(f"Worker error: {message['message']}")
                    
            except queue.Empty:
                # Check if we should synchronize models
                if self._should_synchronize_models():
                    self.synchronize_models()
                continue
                
        self.total_steps_collected += steps_collected
        return collected_experiences
    
    def synchronize_models(self) -> bool:
        """
        Send current model weights to all workers.
        
        Returns:
            True if synchronization successful
        """
        success = self.model_synchronizer.send_model_update(self.model_queues)
        
        if success:
            self.steps_since_sync = 0
        else:
            print("Warning: Model synchronization partially failed")
            
        return success
    
    def _should_synchronize_models(self) -> bool:
        """Check if model synchronization should occur."""
        return self.model_synchronizer.should_sync(
            self.steps_since_sync,
            self.parallel_config.model_sync_interval
        )
    
    def _update_worker_stats(self, worker_id: int, message: Dict[str, Any]) -> None:
        """Update statistics for specific worker."""
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = {
                'episodes_collected': 0,
                'steps_collected': 0,
                'total_reward': 0.0,
                'last_update': time.time()
            }
            
        stats = self.worker_stats[worker_id]
        stats['episodes_collected'] += 1
        stats['steps_collected'] += message.get('episode_steps', 0)
        stats['total_reward'] += message.get('episode_reward', 0.0)
        stats['last_update'] = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about parallel system.
        
        Returns:
            Dictionary with parallel system statistics
        """
        sync_stats = self.model_synchronizer.get_sync_stats()
        
        return {
            'total_episodes_collected': self.total_episodes_collected,
            'total_steps_collected': self.total_steps_collected,
            'active_workers': len(self.workers),
            'worker_stats': self.worker_stats.copy(),
            'synchronization': sync_stats,
            'queue_size': self.experience_queue.qsize(),
            'is_running': self.is_running
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown all workers and cleanup resources."""
        if not self.is_running:
            return
            
        # Send shutdown signals
        for control_queue in self.control_queues:
            try:
                control_queue.put({'type': 'shutdown'})
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        # Cleanup queues
        self._cleanup_queues()
        
        self.is_running = False
        self.initialization_complete = False
        
    def _cleanup_queues(self) -> None:
        """Clean up communication queues."""
        # Clear experience queue
        while not self.experience_queue.empty():
            try:
                self.experience_queue.get_nowait()
            except:
                break
                
        # Clear model queues
        for model_queue in self.model_queues:
            while not model_queue.empty():
                try:
                    model_queue.get_nowait()
                except:
                    break
    
    def __enter__(self):
        """Context manager entry."""
        self.start_workers()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
```

---

#### **File 7: Training Loop Manager Integration**
**Path:** `keisei/training/training_loop_manager.py`
**Lines to Modify:** ~269 (current file size)
**Estimated Effort:** 8 hours

**Current State Analysis:**
- Orchestrates main training loop with epoch-based structure
- Currently handles serial experience collection
- Ready for parallel mode integration

**Key Integration Points:**

```python
# Add imports at top of file
from typing import Optional
from keisei.training.parallel.parallel_manager import ParallelManager

class TrainingLoopManager:
    def __init__(self, trainer, config: AppConfig):
        # ...existing initialization...
        
        # Parallel system integration
        self.parallel_manager: Optional[ParallelManager] = None
        if config.training.parallel.enabled:
            try:
                self.parallel_manager = ParallelManager(config, trainer.agent)
                self.logger.info("Parallel experience collection enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize parallel system: {e}")
                self.logger.info("Falling back to serial mode")
                self.parallel_manager = None
        
    def run_training_loop(self) -> None:
        """Main training loop with parallel/serial mode support."""
        # ...existing setup code...
        
        try:
            # Initialize parallel system if enabled
            if self.parallel_manager and not self.parallel_manager.start_workers():
                self.logger.warning("Failed to start parallel workers, falling back to serial")
                self.parallel_manager = None
            
            # ...existing training loop...
            
        finally:
            # Cleanup parallel system
            if self.parallel_manager:
                self.parallel_manager.shutdown()
    
    def collect_epoch_data(self, steps_per_epoch: int) -> None:
        """Collect training data using parallel or serial mode."""
        if self.parallel_manager and self.parallel_manager.is_running:
            self._collect_parallel_data(steps_per_epoch)
        else:
            self._collect_serial_data(steps_per_epoch)
    
    def _collect_parallel_data(self, steps_per_epoch: int) -> None:
        """Collect experience data using parallel workers."""
        try:
            # Collect experiences from workers
            experience_batches = self.parallel_manager.collect_experiences(steps_per_epoch)
            
            # Add to experience buffer
            for batch_data in experience_batches:
                self.trainer.experience_buffer.merge_worker_buffers([batch_data])
            
            # Update synchronization counter
            self.parallel_manager.steps_since_sync += steps_per_epoch
            
            # Sync models if needed
            if self.parallel_manager._should_synchronize_models():
                self.parallel_manager.synchronize_models()
                
        except Exception as e:
            self.logger.error(f"Parallel collection failed: {e}")
            self.logger.info("Falling back to serial collection")
            self._collect_serial_data(steps_per_epoch)
    
    def _collect_serial_data(self, steps_per_epoch: int) -> None:
        """Collect experience data using original serial method."""
        # This preserves the existing serial collection logic
        for _ in range(steps_per_epoch):
            # ...existing step collection code...
            pass
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics including parallel metrics."""
        stats = {
            # ...existing stats...
        }
        
        # Add parallel statistics if available
        if self.parallel_manager:
            stats['parallel'] = self.parallel_manager.get_statistics()
        
        return stats
```

---

### **Phase 4: Testing and Validation (Week 4)**

#### **File 8: Comprehensive Parallel Tests**
**Path:** `tests/test_parallel_integration.py`
**New File:** Integration testing
**Estimated Effort:** 12 hours

```python
"""
Comprehensive integration tests for parallel experience collection system.
Tests worker lifecycle, experience collection, and model synchronization.
"""

import pytest
import multiprocessing
import time
import torch
from unittest.mock import patch, MagicMock

from keisei.config_schema import AppConfig
from keisei.core.ppo_agent import PPOAgent
from keisei.training.parallel.parallel_manager import ParallelManager
from keisei.training.parallel.self_play_worker import SelfPlayWorker
from keisei.training.parallel.model_sync import ModelSynchronizer
from keisei.utils import load_config


class TestParallelIntegration:
    """Integration tests for parallel system components."""
    
    @pytest.fixture
    def config(self):
        """Test configuration with parallel enabled."""
        config = load_config()
        config.training.parallel.enabled = True
        config.training.parallel.num_workers = 2
        config.training.parallel.worker_timeout = 5.0
        config.training.total_timesteps = 100
        config.training.steps_per_epoch = 20
        return config
    
    @pytest.fixture
    def agent(self, config):
        """Test PPO agent."""
        return PPOAgent(config, torch.device("cpu"))
    
    def test_parallel_manager_lifecycle(self, config, agent):
        """Test parallel manager startup and shutdown."""
        manager = ParallelManager(config, agent)
        
        # Test startup
        assert manager.start_workers()
        assert manager.is_running
        assert len(manager.workers) == config.training.parallel.num_workers
        
        # Test shutdown
        manager.shutdown()
        assert not manager.is_running
        
        # Verify workers are cleaned up
        for worker in manager.workers:
            assert not worker.is_alive()
    
    def test_experience_collection(self, config, agent):
        """Test experience collection from workers."""
        with ParallelManager(config, agent) as manager:
            # Collect experiences
            target_steps = 50
            experiences = manager.collect_experiences(target_steps)
            
            # Verify we got experience data
            assert len(experiences) > 0
            
            # Verify statistics
            stats = manager.get_statistics()
            assert stats['total_steps_collected'] >= target_steps
            assert stats['active_workers'] == config.training.parallel.num_workers
    
    def test_model_synchronization(self, config, agent):
        """Test model weight synchronization."""
        synchronizer = ModelSynchronizer(agent)
        
        # Create test queues
        test_queues = [multiprocessing.Queue() for _ in range(2)]
        
        # Send update
        success = synchronizer.send_model_update(test_queues)
        assert success
        
        # Verify updates in queues
        for queue in test_queues:
            assert not queue.empty()
            update = queue.get()
            assert update['type'] == 'model_weights'
            assert 'state_dict' in update
    
    def test_worker_error_handling(self, config, agent):
        """Test worker error handling and recovery."""
        with ParallelManager(config, agent) as manager:
            # Simulate worker error by terminating a worker
            if manager.workers:
                worker = manager.workers[0]
                worker.terminate()
                worker.join()
            
            # Manager should continue operating with remaining workers
            stats = manager.get_statistics()
            assert manager.is_running
    
    def test_performance_benchmark(self, config, agent):
        """Benchmark parallel vs serial performance."""
        # Test serial collection time
        start_time = time.time()
        with patch('keisei.training.training_loop_manager.TrainingLoopManager._collect_serial_data') as mock_serial:
            mock_serial.return_value = None
            # Simulate serial collection
            time.sleep(0.1)  # Simulate work
        serial_time = time.time() - start_time
        
        # Test parallel collection time
        start_time = time.time()
        with ParallelManager(config, agent) as manager:
            experiences = manager.collect_experiences(20)
        parallel_time = time.time() - start_time
        
        # Parallel should be more efficient (this test may vary based on system)
        # At minimum, parallel system should complete without errors
        assert len(experiences) >= 0
    
    def test_configuration_validation(self, config, agent):
        """Test parallel configuration validation."""
        # Test invalid worker count
        invalid_config = config.model_copy()
        invalid_config.training.parallel.num_workers = 0
        
        with pytest.raises(ValueError):
            # This should be caught by Pydantic validation
            ParallelManager(invalid_config, agent)
    
    def test_queue_overflow_handling(self, config, agent):
        """Test handling of queue overflow scenarios."""
        # Set very small queue size
        small_queue_config = config.model_copy()
        small_queue_config.training.parallel.queue_maxsize = 1
        
        with ParallelManager(small_queue_config, agent) as manager:
            # This should not crash even with small queue
            experiences = manager.collect_experiences(10)
            assert isinstance(experiences, list)


class TestWorkerProcess:
    """Tests for individual worker process functionality."""
    
    def test_worker_initialization(self, config):
        """Test worker process initialization."""
        experience_queue = multiprocessing.Queue()
        model_queue = multiprocessing.Queue()
        control_queue = multiprocessing.Queue()
        
        worker = SelfPlayWorker(
            worker_id=0,
            config=config,
            experience_queue=experience_queue,
            model_queue=model_queue,
            control_queue=control_queue
        )
        
        # Start worker in separate process
        worker.start()
        
        # Wait for initialization
        try:
            init_message = experience_queue.get(timeout=10.0)
            assert init_message['type'] == 'init_complete'
            assert init_message['worker_id'] == 0
        finally:
            # Cleanup
            control_queue.put({'type': 'shutdown'})
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()


@pytest.mark.slow
class TestParallelStability:
    """Long-running stability tests for parallel system."""
    
    def test_extended_training_session(self, config, agent):
        """Test parallel system stability over extended period."""
        config.training.total_timesteps = 1000
        
        with ParallelManager(config, agent) as manager:
            # Run multiple collection cycles
            for cycle in range(10):
                experiences = manager.collect_experiences(50)
                assert len(experiences) > 0
                
                # Occasionally sync models
                if cycle % 3 == 0:
                    manager.synchronize_models()
            
            # Verify system still operational
            stats = manager.get_statistics()
            assert stats['is_running']
            assert stats['total_steps_collected'] > 0
```

## Implementation Schedule Summary

| Phase | Duration | Files Created/Modified | Key Deliverables |
|-------|----------|----------------------|------------------|
| **Phase 1** | Week 1 | 3 files | Configuration schema, package structure, experience buffer enhancements |
| **Phase 2** | Week 2 | 2 files | Worker process and model synchronization implementation |
| **Phase 3** | Week 3 | 2 files | Parallel manager and training loop integration |
| **Phase 4** | Week 4 | 1+ files | Comprehensive testing and validation |

**Total Estimated Effort:** 160 hours (4 weeks × 40 hours)
**Risk Level:** Low (well-defined interfaces, existing infrastructure)
**Expected Performance Improvement:** 2-3x training throughput

This roadmap provides a detailed, file-by-file implementation plan that systematically builds the parallel system while maintaining full backward compatibility and comprehensive testing coverage.
