"""
Example usage of the parallel training system.

This example demonstrates how to use the parallel experience collection system
in the Keisei Shogi training pipeline.
"""

import logging
import time

from keisei.core.neural_network import ActorCritic
from keisei.training.parallel.parallel_manager import ParallelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_model() -> ActorCritic:
    """Create a sample ActorCritic model for testing."""
    # This would typically match your actual model architecture
    # For demonstration, using simplified dimensions
    input_dim = 81 * 14  # Shogi board representation
    action_dim = 2187  # Approximate number of possible moves in Shogi
    
    return ActorCritic(input_dim, action_dim)


def main():
    """Main training loop with parallel experience collection."""
    logger.info("Starting parallel training example")
    
    # Initialize model and configuration
    model = create_sample_model()
    
    # Create configurations
    env_config = {'board_size': 9, 'max_moves': 200}
    model_config = {'input_dim': 81*14, 'action_dim': 2187}
    parallel_config = {
        'num_workers': 4,
        'batch_size': 64,
        'enabled': True,
        'max_queue_size': 1000,
        'timeout_seconds': 30.0,
        'sync_interval': 100,
        'compression_enabled': True
    }
    
    # Create parallel manager
    parallel_manager = ParallelManager(env_config, model_config, parallel_config, device='cpu')
    
    try:
        # Start worker processes
        logger.info("Starting %d worker processes", parallel_config['num_workers'])
        parallel_manager.start_workers(model)
        
        # Training loop
        total_steps = 0
        max_steps = 1000
        
        # Mock experience buffer for the example
        from keisei.core.experience_buffer import ExperienceBuffer
        experience_buffer = ExperienceBuffer(buffer_size=10000, gamma=0.99, lambda_gae=0.95, device='cpu')
        
        while total_steps < max_steps:
            # Collect experiences from workers
            num_collected = parallel_manager.collect_experiences(experience_buffer)
            
            if num_collected > 0:
                logger.info("Collected %d experiences from workers", num_collected)
                
                # Here you would typically:
                # 1. Process the experiences
                # 2. Update the model using the experiences
                # 3. Sync the updated model back to workers
                
                # Simulate model training step
                time.sleep(0.1)  # Simulate training time
                
                # Update model in workers (every N steps)
                if total_steps % parallel_config['sync_interval'] == 0:
                    logger.info("Syncing model weights to workers at step %d", total_steps)
                    parallel_manager.sync_model_if_needed(model, total_steps)
                
                total_steps += 1
            else:
                # No experiences yet, wait a bit
                time.sleep(0.1)
            
            # Optional: Reset workers periodically
            if total_steps % 1000 == 0:
                logger.info("Resetting workers at step %d", total_steps)
                parallel_manager.reset_workers()
            
            # Monitor system
            if total_steps % 100 == 0:
                queue_info = parallel_manager.communicator.get_queue_info()
                logger.info("Training step %d - Queue info: %s", total_steps, queue_info)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error("Training failed with error: %s", str(e))
        raise
    
    finally:
        # Clean shutdown
        logger.info("Shutting down parallel training system")
        parallel_manager.stop_workers()
        logger.info("Parallel training example completed")


if __name__ == '__main__':
    main()
