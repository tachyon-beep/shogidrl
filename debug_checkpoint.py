#!/usr/bin/env python3
"""Debug script to test checkpoint loading issue"""

import os
import tempfile
from unittest.mock import patch, Mock

# Set up path
import sys
sys.path.insert(0, '/home/john/keisei')

from keisei.config_schema import AppConfig
from keisei.training.trainer import Trainer


class MockArgs:
    def __init__(self, resume=None, evaluate=False):
        self.resume = resume
        self.evaluate = evaluate


def test_checkpoint_loading():
    """Test checkpoint loading to debug the issue."""
    
    try:
        # Create a minimal test config
        config = AppConfig()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Update config paths for test isolation
            config.logging.model_dir = f"{tmp_dir}/models"
            config.wandb.enabled = False
            config.training.total_timesteps = 5
            config.training.steps_per_epoch = 2
            
            # Create checkpoint path
            checkpoint_path = f"{tmp_dir}/test_checkpoint.pth"
            args = MockArgs(resume=checkpoint_path)
            
            # Mock external dependencies
            with patch("torch.load") as mock_torch_load, \
                 patch("os.path.exists", return_value=True), \
                 patch("builtins.open", create=True), \
                 patch("os.makedirs"), \
                 patch("torch.save"):
                
                # Mock checkpoint data
                checkpoint_data = {
                    "model_state_dict": {},
                    "optimizer_state_dict": {},
                    "global_timestep": 1500,
                    "total_episodes_completed": 100,
                    "black_wins": 40,
                    "white_wins": 35,
                    "draws": 25,
                    "config": config.model_dump(),
                }
                mock_torch_load.return_value = checkpoint_data
                
                print("Creating trainer...")
                # Create trainer
                trainer = Trainer(config, args)
                
                # Debug: print what we got
                print(f"Trainer global_timestep: {trainer.global_timestep}")
                print(f"Trainer total_episodes: {trainer.total_episodes_completed}")
                print(f"Trainer black_wins: {trainer.black_wins}")
                print(f"Trainer resumed_from_checkpoint: {trainer.resumed_from_checkpoint}")
                
                # Check metrics manager state
                print(f"MetricsManager global_timestep: {trainer.metrics_manager.global_timestep}")
                print(f"MetricsManager stats.global_timestep: {trainer.metrics_manager.stats.global_timestep}")
                
                # Check model manager state
                print(f"ModelManager checkpoint_data: {trainer.model_manager.checkpoint_data}")
                print(f"ModelManager resumed_from_checkpoint: {trainer.model_manager.resumed_from_checkpoint}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_checkpoint_loading()
