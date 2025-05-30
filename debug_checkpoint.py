#!/usr/bin/env python3

"""Debug script to trace checkpoint resume flow."""

import os
import sys
from unittest.mock import Mock, patch

# Add keisei to the path
sys.path.insert(0, '/home/john/keisei')

try:
    from keisei.core.config import AppConfig, EnvConfig, TrainingConfig, PPOConfig, EvaluationConfig, LoggingConfig, WandBConfig, DemoConfig
    from keisei.training.trainer import Trainer
    print("Successfully imported keisei modules")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

class MockArgs:
    def __init__(self, resume=None):
        self.resume = resume

def test_checkpoint_debug():
    try:
        print("=== Creating mock config ===")
        # Create a mock config
        mock_config = AppConfig(
            env=EnvConfig(device='cpu', input_channels=46, num_actions_total=13527, seed=42),
            training=TrainingConfig(
                total_timesteps=1000000,
                episodes_between_saves=10000,
                num_eval_episodes=100,
                ppo=PPOConfig(
                    learning_rate=0.0003,
                    clip_range=0.2,
                    value_function_coeff=0.5,
                    entropy_coeff=0.01,
                    gae_lambda=0.95,
                    discount_factor=0.99,
                    ppo_epochs=4,
                    minibatch_size=64,
                    normalize_advantages=True,
                    clip_value_loss=True,
                    max_grad_norm=0.5,
                    target_kl=0.01,
                ),
                evaluation=EvaluationConfig(evaluation_interval_timesteps=50000),
                logging=LoggingConfig(run_name="test_run", run_name_prefix="test"),
                wandb=WandBConfig(watch_model=False, watch_log_freq=1000, watch_log_type="all"),
            ),
            demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
        )

        args = MockArgs(resume="/path/to/checkpoint.pth")

        checkpoint_data = {
            "global_timestep": 2500,
            "total_episodes_completed": 150,
            "black_wins": 60,
            "white_wins": 55,
            "draws": 35,
            "model_state_dict": {},
            "optimizer_state_dict": {},
        }

        print("=== Setting up mocks ===")
        
        # Just test the ModelManager part directly
        from keisei.training.model_manager import ModelManager
        
        # Create a mock logger function
        def mock_logger(msg):
            print(f"[ModelManager] {msg}")
        
        # Create ModelManager with mock args
        mock_args = MockArgs(resume="/path/to/checkpoint.pth")
        model_manager = ModelManager(mock_args, mock_logger)
        
        # Create a mock agent
        mock_agent = Mock()
        mock_agent.load_model.return_value = checkpoint_data
        
        with patch("os.path.exists", return_value=True):
            print("=== Calling handle_checkpoint_resume ===")
            result = model_manager.handle_checkpoint_resume(
                agent=mock_agent,
                model_dir="/tmp/models",
                resume_path_override="/path/to/checkpoint.pth"
            )
            
            print(f"handle_checkpoint_resume returned: {result}")
            print(f"model_manager.checkpoint_data: {model_manager.checkpoint_data}")
            print(f"model_manager.resumed_from_checkpoint: {model_manager.resumed_from_checkpoint}")
            print(f"mock_agent.load_model.called: {mock_agent.load_model.called}")
            
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_checkpoint_debug()
