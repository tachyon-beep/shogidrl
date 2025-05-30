#!/usr/bin/env python3

"""Minimal debug test for checkpoint resume."""

import pytest
from unittest.mock import Mock, patch
from tests.test_trainer_resume_state import TestTrainerResumeState

def test_debug_checkpoint():
    """Debug checkpoint resume issue."""
    test_instance = TestTrainerResumeState()
    
    # Use the existing test but add debug prints
    with patch("keisei.training.trainer.SessionManager") as mock_session_manager_class, \
         patch("keisei.shogi.ShogiGame"), \
         patch("keisei.shogi.features.FEATURE_SPECS") as mock_feature_specs, \
         patch("keisei.utils.PolicyOutputMapper"), \
         patch("keisei.core.experience_buffer.ExperienceBuffer"), \
         patch("keisei.training.models.model_factory") as mock_model_factory, \
         patch("keisei.core.ppo_agent.PPOAgent") as mock_ppo_agent_class:

        from keisei.training.trainer import Trainer
        from tests.test_trainer_resume_state import MockArgs
        
        # Setup all the mocks as in the test
        feature_spec_mock = Mock()
        feature_spec_mock.num_planes = 46
        mock_feature_specs.__getitem__.return_value = feature_spec_mock

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_factory.return_value = mock_model

        mock_agent_instance = Mock()
        mock_ppo_agent_class.return_value = mock_agent_instance
        
        mock_agent_instance.model = Mock()
        mock_agent_instance.optimizer = Mock()
        mock_agent_instance.model.load_state_dict = Mock()
        mock_agent_instance.optimizer.load_state_dict = Mock()
        mock_agent_instance.device = "cpu"

        mock_session_instance = mock_session_manager_class.return_value
        mock_session_instance.run_name = "test_run"
        mock_session_instance.run_artifact_dir = "/tmp/artifacts"
        mock_session_instance.model_dir = "/tmp/models"
        mock_session_instance.log_file_path = "/tmp/train.log"
        mock_session_instance.eval_log_file_path = "/tmp/eval.log"
        mock_session_instance.is_wandb_active = False

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

        # Mock load_model to return our checkpoint data
        mock_agent_instance.load_model.return_value = checkpoint_data

        # Add spy to the Trainer _handle_checkpoint_resume method
        original_handle_checkpoint_resume = Trainer._handle_checkpoint_resume
        
        def debug_handle_checkpoint_resume(self):
            print(f"[DEBUG] _handle_checkpoint_resume called")
            print(f"[DEBUG] self.args.resume = {self.args.resume}")
            print(f"[DEBUG] Before: self.model_manager.checkpoint_data = {getattr(self.model_manager, 'checkpoint_data', 'NOT SET')}")
            
            result = original_handle_checkpoint_resume(self)
            
            print(f"[DEBUG] After: self.model_manager.checkpoint_data = {getattr(self.model_manager, 'checkpoint_data', 'NOT SET')}")
            print(f"[DEBUG] After: self.global_timestep = {self.global_timestep}")
            
            return result
        
        with patch.object(Trainer, '_handle_checkpoint_resume', debug_handle_checkpoint_resume), \
             patch("torch.load", return_value=checkpoint_data), \
             patch("os.path.exists", return_value=True):
            
            print("=== Creating Trainer ===")
            trainer = Trainer(test_instance.mock_config(), args)
            
            print(f"=== Final Results ===")
            print(f"trainer.global_timestep = {trainer.global_timestep}")
            print(f"trainer.model_manager.checkpoint_data = {getattr(trainer.model_manager, 'checkpoint_data', 'NOT SET')}")
            print(f"mock_agent_instance.load_model.called = {mock_agent_instance.load_model.called}")
            
            if mock_agent_instance.load_model.called:
                print(f"mock_agent_instance.load_model.call_args = {mock_agent_instance.load_model.call_args}")
                print(f"mock_agent_instance.load_model.return_value = {mock_agent_instance.load_model.return_value}")

if __name__ == "__main__":
    test_debug_checkpoint()
