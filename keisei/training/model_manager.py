"""
training/model_manager.py: Model lifecycle management for Shogi RL training.

This module handles model-related concerns including:
- Model configuration and factory instantiation
- Mixed precision training setup
- Checkpoint loading and resuming
- Model artifact creation for WandB
- Final model saving and persistence
"""

import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
from torch.cuda.amp import GradScaler

from keisei.config_schema import AppConfig
from keisei.core.ppo_agent import PPOAgent

from . import utils


class ModelManager:
    """Manages model lifecycle for training runs."""

    def __init__(self, config: AppConfig, args: Any, device: torch.device, logger_func=None):
        """
        Initialize the ModelManager.

        Args:
            config: Application configuration
            args: Command-line arguments
            device: PyTorch device for model operations
            logger_func: Optional logging function for status messages
        """
        self.config = config
        self.args = args
        self.device = device
        self.logger_func = logger_func or (lambda msg: None)

        # Model configuration from args or config
        self.input_features = (
            getattr(args, "input_features", None) or config.training.input_features
        )
        self.model_type = getattr(args, "model", None) or config.training.model_type
        self.tower_depth = (
            getattr(args, "tower_depth", None) or config.training.tower_depth
        )
        self.tower_width = (
            getattr(args, "tower_width", None) or config.training.tower_width
        )
        self.se_ratio = getattr(args, "se_ratio", None) or config.training.se_ratio

        # Initialize feature spec and observation shape
        self._setup_feature_spec()

        # Initialize mixed precision
        self._setup_mixed_precision()

        # Create the model
        self.model = self._create_model()

        # Track checkpoint resume status
        self.resumed_from_checkpoint: Optional[str] = None
        self.checkpoint_data: Optional[Dict[str, Any]] = None

    def _setup_feature_spec(self):
        """Setup feature specification and observation shape."""
        from keisei.shogi import features

        self.feature_spec = features.FEATURE_SPECS[self.input_features]
        self.obs_shape = (self.feature_spec.num_planes, 9, 9)

    def _setup_mixed_precision(self):
        """Setup mixed precision training if enabled and supported."""
        self.use_mixed_precision = (
            self.config.training.mixed_precision and self.device.type == "cuda"
        )
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger_func("Mixed precision training enabled (CUDA).")
        elif self.config.training.mixed_precision and self.device.type != "cuda":
            self.logger_func(
                "Mixed precision training requested but CUDA is not available/selected. "
                "Proceeding without mixed precision."
            )
            self.use_mixed_precision = False
            self.scaler = None
        else:
            self.scaler = None

    def _create_model(self) -> torch.nn.Module:
        """Create the model using the model factory."""
        from keisei.training.models import model_factory

        model = model_factory(
            model_type=self.model_type,
            obs_shape=self.obs_shape,
            num_actions=self.config.env.num_actions_total,
            tower_depth=self.tower_depth,
            tower_width=self.tower_width,
            se_ratio=self.se_ratio if self.se_ratio > 0 else None,
        )
        
        return model.to(self.device)

    def create_agent(self) -> PPOAgent:
        """Create a PPO agent with the configured model."""
        agent = PPOAgent(
            config=self.config,
            device=self.device,
        )
        # Replace the agent's default model with our configured model
        agent.model = self.model
        return agent

    def handle_checkpoint_resume(self, agent: PPOAgent, model_dir: str) -> bool:
        """
        Handle resuming from checkpoint if specified or auto-detected.
        
        Args:
            agent: PPO agent to load checkpoint into
            model_dir: Directory to search for checkpoints
            
        Returns:
            bool: True if resumed from checkpoint, False otherwise
        """
        resume_path = self.args.resume

        def find_ckpt_in_dir(directory):
            return utils.find_latest_checkpoint(directory)

        if resume_path == "latest" or resume_path is None:
            # Try to find latest checkpoint in the run's model_dir
            latest_ckpt = find_ckpt_in_dir(model_dir)
            # If not found, try the parent directory (savedir)
            if not latest_ckpt:
                parent_dir = os.path.dirname(model_dir.rstrip(os.sep))
                parent_ckpt = find_ckpt_in_dir(parent_dir)
                if parent_ckpt:
                    # Copy the checkpoint into the run's model_dir for consistency
                    dest_ckpt = os.path.join(model_dir, os.path.basename(parent_ckpt))
                    shutil.copy2(parent_ckpt, dest_ckpt)
                    latest_ckpt = dest_ckpt
                    
            if latest_ckpt:
                checkpoint_data = agent.load_model(latest_ckpt)
                self.resumed_from_checkpoint = latest_ckpt
                self.checkpoint_data = checkpoint_data
                # Resume logging will be handled by trainer's run_training_loop
                return True
            else:
                self.resumed_from_checkpoint = None
                self.checkpoint_data = None
                return False
                
        elif resume_path:
            checkpoint_data = agent.load_model(resume_path)
            self.resumed_from_checkpoint = resume_path
            self.checkpoint_data = checkpoint_data
            # Resume logging will be handled by trainer's run_training_loop
            return True
        else:
            self.resumed_from_checkpoint = None
            self.checkpoint_data = None
            return False

    def create_model_artifact(
        self,
        model_path: str,
        artifact_name: str,
        run_name: str,
        is_wandb_active: bool,
        artifact_type: str = "model",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """
        Create and upload a W&B artifact for a model checkpoint.

        Args:
            model_path: Path to the model file to upload
            artifact_name: Name for the artifact (without run prefix)
            run_name: Current run name for artifact prefixing
            is_wandb_active: Whether WandB is currently active
            artifact_type: Type of artifact (default: "model")
            description: Optional description for the artifact
            metadata: Optional metadata dict to attach to the artifact
            aliases: Optional list of aliases (e.g., ["latest", "best"])

        Returns:
            bool: True if artifact was created successfully, False otherwise
        """
        if not (is_wandb_active and wandb.run):
            return False

        if not os.path.exists(model_path):
            self.logger_func(
                f"Warning: Model file {model_path} does not exist, skipping artifact creation."
            )
            return False

        try:
            # Create artifact with run name prefix for uniqueness
            full_artifact_name = f"{run_name}-{artifact_name}"
            artifact = wandb.Artifact(
                name=full_artifact_name,
                type=artifact_type,
                description=description or f"Model checkpoint from run {run_name}",
                metadata=metadata or {},
            )

            # Add the model file
            artifact.add_file(model_path)

            # Log the artifact with optional aliases
            wandb.log_artifact(artifact, aliases=aliases)

            aliases_str = f" with aliases {aliases}" if aliases else ""
            self.logger_func(
                f"Model artifact '{full_artifact_name}' created and uploaded{aliases_str}"
            )

            return True

        except (OSError, RuntimeError, TypeError, ValueError) as e:
            self.logger_func(f"Error creating W&B artifact for {model_path}: {e}")
            return False

    def save_final_model(
        self,
        agent: PPOAgent,
        model_dir: str,
        global_timestep: int,
        total_episodes_completed: int,
        game_stats: Dict[str, int],
        run_name: str,
        is_wandb_active: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Save the final trained model and create associated artifacts.
        
        Args:
            agent: PPO agent to save
            model_dir: Directory to save model in
            global_timestep: Current training timestep
            total_episodes_completed: Total episodes completed
            game_stats: Dictionary with black_wins, white_wins, draws
            run_name: Current run name
            is_wandb_active: Whether WandB is active
            
        Returns:
            Tuple[bool, Optional[str]]: (success, model_path)
        """
        final_model_path = os.path.join(model_dir, "final_model.pth")
        
        try:
            agent.save_model(
                final_model_path,
                global_timestep,
                total_episodes_completed,
            )
            self.logger_func(f"Final model saved to {final_model_path}")

            # Create W&B artifact for final model
            final_metadata = {
                "training_timesteps": global_timestep,
                "total_episodes": total_episodes_completed,
                "black_wins": game_stats.get("black_wins", 0),
                "white_wins": game_stats.get("white_wins", 0),
                "draws": game_stats.get("draws", 0),
                "training_completed": True,
                "model_type": getattr(self.config.training, "model_type", "resnet"),
                "feature_set": getattr(self.config.env, "feature_set", "core"),
            }
            
            self.create_model_artifact(
                model_path=final_model_path,
                artifact_name="final-model",
                run_name=run_name,
                is_wandb_active=is_wandb_active,
                description=f"Final trained model after {global_timestep} timesteps",
                metadata=final_metadata,
                aliases=["latest", "final"],
            )
            
            return True, final_model_path
            
        except (OSError, RuntimeError) as e:
            self.logger_func(f"Error saving final model {final_model_path}: {e}")
            return False, None

    def save_final_checkpoint(
        self,
        agent: PPOAgent,
        model_dir: str,
        global_timestep: int,
        total_episodes_completed: int,
        game_stats: Dict[str, int],
        run_name: str,
        is_wandb_active: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Save a final checkpoint with game statistics.
        
        Args:
            agent: PPO agent to save
            model_dir: Directory to save checkpoint in
            global_timestep: Current training timestep
            total_episodes_completed: Total episodes completed
            game_stats: Dictionary with black_wins, white_wins, draws
            run_name: Current run name
            is_wandb_active: Whether WandB is active
            
        Returns:
            Tuple[bool, Optional[str]]: (success, checkpoint_path)
        """
        if global_timestep <= 0:
            return False, None
            
        checkpoint_filename = os.path.join(
            model_dir, f"checkpoint_ts{global_timestep}.pth"
        )
        
        # Don't save if checkpoint already exists
        if os.path.exists(checkpoint_filename):
            return True, checkpoint_filename
            
        try:
            agent.save_model(
                checkpoint_filename,
                global_timestep,
                total_episodes_completed,
                stats_to_save=game_stats,
            )
            self.logger_func(f"Final checkpoint saved to {checkpoint_filename}")

            # Create W&B artifact for final checkpoint
            checkpoint_metadata = {
                "training_timesteps": global_timestep,
                "total_episodes": total_episodes_completed,
                "black_wins": game_stats.get("black_wins", 0),
                "white_wins": game_stats.get("white_wins", 0),
                "draws": game_stats.get("draws", 0),
                "checkpoint_type": "final",
                "model_type": getattr(self.config.training, "model_type", "resnet"),
                "feature_set": getattr(self.config.env, "feature_set", "core"),
            }
            
            self.create_model_artifact(
                model_path=checkpoint_filename,
                artifact_name="final-checkpoint",
                run_name=run_name,
                is_wandb_active=is_wandb_active,
                description=f"Final checkpoint at timestep {global_timestep}",
                metadata=checkpoint_metadata,
                aliases=["latest-checkpoint"],
            )
            
            return True, checkpoint_filename
            
        except (OSError, RuntimeError) as e:
            self.logger_func(f"Error saving final checkpoint {checkpoint_filename}: {e}")
            return False, None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_type": self.model_type,
            "input_features": self.input_features,
            "tower_depth": self.tower_depth,
            "tower_width": self.tower_width,
            "se_ratio": self.se_ratio,
            "obs_shape": self.obs_shape,
            "num_planes": self.feature_spec.num_planes,
            "use_mixed_precision": self.use_mixed_precision,
            "device": str(self.device),
        }
