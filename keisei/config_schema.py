"""
Pydantic configuration schema for Keisei DRL Shogi Client.
Defines all configuration sections and their defaults.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, validator


class EnvConfig(BaseModel):
    device: str = Field("cpu", description="Device to use: 'cpu' or 'cuda'.")
    input_channels: int = Field(
        46, description="Number of input channels for the neural network."
    )
    num_actions_total: int = Field(
        13527, description="Total number of possible actions."
    )
    seed: int = Field(42, description="Random seed for reproducibility.")


class TrainingConfig(BaseModel):
    total_timesteps: int = Field(
        500_000, description="Total number of environment steps for training."
    )
    steps_per_epoch: int = Field(2048, description="Steps per PPO buffer/epoch.")
    ppo_epochs: int = Field(10, description="Number of PPO update epochs per buffer.")
    minibatch_size: int = Field(64, description="Minibatch size for PPO updates.")
    learning_rate: float = Field(3e-4, description="Learning rate for optimizer.")
    gamma: float = Field(0.99, description="Discount factor.")
    clip_epsilon: float = Field(0.2, description="PPO clip epsilon.")
    value_loss_coeff: float = Field(0.5, description="Value loss coefficient.")
    entropy_coef: float = Field(0.01, description="Entropy regularization coefficient.")
    render_every_steps: int = Field(
        1, description="Update expensive display elements (metrics, logs) every N steps to reduce flicker."
    )
    refresh_per_second: int = Field(
        4, description="Rich Live refresh rate per second."
    )
    enable_spinner: bool = Field(
        True, description="Enable spinner column in progress bar (looks cool!)."
    )
    # --- Model/feature config additions ---
    input_features: str = Field("core46", description="Feature set for observation builder (e.g. 'core46', 'core46+all').")
    tower_depth: int = Field(9, description="Number of residual blocks in ResNet tower.")
    tower_width: int = Field(256, description="Width (channels) of ResNet tower.")
    se_ratio: float = Field(0.25, description="SE block squeeze ratio (0 disables SE blocks).")
    model_type: str = Field("resnet", description="Model type to use (e.g. 'resnet').")
    mixed_precision: bool = Field(False, description="Enable mixed-precision training.")
    ddp: bool = Field(False, description="Enable DistributedDataParallel training.")
    gradient_clip_max_norm: float = Field(0.5, description="Maximum norm for gradient clipping.")
    lambda_gae: float = Field(0.95, description="Lambda for Generalized Advantage Estimation (GAE).")
    checkpoint_interval_timesteps: int = Field(10000, description="Save a model checkpoint every N timesteps.")
    evaluation_interval_timesteps: int = Field(50000, description="Run evaluation every N timesteps.")

    @validator("learning_rate")
    # pylint: disable=no-self-argument
    def lr_positive(cls, v):
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v


class EvaluationConfig(BaseModel):
    num_games: int = Field(20, description="Number of games to play during evaluation.")
    opponent_type: str = Field(
        "random", description="Type of opponent: 'random', 'heuristic', etc."
    )
    evaluation_interval_timesteps: int = Field(50000, description="Run evaluation every N timesteps.")


class LoggingConfig(BaseModel):
    log_file: str = Field(
        "logs/training_log.txt", description="Path for the main training log."
    )
    model_dir: str = Field(
        "models/", description="Directory to save model checkpoints."
    )
    run_name: Optional[str] = Field(
        None, description="Optional name for this run (overrides auto-generated name if set)."
    )


class WandBConfig(BaseModel):
    enabled: bool = Field(True, description="Enable Weights & Biases logging.")
    project: Optional[str] = Field("keisei-shogi-rl", description="W&B project name.")
    entity: Optional[str] = Field(None, description="W&B entity (username or team).")
    run_name_prefix: Optional[str] = Field("keisei", description="Prefix for W&B run names.")
    watch_model: bool = Field(True, description="Use wandb.watch() to log model gradients and parameters.")
    watch_log_freq: int = Field(1000, description="Frequency for wandb.watch() logging.")
    watch_log_type: Literal["gradients", "parameters", "all"] = Field(
        "all", description="Type of data to log with wandb.watch()."
    )


class DemoConfig(BaseModel):
    enable_demo_mode: bool = Field(
        False,
        description="If True, enables demo mode with per-move delay and extra logging.",
    )
    demo_mode_delay: float = Field(
        0.5, description="Delay in seconds between moves in demo mode."
    )


class AppConfig(BaseModel):
    env: EnvConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    wandb: WandBConfig
    demo: DemoConfig

    class Config:
        extra = "forbid"  # Disallow unknown fields for strict validation
