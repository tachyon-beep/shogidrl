"""
Pydantic configuration schema for Keisei DRL Shogi Client.
Defines all configuration sections and their defaults.
"""

from typing import Optional

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


class LoggingConfig(BaseModel):
    log_file: str = Field(
        "logs/training_log.txt", description="Path for the main training log."
    )
    model_dir: str = Field(
        "models/", description="Directory to save model checkpoints."
    )


class WandBConfig(BaseModel):
    enabled: bool = Field(True, description="Enable Weights & Biases logging.")
    project: str = Field("keisei-shogi", description="W&B project name.")
    entity: Optional[str] = Field(None, description="W&B entity (user or team).")


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
