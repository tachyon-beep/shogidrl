\
from typing import Optional, List

from pydantic import BaseModel, Field

class NetworkConfig(BaseModel):
    input_channels: int = 46
    num_actions_total: int = 3159
    conv_out_channels: int = 16
    conv_kernel_size: int = 3
    conv_padding: int = 1
    # linear_in_features will be calculated based on conv output and board size (e.g., 16 * 9 * 9 for Shogi).
    # This implies a fixed board input size for the conv layer's feature maps.
    # It's better to calculate this dynamically or assert if board dimensions change.
    value_head_out_features: int = 1

class PathsConfig(BaseModel):
    model_dir: str = "models/"
    log_file: str = "logs/training_log.txt"

class PPOAgentConfig(BaseModel):
    learning_rate: float = 0.0003
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    lambda_gae: float = 0.95  # Lambda for Generalized Advantage Estimation
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5

class ExperienceBufferConfig(BaseModel):
    steps_per_epoch: int = 2048  # Corresponds to PPO buffer size
    minibatch_size: int = 64

class TrainingConfig(BaseModel):
    total_timesteps: int = 500000
    ppo_epochs: int = 10  # Number of optimization epochs over the collected batch per PPO update
    save_freq_episodes: int = 200

class EvaluationConfig(BaseModel):
    eval_during_training: bool = True
    eval_freq_episodes: int = 100  # Kept for reference, but periodic eval is tied to save_freq_episodes
    eval_num_games: int = 10
    eval_opponent_type: str = "heuristic"
    eval_opponent_checkpoint_path: Optional[str] = None
    eval_device: str = "cpu"
    max_moves_per_game_eval: int = 256

class WandBTrainConfig(BaseModel):
    log: bool = True
    project: str = "shogi-drl"
    entity: Optional[str] = None
    # run_name will typically be generated dynamically (e.g., including timestamp or experiment details)

class WandBEvalConfig(BaseModel):
    log: bool = True
    project: str = "shogi-drl-periodic-evaluation"
    entity: Optional[str] = None
    run_name_prefix: str = "periodic_eval_" # Used to construct run names for periodic evaluations

class DebugConfig(BaseModel):
    print_game_real_time: bool = True
    real_time_print_delay: float = 0.5

class GeneralConfig(BaseModel):
    device: str = "cpu"  # Default device, can be overridden by train.py logic if CUDA is available
    max_moves_per_game_training: int = 500 # Max moves in a training game before timeout

class AppConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    ppo_agent: PPOAgentConfig = Field(default_factory=PPOAgentConfig)
    experience_buffer: ExperienceBufferConfig = Field(default_factory=ExperienceBufferConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    wandb_train: WandBTrainConfig = Field(default_factory=WandBTrainConfig)
    wandb_eval: WandBEvalConfig = Field(default_factory=WandBEvalConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    # Example of how to load from YAML and merge with CLI args (to be implemented in config_loader.py)
    # @classmethod
    # def load_config(cls, config_path: str, cli_args: Optional[dict] = None) -> 'AppConfig':
    #     import yaml
    #     with open(config_path, 'r') as f:
    #         yaml_data = yaml.safe_load(f)
    #
    #     # For deep merging of dictionaries if necessary, or Pydantic's own update mechanisms
    #     # merged_data = {**yaml_data, **(cli_args or {})} # Simplified; needs careful handling for nested models
    #
    #     # Pydantic v2 allows model_validate(obj) for dicts
    #     # For nested updates, you might load base, then update fields from CLI
    #     # This is a placeholder for the actual loading logic in config_loader.py
    #     if cli_args:
    #         # A more robust merge would be needed here for nested models
    #         # For example, update yaml_data with cli_args before validation
    #         pass # Placeholder
    #
    #     return cls.model_validate(yaml_data if yaml_data else {})

# To generate a default config.yaml:
# if __name__ == '__main__':
#     import yaml
#     default_config = AppConfig()
#     with open('default_config.yaml', 'w') as f:
#         yaml.dump(default_config.model_dump(), f, sort_keys=False)
#     print("Generated default_config.yaml with Pydantic model defaults.")
