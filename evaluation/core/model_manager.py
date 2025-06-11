from pathlib import Path
import torch
from keisei.config_schema import (
    AppConfig,
    TrainingConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    WandBConfig,
    ParallelConfig,
    DemoConfig,
    _create_display_config, 
)
from keisei.core.ppo_agent import PPOAgent
from keisei.core.neural_network import ActorCritic


class ModelWeightManager:
    def __init__(self, num_input_channels: int, num_actions_total: int, device: torch.device | str = "cpu"):
        self.num_input_channels = num_input_channels
        self.num_actions_total = num_actions_total
        self.device = torch.device(device) if isinstance(device, str) else device

    def _create_minimal_config(self) -> AppConfig:
        minimal_training_config = TrainingConfig(
            total_timesteps=1,
            steps_per_epoch=1,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
            normalize_advantages=True,
            enable_value_clipping=False,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
        )

        minimal_env_config = EnvConfig(
            device=str(self.device),
            input_channels=self.num_input_channels,
            num_actions_total=self.num_actions_total,
            seed=42,
            max_moves_per_game=500
        )

        minimal_evaluation_config = EvaluationConfig(
            enable_periodic_evaluation=False,
            evaluation_interval_timesteps=50000,
            strategy="single_opponent",
            num_games=1,
            max_moves_per_game=100,
            opponent_type="random",
            log_level="INFO",
            max_concurrent_games=1,
            timeout_per_game=None,
            randomize_positions=False,
            random_seed=None,
            save_games=False,
            save_path=None,
            log_file_path_eval="logs/minimal_eval.log",
            wandb_log_eval=False,
            update_elo=False,
            elo_registry_path=None,
            agent_id=None,
            opponent_id=None,
            previous_model_pool_size=1,
            enable_in_memory_evaluation=True,
            model_weight_cache_size=2,
            enable_parallel_execution=False,
            process_restart_threshold=100,
            temp_agent_device=str(self.device),
            clear_cache_after_evaluation=True,
        )
        
        minimal_logging_config = LoggingConfig(
            log_file="logs/minimal_eval_log.txt",
            model_dir="models/minimal_eval/",
            run_name="minimal_eval_run"
        )
        minimal_wandb_config = WandBConfig(
            enabled=False,
            project="keisei-shogi-rl",
            entity=None,
            run_name_prefix="keisei-minimal",
            watch_model=False,
            watch_log_freq=1000,
            watch_log_type="all",
            log_model_artifact=False
        )
        minimal_parallel_config = ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000
        )
        minimal_demo_config = DemoConfig(
            enable_demo_mode=False,
            demo_mode_delay=0.5
        )

        return AppConfig(
            env=minimal_env_config,
            training=minimal_training_config,
            evaluation=minimal_evaluation_config,
            logging=minimal_logging_config,
            wandb=minimal_wandb_config,
            parallel=minimal_parallel_config,
            demo=minimal_demo_config,
        )

    def create_agent_from_weights(
        self,
        weights_path: Path | str,
        device: torch.device | str | None = None,
    ) -> PPOAgent:
        if device is None:
            resolved_device = self.device
        elif isinstance(device, str):
            resolved_device = torch.device(device)
        else:
            resolved_device = device
        
        model_instance = ActorCritic(
            input_channels=self.num_input_channels,
            num_actions_total=self.num_actions_total # Corrected parameter name
        ).to(resolved_device)

        final_app_config = self._create_minimal_config()

        agent = PPOAgent(config=final_app_config, model=model_instance, device=resolved_device)
        agent.load_model(str(weights_path))
        return agent

    def _infer_input_channels_from_checkpoint(self, checkpoint_path: Path | str) -> int:
        print(f"Warning: Inferring num_input_channels from checkpoint at {checkpoint_path} is not fully implemented. Using manager's num_input_channels.")
        return self.num_input_channels

    # _infer_total_actions_from_checkpoint would be similar if needed
    # def _infer_total_actions_from_checkpoint(self, checkpoint_path: Path | str) -> int:
    #     print(f"Warning: Inferring num_actions_total from checkpoint at {checkpoint_path} is not fully implemented. Using manager's num_actions_total.")
    #     return self.num_actions_total

# Example of how ModelWeightManager might be instantiated if you know the model arch:
# manager = ModelWeightManager(num_input_channels=46, num_actions_total=13527, device="cpu")
# agent = manager.create_agent_from_weights("path/to/weights.pth")