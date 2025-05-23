# train.py: Thin shim to call the real trainer in keisei.train

import sys
import os
import subprocess
import datetime
import keisei.train  # Moved import to the top
# Correctly import the config module from the root of the project
import config as app_config # Use app_config to avoid conflict with this script's name if it were config.py

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def run_evaluation(checkpoint_path: str, current_timestep: int, current_episode: int):
    """Runs the evaluation script as a subprocess."""
    if not app_config.EVAL_DURING_TRAINING:
        print("Periodic evaluation during training is disabled in config.")
        return

    command = [
        sys.executable, # Use the same python interpreter
        os.path.join(os.path.dirname(__file__), "evaluate.py"),
        "--agent-checkpoint", checkpoint_path,
        "--opponent-type", app_config.EVAL_OPPONENT_TYPE,
        "--num-games", str(app_config.EVAL_NUM_GAMES),
        "--max-moves-per-game", str(app_config.MAX_MOVES_PER_GAME_EVAL),
        "--device", app_config.EVAL_DEVICE,
        "--log-file", os.path.join("logs", f"periodic_eval_ts{current_timestep}_ep{current_episode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
    if app_config.EVAL_OPPONENT_TYPE == "ppo" and app_config.EVAL_OPPONENT_CHECKPOINT_PATH:
        command.extend(["--opponent-checkpoint", app_config.EVAL_OPPONENT_CHECKPOINT_PATH])
    
    if app_config.EVAL_WANDB_LOG:
        command.append("--wandb-log")
        if app_config.EVAL_WANDB_PROJECT:
            command.extend(["--wandb-project", app_config.EVAL_WANDB_PROJECT])
        if app_config.EVAL_WANDB_ENTITY:
            command.extend(["--wandb-entity", app_config.EVAL_WANDB_ENTITY])
        
        run_name = f"{app_config.EVAL_WANDB_RUN_NAME_PREFIX}ts{current_timestep}_ep{current_episode}"
        command.extend(["--wandb-run-name", run_name])

    print(f"Running periodic evaluation: {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        print("--- Periodic Evaluation Output ---")
        print(process.stdout)
        if process.stderr:
            print("--- Periodic Evaluation Errors ---")
            print(process.stderr)
        if process.returncode != 0:
            print(f"Periodic evaluation script exited with error code {process.returncode}.", file=sys.stderr)
        else:
            print("Periodic evaluation completed successfully.")
    except Exception as e:
        print(f"Error running periodic evaluation: {e}", file=sys.stderr)


# Patch keisei.ppo_agent.PPOAgent.save_model
try:
    from keisei.ppo_agent import PPOAgent as ActualPPOAgent
    _original_save_model_actual = ActualPPOAgent.save_model

    # The signature of PPOAgent.save_model is (self, file_path: str, global_timestep: int = 0, total_episodes_completed: int = 0)
    def patched_save_model_actual(self, file_path: str, global_timestep: int = 0, total_episodes_completed: int = 0):
        # Call the original save_model method with all its arguments
        result = _original_save_model_actual(self, file_path, global_timestep, total_episodes_completed)
        print(f"Model saved to {file_path} (via ppo_agent.PPOAgent). Triggering periodic evaluation.")
        print(f"Calling run_evaluation with timestep: {global_timestep}, episode: {total_episodes_completed}")
        run_evaluation(file_path, global_timestep, total_episodes_completed)
        return result

    ActualPPOAgent.save_model = patched_save_model_actual
    print("Successfully patched keisei.ppo_agent.PPOAgent.save_model for periodic evaluation.")
except ImportError:
    print("Could not import keisei.ppo_agent.PPOAgent for patching. Periodic evaluation hook might not work.", file=sys.stderr)
except AttributeError:
    print("Could not find save_model method on keisei.ppo_agent.PPOAgent for patching.", file=sys.stderr)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    keisei.train.main()
