"""
Configuration for Deep Reinforcement Learning (DRL) Shogi Client (Keisei).

This module consolidates all configuration parameters for the project,
including core environment settings, training hyperparameters, evaluation setup,
file paths, and debugging/display options.
"""

# --- Core Environment & Agent Settings ---
DEVICE = "cpu"  # Default device ('cpu' or 'cuda'). Dynamically updated in train.py if CUDA is available.
INPUT_CHANNELS = 46  # Number of input channels for the neural network (from observation space).
NUM_ACTIONS_TOTAL = 3159  # Total number of possible actions (from PolicyOutputMapper).

# --- Paths & File Logging ---
MODEL_DIR = "models/"  # Directory to save model checkpoints.
LOG_FILE = "logs/training_log.txt"  # Path for the main file-based training log.

# --- Training Loop Hyperparameters ---
TOTAL_TIMESTEPS = 500000  # Total number of environment steps for training.
STEPS_PER_EPOCH = 2048  # Number of steps collected per agent-environment interaction cycle before learning (PPO buffer size).
PPO_EPOCHS = 10  # Number of optimization epochs over the collected batch per PPO update.
MINIBATCH_SIZE = 64  # Size of minibatches used during PPO updates.

# --- PPO Algorithm Hyperparameters ---
LEARNING_RATE = 0.0003  # Learning rate for the Adam optimizer (3e-4).
GAMMA = 0.99  # Discount factor for future rewards.
CLIP_EPSILON = 0.2  # PPO clipping parameter for the surrogate objective.
LAMBDA_GAE = 0.95  # Lambda parameter for Generalized Advantage Estimation (GAE).
ENTROPY_COEFF = 0.01  # Coefficient for the entropy bonus in the PPO loss, encouraging exploration.
VALUE_LOSS_COEFF = 0.5  # Coefficient for the value function loss in the PPO objective.

# --- Game Rules & Constraints ---
MAX_MOVES_PER_GAME = 500  # Maximum moves allowed in a single training game before it's considered a draw/timeout.
MAX_MOVES_PER_GAME_EVAL = 256 # Maximum moves for evaluation games (can differ from training).

# --- Model Saving & Checkpointing ---
SAVE_FREQ_EPISODES = 200  # Frequency (in episodes) for saving model checkpoints during training.

# --- Periodic Evaluation Settings (during training) ---
EVAL_DURING_TRAINING = True  # Master switch to enable/disable periodic evaluation during training.
# Note: Periodic evaluation is triggered after model saving, i.e., every `SAVE_FREQ_EPISODES`.
# The `EVAL_FREQ_EPISODES` below is kept for reference but currently not the primary trigger for periodic eval.
EVAL_FREQ_EPISODES = 100  # If a separate evaluation frequency independent of saving is ever needed.
EVAL_NUM_GAMES = 10  # Number of games to play during each periodic evaluation session.
EVAL_OPPONENT_TYPE = "heuristic"  # Opponent for periodic evals: "random", "heuristic", or "ppo".
EVAL_OPPONENT_CHECKPOINT_PATH = None  # Path to opponent PPO checkpoint if EVAL_OPPONENT_TYPE is "ppo".
EVAL_DEVICE = "cpu"  # Device for periodic evaluation runs ("cpu" or "cuda").

# --- Weights & Biases (W&B) Integration Settings ---
# For the Main Training Loop (in keisei/train.py)
WANDB_LOG_TRAIN = True  # Master switch to enable W&B logging for the main training process.
WANDB_PROJECT_TRAIN = "shogi-drl"  # Default W&B project name for main training runs.
WANDB_ENTITY_TRAIN = None  # Your W&B entity (username/team). None uses W&B default.

# For Periodic Evaluation Runs (triggered by keisei/train.py calling evaluate.py)
EVAL_WANDB_LOG = True  # Enable W&B logging for periodic evaluation runs.
EVAL_WANDB_PROJECT = "shogi-drl-periodic-evaluation"  # W&B project for periodic evaluation runs.
EVAL_WANDB_ENTITY = None  # W&B entity for periodic evaluation runs. None uses W&B default.
EVAL_WANDB_RUN_NAME_PREFIX = "periodic_eval_"  # Prefix for W&B run names for evaluation runs.

# --- Debugging & Display Settings ---
PRINT_GAME_REAL_TIME = False  # If True, prints game states/moves to console. Significantly slows down training.
REAL_TIME_PRINT_DELAY = 0.5  # Delay in seconds between moves when PRINT_GAME_REAL_TIME is True.
