"""
Configuration for Deep Reinforcement Learning (DRL) Shogi Client.

This module contains all configuration parameters used throughout the Keisei project,
including model hyperparameters, training settings, file paths, and game constants.
"""

# --- Training Hyperparameters ---
TOTAL_TIMESTEPS = 500000
STEPS_PER_EPOCH = 2048         # Aka Rollout buffer size / PPO Buffer size
PPO_EPOCHS = 10                # Number of optimization epochs per PPO update
MINIBATCH_SIZE = 64
LEARNING_RATE = 0.0003         # 3e-4
GAMMA = 0.99                   # Discount factor
CLIP_EPSILON = 0.2             # PPO clipping parameter
LAMBDA_GAE = 0.95              # GAE lambda parameter
ENTROPY_COEFF = 0.01           # Entropy bonus coefficient
VALUE_LOSS_COEFF = 0.5         # Coefficient for the value loss in PPO (already present, kept)

# --- Game Specific ---
MAX_MOVES_PER_GAME = 512

# --- Network Architecture Related (Verification, not typically tuned here directly) ---
INPUT_CHANNELS = 46  # As per DESIGN.md - determined by get_observation()
NUM_ACTIONS_TOTAL = 3159  # As per DESIGN.md - determined by PolicyOutputMapper
# NUM_RESNET_BLOCKS and NUM_FILTERS are part of the model definition in neural_network.py

# --- Model Saving ---
SAVE_FREQ_EPISODES = 200       # Save model every N episodes

# --- Environment & Paths ---
DEVICE = "cuda"  # or 'cpu' - will be dynamically set to cuda if available, else cpu in train.py typically
MODEL_DIR = "models/"
LOG_FILE = "logs/training_log.txt"

# --- Evaluation parameters ---
EVAL_FREQ_TIMESTEPS = 20000  # Evaluate every N global timesteps (adjust if needed based on new TOTAL_TIMESTEPS)
EVAL_NUM_GAMES = 10          # Number of games to play during evaluation
