"""
Configuration for Deep Reinforcement Learning (DRL) Shogi Client.

This module contains all configuration parameters used throughout the Keisei project,
including model hyperparameters, training settings, file paths, and game constants.
"""

TOTAL_TIMESTEPS = 10_000_000
STEPS_PER_EPOCH = 4096
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
LAMBDA_GAE = 0.95
ENTROPY_COEFF = 0.01
MAX_MOVES_PER_GAME = 512
INPUT_CHANNELS = 46  # As per DESIGN.md
NUM_ACTIONS_TOTAL = 3159  # As per DESIGN.md (update if PolicyOutputMapper changes)
SAVE_FREQ_EPISODES = 100
DEVICE = "cuda"  # or 'cpu'
VALUE_LOSS_COEFF = 0.5  # Coefficient for the value loss in PPO

MODEL_DIR = "models/"
LOG_FILE = "logs/training_log.txt"

# Evaluation parameters
EVAL_FREQ_TIMESTEPS = 20000  # Evaluate every N global timesteps
EVAL_NUM_GAMES = 10  # Number of games to play during evaluation
