# Configuration for DRL Shogi Client

TOTAL_TIMESTEPS = 10_000_000
STEPS_PER_EPOCH = 4096
PPO_EPOCHS = 15  # Increased for more stable updates
MINIBATCH_SIZE = 128  # Larger batch for stability
LEARNING_RATE = 1e-4  # Lower learning rate for deeper net
GAMMA = 0.995  # Slightly higher discount for longer games
CLIP_EPSILON = 0.2
LAMBDA_GAE = 0.95
ENTROPY_COEFF = 0.01
MAX_MOVES_PER_GAME = 512
INPUT_CHANNELS = 46  # As per DESIGN.md
NUM_ACTIONS_TOTAL = 3159  # As per DESIGN.md (update if PolicyOutputMapper changes)
SAVE_FREQ_EPISODES = 100
DEVICE = "cpu"  # Use 'cuda' if GPU is available

MODEL_DIR = "models/"
LOG_DIR = "logs/"
LOG_FILE = LOG_DIR + "shogi_training_log.txt"
