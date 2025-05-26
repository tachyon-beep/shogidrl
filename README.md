<!-- filepath: /home/john/keisei/README.md -->
# Keisei: A Deep Reinforcement Learning Shogi AI

**Keisei** (形勢, Japanese for "position" or "situation" in a game like Shogi or Go) is a Deep Reinforcement Learning (DRL) project designed to master the game of Shogi from scratch. It learns exclusively through self-play, utilizing the Proximal Policy Optimization (PPO) algorithm. This project aims to explore the capabilities of DRL in complex, perfect-information games without relying on human-defined heuristics, opening books, or traditional evaluation functions beyond win/loss/draw.

## Table of Contents

- [Project Overview](#project-overview)
- [Core Philosophy](#core-philosophy)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
  - [Core Components](#core-components)
  - [Data Flow & Training Loop](#data-flow--training-loop)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Configuration](#configuration)
  - [Running Training](#running-training)
  - [Monitoring Training](#monitoring-training)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
  - [State Representation](#state-representation)
  - [Action Representation](#action-representation)
  - [Neural Network Architecture](#neural-network-architecture)
- [Operations and Future Development](#operations-and-future-development)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Keisei implements a complete Shogi game environment and a PPO-based agent that learns to play Shogi. The AI starts with no prior knowledge of Shogi strategy and improves by playing games against itself, learning from the outcomes of those games.

For a comprehensive understanding of the project's design and operational plans, please refer to:
- [Design Document (`docs/DESIGN.md`)](docs/DESIGN.md)
- [Operations & Optimization Plan (`OPS_PLAN.md`)](OPS_PLAN.md)
- [Code Audit Report (`docs/code_audit/`)](docs/code_audit/)

## Core Philosophy

- **Learn from Scratch:** The AI does not use any hardcoded opening books or human-designed evaluation functions.
- **Self-Play:** Strategies are discovered solely through self-play and reinforcement learning.
- **PPO Algorithm:** Utilizes Proximal Policy Optimization for stable and efficient learning.

## Key Features

- **Full Shogi Implementation:** Supports all standard Shogi rules, including drops, promotions, and repetition detection (Sennichite).
- **Modular Codebase:** Clearly defined modules for the Shogi engine, neural network, PPO agent, experience buffer, and utilities.
- **PyTorch-Based:** The neural network and PPO algorithm are implemented using PyTorch.
- **Configurable Training:** Hyperparameters and training settings are centralized in `config.py`.
- **Logging & Monitoring:** Comprehensive logging to console, log files, and integration with Weights & Biases (W&B) for experiment tracking.
- **Model Checkpointing:** Saves model progress periodically, allowing for resumption of training and evaluation of different versions.
- **Reproducible Training:** Designed to support reproducible training runs.

## System Architecture

The system follows a standard DRL loop:

1.  **Environment (Shogi Game):** The AI plays against itself.
2.  **Agent (PPO):** Observes the game state, chooses a move.
3.  **Experience Collection:** Game transitions `(state, action, reward, next_state, done, log_prob_action, value_estimate)` are stored.
4.  **Learning:** The PPO agent uses collected experiences to update its policy and value networks.

### Core Components

-   **Shogi Game Environment (`keisei/shogi/shogi_engine.py`):** Manages Shogi game logic, state, legal moves, and rule enforcement.
-   **Neural Network - ActorCritic (`keisei/neural_network.py`):** A PyTorch-based model (ResNet-like) that outputs a policy (move probabilities) and a value (expected future reward).
-   **Policy Output Mapper (`keisei/utils.py` - `PolicyOutputMapper`):** Maps the neural network's output to legal Shogi moves and vice-versa.
-   **PPO Agent (`keisei/ppo_agent.py`):** Implements the PPO algorithm, managing action selection and model learning.
-   **Experience Buffer (`keisei/experience_buffer.py`):** Stores game experiences for training, implementing Generalized Advantage Estimation (GAE).
-   **Training Orchestration (`train.py`):** The main script that drives the training loop, self-play, learning updates, logging, and model saving.
-   **Configuration (`config.py`):** Centralizes all hyperparameters and settings.

### Data Flow & Training Loop

1.  **Initialization:** `train.py` sets up components, loads configurations, and initializes W&B.
2.  **Self-Play & Experience Collection:** The agent plays Shogi games, storing experiences in the `ExperienceBuffer`.
3.  **Learning Phase:** When enough experiences are collected, the `PPOAgent` updates the `ActorCritic` network.
4.  **Iteration & Checkpointing:** The cycle repeats. Model checkpoints and metrics are logged.

For a detailed architectural overview, see the [Operations & Optimization Plan (`OPS_PLAN.md`)](OPS_PLAN.md).

## Getting Started

Refer to [HOW_TO_USE.md](HOW_TO_USE.md) for detailed instructions.

### Prerequisites

-   Python 3.10+
-   PyTorch (CUDA recommended for GPU acceleration)
-   Dependencies listed in `requirements.txt`

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> # Replace <your-repo-url> with the actual URL
    cd keisei
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Linux/macOS
    # env\\Scripts\\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # For development dependencies (e.g., linters, testing tools)
    pip install -r requirements-dev.txt
    ```
4.  **(Optional) Weights & Biases Setup:**
    If you plan to use W&B for logging:
    - Create a `.env` file in the root directory.
    - Add your W&B API key: `WANDB_API_KEY='your_api_key_here'`
    - Ensure `wandb` is installed (it's in `requirements.txt`).

### Configuration

-   All primary training parameters are located in `config.py`.
-   Key parameters to review: `TOTAL_TIMESTEPS`, `STEPS_PER_EPOCH`, `LEARNING_RATE`, `DEVICE` ("cuda" or "cpu"), `SAVE_FREQ_EPISODES`.
-   W&B settings (project name, entity) are also in `config.py`.

### Running Training

1.  **Activate the virtual environment.**
2.  **Start training:**
    ```bash
    python train.py
    ```

### Monitoring Training

-   **Console Output:** Real-time progress, episode rewards, losses.
-   **Log Files:** Detailed logs are saved in the `logs/` directory (e.g., `logs/shogi_run_<timestamp>/training_log.txt`).
-   **Weights & Biases Dashboard:** If enabled, provides comprehensive experiment tracking, visualizations, and model artifact storage.
-   **Model Checkpoints:** Saved periodically to the `models/` directory (e.g., `models/ppo_shogi_<timestamp>/ppo_shogi_agent_step_X.pth`).

### Evaluation

-   The `train.py` script includes basic evaluation during training.
-   A dedicated `evaluate.py` script can be used for more thorough evaluation against specific model checkpoints or baselines (development may be ongoing).
-   Tests are located in the `tests/` directory and can be run using `pytest`.

## Project Structure

```
keisei/
├── config.py             # Hyperparameters and configuration
├── train.py              # Main training script
├── HOW_TO_USE.md         # Detailed usage instructions
├── OPS_PLAN.md           # Operations and optimization plan
├── README.md             # This file
├── requirements.txt      # Main dependencies
├── requirements-dev.txt  # Development dependencies
├── pyproject.toml        # Project metadata and build configuration
├── pytest.ini            # Pytest configuration
├── .env.example          # Example for .env file (for WANDB_API_KEY)
│
├── keisei/                 # Core library code
│   ├── __init__.py
│   ├── shogi/              # Shogi game engine and rules
│   │   ├── __init__.py
│   │   └── shogi_engine.py # Core game logic, board, pieces, moves
│   │   └── (other shogi related files like constants, move_logic)
│   ├── neural_network.py   # ActorCritic NN model (PyTorch)
│   ├── ppo_agent.py        # PPO algorithm, action selection, learning
│   ├── experience_buffer.py# Replay buffer with GAE
│   ├── utils.py            # PolicyOutputMapper, logging setup, other helpers
│   └── evaluate.py         # (Potentially) Script for evaluating trained models
│
├── docs/                   # Documentation
│   ├── DESIGN.md           # Detailed design document
│   ├── code_audit/         # Code audit reports and related files
│   └── (other markdown files)
│
├── models/                 # Saved trained models
│   └── ppo_shogi_<timestamp>/ # Models from a specific run
│       └── ppo_shogi_agent_step_X.pth
│
├── logs/                   # Training logs
│   └── shogi_run_<timestamp>/   # Logs from a specific run
│       └── training_log.txt
│
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py            # Individual test files
│
├── wandb/                  # Weights & Biases local files (if used)
└── env/                    # Python virtual environment (if created here)
```

## Technical Details

### State Representation

The game state is represented as a multi-channel 3D NumPy array `(Channels, 9, 9)` fed into the neural network. This includes:
- Planes for current player's pieces (promoted and unpromoted).
- Planes for opponent's pieces (promoted and unpromoted).
- Planes for piece counts in hand for both players.
- Planes indicating player to move, repetition counts, etc.
(Refer to `docs/DESIGN.md` for the exact channel definition).

### Action Representation

The policy network outputs logits for a flat vector of all possible actions. This includes:
- Board moves (from each square to each relative direction, with and without promotion).
- Drop moves (each piece type to each square).
The `PolicyOutputMapper` handles the mapping between these canonical actions and legal Shogi moves.
(Refer to `docs/DESIGN.md` for details).

### Neural Network Architecture

An AlphaZero-like architecture is used for the `ActorCritic` model:
- An initial convolutional layer.
- A stack of ResNet blocks.
- Separate policy and value heads:
    - **Policy Head:** Outputs logits for actions.
    - **Value Head:** Outputs a scalar value estimating expected reward (tanh activated to [-1, 1]).

## Operations and Future Development

The `OPS_PLAN.md` outlines ongoing operational tasks, monitoring, optimization strategies, and potential future enhancements, such as:
- Hyperparameter tuning.
- Advanced evaluation against other engines.
- Implementation of a USI (Universal Shogi Interface) for playing against other Shogi GUIs/engines.
- Further rule refinements (e.g., Nyugyoku - King entering opponent's camp).

## Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` (to be created) for guidelines on how to contribute to the project. This would typically include:
- Reporting bugs or suggesting features via GitHub Issues.
- Forking the repository and submitting Pull Requests for code changes.
- Adhering to coding standards (e.g., using `black` for formatting, `flake8` for linting).
- Writing unit tests for new features or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file (to be created, assuming MIT from original README) for details.
