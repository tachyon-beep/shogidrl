# DRL Shogi Client: Operations & Optimization Plan (Post-Green)

## Introduction: System Architecture Overview

This section provides an overview of the DRL Shogi Client's architecture. It is intended to equip an AI agent (or a human developer) with the necessary understanding to effectively implement the operational tasks outlined in this plan.

The DRL Shogi Client is a project aimed at training a Reinforcement Learning agent to play Shogi. The system is built using Python and PyTorch, employing the Proximal Policy Optimization (PPO) algorithm.

### Core Components & Modules:

The system is modular, with key responsibilities handled by distinct components:

1.  **Shogi Game Environment (`keisei/shogi/shogi_engine.py` & related files in `keisei/shogi/`):**
    *   **Role:** Manages the complete Shogi game logic, including board representation, piece movements, legal move generation (`get_legal_moves()`), move execution (`make_move()`), game state tracking (win/loss/draw, `game_over`), and rule enforcement (e.g., checks, checkmates, repetition).
    *   **Key for Ops:** Understanding its API is crucial for implementing new game rules (like Nyugyoku), evaluating the agent, and potentially modifying game dynamics. `ShogiGame.get_observation()` provides the state representation for the RL agent.

2.  **Neural Network - ActorCritic (`keisei/neural_network.py`):**
    *   **Role:** Defines the `ActorCritic` model, a deep neural network (likely a ResNet-based architecture) implemented in PyTorch.
    *   **Input:** Takes the game state observation from `ShogiGame.get_observation()`.
    *   **Output:** Produces two outputs:
        *   **Policy (Actor):** A probability distribution over all possible legal moves.
        *   **Value (Critic):** An estimate of the expected future reward from the current state.
    *   **Key for Ops:** Architectural changes (number of layers, types of layers, activation functions) and hyperparameter tuning related to the network are performed here.

3.  **Policy Output Mapper (`keisei/utils.py` - `PolicyOutputMapper` class):**
    *   **Role:** This crucial component acts as an interface between the raw output of the neural network's policy head and the Shogi game engine's action space. It maps the network's output tensor (e.g., a flat vector of probabilities for all possible moves in a canonical representation) to specific, legal Shogi moves that can be understood and executed by `ShogiGame`. It also handles the reverse: mapping game actions to a format suitable for network input/training if needed.
    *   **Key for Ops:** Ensuring its correct alignment with `ShogiGame.get_observation()` and the `ActorCritic` network's input/output dimensions is vital, especially when modifying the network or observation space.

4.  **PPO Agent (`keisei/ppo_agent.py`):**
    *   **Role:** Implements the Proximal Policy Optimization (PPO) algorithm. It orchestrates the interaction between the agent and the game environment.
    *   **Functions:**
        *   `select_action()`: Chooses an action based on the policy from the `ActorCritic` network during self-play or evaluation.
        *   `learn()`: Updates the `ActorCritic` network parameters using experiences collected in the `ExperienceBuffer`. (Now returns average losses for logging).
    *   **Key for Ops:** PPO-specific hyperparameters (e.g., `CLIP_EPSILON`, `PPO_EPOCHS`, `ENTROPY_COEFF`, `VALUE_LOSS_COEFF`) are managed here or passed from `config.py`.

5.  **Experience Buffer (`keisei/experience_buffer.py`):**
    *   **Role:** Stores trajectories of experiences (state, action, reward, next state, done, log probability of action, value estimate, advantages, returns) collected during self-play.
    *   **Function:** Provides batches of these experiences to the `PPOAgent` for learning. It now handles the calculation of Generalized Advantage Estimation (GAE) and returns via `compute_advantages_and_returns()`.
    *   **Key for Ops:** Its size (`STEPS_PER_EPOCH` or rollout buffer size) is an important hyperparameter. Optimized `get_batch` method.

6.  **Training Orchestration (`train.py`):**
    *   **Role:** The main script that drives the entire training loop.
    *   **Responsibilities:**
        *   Initializes the game environment, `ActorCritic` network, `PPOAgent`, and `ExperienceBuffer`.
        *   Manages the self-play loop where the agent plays against itself to generate experiences (now timestep-based).
        *   Calls `PPOAgent.learn()` to update the model and logs returned losses.
        *   Handles model checkpoint saving (e.g., to the `models/` directory) at specified frequencies (`SAVE_FREQ_EPISODES`).
        *   **Integrated with Weights & Biases (W&B):** Configures and manages logging to W&B, including API key loading from `.env`, `wandb.init()`, logging of episode metrics, PPO update metrics, and model artifacts. Calls `wandb.finish()`.
    *   **Key for Ops:** This is the primary entry point for starting training runs, adjusting high-level training parameters, and monitoring via W&B.

7.  **Configuration (`config.py` - root-level at `/home/john/keisei/config.py`):**
    *   **Role:** Centralizes hyperparameters and settings for the training process, neural network architecture, PPO algorithm, and game environment.
    *   **Examples:** `TOTAL_TIMESTEPS`, `STEPS_PER_EPOCH`, `LEARNING_RATE`, `GAMMA`, `BATCH_SIZE`, network layer definitions, `MAX_MOVES_PER_GAME`, `VALUE_LOSS_COEFF`, `EVAL_FREQ_TIMESTEPS`, `EVAL_NUM_GAMES`.
    *   **Key for Ops:** Most hyperparameter tuning tasks will involve modifying values in this file. Hyperparameters updated for initial PPO run.

### Data Flow & Training Loop Overview:

1.  **Initialization:** `train.py` sets up all components, loads configurations from `config.py`, and initializes W&B (loading API key from `.env` via `python-dotenv`).
2.  **Self-Play & Experience Collection (Timestep-based):**
    *   The `PPOAgent`, using the current `ActorCritic` network, plays Shogi games (via `ShogiGame`).
    *   For each step, `ShogiGame.get_observation()` provides the current state.
    *   The `ActorCritic` network processes this observation to output a policy and a value.
    *   `PPOAgent.select_action()` (using `PolicyOutputMapper`) chooses a move.
    *   `ShogiGame.make_move()` applies the move, and the game returns the next state, reward, and done signal.
    *   This (state, action, reward, next_state, done, log_prob, value) tuple is stored in the `ExperienceBuffer`.
3.  **Learning Phase (Triggered by `STEPS_PER_EPOCH`):**
    *   When the `ExperienceBuffer` collects `STEPS_PER_EPOCH` experiences:
        *   The value of the last state is estimated using `agent.get_value()`.
        *   `ExperienceBuffer.compute_advantages_and_returns()` is called.
        *   `PPOAgent.learn()` is called, which iterates over the data in minibatches for `PPO_EPOCHS`, calculates losses, and updates the `ActorCritic` network. Average losses are returned.
        *   The `ExperienceBuffer` is cleared.
4.  **Iteration & Checkpointing:**
    *   The self-play and learning phases repeat for `TOTAL_TIMESTEPS`.
    *   Model checkpoints are saved periodically by `train.py` (and logged as W&B artifacts).
    *   Metrics (episode stats, PPO losses) are logged to W&B throughout the process.
5.  **Termination:** `wandb.finish()` is called at the end of training.

### Key Files & Entry Points for Operations:

*   **Configuration:** `/home/john/keisei/config.py`
*   **Main Training Script:** `/home/john/keisei/train.py`
*   **Shogi Game Logic:** `/home/john/keisei/keisei/shogi/shogi_engine.py` (and other files in `/home/john/keisei/keisei/shogi/`)
*   **Neural Network Definition:** `/home/john/keisei/keisei/neural_network.py`
*   **PPO Algorithm:** `/home/john/keisei/keisei/ppo_agent.py`
*   **Experience Storage:** `/home/john/keisei/keisei/experience_buffer.py`
*   **Utilities (incl. PolicyOutputMapper):** `/home/john/keisei/keisei/utils.py`
*   **Evaluation Script (to be enhanced/created):** `/home/john/keisei/evaluate.py` (expected location)
*   **USI Interface (to be created):** `/home/john/keisei/usi_engine.py` (expected location)
*   **Logs:** `/home/john/keisei/logs/training_log.txt` (and W&B Dashboard)
*   **Saved Models:** `/home/john/keisei/models/` (and W&B Artifacts)
*   **Environment Variables:** `.env` (for `WANDB_API_KEY`)
*   **Requirements:** `requirements.txt` (includes `wandb`, `python-dotenv`)

Understanding these components and their interactions is fundamental to executing the tasks in this Operations & Optimization Plan.

This document outlines the steps for operating, monitoring, and optimizing the DRL Shogi Client now that the core PPO implementation and W&B integration are complete.
**Note:** Prioritization of these steps will be crucial. Consider impact vs. effort for each.

## Current Status (as of 2025-05-18)

*   **PPO Algorithm Implementation:** Completed (Phases 1-3 of original plan). `ExperienceBuffer` enhanced, `PPOAgent` updated with `learn` method, `train.py` refactored for PPO loop.
*   **Configuration Management:** Consolidated into `/home/john/keisei/config.py`. New config values (`VALUE_LOSS_COEFF`, `EVAL_FREQ_TIMESTEPS`, `EVAL_NUM_GAMES`) added. Hyperparameters updated for an initial PPO run.
*   **Weights & Biases Integration:** Completed. API key in `.env`, `wandb` and `python-dotenv` in `requirements.txt`, W
