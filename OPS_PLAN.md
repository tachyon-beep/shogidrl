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

3.  **Policy Output Mapper (Likely within `keisei/utils.py`, `keisei/ppo_agent.py` or `keisei/neural_network.py`):**
    *   **Role:** This crucial component acts as an interface between the raw output of the neural network's policy head and the Shogi game engine's action space. It maps the network's output tensor (e.g., a flat vector of probabilities for all possible moves in a canonical representation) to specific, legal Shogi moves that can be understood and executed by `ShogiGame`. It also handles the reverse: mapping game actions to a format suitable for network input/training if needed.
    *   **Key for Ops:** Ensuring its correct alignment with `ShogiGame.get_observation()` and the `ActorCritic` network's input/output dimensions is vital, especially when modifying the network or observation space.

4.  **PPO Agent (`keisei/ppo_agent.py`):**
    *   **Role:** Implements the Proximal Policy Optimization (PPO) algorithm. It orchestrates the interaction between the agent and the game environment.
    *   **Functions:**
        *   `select_action()`: Chooses an action based on the policy from the `ActorCritic` network during self-play or evaluation.
        *   `learn()`: Updates the `ActorCritic` network parameters using experiences collected in the `ExperienceBuffer`.
    *   **Key for Ops:** PPO-specific hyperparameters (e.g., `CLIP_EPSILON`, `PPO_EPOCHS`, `ENTROPY_COEFF`) are managed here or passed from `config.py`. Advanced RL techniques might involve modifying this agent.

5.  **Experience Buffer (`keisei/experience_buffer.py`):**
    *   **Role:** Stores trajectories of experiences (state, action, reward, next state, done, log probability of action, value estimate) collected during self-play.
    *   **Function:** Provides batches of these experiences to the `PPOAgent` for learning. It also typically handles the calculation of Generalized Advantage Estimation (GAE).
    *   **Key for Ops:** Its size (`STEPS_PER_EPOCH` or rollout buffer size) is an important hyperparameter. Profiling its operations can be important for performance.

6.  **Training Orchestration (`train.py`):**
    *   **Role:** The main script that drives the entire training loop.
    *   **Responsibilities:**
        *   Initializes the game environment, `ActorCritic` network, `PPOAgent`, and `ExperienceBuffer`.
        *   Manages the self-play loop where the agent plays against itself (or a version of itself) to generate experiences.
        *   Calls `PPOAgent.learn()` to update the model.
        *   Handles model checkpoint saving (e.g., to the `models/` directory) at specified frequencies (`SAVE_FREQ_EPISODES`).
        *   Configures and manages logging (e.g., to `logs/train.log` and potentially TensorBoard/Weights & Biases).
    *   **Key for Ops:** This is the primary entry point for starting training runs, adjusting high-level training parameters, and integrating monitoring tools.

7.  **Configuration (`config.py` - primarily the root-level one at `/home/john/keisei/config.py`):**
    *   **Role:** Centralizes hyperparameters and settings for the training process, neural network architecture, PPO algorithm, and game environment.
    *   **Examples:** `TOTAL_TIMESTEPS`, `STEPS_PER_EPOCH`, `LEARNING_RATE`, `GAMMA`, `BATCH_SIZE`, network layer definitions, `MAX_MOVES_PER_GAME`.
    *   **Key for Ops:** Most hyperparameter tuning tasks will involve modifying values in this file.

### Data Flow & Training Loop Overview:

1.  **Initialization:** `train.py` sets up all components and loads configurations from `config.py`.
2.  **Self-Play & Experience Collection:**
    *   The `PPOAgent`, using the current `ActorCritic` network, plays Shogi games (via `ShogiGame`).
    *   For each step, `ShogiGame.get_observation()` provides the current state.
    *   The `ActorCritic` network processes this observation to output a policy and a value.
    *   `PPOAgent.select_action()` (potentially using `PolicyOutputMapper`) chooses a move.
    *   `ShogiGame.make_move()` applies the move, and the game returns the next state, reward, and done signal.
    *   This (state, action, reward, next_state, done, log_prob, value) tuple is stored in the `ExperienceBuffer`.
3.  **Learning Phase:**
    *   Once the `ExperienceBuffer` collects enough data (e.g., `STEPS_PER_EPOCH`), the `PPOAgent.learn()` method is called.
    *   It computes advantages (e.g., GAE) using the collected rewards and value estimates.
    *   It iterates over the data in minibatches for a certain number of `PPO_EPOCHS`.
    *   In each iteration, it calculates policy and value losses and updates the `ActorCritic` network weights via backpropagation.
4.  **Iteration & Checkpointing:**
    *   The self-play and learning phases repeat for `TOTAL_TIMESTEPS`.
    *   Model checkpoints are saved periodically by `train.py`.
    *   Metrics are logged throughout the process.

### Key Files & Entry Points for Operations:

*   **Configuration:** `/home/john/keisei/config.py`
*   **Main Training Script:** `/home/john/keisei/train.py`
*   **Shogi Game Logic:** `/home/john/keisei/keisei/shogi/shogi_engine.py` (and other files in `/home/john/keisei/keisei/shogi/`)
*   **Neural Network Definition:** `/home/john/keisei/keisei/neural_network.py`
*   **PPO Algorithm:** `/home/john/keisei/keisei/ppo_agent.py`
*   **Experience Storage:** `/home/john/keisei/keisei/experience_buffer.py`
*   **Evaluation Script (to be enhanced/created):** `/home/john/keisei/evaluate.py` (expected location)
*   **USI Interface (to be created):** `/home/john/keisei/usi_engine.py` (expected location)
*   **Logs:** `/home/john/keisei/logs/train.log`
*   **Saved Models:** `/home/john/keisei/models/` (expected location)

Understanding these components and their interactions is fundamental to executing the tasks in this Operations & Optimization Plan.

This document outlines the steps for operating, monitoring, and optimizing the DRL Shogi Client now that the core implementation is complete and all initial tests are passing ("Green").
**Note:** Prioritization of these steps will be crucial. Consider impact vs. effort for each.

## 1. Initial Training & Baseline Performance Monitoring

**Goal:** Establish a baseline performance and identify initial areas for optimization.

- **1.1. Configure Initial Training Run:**
    - Verify default hyperparameters in `config.py` and `train.py` are sensible starting points (e.g., `TOTAL_TIMESTEPS`, `STEPS_PER_EPOCH`, `LEARNING_RATE`, `PPO_EPOCHS`, `MINIBATCH_SIZE`, `GAMMA`, `CLIP_EPSILON`, `LAMBDA_GAE`, `ENTROPY_COEFF`).
    - **Final Check:** Ensure `PolicyOutputMapper` and `ShogiGame.get_observation()` outputs are correctly aligned with the `ActorCritic` network's input/output dimensions.
    - Ensure logging is configured correctly in `train.py` to capture key metrics (episode rewards, win/loss/draw rates, policy loss, value loss, entropy).
    - Set up `SAVE_FREQ_EPISODES` in `train.py` to save model checkpoints regularly. Consider also saving based on `TOTAL_TIMESTEPS` (e.g., every 1M steps) for flexibility, as episode length can vary.
- **1.2. Execute Initial Training Run:**
    - Run `train.py` on the target environment (local machine with/without GPU, or cloud instance).
    - Monitor the console output and log files (`logs/train.log`) for any immediate errors or unexpected behavior.
- **1.3. Basic Performance Analysis:**
    - After a significant number of timesteps (e.g., 1M-5M, or a few hundred episodes), analyze the logs:
        - Is the agent learning at all? (e.g., are average rewards increasing, is win rate against a random or fixed opponent improving if applicable?).
        - **Self-Play Proxies:** If only self-play, look for more decisive games (fewer draws due to max moves) or changes in average game length.
        - Are policy loss, value loss, and entropy behaving as expected? (e.g., losses decreasing, entropy decreasing slowly).
        - **Log Critic's Average Value Estimate:** Monitor if it correlates with outcomes or stays near 0/fluctuates wildly.
        - How long does training take per N steps/episodes?
- **1.4. Implement Robust Evaluation Script (Highly Recommended):**
    - Create/enhance `evaluate.py` to load any model checkpoint and play a set number of games against a fixed baseline (e.g., random agent, very early checkpoint).
    - This script should clearly log its results (win/loss/draw counts, average rewards per game) to objectively track ELO-like progress, decoupled from training dynamics.

## 2. Hyperparameter Tuning & Neural Network Architecture Refinement

**Goal:** Systematically improve learning efficiency and final agent performance. This is an iterative process.

- **2.1. Identify Key Hyperparameters for Tuning:**
    - **PPO Specific:** `LEARNING_RATE`, `CLIP_EPSILON`, `GAMMA`, `LAMBDA_GAE`, `ENTROPY_COEFF`, `PPO_EPOCHS`, `MINIBATCH_SIZE`.
    - **Network Architecture:** Number of ResNet blocks, number of filters in convolutional layers, activation functions.
    - **Training Regime:** `STEPS_PER_EPOCH` (rollout buffer size).
    - **Game Dynamics:** `MAX_MOVES_PER_GAME` (if games end too often/late due to this limit).
- **2.2. Choose a Tuning Strategy:**
    - **Manual/Grid Search:** For a few key parameters.
    - **Random Search:** Often more effective than grid search.
    - **Automated Hyperparameter Optimization (HPO) tools:** (e.g., Optuna, Ray Tune, Weights & Biases Sweeps). If W&B is integrated, Sweeps is a natural fit.
- **2.3. Iterative Tuning Cycles:**
    - For each set of hyperparameters/architecture change:
        - Run training for a fixed number of timesteps (shorter than a full run, but long enough to see trends).
        - Evaluate performance using the evaluation script (1.4) or by observing training metrics.
        - Log results meticulously (e.g., using W&B/TensorBoard, with clear run naming/tagging).
        - Adjust parameters based on results and repeat.
- **2.4. Neural Network Architecture Experiments:**
    - Based on literature or intuition, try modifications to `ActorCritic` in `neural_network.py`:
        - Start with simpler changes: Varying number of residual blocks, filter sizes, adding/removing batch norm layers.
        - Then consider: Deeper or wider networks.
        - Changes to the structure of policy and value heads.
        - (Advanced) Different types of layers (e.g., attention mechanisms).
    - Ensure `input_channels` and `num_actions_total` are correctly passed.
    - Test any architectural changes thoroughly.
    - (Advanced) Consider if the current observation space (`ShogiGame.get_observation()`) is optimal or if more/different features could help.

## 3. Performance Profiling & Optimization of Critical Code Paths

**Goal:** Reduce training time by identifying and optimizing bottlenecks.

- **3.1. Profile Key Components:**
    - Use Python\'s `cProfile` or other profiling tools (e.g., `py-spy`, `line_profiler`).
    - Focus on:
        - `ShogiGame.get_legal_moves()`
        - `ShogiGame.make_move()`
        - `ShogiGame.get_observation()`
        - `PolicyOutputMapper` methods
        - `PPOAgent.learn()` (especially the minibatch loop and network forward/backward passes)
        - `ExperienceBuffer` operations (especially GAE calculation if not fully vectorized).
        - Data transfer between CPU/GPU if applicable.
- **3.2. Identify Bottlenecks:**
    - Analyze profiling results to find functions or operations consuming the most time.
- **3.3. Implement Optimizations:**
    - **Algorithmic improvements:** Can `get_legal_moves` be made more efficient? Are there redundant calculations?
    - **Vectorization:** Utilize NumPy/PyTorch vectorized operations where possible instead of Python loops. Optimize Python logic *around* PyTorch calls first.
    - **Caching:** If pure functions are called repeatedly with the same arguments.
    - **(Advanced) Consider JIT compilation (e.g., Numba) for critical pure Python/NumPy loops if appropriate and PyTorch compatibility is maintained. Be cautious with PyTorch tensor operations.**
    - **(Advanced) Consider parallelizing self-play game generation** if multiple CPU cores are available and game generation is a bottleneck.
- **3.4. Re-profile and Validate:**
    - After optimizations, re-profile to confirm improvements and ensure no new bottlenecks were introduced.
    - Ensure all tests still pass.

## 4. Advanced Game Termination Rules & Features

**Goal:** Enhance the Shogi simulation with more complete rules and add usability features.

- **4.1. Implement Nyugyoku (Try Rule):**
    - **Design:**
        - Define the conditions for Nyugyoku based on official Shogi rules (king in promotion zone, piece counts, etc.).
        - Define the reward signal for Nyugyoku outcomes (e.g., +0.5 for a point win, 0 for a draw, or based on game rules).
    - **Implementation in `ShogiGame`:**
        - Add logic to `make_move()` or a dedicated check function to detect Nyugyoku conditions.
        - Update game termination (`self.game_over`, `self.winner`) accordingly.
    - **Testing:** Write new unit tests in `test_shogi_engine.py` or `test_shogi_rules_logic.py` for various Nyugyoku scenarios.
- **4.2. Add Support for USI (Universal Shogi Interface) Protocol:**
    - **Research:** Understand the USI protocol commands (e.g., `usi`, `isready`, `setoption`, `usinewgame`, `position`, `go`, `stop`).
    - **Design:** Create a new module or class to handle USI communication. This will involve parsing USI commands and translating them into agent actions, and formatting agent moves/info back into USI.
    - **Integration:**
        - The USI handler will need to interact with `ShogiGame` (to set up positions) and `PPOAgent` (to get moves).
        - A new entry point script (e.g., `usi_engine.py`) will likely be needed, distinct from `train.py`.
    - **Testing:** Requires a USI-compatible GUI or test harness.
- **4.3. Implement Self-Play Against Older Versions of the Agent:**
    - **Mechanism:**
        - Periodically save model checkpoints (already in place).
        - During experience generation, a certain percentage of games can be played against a previously saved checkpoint (the "opponent").
        - The opponent model would also use `PPOAgent.select_action` but with its loaded weights and in `eval` mode.
    - **Implementation in `train.py`:**
        - Modify the game setup part of the training loop.
        - Load an older model for the opponent.
        - Alternate which agent\'s turn it is and feed observations accordingly.
    - **Considerations:**
        - How often to update the opponent pool.
        - How many distinct older opponents to maintain.
        - The selection strategy for choosing an opponent (e.g., uniformly random, bias towards stronger/more recent).

## 5. Advanced RL Techniques & Exploration

**Goal:** Further improve learning stability and sample efficiency.

- **5.1. Implement a More Sophisticated MCTS for Action Selection (if PPO alone is insufficient):**
    - **Research:** Review AlphaZero-style MCTS integration with policy/value networks.
    - **Design:**
        - `MCTSNode` class.
        - MCTS search function that uses the `ActorCritic` network to guide simulations (policy network provides priors for MCTS, value network evaluates leaf nodes).
    - **Integration with `PPOAgent`:**
        - `PPOAgent.select_action` would invoke the MCTS search for a number of simulations, then choose an action based on visit counts.
        - The policy target for training the network would be derived from MCTS visit counts.
    - **Complexity:** This is a very significant addition.
- **5.2. Implement/Enhance Detailed Experiment Logging (e.g., TensorBoard/Weights & Biases):**
    - **Setup:** If not already fully integrated, set up or enhance integration with a tool like TensorBoard (`torch.utils.tensorboard.SummaryWriter`) or Weights & Biases in `train.py`. Choose one and standardize.
    - **Metrics to Log:**
        - Episode rewards (scalar, histogram).
        - Win/loss/draw rates (against baseline and in self-play).
        - Policy loss, value loss, entropy (scalars).
        - Hyperparameters for each run.
        - Average value estimates from the critic.
        - Distributions of action probabilities.
        - (Advanced) Model graph, weight distributions/histograms, gradients or weight norms.
        - (Advanced) Custom charts or reports.
    - **Usage:** Regularly review these logs to understand training dynamics.

## 6. Ongoing Maintenance & Iteration
- Regularly review training progress and logged metrics.
- Revisit previous steps (hyperparameter tuning, profiling) as the agent improves or plateaus.
- Keep dependencies updated (e.g., PyTorch, NumPy).
- Address any new bugs or issues that arise.
- **Version Control:** Maintain good git hygiene: commit changes regularly with clear messages, use branches for experiments, and tag important releases or model checkpoints.
- **Documentation:** Update design documents and code comments as new features or significant changes are made.
- **(Future) Automated Testing/CI:** Consider setting up a Continuous Integration pipeline (e.g., GitHub Actions) to automatically run tests on commits/PRs if the project grows or involves multiple contributors.
