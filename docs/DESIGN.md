Okay, this is an excellent consolidation of requirements and design considerations. I will now integrate both sets of information into a single, highly detailed design document.

**DRL Shogi Client: Detailed Design Document**

**1. Core Philosophy: "Learn from Scratch"**

*   No hardcoded open*   `reset(self)`: Initializes board to starting position (see **4.A.1 Initial Board Setup**), clears hands, sets `current_player` to 0 (Black), clears `move_history`, `game_over` to False, `winner` to None, `move_count` to 0.ng books.
*   No human-designed evaluation functions (other than win/loss/draw for rewards).
*   The AI should discover strategies solely through self-play and reinforcement learning.

**2. Overall Architecture**

The system revolves around the standard Reinforcement Learning (RL) loop using Proximal Policy Optimization (PPO):

1.  **Environment (Shogi Game):** The AI plays against itself (or a previous version of itself).
2.  **Agent (PPO):** Observes the game state, chooses a move.
3.  **Experience Collection:** Transitions `(state, action, reward, next_state, done, log_prob_action, value_estimate)` are stored.
4.  **Learning:** The PPO agent uses collected experiences to update its policy and value networks.

```
+---------------------+      +----------------------+      +---------------------+
|     Shogi Game      |<---->|   PPO Agent (AI)     |<---->| Experience Buffer   |
|   (Environment)     |      | (Policy & Value Nets)|      | (Stores Transitions)|
+---------------------+      +----------------------+      +---------------------+
         ^                                                           |
         | (State, Reward, Done)                                     | (Batches of Experience)
         |                                                           V
         +---------------------------(Action)------------------------+
                                        |
                                        V
                               +---------------------+
                               |   Training Loop     |
                               | (Orchestrates all)  |
                               +---------------------+
```

**3. Configuration Architecture**

The Keisei system employs a sophisticated Pydantic-based configuration architecture that provides type safety, validation, and comprehensive control over all aspects of training, evaluation, and game management. This configuration system supports advanced features including mixed precision training, distributed data parallel (DDP), gradient clipping, and comprehensive experiment tracking.

**3.A. Configuration Schema (`config_schema.py`)**

The configuration system is built around six main configuration sections, each managed by dedicated Pydantic models:

**3.A.1. Environment Configuration (`EnvConfig`)**
```python
class EnvConfig(BaseModel):
    device: str = "cuda"
    input_channels: int = 46
    action_space_size: int = 6480
    seed: Optional[int] = None
    max_moves_per_game: int = 500
```

Controls the fundamental environment parameters:
- **Device Management**: Automatic GPU/CPU detection and allocation
- **Observation Space**: 46-channel board representation (9x9x46 tensor)
- **Action Space**: 6,480 possible moves covering all legal Shogi moves
- **Game Limits**: Configurable maximum moves per game to prevent infinite games
- **Reproducibility**: Optional seeding for deterministic training runs

**3.A.2. Training Configuration (`TrainingConfig`)**
```python
class TrainingConfig(BaseModel):
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 64
    entropy_coeff: float = 0.01
    lambda_gae: float = 0.95
    gradient_clip_norm: float = 0.5
    mixed_precision: bool = False
    distributed: bool = False
    checkpoint_interval: int = 1000
```

Provides comprehensive control over the PPO training algorithm:
- **Core PPO Parameters**: Learning rate, clipping, GAE lambda
- **Advanced Features**: Mixed precision (AMP), distributed training (DDP)
- **Stability Controls**: Gradient clipping, entropy regularization
- **Checkpoint Management**: Automatic model saving intervals

**3.A.3. Evaluation Configuration (`EvaluationConfig`)**
```python
class EvaluationConfig(BaseModel):
    eval_interval: int = 100
    eval_games: int = 10
    eval_timeout: float = 300.0
    save_eval_games: bool = True
```

Controls evaluation and validation processes:
- **Evaluation Frequency**: How often to run evaluation during training
- **Sample Size**: Number of games to play for statistical significance
- **Performance Monitoring**: Timeout controls and game saving options

**3.A.4. Logging Configuration (`LoggingConfig`)**
```python
class LoggingConfig(BaseModel):
    model_dir: str = "models"
    log_file: Optional[str] = None
    log_level: str = "INFO"
```

Manages experiment tracking and output:
- **Model Storage**: Centralized model checkpoint directory
- **Debug Information**: Configurable logging levels and file output
- **Experiment Organization**: Structured directory naming with timestamps

**3.A.5. Weights & Biases Configuration (`WandBConfig`)**
```python
class WandBConfig(BaseModel):
    enabled: bool = False
    project: str = "keisei-shogi"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = []
    notes: Optional[str] = None
```

Comprehensive experiment tracking integration:
- **Cloud Logging**: Automatic metrics, loss, and performance tracking
- **Experiment Management**: Project organization, tagging, and documentation
- **Visualization**: Real-time training curves and model comparison tools

**3.A.6. Demo Configuration (`DemoConfig`)**
```python
class DemoConfig(BaseModel):
    enabled: bool = False
    model_path: Optional[str] = None
    interactive: bool = True
```

Controls demonstration and inference modes:
- **Model Loading**: Specify trained models for gameplay demonstrations
- **Interaction Modes**: Human vs AI or AI vs AI gameplay options

**3.B. Trainer Manager Architecture**

The training system has been refactored into nine specialized managers, each responsible for specific aspects of the training pipeline. This modular architecture provides clean separation of concerns and enables advanced features like distributed training and mixed precision.

**3.B.1. Session Manager (`session_manager.py`)**
- **Responsibility**: Overall training session lifecycle management
- **Key Features**:
  - Run directory creation with timestamps
  - Weights & Biases integration and setup
  - Configuration validation and persistence
  - Session state management and cleanup

**3.B.2. Model Manager (`model_manager.py`)**
- **Responsibility**: Neural network and optimizer management
- **Key Features**:
  - Model instantiation and device placement
  - Checkpoint saving and loading with versioning
  - Mixed precision (AMP) scaler management
  - Model state validation and recovery

**3.B.3. Environment Manager (`env_manager.py`)**
- **Responsibility**: Game environment and policy mapping
- **Key Features**:
  - Shogi game instance creation and configuration
  - Policy output mapper initialization
  - Environment seeding and reproducibility
  - Action space validation

**3.B.4. Step Manager (`step_manager.py`)**
- **Responsibility**: Individual training step execution
- **Key Features**:
  - Action selection with legal move filtering
  - Environment interaction and state transitions
  - Experience collection and storage
  - Reward calculation and processing

**3.B.5. Metrics Manager (`metrics_manager.py`)**
- **Responsibility**: Training statistics and performance tracking
- **Key Features**:
  - Loss component tracking (policy, value, entropy)
  - Performance metrics (episode length, rewards)
  - Learning rate scheduling and monitoring
  - Statistical aggregation and reporting

**3.B.6. Training Loop Manager (`training_loop_manager.py`)**
- **Responsibility**: Main training algorithm orchestration
- **Key Features**:
  - PPO algorithm implementation with clipping
  - Experience buffer management and GAE computation
  - Minibatch processing and gradient updates
  - Advanced training features (DDP, mixed precision)

**3.B.7. Setup Manager (`setup_manager.py`)**
- **Responsibility**: Component initialization and validation
- **Key Features**:
  - Configuration loading and validation
  - Device detection and allocation
  - Component dependency resolution
  - Error handling and recovery

**3.B.8. Display Manager (`display_manager.py`)**
- **Responsibility**: User interface and progress monitoring
- **Key Features**:
  - Training progress visualization
  - Real-time metrics display
  - Game state rendering
  - Performance monitoring dashboards

**3.B.9. Callback Manager (`callback_manager.py`)**
- **Responsibility**: Event-driven training customization
- **Key Features**:
  - Training event hooks (epoch start/end, checkpoint)
  - Custom evaluation schedules
  - Early stopping and learning rate scheduling
  - Extensible callback system for research experiments

**3.C. Shogi Game Manager Configuration**

The `ShogiGame` class integrates seamlessly with the configuration system, providing configurable game parameters while maintaining strict rule compliance.

**3.C.1. Game Configuration Parameters**
```python
class ShogiGame:
    def __init__(self, max_moves_per_game: int = 500, seed: Optional[int] = None):
        self.max_moves_per_game = max_moves_per_game
        self.move_count = 0
        # ... other initialization
```

**Key Features**:
- **Move Limits**: Configurable maximum moves to prevent infinite games
- **Seeding**: Deterministic game sequences for reproducible training
- **SFEN Support**: Standard Forsyth-Edwards Notation for position serialization
- **Observation Space**: 46-channel board representation with piece positions, captured pieces, and game state

**3.C.2. Configuration Integration**
The game manager receives configuration from the `EnvConfig` section:
- `max_moves_per_game`: Prevents runaway games during training
- `seed`: Ensures reproducible training runs when specified
- `input_channels`: Validates observation space dimensions (46 channels)
- `action_space_size`: Validates action space coverage (6,480 moves)

**3.D. Configuration Usage Patterns**

**3.D.1. Configuration Loading**
```python
from keisei.config_schema import KeseiConfig

# Load from YAML file
config = KeiiseiConfig.from_yaml("config.yaml")

# Override specific parameters
config.training.learning_rate = 1e-4
config.training.mixed_precision = True
config.wandb.enabled = True
```

**3.D.2. Manager Integration**
```python
# Managers receive relevant configuration sections
session_manager = SessionManager(config.logging, config.wandb)
model_manager = ModelManager(config.training, config.env)
training_loop = TrainingLoopManager(config.training)
```

**3.D.3. Advanced Configuration Features**
- **Validation**: Pydantic ensures type safety and value constraints
- **Documentation**: Built-in help and parameter descriptions
- **Environment Variables**: Support for runtime configuration overrides
- **CLI Integration**: Command-line parameter parsing and validation

**4. Modules and Key Components**

**4.A. `shogi_engine.py` (The Game Environment)**

This module is responsible for Shogi rules, game state, and move validation.

*   **Internal Representations:**
    *   **Piece Types:** Use consistent integers internally (e.g., PAWN=0, LANCE=1, ..., KING=7; PROMOTED_PAWN=8, ..., PROMOTED_ROOK=13). Mapping to symbols ('P', '+P') will be for display.
    *   **Color:** 0 for Black (Sente), 1 for White (Gote).
    *   **Board Indexing:** Rows and columns will be 0-indexed (0-8).

*   **Class `Piece`**:
    *   `__init__(self, piece_type_int, color_int, is_promoted=False)`
    *   Attributes: `type_int` (integer representation), `color` (0 or 1), `is_promoted` (Boolean).
    *   `symbol(self)`: Returns a character representation (e.g., 'P', '+P', 'p', '+p') based on type and color.

*   **Class `ShogiGame`**:
    *   `__init__(self)`:
        *   `self.board`: 9x9 list of lists (or NumPy array) storing `Piece` objects or `None`.
        *   `self.hands`: Dictionary `{color_int: {piece_type_int: count}}` for captured pieces (unpromoted state).
        *   `self.current_player`: 0 (Black) or 1 (White).
        *   `self.move_history`: List of `(board_state_hash, player_to_move)` tuples for repetition detection (Sennichite). `board_state_hash` should include board, hands.
        *   `self.game_over`: Boolean.
        *   `self.winner`: 0 (Black), 1 (White), or `None`/2 (Draw).
        *   `self.move_count`: Integer.
    *   `reset(self)`: Initializes board to starting position (see **4.A.1 Initial Board Setup**), clears hands, sets `current_player` to 0 (Black), clears `move_history`, `game_over` to False, `winner` to None, `move_count` to 0.
    *   `get_piece(self, row, col)`: Returns piece at `(row, col)`.
    *   `set_piece(self, row, col, piece)`: Sets piece at `(row, col)`.
    *   `_is_on_board(self, row, col)`: Helper: `0 <= row <= 8 and 0 <= col <= 8`.
    *   `_get_individual_piece_moves(self, piece, r_from, c_from)`:
        *   Returns list of `(r_to, c_to)` tuples for a piece, considering *only* its fundamental movement rules (not checks, board boundaries, or captures on own pieces). Handles promoted pieces.
    *   `get_legal_moves(self)`:
        *   **CRITICAL FUNCTION.**
        *   Generates all possible moves for `self.current_player`. Returns a list of `move_tuple` (see **4.A.2 Precise Internal Move Representation**).
        *   **Board moves:**
            *   Iterate through `self.current_player`'s pieces on `self.board`.
            *   For each, call `_get_individual_piece_moves`.
            *   For each potential `(r_to, c_to)`:
                *   Check if `(r_to, c_to)` is on board.
                *   Check if `(r_to, c_to)` is occupied by own piece (illegal).
                *   Determine promotion options based on `_can_promote()` (see **4.A.3 Promotion Rules**). If promotion is optional, generate moves for both promoted and unpromoted states. If mandatory, only the promoted move.
        *   **Drop moves:**
            *   Iterate through `piece_type_int` in `self.hands[self.current_player]` with `count > 0`.
            *   For each piece type, iterate through all 81 squares `(r_to, c_to)`.
            *   Check if drop is valid: square must be empty.
            *   Illegal drop checks:
                *   Pawn/Lance cannot be dropped on the last rank.
                *   Knight cannot be dropped on the last two ranks.
                *   Nifu (Two Pawns): No two unpromoted pawns of the current player can be in the same file. (See **4.A.4 Illegal Move Checks**).
        *   **Universal Legality Checks for each generated move (board or drop):**
            *   Simulate the move temporarily.
            *   Check if it results in the `self.current_player`'s king being in check (illegal). Revert simulation.
            *   Uchi Fu Zume (Dropping Pawn for Checkmate): If the move is a pawn drop that results in checkmate, it's illegal. (See **4.A.4 Illegal Move Checks** - can be a stretch goal for initial implementation).
    *   `make_move(self, move_tuple)`:
        *   Takes a `move_tuple` (as returned by `get_legal_moves`).
        *   Updates `self.board`, `self.hands`.
        *   Handles captures: captured piece (reverted to unpromoted state) is added to opponent's hand.
        *   Handles promotions: based on `promotion_choice` in `move_tuple` or mandatory promotion rules (see **4.A.3 Promotion Rules**).
        *   Updates `self.current_player`.
        *   Updates `self.move_history` with a hash of `(self.board, self.hands, self.current_player)`.
        *   Increments `self.move_count`.
        *   Checks for game end conditions (updates `self.game_over`, `self.winner`):
            *   Checkmate (`_is_checkmate`).
            *   Sennichite (Repetition - see **4.A.5 Game Termination**).
            *   Max moves reached (see **4.A.5 Game Termination**).
    *   `_is_in_check(self, player_color_int)`: Checks if `player_color_int`'s king is attacked by any opponent piece.
    *   `_is_checkmate(self, player_color_int)`: Checks if `player_color_int` is checkmated (is in check and `get_legal_moves()` for `player_color_int` returns an empty list).
    *   `_can_promote(self, piece_type_int, r_from, r_to, player_color_int)`:
        *   Returns `(is_optional, is_mandatory)`.
        *   Checks if a promotion is possible or mandatory based on piece type, origin, destination, and player color (promotion zone: farthest 3 ranks for the player).
    *   `get_observation(self)`: **CRITICAL FOR RL.**
        *   Returns a numerical representation of the game state (NumPy array) suitable for a neural network. (See **3.A.6 State Representation - Exact Definition**).
    *   `get_reward(self)`:
        *   Returns reward for the player *whose turn it just was* (i.e., the player who made the move leading to the current state).
        *   If game over and current player (who just moved) won: +1
        *   If game over and current player (who just moved) lost: -1
        *   If game over and draw: 0
        *   If game ongoing: 0 (sparse reward).
    *   `to_string(self)`: Simple text representation of the board for logging/debugging. (See **3.H Board Printing**).
    *   `sfen_encode_move(self, move_tuple)`: Converts internal `move_tuple` to SFEN/USI string (e.g., "7g7f", "P*5e"). Essential for logging.
    *   `sfen_decode_move(self, sfen_string)`: (Optional) Converts SFEN string to internal `move_tuple`. Useful for testing.

    *   **3.A.1 Initial Board Setup:**
        *   Use standard Shogi starting FEN: `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1`
        *   Or an equivalent explicit array initialization in `reset()`.

    *   **3.A.2 Precise Internal Move Representation:**
        *   **Board Moves:** `(from_row, from_col, to_row, to_col, promotion_choice)`
            *   `from_row, from_col, to_row, to_col`: 0-8 integers.
            *   `promotion_choice`:
                *   `0`: No promotion (if promotion is optional).
                *   `1`: Promote (if promotion is optional or mandatory).
                *   (Internally, `make_move` might infer mandatory promotion if `promotion_choice` is not 1 but conditions are met).
        *   **Drop Moves:** `('DROP', piece_type_int, to_row, to_col)`
            *   `'DROP'`: A sentinel to distinguish from board moves.
            *   `piece_type_int`: Integer for the piece being dropped (unpromoted form).
            *   `to_row, to_col`: 0-8 integers.
        *   Example: `(6, 6, 6, 5, 0)` for 7g7f (no promotion). `('DROP', PAWN_INT, 4, 4)` for P*5e.

    *   **3.A.3 Promotion Rules - Explicit Logic:**
        *   **Mandatory Promotion:** Pawns and Lances landing on the opponent's farthest rank (rank 0 for Black, rank 8 for White) *must* promote. Knights landing on the opponent's two farthest ranks (ranks 0-1 for Black, ranks 7-8 for White) *must* promote. If `make_move` receives a move tuple where promotion is mandatory but `promotion_choice` isn't 1, it should auto-promote.
        *   **Optional Promotion:** Any piece (that can promote) entering, moving within, or leaving the opponent's promotion zone (farthest 3 ranks: 0-2 for Black, 6-8 for White) *may* promote. If `get_legal_moves` identifies such a move, it should generate two move tuples if promotion is optional: one with `promotion_choice=0` and one with `promotion_choice=1`.

    *   **3.A.4 Illegal Move Checks - Explicit Logic in `get_legal_moves`:**
        *   **King Safety:** The most fundamental: a player cannot make any move that leaves or places their own king in check.
        *   **Nifu (Two Pawns):** A player cannot have two unpromoted pawns in the same file. This check is performed during generation of pawn drop moves and pawn board moves (though a pawn moving within a file can't cause Nifu if it wasn't Nifu before).
        *   **Uchi Fu Zume (Dropping Pawn for Checkmate):** A pawn cannot be dropped to give immediate checkmate.
            *   This is a complex lookahead: if a pawn drop results in check, `get_legal_moves` must then (temporarily) simulate the drop, switch to the opponent, and verify if the opponent has *any* legal move to escape the check. If not, the pawn drop is illegal.
            *   **Initial Implementation:** This can be a "stretch goal." Omitting it initially will simplify `get_legal_moves` but means the agent might learn an illegal strategy. The game might need to terminate with a loss for the player making an Uchi Fu Zume if not prevented. Best to implement if possible.

    *   **3.A.5 Game Termination - Precise Definitions:**
        *   **Checkmate:** If `self.current_player` has no legal moves and their king is in check, they are checkmated. The previous player wins.
        *   **Sennichite (Repetition):** Four-fold repetition of the exact same board position, pieces in hand for both players, and player to move.
            *   Outcome: Typically a draw.
            *   Perpetual Check: If Sennichite occurs due to a sequence of perpetual checks, the player delivering checks loses. For simplicity, initially, all Sennichite can be declared a draw.
        *   **Max Moves:** A hard limit (e.g., `MAX_MOVES_PER_GAME = 512`). If reached, the game is a draw.
        *   **No Legal Moves (Stalemate):** If `self.current_player` has no legal moves and their king is *not* in check, it's a stalemate, and that player loses (this is specific to Shogi rules for this situation).
        *   **Try Rule (Nyugyoku):** Omit for "simple as possible" initial implementation due to its complexity.

    *   **3.A.6 State Representation (`ShogiGame.get_observation`) - Exact Definition:**
        *   The observation will be a 3D NumPy array `(Channels, 9, 9)`.
        *   **Piece Planes (Binary: 1 if present, 0 otherwise):**
            *   Planes 0-6: Current player's unpromoted pieces (Pawn, Lance, Knight, Silver, Gold, Bishop, Rook).
            *   Planes 7-12: Current player's promoted pieces (+P, +L, +N, +S, +B, +R). (King doesn't promote, Gold doesn't promote).
            *   Plane 13: Current player's King.
            *   Planes 14-20: Opponent's unpromoted pieces.
            *   Planes 21-26: Opponent's promoted pieces.
            *   Plane 27: Opponent's King.
            *   Total: 28 planes for on-board pieces.
        *   **Hand Piece Counts (Scalar features, to be reshaped into planes or appended):**
            *   Alternative A (Scalar Appended): After the 28x9x9 planes, append 14 scalar values:
                *   7 values for current player's hand counts (P, L, N, S, G, B, R).
                *   7 values for opponent's hand counts.
                *   These counts should be normalized (e.g., divide by max possible: Pawn by 18, Lance/Knight/Silver by 4, Gold/Bishop/Rook by 2).
            *   Alternative B (Constant Value Planes): For each piece type in hand, one plane for current player, one for opponent. Value of the plane is normalized count. 14 planes (9x9) filled with normalized hand counts.
            *   **Recommendation for Initial Implementation:** Alternative B (Constant Value Planes) to keep input purely spatial.
                *   Planes 28-34: Current player's hand counts (P to R), each plane filled with its normalized count.
                *   Planes 35-41: Opponent's hand counts (P to R), each plane filled with its normalized count.
                *   Total: 14 planes for hand pieces.
        *   **Game State Planes:**
            *   Plane 42: Player to move. All 1s if Black (0) to move, all 0s if White (1) to move (or vice-versa, be consistent).
            *   Plane 43: Repetition count 1. All 1s if the current board state (board, hands, player) has occurred once before in `self.move_history`.
            *   Plane 44: Repetition count 2. All 1s if occurred twice before.
            *   Plane 45: Repetition count 3. All 1s if occurred three times before (Sennichite imminent).
            *   (Optional) Plane for move count normalized (e.g., `self.move_count / MAX_MOVES_PER_GAME`).
        *   **Total Channels (Example with Alt B for hands, 3 repetition planes):** 28 (board) + 14 (hands) + 1 (player) + 3 (repetition) = **46 channels**. This needs to be fixed and consistently used.
        *   **Normalization:** Binary planes are 0/1. Hand count planes are normalized 0-1. Repetition planes are 0/1. Player-to-move plane 0/1.

**4.B. `neural_network.py` (Policy and Value Networks)**

*   **Class `ActorCritic(nn.Module)`** (using PyTorch `torch.nn`):
    *   `__init__(self, input_channels, num_actions_total)`:
        *   `input_channels`: Number of channels from `ShogiGame.get_observation()` (e.g., 46).
        *   `num_actions_total`: Total size of the action space representation from `PolicyOutputMapper.get_total_actions()`.
        *   **CNN Architecture (AlphaZero-like suggestion):**
            *   `self.initial_conv`: `nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)`, `nn.BatchNorm2d(256)`, `nn.ReLU()`.
            *   `self.resnet_blocks`: A stack of 10-20 ResNet blocks. Each block:
                *   `Conv2d(256, 256, kernel_size=3, padding=1)`, `BatchNorm2d(256)`, `ReLU`
                *   `Conv2d(256, 256, kernel_size=3, padding=1)`, `BatchNorm2d(256)`
                *   Add residual connection, then `ReLU`.
            *   `self.policy_head_conv`: `nn.Conv2d(256, 2, kernel_size=1, padding=0)` (intermediate policy features), `nn.BatchNorm2d(2)`, `nn.ReLU()`.
            *   `self.policy_head_fc`: `nn.Linear(2 * 9 * 9, num_actions_total)`.
            *   `self.value_head_conv`: `nn.Conv2d(256, 1, kernel_size=1, padding=0)` (intermediate value features), `nn.BatchNorm2d(1)`, `nn.ReLU()`.
            *   `self.value_head_fc1`: `nn.Linear(1 * 9 * 9, 256)`, `nn.ReLU()`.
            *   `self.value_head_fc2`: `nn.Linear(256, 1)`.
    *   `forward(self, x)`:
        *   `x` is the observation tensor `(batch_size, input_channels, 9, 9)`.
        *   Pass `x` through `initial_conv` and `resnet_blocks`.
        *   Policy path: Pass shared features through `policy_head_conv`, flatten, then `policy_head_fc` to get `action_logits`.
        *   Value path: Pass shared features through `value_head_conv`, flatten, then `value_head_fc1`, then `value_head_fc2` to get `value`. Apply `torch.tanh(value)` to constrain value output to [-1, 1].
        *   Return `action_logits`, `value`.

**4.C. `ppo_agent.py` (The Learning Agent)**

*   **Class `PPOAgent`**:
    *   `__init__(self, input_channels, num_actions_total, policy_output_mapper, learning_rate, gamma, clip_epsilon, ppo_epochs, minibatch_size, entropy_coeff, lambda_gae)`:
        *   `self.actor_critic`: Instance of `ActorCritic`.
        *   `self.optimizer`: Adam optimizer for `actor_critic.parameters()`.
        *   `self.policy_output_mapper`: Instance of `PolicyOutputMapper`.
        *   `self.gamma`: Discount factor (e.g., 0.99).
        *   `self.clip_epsilon`: PPO clipping parameter (e.g., 0.1 or 0.2).
        *   `self.ppo_epochs`: Number of optimization epochs per PPO update (e.g., 4-10).
        *   `self.minibatch_size`: Size of minibatches for training (e.g., 64 or 256).
        *   `self.entropy_coeff`: Coefficient for entropy bonus (e.g., 0.01).
        *   `self.lambda_gae`: GAE lambda parameter (e.g., 0.95).
    *   `select_action(self, observation_tensor, legal_shogi_moves)`:
        *   `self.actor_critic.eval()` mode.
        *   Pass `observation_tensor` through `self.actor_critic` to get `all_action_logits` and `value`.
        *   `legal_policy_indices, shogi_move_for_legal_policy_idx_map = self.policy_output_mapper.get_legal_policy_indices_and_map(legal_shogi_moves, observation_tensor.device)`
            *   `shogi_move_for_legal_policy_idx_map`: Maps a global policy index (if legal) to its shogi move tuple.
        *   Create a mask: `mask = torch.ones_like(all_action_logits) * -1e8` (very small number).
        *   `mask[legal_policy_indices] = 0`.
        *   `masked_logits = all_action_logits + mask`.
        *   `action_probs = F.softmax(masked_logits, dim=-1)`.
        *   `distribution = Categorical(probs=action_probs)`.
        *   `selected_global_policy_index = distribution.sample()`.
        *   `log_prob = distribution.log_prob(selected_global_policy_index)`.
        *   `selected_shogi_move = shogi_move_for_legal_policy_idx_map[selected_global_policy_index.item()]`.
        *   Return `selected_shogi_move`, `selected_global_policy_index.detach()`, `log_prob.detach()`, `value.detach()`.
    *   `compute_advantages_gae(self, rewards, values, dones, next_value, device)`:
        *   Calculates Generalized Advantage Estimation (GAE).
        *   `rewards, values, dones` are tensors of collected trajectory data.
        *   `next_value` is the value estimate of the state after the last state in the trajectory.
        *   Returns `advantages` tensor.
    *   `learn(self, experiences)`:
        *   `self.actor_critic.train()` mode.
        *   `experiences`: A structured object or tuple containing `(states, actions, old_log_probs, advantages, returns, old_values)`.
        *   `advantages` should be normalized: `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`.
        *   Loop for `self.ppo_epochs`:
            *   Shuffle data and iterate through minibatches.
            *   For each minibatch `(b_states, b_actions, b_old_log_probs, b_advantages, b_returns, b_old_values)`:
                *   Get `current_logits, current_values_raw` from `self.actor_critic` using `b_states`. `current_values = torch.tanh(current_values_raw)`.
                *   `dist = Categorical(logits=current_logits)`.
                *   `new_log_probs = dist.log_prob(b_actions)`.
                *   `entropy = dist.entropy().mean()`.
                *   `ratio = torch.exp(new_log_probs - b_old_log_probs)`.
                *   `surr1 = ratio * b_advantages`.
                *   `surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages`.
                *   `policy_loss = -torch.min(surr1, surr2).mean()`.
                *   `value_loss = F.mse_loss(current_values.squeeze(), b_returns)`. (PPO variants sometimes clip value loss too).
                *   `total_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy`.
                *   `self.optimizer.zero_grad()`.
                *   `total_loss.backward()`.
                *   `torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)`.
                *   `self.optimizer.step()`.
        *   Return `policy_loss.item()`, `value_loss.item()`, `entropy.item()`.
    *   `save_model(self, path)`: Saves `actor_critic.state_dict()` and `optimizer.state_dict()`.
    *   `load_model(self, path)`: Loads `actor_critic.state_dict()` and `optimizer.state_dict()`.

**4.D. `experience_buffer.py` (Stores Self-Play Data)**

*   **Class `ExperienceBuffer`**:
    *   `__init__(self, buffer_size, device)`
    *   `self.buffer_size = buffer_size`
    *   `self.device = device`
    *   `self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values`: Lists (or pre-allocated tensors) to store transitions.
    *   `self.ptr = 0`, `self.path_start_idx = 0`
    *   `add(self, state, action_idx, reward, done, log_prob, value)`:
        *   Store individual transition components. `state` should be squeezed observation.
    *   `finish_path(self, last_value=torch.tensor(0.0))`:
        *   Called at the end of a trajectory (game over or max steps).
        *   Calculates GAE advantages and returns for the completed path.
        *   `path_slice = slice(self.path_start_idx, self.ptr)`
        *   Extract `rewards_path = self.rewards[path_slice]`, `values_path = self.values[path_slice]`, `dones_path = self.dones[path_slice]`.
        *   Compute GAE: `advantages_path, returns_path = compute_gae_for_path(rewards_path, values_path, dones_path, last_value, gamma, lambda_gae)`
        *   Store these `advantages_path` and `returns_path` in separate lists/tensors in the buffer, aligned with the path.
        *   `self.path_start_idx = self.ptr`
    *   `get_all_data_and_clear(self)`:
        *   Returns all stored experiences as tensors: `(states, actions, log_probs, advantages, returns, values)`.
        *   Concatenate/stack list buffers into tensors on `self.device`.
        *   Clears the buffer (`self.ptr = 0`, `self.path_start_idx = 0`).
    *   `__len__(self)`: Returns `self.ptr`.

**4.E. `train.py` (Main Training Loop)**

*   **Hyperparameters (from `config.py` or CLI):**
    *   `TOTAL_TIMESTEPS`: e.g., 10e6, 50e6 or more.
    *   `STEPS_PER_EPOCH` (Rollout buffer size): e.g., 2048 or 4096 (number of (s,a,r,s') transitions before an update).
    *   `PPO_EPOCHS`: e.g., 10.
    *   `MINIBATCH_SIZE`: e.g., 64 or 256 (ensure `STEPS_PER_EPOCH` is divisible by `MINIBATCH_SIZE`).
    *   `LEARNING_RATE`: e.g., 3e-4.
    *   `GAMMA`: 0.99.
    *   `CLIP_EPSILON`: 0.2.
    *   `LAMBDA_GAE`: 0.95.
    *   `ENTROPY_COEFF`: 0.01.
    *   `MAX_MOVES_PER_GAME`: e.g., 512.
    *   `INPUT_CHANNELS`: Derived from `ShogiGame.get_observation()` (e.g., 46).
    *   `NUM_ACTIONS_TOTAL`: From `PolicyOutputMapper.get_total_actions()`.
    *   `SAVE_FREQ_EPISODES`: e.g., 100.
    *   `DEVICE`: 'cuda' or 'cpu'.
*   **Initialization:**
    *   Set up logging (file, console, potentially TensorBoard).
    *   `game = ShogiGame()`
    *   `policy_mapper = PolicyOutputMapper()` (needs `game` or knowledge of piece types for drop actions).
    *   `agent = PPOAgent(INPUT_CHANNELS, policy_mapper.get_total_actions(), policy_mapper, LEARNING_RATE, ...)`
    *   `buffer = ExperienceBuffer(STEPS_PER_EPOCH, DEVICE)`
    *   `log_file = open("logs/shogi_training_log.txt", "a")`
*   **Main Loop:**
    ```python
    obs_np = game.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    current_episode_moves = 0
    current_episode_reward_p0 = 0 # Sente (Black)
    current_episode_reward_p1 = 0 # Gote (White)
    game_log_sfen_moves = []

    for global_step in range(TOTAL_TIMESTEPS):
        legal_shogi_moves = game.get_legal_moves()
        if not legal_shogi_moves: # Should be caught by game.game_over
            # This case implies stalemate or an issue if game not over
            # Force game end if not already handled
            if not game.game_over: 
                print(f"Warning: No legal moves but game not marked over. State: Player {game.current_player}")
                # Determine outcome based on Shogi stalemate rules (player with no moves loses if not in check)
                # For simplicity here, assume make_move will handle it or it's a draw/loss
                game.game_over = True
                # game.winner might need to be set appropriately based on stalemate rules if not checkmate
            # else: game is already over, handled below
        
        if game.game_over or not legal_shogi_moves: # Episode finished
            buffer.finish_path(last_value=torch.tensor(0.0, device=DEVICE)) # Terminal state value is 0
            
            # Log episode stats
            winner_str = f"Player {game.winner}" if game.winner is not None else "Draw"
            log_msg = (f"GlobalStep: {global_step}, Episode Ended. Winner: {winner_str}, "
                       f"Moves: {current_episode_moves}, P0_R: {current_episode_reward_p0}, P1_R: {current_episode_reward_p1}\n"
                       f"SFEN Moves: {' '.join(game_log_sfen_moves)}\n")
            print(log_msg)
            log_file.write(log_msg)
            log_file.flush()

            # Reset for next episode
            obs_np = game.reset()
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            current_episode_moves = 0
            current_episode_reward_p0 = 0
            current_episode_reward_p1 = 0
            game_log_sfen_moves = []
            
            # Skip action selection if game just ended this step
            if len(buffer) < STEPS_PER_EPOCH: # Only continue if buffer not full
                 continue # To next global_step to collect more data
        
        # If game is ongoing and there are legal moves
        with torch.no_grad():
            shogi_move, global_action_idx, log_prob, value = \
                agent.select_action(obs, legal_shogi_moves)

        prev_obs_np = obs_np # Store current observation before making move
        player_who_moved = game.current_player 
        
        game.make_move(shogi_move)
        game_log_sfen_moves.append(game.sfen_encode_move(shogi_move))

        obs_np = game.get_observation() # New observation
        reward = game.get_reward() # Reward for player_who_moved
        done = game.game_over

        buffer.add(
            torch.tensor(prev_obs_np, dtype=torch.float32), # No unsqueeze, buffer handles batching later
            global_action_idx,
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
            log_prob,
            value
        )

        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0) # Prepare for next select_action

        if player_who_moved == 0: current_episode_reward_p0 += reward
        else: current_episode_reward_p1 += reward
        current_episode_moves += 1
        
        if current_episode_moves >= MAX_MOVES_PER_GAME and not game.game_over:
            # Force game end due to max moves
            game.game_over = True
            game.winner = 2 # Draw by max moves
            done = True 
            # The reward for the last move will be 0, then terminal value 0
            # This path is handled by the episode end logic at the start of the loop on next iteration

        # If buffer is full, perform PPO update
        if len(buffer) >= STEPS_PER_EPOCH:
            with torch.no_grad():
                # If the last game didn't end exactly at STEPS_PER_EPOCH, get value for current obs
                _, last_val = agent.actor_critic(obs) if not done else (None, torch.tensor([[0.0]], device=DEVICE)) 
            buffer.finish_path(last_value=last_val.squeeze()) # Finish potentially incomplete path
            
            experiences = buffer.get_all_data_and_clear() # Returns tuple of tensors
            policy_loss, value_loss, entropy = agent.learn(experiences)
            
            # Log PPO update stats
            update_log_msg = (f"GlobalStep: {global_step}, PPO Update. PolicyLoss: {policy_loss:.4f}, "
                              f"ValueLoss: {value_loss:.4f}, Entropy: {entropy:.4f}\n")
            print(update_log_msg, end='') # No newline if episode log follows
            log_file.write(update_log_msg)
            log_file.flush()

            num_episodes_completed = global_step // (current_episode_moves if current_episode_moves > 0 else 1) # Approx
            if num_episodes_completed % SAVE_FREQ_EPISODES == 0 and num_episodes_completed > 0: # Save periodically
                agent.save_model(f"models/shogi_ppo_agent_step_{global_step}.pth")
    
    log_file.close()
    ```

**3.F. `utils.py` (Helper functions & Action Mapping)**

*   **Class `PolicyOutputMapper`**:
    *   `__init__(self)`: Defines the canonical mapping from Shogi moves to policy network output indices.
        *   **Action Representation (AlphaZero-style for Shogi):**
            The policy output is a flat vector of logits. Each logit corresponds to a potential action.
            The actions are structured conceptually as planes, then flattened.
            *   **Board Move Planes (81 values per plane):**
                *   Each plane represents moving a piece *from* a square *to* a relative destination.
                *   E.g., Plane 0: N (North, (r-1,c)). Plane 1: NE ((r-1,c+1)), ..., Plane 7: NW ((r-1,c-1)).
                *   Need to define directions for all piece types:
                    *   Pawn: 1 (N)
                    *   Lance: N-striding
                    *   Knight: 2 (NNE, NNW)
                    *   Silver: 5 (N, NE, SE, SW, NW)
                    *   Gold/Promoted P,L,N,S: 6 (N, S, E, W, NE, NW)
                    *   Bishop: 4 diagonal striding directions
                    *   Rook: 4 orthogonal striding directions
                    *   King: 8 (N,S,E,W,NE,NW,SE,SW)
                    *   Promoted Bishop: King moves + 4 diagonal striding
                    *   Promoted Rook: King moves + 4 orthogonal striding
                *   A common way to represent this (as in AlphaZero): 73 action planes for board moves.
                    *   56 planes for queen-like moves (8 directions, up to 7 squares distance).
                    *   8 planes for knight-like moves.
                    *   9 planes for underpromotions (specific to chess, adapt for Shogi promotions).
                *   **Simpler for Shogi (Recommended):** Use a fixed set of policy "move type" planes.
                    *   1 plane: Move Pawn forward (81 cells, applies if a pawn is on `from_sq`).
                    *   1 plane: Move Lance forward (striding) (81 cells).
                    *   2 planes: Move Knight (81 cells per direction).
                    *   5 planes: Move Silver (81 cells per direction).
                    *   6 planes: Move Gold (81 cells per direction).
                    *   For Bishop/Rook, either define max N moves per direction (e.g., 7 * 4 directions * 81) or fewer planes and game logic determines extent.
                    *   **Alternative Fixed Relative Moves (Like KataGo):**
                        *   A set of predefined relative coordinates (dx, dy) e.g., (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1) for single steps.
                        *   Knight moves: (1,2), (2,1), etc.
                        *   Lance/Bishop/Rook moves: these are "riders". The policy might output a probability for each direction from each square.
                    *   **Promotion as separate action dimension:**
                        For moves that can promote, the policy output could have an additional dimension or set of actions indicating "promote" vs "don't promote." E.g., if a move from (r,c) to (r',c') is action `k`, then action `k + C` could be the same move with promotion.
                    *   **Final Recommended Action Structure for NN Output:**
                        1.  **Board Moves without Promotion (Max 8 directions for King-like, plus Knight moves):**
                            *   `from_sq` (81) * `to_sq_relative_direction` (e.g., 8 queen directions + 8 knight directions = 16).
                            *   This part represents selecting a source square and a "type" of move. `get_legal_moves` filters by what piece is actually on `from_sq`.
                            *   E.g., `81 * 16 = 1296` actions.
                        2.  **Board Moves with Promotion (Same as above, but signifying promotion):**
                            *   If a move from #1 can promote, this equivalent action means "do it and promote."
                            *   `81 * 16 = 1296` actions.
                        3.  **Drop Moves (7 piece types * 81 squares):**
                            *   `7 * 81 = 567` actions.
                        *   Total actions = `1296 (no promo) + 1296 (promo) + 567 (drop) = 3159`.
                        *   This needs careful implementation to map actual piece movements (e.g. Rook moving 3 squares North) to one of these canonical action indices.
        *   `self.idx_to_shogi_move_spec`: List mapping global policy index to a specification like `('board', from_r, from_c, to_r_or_direction, to_c_or_direction, promotion_flag)` or `('drop', piece_type_int, to_r, to_c)`. The `promotion_flag` distinguishes between the two blocks of board moves.
        *   `self.shogi_move_spec_to_idx`: Reverse mapping.
    *   `get_total_actions(self)`: Returns `self.total_actions` (e.g., 3159).
    *   `shogi_move_to_policy_index(self, shogi_move_tuple)`: Converts a game engine `move_tuple` to its global policy index. This requires knowing the piece type on `from_sq` for board moves to map its movement to the canonical direction indices.
    *   `get_legal_policy_indices_and_map(self, legal_shogi_moves_from_engine, device)`:
        *   Iterates `legal_shogi_moves_from_engine` (list of tuples from `ShogiGame`).
        *   For each `shogi_move_tuple`, calls `shogi_move_to_policy_index` to find its global policy index.
        *   Returns:
            *   `legal_indices_tensor`: A `torch.LongTensor` of global policy indices that are legal, on `device`.
            *   `index_to_shogi_move_dict`: A Python dictionary mapping these `legal_indices` (as integers) back to their `shogi_move_tuple`.

**3.G. Configuration (`config.py` or similar)**

*   Store all hyperparameters listed in `train.py`.
*   Store `INPUT_CHANNELS`, `MAX_MOVES_PER_GAME`.
*   Paths for saving models and logs.
*   `DEVICE` ('cuda', 'cpu').

**3.H. Logging and Debugging**

*   **Move Log Format:** SFEN/USI (e.g., `7g7f 3c3d ...`) for each game.
*   **Training Log (`logs/shogi_training_log.txt`):**
    *   Global step, episode number.
    *   Episode details: winner, number of moves, P0_Reward, P1_Reward.
    *   PPO Update details: Policy Loss, Value Loss, Entropy.
    *   (Optional) Average value estimate from network.
*   **Board Printing (`ShogiGame.to_string()`):** Human-readable for debugging.
    Example:
    ```
    Player to move: Black (Sente)
    Gote Hand: Px2 Lx1
      9  8  7  6  5  4  3  2  1
    +---------------------------+
    | l  n  s  g  k  g  s  n  l | P1
    | .  R  .  .  .  .  .  B  . | P2
    | p  p  p  p  p  p  p  p  p | P3
    | .  .  .  .  .  .  .  .  . | P4
    | .  .  .  .  .  .  .  .  . | P5
    | .  .  .  .  .  .  .  .  . | P6
    | P  P  P  P  P  P  P  P  P | P7
    | .  b  .  .  .  .  .  r  . | P8
    | L  N  S  G  K  G  S  N  L | P9
    +---------------------------+
    Sente Hand: Gx1
    Move Count: 10
    ```
    (Use standard Shogi notation: Black at bottom, White at top. Ranks 1-9, Files 9-1.)

**5. Development Environment & Dependencies**

*   Python: 3.8+
*   PyTorch: 1.10+ (or latest stable)
*   NumPy: Latest stable
*   Version Control: Git
*   (Optional) TensorBoard for richer logging.

**6. Project Structure (Suggested Directory Layout)**

```
shogi_drl/
├── shogi_engine.py       # Game rules, state
├── neural_network.py     # ActorCritic NN model
├── ppo_agent.py          # PPO algorithm, action selection, learning
├── experience_buffer.py  # Replay buffer
├── utils.py              # PolicyOutputMapper, other helpers
├── train.py              # Main training script
├── config.py             # Hyperparameters and configuration
├── README.md
├── requirements.txt
├── models/                 # To save trained models (e.g., shogi_ppo_agent_step_XXXX.pth)
└── logs/                   # To save training logs (e.g., shogi_training_log.txt)
```

**7. Phased Approach for Coder**

1.  **Phase 1: `shogi_engine.py` - Core Game Mechanics.**
    *   Implement `Piece` class.
    *   Implement `ShogiGame` structure, `reset()`, board representation, piece placement, hand management.
    *   Basic piece movement rules (`_get_individual_piece_moves`), captures.
    *   Promotion logic (`_can_promote`), including mandatory/optional.
    *   `get_legal_moves` (initially without Nifu, Uchi Fu Zume, but WITH king safety and basic drop rules).
    *   `make_move` to update state.
    *   Check detection (`_is_in_check`), Checkmate (`_is_checkmate`).
    *   Basic game termination (checkmate, max moves).
    *   `sfen_encode_move` and `to_string`.
    *   **Crucial:** Develop a text-based player interface or robust unit tests to manually play/test scenarios.

2.  **Phase 2: State Representation & Action Mapping.**
    *   Finalize and implement `ShogiGame.get_observation()` as per section **3.A.6**.
    *   Design and implement `PolicyOutputMapper` in `utils.py` (action to index, index to action, legal index filtering) as per section **3.F**. This is complex and needs thorough testing.

3.  **Phase 3: Basic RL Loop Structure & Random Agent.**
    *   Implement `neural_network.py` (`ActorCritic` structure only, forward pass returning dummy logits/value).
    *   Implement `ppo_agent.py`:
        *   `__init__` structure.
        *   `select_action` to pick a *random valid action* from `legal_shogi_moves` (ignoring NN output for now).
    *   Implement `experience_buffer.py` structure.
    *   Implement `train.py` main loop to play games using this random agent.
        *   Collect `(state, action_idx, reward, next_state, done, dummy_log_prob, dummy_value)` into buffer.
        *   Test game flow, state/observation generation, reward calculation, game termination, logging.

4.  **Phase 4: Full PPO Implementation.**
    *   Complete `neural_network.py` with specified architecture.
    *   Complete `ppo_agent.py`:
        *   Wire up `select_action` to use the NN.
        *   Implement `compute_advantages_gae`.
        *   Implement the full `learn` method (PPO loss calculations, optimization steps).
    *   Complete `experience_buffer.py` with GAE calculation logic (`finish_path`).
    *   Refine `train.py` to correctly use the agent's learning capabilities.
    *   Start training. Debug PPO specific issues (exploding/vanishing gradients, loss components, etc.).

5.  **Phase 5: Refinements and Advanced Rules.**
    *   Implement full Nifu and Uchi Fu Zume checks in `ShogiGame.get_legal_moves`.
    *   Implement Sennichite repetition rule.
    *   Tune hyperparameters based on initial training runs.
    *   Iterate on NN architecture or state representation if performance is poor.
    *   Consider adding opponent snapshots for more stable training.
    *   Implement model saving/loading and evaluation against checkpoints.

This comprehensive document should provide a solid foundation for development. The most challenging parts will be the bug-free implementation of `ShogiGame` (especially `get_legal_moves`), the `PolicyOutputMapper`, and correctly implementing the PPO algorithm.