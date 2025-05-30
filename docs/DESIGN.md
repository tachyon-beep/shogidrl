Okay, this is an excellent consolidation of requirements and design considerations. I will now integrate both sets of information into a single, highly detailed design document.

**DRL Shogi Client: Detailed Design Document**

**1. Core Philosophy: "Learn from Scratch"**

*   No hardcoded opening books.
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

**3. Modules and Key Components**

**3.A. `shogi_engine.py` (The Game Environment)**

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
    *   `reset(self)`: Initializes board to starting position (see **3.A.1 Initial Board Setup**), clears hands, sets `current_player` to 0 (Black), clears `move_history`, `game_over` to False, `winner` to None, `move_count` to 0.
    *   `get_piece(self, row, col)`: Returns piece at `(row, col)`.
    *   `set_piece(self, row, col, piece)`: Sets piece at `(row, col)`.
    *   `_is_on_board(self, row, col)`: Helper: `0 <= row <= 8 and 0 <= col <= 8`.
    *   `_get_individual_piece_moves(self, piece, r_from, c_from)`:
        *   Returns list of `(r_to, c_to)` tuples for a piece, considering *only* its fundamental movement rules (not checks, board boundaries, or captures on own pieces). Handles promoted pieces.
    *   `get_legal_moves(self)`:
        *   **CRITICAL FUNCTION.**
        *   Generates all possible moves for `self.current_player`. Returns a list of `move_tuple` (see **3.A.2 Precise Internal Move Representation**).
        *   **Board moves:**
            *   Iterate through `self.current_player`'s pieces on `self.board`.
            *   For each, call `_get_individual_piece_moves`.
            *   For each potential `(r_to, c_to)`:
                *   Check if `(r_to, c_to)` is on board.
                *   Check if `(r_to, c_to)` is occupied by own piece (illegal).
                *   Determine promotion options based on `_can_promote()` (see **3.A.3 Promotion Rules**). If promotion is optional, generate moves for both promoted and unpromoted states. If mandatory, only the promoted move.
        *   **Drop moves:**
            *   Iterate through `piece_type_int` in `self.hands[self.current_player]` with `count > 0`.
            *   For each piece type, iterate through all 81 squares `(r_to, c_to)`.
            *   Check if drop is valid: square must be empty.
            *   Illegal drop checks:
                *   Pawn/Lance cannot be dropped on the last rank.
                *   Knight cannot be dropped on the last two ranks.
                *   Nifu (Two Pawns): No two unpromoted pawns of the current player can be in the same file. (See **3.A.4 Illegal Move Checks**).
        *   **Universal Legality Checks for each generated move (board or drop):**
            *   Simulate the move temporarily.
            *   Check if it results in the `self.current_player`'s king being in check (illegal). Revert simulation.
            *   Uchi Fu Zume (Dropping Pawn for Checkmate): If the move is a pawn drop that results in checkmate, it's illegal. (See **3.A.4 Illegal Move Checks** - can be a stretch goal for initial implementation).
    *   `make_move(self, move_tuple)`:
        *   Takes a `move_tuple` (as returned by `get_legal_moves`).
        *   Updates `self.board`, `self.hands`.
        *   Handles captures: captured piece (reverted to unpromoted state) is added to opponent's hand.
        *   Handles promotions: based on `promotion_choice` in `move_tuple` or mandatory promotion rules (see **3.A.3 Promotion Rules**).
        *   Updates `self.current_player`.
        *   Updates `self.move_history` with a hash of `(self.board, self.hands, self.current_player)`.
        *   Increments `self.move_count`.
        *   Checks for game end conditions (updates `self.game_over`, `self.winner`):
            *   Checkmate (`_is_checkmate`).
            *   Sennichite (Repetition - see **3.A.5 Game Termination**).
            *   Max moves reached (see **3.A.5 Game Termination**).
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

**3.B. `neural_network.py` (Policy and Value Networks)**

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

**3.C. `ppo_agent.py` (The Learning Agent)**

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

**3.D. `experience_buffer.py` (Stores Self-Play Data)**

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

**3.E. `train.py` (Main Training Loop)**

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

**4. Development Environment & Dependencies**

*   Python: 3.8+
*   PyTorch: 1.10+ (or latest stable)
*   NumPy: Latest stable
*   Version Control: Git
*   (Optional) TensorBoard for richer logging.

**5. Project Structure (Suggested Directory Layout)**

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

**6. Phased Approach for Coder**

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

---

## 6. Advanced Configuration System

The Keisei project implements a sophisticated, type-safe configuration system using Pydantic models that provides comprehensive control over all aspects of training, evaluation, and environment setup.

### 6.1 Configuration Architecture Overview

The configuration system is built around a central `config_schema.py` file that defines six main configuration sections:

```
Configuration Schema
├── EnvConfig: Environment and game setup
├── TrainingConfig: Advanced training features and optimization
├── EvaluationConfig: Model evaluation parameters
├── LoggingConfig: Output directories and logging
├── WandBConfig: Weights & Biases integration
└── DemoConfig: Demo mode configuration
```

### 6.2 Environment Configuration (EnvConfig)

Controls the core game environment and neural network setup:

- **Device Management**: Automatic GPU detection with fallback to CPU
- **Input Channels**: Configurable observation space (default: 46 channels for 9x9 Shogi board)
- **Action Space**: Total possible actions in the policy output space
- **Seeding**: Reproducible training runs with configurable random seeds
- **Game Parameters**: 
  - `max_moves_per_game`: Maximum moves before draw (default: 500)
  - Board representation and SFEN serialization support

### 6.3 Advanced Training Configuration (TrainingConfig)

Provides comprehensive control over modern deep learning training techniques:

**Core Training Parameters:**
- Learning rate scheduling and optimization
- Batch size management for different hardware configurations
- Total timesteps and training duration control

**Advanced Features:**
- **Mixed Precision Training**: Automatic mixed precision (AMP) support for faster training on modern GPUs
- **Distributed Data Parallel (DDP)**: Multi-GPU training coordination
- **Gradient Management**: 
  - Gradient clipping with configurable thresholds
  - Gradient accumulation for effective larger batch sizes
- **GAE Configuration**: Generalized Advantage Estimation parameters (lambda, gamma)
- **PPO Hyperparameters**: Clip epsilon, entropy coefficients, value function coefficients

**Checkpoint and Resume:**
- Configurable checkpoint intervals
- Automatic model saving and loading
- Training resume capabilities with state preservation

### 6.4 Evaluation Configuration (EvaluationConfig)

Controls model assessment and performance monitoring:

- **Evaluation Frequency**: Configurable intervals for model evaluation
- **Opponent Selection**: Self-play against previous model versions
- **Performance Metrics**: Win rate tracking, game length analysis
- **Statistical Analysis**: Confidence intervals and significance testing

### 6.5 Logging and Monitoring (LoggingConfig)

Comprehensive logging system configuration:

- **Directory Management**: Automatic creation of timestamped run directories
- **Log Levels**: Configurable verbosity for different components
- **File Organization**: Structured output for models, logs, and artifacts
- **Performance Logging**: Training metrics, loss curves, and system statistics

### 6.6 Weights & Biases Integration (WandBConfig)

Full integration with W&B for experiment tracking:

- **Project Organization**: Configurable project names and entity settings
- **Experiment Tracking**: Automatic logging of hyperparameters, metrics, and artifacts
- **Model Versioning**: Integration with W&B model registry
- **Visualization**: Real-time training curves and performance dashboards
- **Collaboration**: Team sharing and experiment comparison

### 6.7 Demo Mode Configuration (DemoConfig)

Specialized configuration for demonstration and testing:

- **Interactive Play**: Human vs AI game modes
- **Visualization**: Board state rendering and move highlighting
- **Performance Testing**: Benchmarking and profiling modes

---

## 7. Refactored Trainer Architecture

The training system has been completely refactored into a modular, manager-based architecture that separates concerns and provides clean interfaces between components.

### 7.1 Manager System Overview

The trainer is organized into 9 specialized managers, each responsible for a specific aspect of the training process:

```
Trainer Architecture
├── SessionManager: Run lifecycle and directory management
├── ModelManager: Neural network and checkpoint handling
├── EnvManager: Game environment and policy mapping
├── StepManager: Individual training step execution
├── MetricsManager: Statistics collection and analysis
├── TrainingLoopManager: Main training orchestration
├── SetupManager: Component initialization and validation
├── DisplayManager: User interface and progress visualization
└── CallbackManager: Event-driven callback system
```

### 7.2 SessionManager

**Responsibilities:**
- Training session lifecycle management
- Directory structure creation and organization
- Weights & Biases session initialization and cleanup
- Configuration validation and environment setup

**Key Features:**
- Automatic timestamped run directory creation
- W&B experiment tracking integration
- Configuration serialization and backup
- Resource cleanup and graceful shutdown

**Configuration Integration:**
- Uses `LoggingConfig` for directory management
- Integrates with `WandBConfig` for experiment tracking
- Validates all configuration sections before session start

### 7.3 ModelManager

**Responsibilities:**
- Neural network instantiation and management
- Model checkpointing and loading
- Mixed precision training coordination
- Optimizer state management

**Advanced Features:**
- **Mixed Precision Support**: Automatic mixed precision (AMP) integration
- **Checkpoint Management**: Configurable save intervals and retention policies
- **Model Versioning**: Integration with experiment tracking systems
- **State Preservation**: Complete training state serialization

**Configuration Integration:**
- Uses `TrainingConfig` for mixed precision and checkpoint settings
- Implements device management from `EnvConfig`
- Supports distributed training coordination

### 7.4 EnvManager

**Responsibilities:**
- Shogi game environment initialization
- Policy output mapper configuration
- Environment reset and state management
- Action space validation

**Shogi Game Configuration:**
- Configurable `max_moves_per_game` parameter
- SFEN notation support for game state serialization
- Seeding capabilities for reproducible games
- Observation space configuration (46 channels, 9x9 board)

**Policy Integration:**
- Action space mapping between Shogi moves and neural network outputs
- Legal move filtering and validation
- Move encoding/decoding for network communication

### 7.5 StepManager

**Responsibilities:**
- Individual training step execution
- Experience collection and batching
- Reward calculation and processing
- Transition state management

**Features:**
- Efficient batch processing of game experiences
- Automatic advantage estimation (GAE) calculation
- Reward normalization and scaling
- Memory-efficient experience buffering

### 7.6 MetricsManager

**Responsibilities:**
- Training statistics collection and aggregation
- Performance metric calculation
- Loss tracking and analysis
- System resource monitoring

**Metrics Tracked:**
- Policy and value loss components
- Entropy and exploration metrics
- Game outcome statistics (win/loss/draw rates)
- Training performance (steps/second, GPU utilization)
- System metrics (memory usage, CPU load)

### 7.7 TrainingLoopManager

**Responsibilities:**
- Main training loop orchestration
- Component coordination and scheduling
- Training phase management
- Error handling and recovery

**Advanced Features:**
- **Distributed Training**: Multi-GPU coordination when using DDP
- **Dynamic Scheduling**: Adaptive learning rate and batch size management
- **Fault Tolerance**: Automatic recovery from training interruptions
- **Resource Management**: Memory optimization and cleanup

### 7.8 SetupManager

**Responsibilities:**
- Component initialization and dependency injection
- Configuration validation and consistency checking
- Hardware detection and optimization
- Pre-training system validation

**Validation Features:**
- Configuration schema validation using Pydantic
- Hardware capability detection (GPU, memory, CUDA)
- Dependency version checking
- Pre-flight system tests

### 7.9 DisplayManager

**Responsibilities:**
- Training progress visualization
- Real-time metrics display
- User interface coordination
- Progress reporting and notifications

**Features:**
- Rich terminal interface with progress bars
- Real-time loss and metric visualization
- Training time estimation and completion forecasting
- Interactive training control (pause/resume)

### 7.10 CallbackManager

**Responsibilities:**
- Event-driven callback system
- Custom hook integration
- Training event handling
- Extension point management

**Callback Types:**
- Training phase callbacks (epoch start/end, batch processing)
- Model callbacks (checkpoint save/load, evaluation)
- Metrics callbacks (logging, visualization updates)
- Custom user-defined callbacks

---

## 8. Configuration and Manager Integration

### 8.1 Configuration Flow

The configuration system integrates seamlessly with the manager architecture:

1. **Schema Validation**: Pydantic models validate all configuration parameters
2. **Manager Injection**: Configuration sections are injected into relevant managers
3. **Runtime Adaptation**: Managers adapt behavior based on configuration settings
4. **State Persistence**: Configuration state is preserved across training resumption

### 8.2 Manager Communication

Managers communicate through well-defined interfaces:

- **Event System**: Managers publish and subscribe to training events
- **Shared State**: Thread-safe shared state objects for coordination
- **Configuration Registry**: Centralized configuration access
- **Dependency Injection**: Automatic dependency resolution between managers

### 8.3 Extensibility and Customization

The architecture supports extensive customization:

- **Custom Managers**: Easy addition of new manager types
- **Configuration Extensions**: Plugin-based configuration schema extensions
- **Callback Hooks**: User-defined callbacks for custom training logic
- **Component Replacement**: Swappable implementations for core components

---

## 9. Modern Training Features

### 9.1 Mixed Precision Training

Automatic mixed precision (AMP) support provides:
- **Performance**: 1.5-2x training speedup on modern GPUs
- **Memory Efficiency**: Reduced GPU memory usage
- **Numerical Stability**: Automatic loss scaling and overflow detection
- **Backward Compatibility**: Graceful fallback for older hardware

### 9.2 Distributed Training

Multi-GPU training support includes:
- **Data Parallel**: Distributed data parallel (DDP) training
- **Gradient Synchronization**: Efficient gradient aggregation across GPUs
- **Load Balancing**: Automatic batch distribution and workload balancing
- **Fault Tolerance**: Resilient training with node failure recovery

### 9.3 Advanced Optimization

Sophisticated optimization features:
- **Gradient Clipping**: Configurable gradient norm clipping
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Batch Size Scaling**: Dynamic batch size optimization
- **Memory Management**: Efficient memory usage and garbage collection

### 9.4 Checkpoint and Resume

Comprehensive training state management:
- **Full State Preservation**: Complete training state serialization
- **Incremental Checkpoints**: Efficient incremental state saving
- **Version Compatibility**: Forward and backward compatible checkpoint formats
- **Cloud Integration**: Support for cloud-based checkpoint storage

This advanced configuration and training system provides a robust foundation for scalable, reproducible, and efficient deep reinforcement learning training for Shogi AI development.