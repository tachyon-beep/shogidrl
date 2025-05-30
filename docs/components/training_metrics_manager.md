# Software Documentation Template for Subsystems - Training Metrics Manager

## 📘 training_metrics_manager.py as of 2024-12-19

**Project Name:** `Keisei Shogi DRL System`
**Folder Path:** `/keisei/training/metrics_manager.py`
**Documentation Version:** `1.0`
**Date:** `2024-12-19`
**Responsible Author or Agent ID:** `Documentation Agent`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Manages training statistics, metrics tracking, PPO metrics formatting, and progress update handling with comprehensive game outcome tracking.

* **Key Responsibilities:**
  - Track game outcome statistics (wins, losses, draws by color)
  - Process and format PPO learning metrics for display and logging
  - Manage progress updates for real-time display
  - Calculate win rates and episode metrics
  - Handle checkpoint/restore operations for statistics persistence

* **Domain Context:**
  Training metrics and statistics management in PPO-based DRL system, specifically handling Shogi game outcomes and reinforcement learning metrics.

* **High-Level Architecture / Interaction Summary:**
  
  The MetricsManager serves as the central repository for training statistics and metrics formatting. It tracks game outcomes, processes PPO learning metrics, and provides formatted output for display and logging. The manager integrates with the training loop to provide real-time progress updates and maintains persistent statistics across training sessions.

---

### 2. Modules 📦

* **Module Name:** `metrics_manager.py`

  * **Purpose:** Centralized training statistics and metrics management
  * **Design Patterns Used:** Manager pattern for metrics lifecycle, Data container pattern with TrainingStats
  * **Key Functions/Classes Provided:** 
    - `MetricsManager` class for metrics orchestration
    - `TrainingStats` dataclass for statistics storage
    - PPO metrics formatting and progress tracking
  * **Configuration Surface:** No direct configuration; uses metrics data from training loop

---

### 3. Classes and Functions 🏗️

#### Dataclass: `TrainingStats`

**Purpose:** Container for training statistics with default values.

**Key Attributes:**
- `global_timestep`: Current global timestep counter (default: 0)
- `total_episodes_completed`: Total completed episodes (default: 0)
- `black_wins`: Number of Black player wins (default: 0)
- `white_wins`: Number of White player wins (default: 0)
- `draws`: Number of drawn games (default: 0)

#### Class: `MetricsManager`

**Purpose:** Central manager for training statistics, PPO metrics processing, and progress tracking.

**Key Attributes:**
- `stats`: TrainingStats instance containing all statistics
- `pending_progress_updates`: Dictionary of pending progress updates for display

**Key Methods:**

##### `__init__()`
- **Purpose:** Initialize metrics manager with zero statistics
- **Parameters:** None
- **Return Type:** None
- **Key Behavior:**
  - Creates TrainingStats instance with default values
  - Initializes empty pending progress updates dictionary
- **Usage:** Called during trainer initialization

##### `update_episode_stats(winner_color: Optional[Color]) -> Dict[str, float]`
- **Purpose:** Update episode statistics based on game outcome
- **Parameters:**
  - `winner_color`: Color of winner (Black/White) or None for draw
- **Return Type:** Dictionary with updated win rates
- **Key Behavior:**
  - Increments total episodes completed
  - Updates appropriate win/draw counter based on outcome
  - Returns current win rates as dictionary
- **Usage:** Called at end of each game episode

##### `get_win_rates() -> Tuple[float, float, float]`
- **Purpose:** Calculate win rates as percentages
- **Parameters:** None
- **Return Type:** Tuple of (black_rate, white_rate, draw_rate) percentages
- **Key Behavior:**
  - Calculates percentages based on total episodes
  - Returns (0.0, 0.0, 0.0) if no episodes completed
  - All rates sum to 100% (within floating point precision)
- **Usage:** For display and logging of training progress

##### `get_win_rates_dict() -> Dict[str, float]`
- **Purpose:** Get win rates as a dictionary for logging
- **Parameters:** None
- **Return Type:** Dictionary with win rate keys and percentage values
- **Key Structure:**
  ```python
  {
      "win_rate_black": float,
      "win_rate_white": float,
      "win_rate_draw": float
  }
  ```
- **Usage:** For structured logging and metrics reporting

##### `format_episode_metrics(episode_length: int, episode_reward: float) -> str`
- **Purpose:** Format episode completion metrics for display
- **Parameters:**
  - `episode_length`: Number of steps in the episode
  - `episode_reward`: Total reward for the episode
- **Return Type:** Formatted string with episode metrics
- **Format:** `"Ep {num}: Len={length}, R={reward:.3f}, B={black}%, W={white}%, D={draw}%"`
- **Usage:** For console output and progress display

##### `format_ppo_metrics(learn_metrics: Dict[str, float]) -> str`
- **Purpose:** Format PPO learning metrics for compact display
- **Parameters:**
  - `learn_metrics`: Dictionary of PPO metrics from agent.learn()
- **Return Type:** Formatted string with key PPO metrics
- **Key Metrics:**
  - KL divergence approximation (`ppo/kl_divergence_approx`)
  - Policy loss (`ppo/policy_loss`)
  - Value loss (`ppo/value_loss`)
  - Entropy (`ppo/entropy`)
- **Format:** Space-separated abbreviated metrics (e.g., "KL:0.0123 PolL:0.0456")
- **Usage:** For compact console display during training

##### `format_ppo_metrics_for_logging(learn_metrics: Dict[str, float]) -> str`
- **Purpose:** Format PPO metrics for detailed logging in JSON format
- **Parameters:**
  - `learn_metrics`: Dictionary of PPO metrics
- **Return Type:** JSON-formatted string of metrics
- **Key Behavior:**
  - Formats all metric values to 4 decimal places
  - Converts to JSON string for structured logging
- **Usage:** For detailed log files and external monitoring

##### `update_progress_metrics(key: str, value: Any) -> None`
- **Purpose:** Store a progress update for later display
- **Parameters:**
  - `key`: Update identifier (e.g., 'ppo_metrics', 'speed')
  - `value`: Update value of any type
- **Return Type:** None
- **Usage:** For accumulating display updates before rendering

##### `get_progress_updates() -> Dict[str, Any]`
- **Purpose:** Retrieve current pending progress updates
- **Parameters:** None
- **Return Type:** Copy of pending progress updates dictionary
- **Usage:** For accessing updates without modifying internal state

##### `clear_progress_updates() -> None`
- **Purpose:** Clear pending progress updates after display
- **Parameters:** None
- **Return Type:** None
- **Usage:** Called after progress updates have been displayed

##### `get_final_stats() -> Dict[str, int]`
- **Purpose:** Get final game statistics for saving with model
- **Parameters:** None
- **Return Type:** Dictionary with all final statistics
- **Key Structure:**
  ```python
  {
      "black_wins": int,
      "white_wins": int,
      "draws": int,
      "total_episodes_completed": int,
      "global_timestep": int
  }
  ```
- **Usage:** For model checkpointing and final results recording

##### `restore_from_checkpoint(checkpoint_data: Dict[str, Any]) -> None`
- **Purpose:** Restore statistics from checkpoint data
- **Parameters:**
  - `checkpoint_data`: Dictionary with saved statistics
- **Return Type:** None
- **Key Behavior:**
  - Restores all statistics with fallback to default values
  - Handles missing keys gracefully
- **Usage:** For resuming training from saved checkpoints

##### `increment_timestep() -> None`
- **Purpose:** Increment the global timestep counter
- **Parameters:** None
- **Return Type:** None
- **Usage:** Called at each training step to track progress

##### Properties (Backward Compatibility)
- **`global_timestep`** - Get/set current global timestep
- **`total_episodes_completed`** - Get/set total completed episodes
- **`black_wins`** - Get/set Black player wins
- **`white_wins`** - Get/set White player wins
- **`draws`** - Get/set drawn games

---

### 4. Data Structures 📊

#### TrainingStats Structure

```python
@dataclass
class TrainingStats:
    global_timestep: int = 0              # Current training step
    total_episodes_completed: int = 0     # Total episodes finished
    black_wins: int = 0                   # Black player victories
    white_wins: int = 0                   # White player victories
    draws: int = 0                        # Drawn games
```

#### Progress Updates Dictionary

```python
pending_progress_updates: Dict[str, Any] = {
    "ppo_metrics": str,                   # Formatted PPO metrics
    "speed": float,                       # Training speed (steps/sec)
    "episode_metrics": str,               # Episode completion summary
    # ... other progress indicators
}
```

#### Win Rates Dictionary

```python
win_rates: Dict[str, float] = {
    "win_rate_black": float,              # Black win percentage
    "win_rate_white": float,              # White win percentage
    "win_rate_draw": float                # Draw percentage
}
```

---

### 5. Inter-Module Relationships 🔗

#### Dependencies:
- **`keisei.shogi.shogi_core_definitions`** - Color enum for game outcomes
- **`json`** - JSON formatting for detailed metrics logging
- **`dataclasses`** - TrainingStats container implementation

#### Used By:
- **`trainer.py`** - Main training orchestrator for metrics management
- **`training_loop_manager.py`** - Episode completion and progress tracking
- **`step_manager.py`** - Timestep incrementing and PPO metrics processing
- **`display_manager.py`** - Progress updates for visual display

#### Provides To:
- **Training Infrastructure** - Centralized statistics and metrics formatting
- **Display System** - Formatted metrics for console and UI display
- **Checkpointing System** - Statistics persistence and restoration
- **Logging System** - Structured metrics for external monitoring

---

### 6. Implementation Notes 🔧

#### Statistics Management:
- Uses dataclass for clean statistics storage
- Property-based interface for backward compatibility
- Immutable copy semantics for progress updates

#### Metrics Formatting:
- Two-tier formatting: compact for display, detailed for logging
- Robust handling of missing metrics keys
- Consistent decimal precision for metrics display

#### Progress Updates:
- Accumulator pattern for batching display updates
- Key-value storage allows flexible update types
- Clear operation prevents stale updates

---

### 7. Testing Strategy 🧪

#### Unit Tests:
- Test statistics updates with various game outcomes
- Verify win rate calculations with edge cases (zero episodes)
- Test metrics formatting with complete and partial PPO metrics
- Validate checkpoint save/restore functionality

#### Integration Tests:
- Test metrics integration with actual training loops
- Verify progress update lifecycle during training
- Test metrics persistence across training sessions

#### Edge Cases:
- Zero episodes completed (division by zero protection)
- Missing PPO metrics keys in formatting
- Invalid checkpoint data during restoration
- Large number handling for long training runs

---

### 8. Performance Considerations ⚡

#### Memory Usage:
- TrainingStats uses minimal memory for statistics
- Progress updates dictionary may accumulate if not cleared
- JSON formatting creates temporary string objects

#### Computation Overhead:
- Win rate calculations involve division operations
- String formatting for display has minimal overhead
- Dictionary copying for progress updates has small cost

#### Optimization Strategies:
- Cache win rate calculations if updated frequently
- Limit progress update dictionary size
- Use efficient string formatting for large metric sets

---

### 9. Security Considerations 🔒

#### Data Integrity:
- No validation of statistics consistency or plausibility
- External code can modify statistics through property setters
- Checkpoint data trusted without validation

#### Information Disclosure:
- Metrics may reveal training details or performance characteristics
- Progress updates could contain sensitive information
- JSON formatting exposes all metrics without filtering

#### Mitigation Strategies:
- Implement statistics validation and bounds checking
- Add access controls for sensitive metrics
- Filter or redact sensitive information in formatted output

---

### 10. Error Handling 🚨

#### Missing Data Handling:
- Graceful defaults for missing checkpoint data
- Robust handling of missing PPO metrics keys
- Zero-division protection in win rate calculations

#### Invalid Input Handling:
- No validation of episode outcomes or metrics values
- Property setters allow invalid negative values
- Progress update values not validated

#### Recovery Strategies:
- Default values allow continued operation with incomplete data
- Missing metrics keys don't break formatting operations
- Statistics can be manually corrected if needed

---

### 11. Configuration ⚙️

#### No Direct Configuration:
- MetricsManager requires no configuration parameters
- Operates on data provided by training components
- Formatting options are hard-coded but could be configurable

#### Implicit Configuration:
- PPO metrics keys defined by agent implementation
- Progress update keys defined by usage patterns
- Display format determined by console requirements

---

### 12. Future Enhancements 🚀

#### Advanced Statistics:
- Elo rating calculation for player strength estimation
- Moving averages for smooth win rate trends
- Statistical significance testing for performance changes
- Advanced episode analysis (length distributions, reward patterns)

#### Enhanced Formatting:
- Configurable metrics display formats
- Templated output for different audiences
- Real-time metrics streaming to external systems
- Custom metrics aggregation and reporting

#### Performance Monitoring:
- Metrics collection performance profiling
- Memory usage tracking for long training runs
- Automated anomaly detection in training metrics
- Comparative analysis across training runs

---

### 13. Usage Examples 💡

#### Basic Episode Tracking:
```python
# Initialize metrics manager
metrics = MetricsManager()

# Track episode completion
winner_color = Color.BLACK  # or Color.WHITE or None for draw
win_rates = metrics.update_episode_stats(winner_color)

# Format episode summary
episode_summary = metrics.format_episode_metrics(
    episode_length=45, 
    episode_reward=12.5
)
print(episode_summary)  # "Ep 1: Len=45, R=12.500, B=100.0%, W=0.0%, D=0.0%"
```

#### PPO Metrics Processing:
```python
# Process PPO learning metrics
ppo_metrics = {
    'ppo/kl_divergence_approx': 0.0123,
    'ppo/policy_loss': 0.0456,
    'ppo/value_loss': 0.0789,
    'ppo/entropy': 1.2345
}

# Format for display
display_metrics = metrics.format_ppo_metrics(ppo_metrics)
print(display_metrics)  # "KL:0.0123 PolL:0.0456 ValL:0.0789 Ent:1.2345"

# Format for logging
log_metrics = metrics.format_ppo_metrics_for_logging(ppo_metrics)
print(log_metrics)  # JSON string with 4 decimal places
```

#### Progress Updates Management:
```python
# Accumulate progress updates
metrics.update_progress_metrics('speed', 15.2)
metrics.update_progress_metrics('ppo_metrics', display_metrics)

# Get all pending updates
updates = metrics.get_progress_updates()

# Clear after display
metrics.clear_progress_updates()
```

#### Checkpoint Operations:
```python
# Save final statistics
final_stats = metrics.get_final_stats()
save_checkpoint({'metrics': final_stats})

# Restore from checkpoint
checkpoint_data = load_checkpoint()
metrics.restore_from_checkpoint(checkpoint_data.get('metrics', {}))
```

---

### 14. Maintenance Notes 📝

#### Regular Maintenance:
- Monitor memory usage of progress updates dictionary
- Review metrics formatting for new PPO metrics keys
- Validate statistics accuracy during long training runs

#### Version Compatibility:
- PPO metrics keys may change with agent implementation updates
- Property interface maintained for backward compatibility
- Checkpoint format should remain stable across versions

#### Code Quality:
- Maintain consistent decimal precision in formatting
- Keep statistics operations atomic and consistent
- Document expected metrics keys and formats

---

### 15. Related Documentation 📚

- **`training_trainer.md`** - Main trainer integration with metrics manager
- **`training_step_manager.md`** - PPO metrics generation and timestep tracking
- **`training_display_manager.md`** - Progress update display and formatting
- **`core_ppo_agent.md`** - PPO metrics generation from agent.learn()
- **`shogi_core_definitions.md`** - Color enum and game outcome definitions
