# Phase 2 – Core Feature Development

In this stage the display components gain real functionality.  The metrics manager feeds real-time data to the new panels.

## Goals

1. Show an ASCII representation of the current game board.
2. Visualize metric trends with sparklines.
3. Maintain and display Elo ratings for black and white.

## Tasks

### 1. Implement `ShogiBoard`
- **File**: `training/display_components.py`.
- Replace the stub with methods described in `TUI_ENHANCEMENT_PLAN.md`.
- Access the board state via `MetricsManager` or the training loop as appropriate.  Track the last position to avoid unnecessary rendering work.

### 2. Implement `Sparkline`
- Extend the existing placeholder to generate Unicode sparklines.  Follow the normalization algorithm from the plan.
- Parameterize width using `DisplayConfig.sparkline_width`.

### 3. Create `EloRatingSystem`
- **File**: new `training/elo_rating.py`.
- Implement rating calculation, update method and trend history as outlined.
- Add a helper `get_strength_assessment()` returning a human readable summary.

### 4. Integrate Metric History
- Update `MetricsManager` to record episode outcomes and PPO metrics by calling `MetricsHistory` methods.
- Add an `elo_system` attribute and invoke `update_ratings()` when episodes finish.

### 5. Update Display
- In `training/display.py`, instantiate the components when `DisplayConfig` enables them.
- Add rendering logic to output the board, trends and Elo panels using Rich `Panel` objects.

## Testing Notes

- Unit-test `EloRatingSystem` for win, loss and draw scenarios.
- Add tests for sparkline generation edge cases (flat data, varying ranges).
- Ensure board rendering handles an empty or missing board gracefully.
