# Phase 1 – Infrastructure Setup

This phase establishes the foundations required by later features.  Files referenced here live under the `training/` directory unless otherwise noted.

## Goals

1. Create a reusable display component module.
2. Track historical metrics required for sparklines and Elo calculations.
3. Provide a configuration model for optional TUI features.

## Tasks

### 1. Display Component Framework
- **Create File**: `training/display_components.py`.
- **Purpose**: Host classes such as `ShogiBoard`, `Sparkline`, and future display widgets.
- **Initial Content**:
  - Define a base `DisplayComponent` protocol with a `render()` method returning a Rich `RenderableType`.
  - Implement minimal `ShogiBoard` and `Sparkline` stubs returning placeholder panels.  Real logic is added in Phase 2.

### 2. Metrics History Tracking
- **Modify**: `training/metrics_manager.py`.
- **Add Class**: `MetricsHistory` with attributes for win rate history, learning rates, policy/value losses and KL divergence.
- Provide `add_episode_data()` and `add_ppo_data()` methods to append metrics and trim history length.
- Instantiate `MetricsHistory` within `MetricsManager.__init__`.

### 3. Configuration Model
- **Add Model**: `DisplayConfig` in `config_schema.py` (or a new module if preferred) with fields outlined in the enhancement plan.
- **Default Values**: Reflect defaults from `TUI_ENHANCEMENT_PLAN.md`.
- **Integration**: Extend `AppConfig` with a `display: DisplayConfig` section. Ensure loading from YAML continues to validate correctly.

### 4. TrainingDisplay Preparation
- **File**: `training/display.py`.
- Introduce optional attributes for the new components (`board_component`, `trend_component`, `elo_component`). Use `DisplayConfig` to determine which are active.
- Implement placeholder slots in the current layout so later phases can insert new panels.

## Testing Notes

- Add unit tests verifying `MetricsHistory` trimming behaviour and default values of `DisplayConfig`.
- Existing tests should continue to pass without enabling the new features.
