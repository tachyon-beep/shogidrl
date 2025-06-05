# Phase 3 – Enhanced Layout and Adaptation

This phase reorganizes the interface into a dashboard style layout and adapts to different terminal sizes.

## Goals

1. Add a multi-panel dashboard showing board, trends and Elo information.
2. Provide fallbacks for small terminals.
3. Preserve or improve overall refresh performance.

## Tasks

### 1. Layout Construction
- **File**: `training/display.py`.
- Create `_setup_enhanced_layout()` as proposed in the enhancement plan.
- Divide the screen into three dashboard columns plus the log and progress bar sections.
- Expose a method to choose between the enhanced and legacy layouts.

### 2. Adaptive Display Manager
- **File**: `training/display_manager.py` or a new module.
- Implement `AdaptiveDisplayManager` using `DisplayConfig` to decide which layout to create based on `os.get_terminal_size()` results.
- Provide safe fallbacks for terminals that lack Unicode or have limited width/height.

### 3. Component Integration
- Ensure the `TrainingDisplay` refresh loop renders the three new panels when enabled.
- Panels should degrade gracefully if a component is disabled or unsupported.

## Testing Notes

- Write integration tests to instantiate the display with varying terminal sizes and verify which layout is chosen.
- Profile update speed with the enhanced layout to ensure refresh intervals remain under the configured limit.
