# Display Fixes and Metrics Enhancement Plan

## Overview
This document outlines the implementation plan for enhancing the Keisei DRL Shogi training system's metric trends display with additional metrics, rolling averages, and trendlines.

## Completed Fixes

### 1. TrainingConfig Parameter Fix
**File:** `/home/john/keisei/tests/test_model_save_load.py`
**Issue:** Missing `enable_value_clipping` parameter in TrainingConfig instantiation
**Fix:** Added `enable_value_clipping=False` parameter to satisfy Pydantic validation
**Status:** ✅ Complete

### 2. EvaluationConfig Parameter Fix
**File:** `/home/john/keisei/tests/test_model_save_load.py`
**Issue:** Missing Elo-related parameters in EvaluationConfig instantiation
**Fix:** Added the following parameters:
- `elo_registry_path=None`
- `agent_id=None`
- `opponent_id=None`
- `previous_model_pool_size=5`
**Status:** ✅ Complete

### 3. DisplayConfig Instantiation Fix
**File:** `/home/john/keisei/keisei/training/display.py`
**Issue:** Pylance error with `getattr(config, "display", DisplayConfig())` pattern
**Fix:** Changed to direct access `config.display` since AppConfig already includes DisplayConfig with default_factory
**Status:** ✅ Complete

## Pending Enhancement: Metrics Trends Display

### Current Metrics Infrastructure Analysis

#### Core Classes and Files:
1. **MetricsManager** (`/home/john/keisei/keisei/training/metrics_manager.py`)
   - Current metrics: PPO metrics, win rates, ELO ratings, game statistics
   - Methods: `update_metrics()`, `get_recent_metrics()`, `log_episode_metrics()`
   - Storage: In-memory with optional W&B logging

2. **TrainingDisplay** (`/home/john/keisei/keisei/training/display.py`)
   - Current display: Progress bars, win rate trends, ELO progression
   - Components: Uses Sparkline for trend visualization
   - Update frequency: Real-time during training

3. **Display Components** (`/home/john/keisei/keisei/training/display_components.py`)
   - Sparkline class for trend visualization
   - Progress indicators and status displays

### Enhancement Specifications

#### New Metrics to Add:
1. **Moves Made per Game**
   - Track total moves in each game
   - Calculate rolling average over configurable window (default: 100 games)
   - Display trend to show game complexity evolution

2. **Games Completed Rate**
   - Track games completed per hour/episode batch
   - Calculate throughput metrics
   - Display training efficiency trends

3. **Enhanced Win/Draw Rates**
   - Current: Basic win rate tracking
   - Enhancement: Separate win, loss, draw rate tracking
   - Add opponent-specific win rates if available

4. **Average Turns per Game**
   - Track game length in turns
   - Rolling average with trendlines
   - Correlation analysis with model performance

#### Implementation Plan

##### Phase 1: Extend MetricsManager
**File:** `/home/john/keisei/keisei/training/metrics_manager.py`

**Modifications Required:**
1. **Add new metric tracking fields:**
   ```python
   # Add to __init__ method
   self.moves_per_game: List[int] = []
   self.games_completed_timestamps: List[float] = []
   self.win_loss_draw_history: List[Tuple[str, float]] = []  # (result, timestamp)
   self.turns_per_game: List[int] = []
   ```

2. **Extend `log_episode_metrics()` method:**
   - Add parameters for moves_made, game_result, turns_count
   - Store timestamp for throughput calculations
   - Update rolling averages

3. **Add new calculation methods:**
   ```python
   def get_moves_per_game_trend(self, window_size: int = 100) -> List[float]
   def get_games_completion_rate(self, time_window_hours: float = 1.0) -> float
   def get_win_loss_draw_rates(self, window_size: int = 100) -> Dict[str, float]
   def get_average_turns_trend(self, window_size: int = 100) -> List[float]
   ```

**Key Integration Points:**
- Line ~150-200: `log_episode_metrics()` method enhancement
- Line ~250-300: Add new getter methods for trend data
- Line ~50-100: Update `__init__` and class attributes

##### Phase 2: Enhance Display Components
**File:** `/home/john/keisei/keisei/training/display_components.py`

**New Components Required:**
1. **MultiMetricSparkline Class:**
   ```python
   class MultiMetricSparkline:
       def __init__(self, width: int, height: int, metrics: List[str])
       def add_data_point(self, metric_name: str, value: float)
       def render_with_trendlines(self) -> str
   ```

2. **RollingAverageCalculator:**
   ```python
   class RollingAverageCalculator:
       def __init__(self, window_size: int)
       def add_value(self, value: float) -> float  # Returns current rolling avg
       def get_trend_direction(self) -> str  # "↑", "↓", "→"
   ```

**Integration Points:**
- Line ~1-50: Import and class definitions
- Line ~100-150: Extend existing Sparkline functionality
- Line ~200-250: Add trend calculation utilities

##### Phase 3: Update Training Display
**File:** `/home/john/keisei/keisei/training/display.py`

**Modifications Required:**
1. **Extend `update_display()` method (around line 200-250):**
   ```python
   # Add new metric displays
   moves_trend = self.metrics_manager.get_moves_per_game_trend()
   completion_rate = self.metrics_manager.get_games_completion_rate()
   wld_rates = self.metrics_manager.get_win_loss_draw_rates()
   turns_trend = self.metrics_manager.get_average_turns_trend()
   ```

2. **Add new display sections:**
   - Game complexity metrics panel
   - Training efficiency panel
   - Enhanced performance breakdown

3. **Update layout configuration in `_create_layout()` method (around line 100-150):**
   - Add new metric panels to existing layout
   - Ensure responsive design for different terminal sizes

**Key Files to Import:**
- Update imports to include new display components
- Add configuration options for metric display preferences

##### Phase 4: Configuration Schema Updates
**File:** `/home/john/keisei/keisei/config_schema.py`

**DisplayConfig Enhancements:**
```python
@dataclass
class DisplayConfig:
    # ...existing fields...
    
    # New metric display options
    show_moves_trend: bool = True
    show_completion_rate: bool = True
    show_enhanced_win_rates: bool = True
    show_turns_trend: bool = True
    
    # Rolling average configuration
    metrics_window_size: int = 100
    trend_smoothing_factor: float = 0.1
    
    # Display layout options
    metrics_panel_height: int = 6
    enable_trendlines: bool = True
```

**Integration Points:**
- Line ~200-250: DisplayConfig class extension
- Line ~50-100: Add imports for new metric types
- Ensure backward compatibility with existing configurations

### Implementation Order and Dependencies

#### Step 1: Core Metrics Collection
1. Modify `MetricsManager.__init__()` to add new data structures
2. Update `log_episode_metrics()` to accept and store new metrics
3. Add basic getter methods for new metrics
4. **Test:** Unit tests for new metric collection

#### Step 2: Calculation and Trend Analysis
1. Implement rolling average calculations
2. Add trend direction detection
3. Implement throughput calculations
4. **Test:** Validate rolling averages and trend calculations

#### Step 3: Display Components
1. Create `MultiMetricSparkline` class
2. Extend existing display components
3. Add trendline rendering capabilities
4. **Test:** Visual validation of new components

#### Step 4: Integration with Training Display
1. Update `TrainingDisplay.update_display()` method
2. Modify layout to accommodate new metrics
3. Add configuration-driven display options
4. **Test:** Full integration testing with training loop

#### Step 5: Configuration and Documentation
1. Update `DisplayConfig` schema
2. Add example configurations
3. Update documentation and help text
4. **Test:** Configuration validation and user acceptance

### Technical Considerations

#### Performance Implications:
- **Memory Usage:** New metrics storage will increase memory footprint
- **Calculation Overhead:** Rolling averages and trend calculations per update
- **Display Refresh:** Additional rendering for new components

#### Mitigation Strategies:
- Use efficient data structures (deque for rolling windows)
- Implement configurable update frequencies
- Add memory cleanup for long-running training sessions

#### Backward Compatibility:
- All new features are opt-in via configuration
- Existing display behavior unchanged by default
- Graceful degradation if new metrics unavailable

### Testing Strategy

#### Unit Tests Required:
1. **MetricsManager Tests:** `/home/john/keisei/tests/test_metrics_manager.py`
   - Test new metric collection methods
   - Validate rolling average calculations
   - Test throughput calculations

2. **Display Component Tests:** `/home/john/keisei/tests/test_display_components.py`
   - Test MultiMetricSparkline rendering
   - Validate trendline generation
   - Test layout responsiveness

3. **Integration Tests:** `/home/john/keisei/tests/test_training_display.py`
   - Test full display update cycle
   - Validate configuration handling
   - Test error handling and edge cases

#### Manual Testing Scenarios:
1. **Short Training Session:** Verify metrics appear correctly with limited data
2. **Long Training Session:** Test memory usage and performance over time
3. **Configuration Variations:** Test different display options and layouts
4. **Terminal Resize:** Verify responsive layout behavior

### Future Enhancements

#### Phase 2 Features (Post-Implementation):
1. **Historical Trend Analysis:** Compare current training session to previous sessions
2. **Predictive Indicators:** ML-based trend prediction for training outcomes
3. **Export Capabilities:** Save trend data for external analysis
4. **Real-time Alerts:** Notifications for significant trend changes

#### Integration Opportunities:
1. **W&B Dashboard:** Sync enhanced metrics to Weights & Biases
2. **TensorBoard:** Export trend data for TensorBoard visualization
3. **REST API:** Expose metrics via HTTP endpoint for external monitoring

## Dependencies and Prerequisites

### Required Libraries:
- No new external dependencies required
- Uses existing Rich, NumPy, and dataclasses infrastructure

### Development Environment:
- Python 3.8+ (current project requirement)
- Rich terminal library (already in requirements.txt)
- Pytest for testing (already in requirements-dev.txt)

### Configuration Files:
- Update example configurations in `/home/john/keisei/examples/`
- Add documentation in `/home/john/keisei/docs/`

## Estimated Implementation Time

- **Phase 1 (Core Metrics):** 2-3 days
- **Phase 2 (Display Components):** 3-4 days  
- **Phase 3 (Integration):** 2-3 days
- **Phase 4 (Configuration):** 1-2 days
- **Testing and Documentation:** 2-3 days

**Total Estimated Time:** 10-15 development days

## Success Criteria

1. **Functional Requirements:**
   - All new metrics display correctly during training
   - Rolling averages and trends update in real-time
   - Configuration options work as specified
   - No performance degradation in training loop

2. **Quality Requirements:**
   - All unit tests pass
   - Integration tests validate full functionality
   - Code coverage maintains current levels
   - Documentation is complete and accurate

3. **User Experience:**
   - Enhanced metrics provide valuable training insights
   - Display remains clean and readable
   - Configuration is intuitive and well-documented
   - Backward compatibility is maintained

---

**Document Status:** Ready for Implementation  
**Last Updated:** June 6, 2025  
**Next Review:** After Phase 1 completion
