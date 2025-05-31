# Software Documentation Template for Subsystems - Training Display UI

## 📘 training_display.py as of 2024-12-19

**Project Name:** `Keisei Shogi DRL`
**Folder Path:** `keisei/training/display.py`
**Documentation Version:** `1.0`
**Date:** `2024-12-19`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Module:**
  Provides Rich-based terminal user interface components for the Shogi RL trainer. Implements comprehensive progress visualization, real-time log display, and training metrics presentation in an interactive console interface. Manages layout, progress bars, and live updating displays for enhanced training monitoring.

* **Key Responsibilities:**
  - Rich UI layout design and management
  - Real-time progress bar with comprehensive metrics display
  - Live log panel with scrolling message display
  - Training statistics visualization (win rates, episode counts, PPO metrics)
  - Console layout orchestration and refresh management
  - Dynamic UI updates based on training state

* **Domain Context:**
  User interface layer in PPO-based deep reinforcement learning system for Shogi gameplay. Provides real-time visual feedback for training progress, performance metrics, and system status to enhance training monitoring and debugging.

* **High-Level Architecture / Interaction Summary:**
  Rich-based UI class that creates and manages terminal interface components for training visualization. Integrates with trainer state and DisplayManager to provide real-time updates of progress, metrics, and log messages through a structured layout system.

---

### 2. Module Details 📦

* **Module Name:** `display.py`
  
  * **Purpose:** Rich-based terminal UI implementation for training visualization
  * **Design Patterns Used:** 
    - Component pattern for UI element organization
    - Observer pattern for real-time updates
    - Layout pattern for UI structure management
    - Factory pattern for UI component creation
  * **Key Functions/Classes Provided:**
    - `TrainingDisplay` - Main UI class for training visualization
  * **Configuration Surface:**
    - Progress bar configuration and column selection
    - Spinner enable/disable options
    - Refresh rate and update frequency
    - Console layout and panel sizing
  * **Dependencies:**
    - **Internal:**
      - Trainer class (for state access and log messages)
      - Configuration objects (for display settings)
    - **External:**
      - `rich` - Terminal UI framework and components
      - `typing` - Type hints for method signatures
  * **External API Contracts:**
    - **Rich Integration:** Uses Rich console, layout, progress, and panel components
    - **Trainer Interface:** Expects specific trainer attributes and log message format

---

### 3. Classes 🏗️

#### `TrainingDisplay`
**Purpose:** Main Rich-based UI class that manages comprehensive training visualization with real-time progress bars, metrics display, and live log panels.

**Initialization Parameters:**
- `config` - Training configuration object
- `trainer` - Trainer instance for state access
- `rich_console: Console` - Rich console instance for rendering

**Key Attributes:**
```python
self.config                 # Training configuration
self.trainer               # Trainer instance reference
self.rich_console          # Rich console for rendering
self.rich_log_messages     # Reference to trainer log messages
self.progress_bar          # Rich Progress instance
self.training_task         # Progress task ID
self.layout               # Rich Layout for UI structure
self.log_panel            # Rich Panel for log display
```

**Key Methods:**

##### `__init__(self, config, trainer, rich_console: Console)`
**Purpose:** Initialize TrainingDisplay with configuration, trainer reference, and Rich console.

**Initialization Process:**
1. Store configuration and trainer references
2. Access trainer's rich_log_messages for live display
3. Setup Rich progress display components via `_setup_rich_progress_display()`

##### `_setup_rich_progress_display()`
**Purpose:** Creates and configures the complete Rich UI layout with progress bar, metrics, and log panel.

**Returns:** 
- `Tuple[Progress, TaskID, Layout, Panel]` - UI components for display management

**UI Component Configuration:**

**Progress Bar Columns:**
```python
progress_columns = [
    "[progress.description]{task.description}",     # Task description
    BarColumn(),                                    # Visual progress bar
    TaskProgressColumn(),                           # Completion percentage
    TextColumn("•"),                               # Separator
    TimeElapsedColumn(),                           # Elapsed time
    TextColumn("•"),                               # Separator
    TimeRemainingColumn(),                         # Estimated remaining time
    TextColumn("• Steps: {task.completed}/{task.total} ({task.fields[speed]:.1f} it/s)"),
    TextColumn("• {task.fields[ep_metrics]}", style="bright_cyan"),
    TextColumn("• {task.fields[ppo_metrics]}", style="bright_yellow"),
    TextColumn("• Wins B:{task.fields[black_wins_cum]} W:{task.fields[white_wins_cum]} D:{task.fields[draws_cum]}", style="bright_green"),
    TextColumn("• Rates B:{task.fields[black_win_rate]:.1%} W:{task.fields[white_win_rate]:.1%} D:{task.fields[draw_rate]:.1%}", style="bright_blue")
]
```

**Spinner Configuration:**
```python
enable_spinner = getattr(config.training, "enable_spinner", True)
if enable_spinner:
    progress_columns = [SpinnerColumn()] + base_columns
```

**Training Task Initialization:**
```python
training_task = progress_bar.add_task(
    "Training",
    total=config.training.total_timesteps,
    completed=trainer.global_timestep,
    ep_metrics="Ep L:0 R:0.0",
    ppo_metrics="",
    black_wins_cum=trainer.black_wins,
    white_wins_cum=trainer.white_wins,
    draws_cum=trainer.draws,
    black_win_rate=initial_black_win_rate,
    white_win_rate=initial_white_win_rate,
    draw_rate=initial_draw_rate,
    speed=0.0,
    start=(trainer.global_timestep < config.training.total_timesteps)
)
```

**Layout Structure:**
```python
layout = Layout(name="root")
layout.split_column(
    Layout(name="main_log", ratio=1),      # Log panel (expandable)
    Layout(name="progress_display", size=4) # Progress bar (fixed height)
)
layout["main_log"].update(log_panel)
layout["progress_display"].update(progress_bar)
```

##### `update_progress(self, trainer, speed, pending_updates)`
**Purpose:** Updates progress bar with current training state, speed metrics, and pending UI updates.

**Parameters:**
- `trainer` - Trainer instance with current state
- `speed: float` - Current training speed (steps/second)
- `pending_updates: Dict[str, Any]` - Additional UI field updates

**Update Process:**
1. Create base update data with timestep and speed
2. Merge with pending updates from caller
3. Apply all updates to progress bar task

**Update Data Structure:**
```python
update_data = {
    "completed": trainer.global_timestep,
    "speed": speed,
    **pending_updates  # Additional metrics and status updates
}
```

##### `update_log_panel(self, trainer)`
**Purpose:** Updates the log panel with recent log messages, managing scrolling and display size.

**Parameters:**
- `trainer` - Trainer instance with rich_log_messages

**Update Logic:**
1. **Size Calculation:** `visible_rows = max(0, console.size.height - 6)`
2. **Message Selection:** Display most recent messages that fit in visible area
3. **Panel Update:** Update log panel content with Rich Group of messages
4. **Empty Handling:** Display empty Text when no messages available

**Scrolling Behavior:**
```python
if trainer.rich_log_messages:
    display_messages = trainer.rich_log_messages[-visible_rows:]
    updated_panel_content = Group(*display_messages)
    log_panel.renderable = updated_panel_content
else:
    log_panel.renderable = Text("")
```

##### `start(self)`
**Purpose:** Creates and returns a Rich Live instance for real-time UI rendering.

**Returns:** 
- `Live` - Rich Live instance configured for training display

**Live Configuration:**
```python
return Live(
    self.layout,
    console=self.rich_console,
    refresh_per_second=self.config.training.refresh_per_second,
    transient=False
)
```

**Live Display Features:**
- **Real-time Updates:** Automatic refresh based on configuration
- **Non-transient:** Display persists after training completion
- **Layout Management:** Handles complex multi-panel layout
- **Console Integration:** Integrates with Rich console system

---

### 4. Data Structures 🗂️

#### Progress Bar Configuration
```python
progress_config = {
    "columns": [
        "[progress.description]{task.description}",
        "BarColumn()",
        "TaskProgressColumn()",
        "TimeElapsedColumn()",
        "TimeRemainingColumn()",
        "Custom metrics columns"
    ],
    "console": Console,
    "transient": False,
    "spinner_enabled": bool
}
```

#### Training Task Fields
```python
task_fields = {
    "ep_metrics": str,           # Episode metrics (length, reward)
    "ppo_metrics": str,          # PPO algorithm metrics
    "black_wins_cum": int,       # Cumulative black wins
    "white_wins_cum": int,       # Cumulative white wins
    "draws_cum": int,            # Cumulative draws
    "black_win_rate": float,     # Black win percentage
    "white_win_rate": float,     # White win percentage
    "draw_rate": float,          # Draw percentage
    "speed": float               # Training speed (steps/second)
}
```

#### Layout Structure
```python
layout_structure = {
    "root": {
        "type": "Layout",
        "split": "column",
        "children": {
            "main_log": {
                "type": "Layout",
                "ratio": 1,              # Expandable log area
                "content": "Panel"       # Log message panel
            },
            "progress_display": {
                "type": "Layout", 
                "size": 4,               # Fixed height progress area
                "content": "Progress"    # Progress bar component
            }
        }
    }
}
```

#### Update Data Format
```python
update_data_format = {
    "completed": int,            # Current timestep
    "speed": float,              # Steps per second
    "ep_metrics": str,           # Episode information
    "ppo_metrics": str,          # PPO training metrics
    "black_wins_cum": int,       # Game result counts
    "white_wins_cum": int,
    "draws_cum": int,
    "black_win_rate": float,     # Win rate percentages
    "white_win_rate": float,
    "draw_rate": float
}
```

---

### 5. Inter-Module Relationships 🔗

#### Dependencies
- **`keisei.training.trainer.Trainer`** - State access and log message source
- **`keisei.training.display_manager.DisplayManager`** - UI orchestration and update coordination
- **Rich Framework** - Terminal UI components and rendering

#### Integration Points
- **DisplayManager:** TrainingDisplay is managed and updated by DisplayManager
- **Trainer State:** Direct access to trainer attributes and log messages
- **Configuration System:** Display settings from training configuration
- **Console System:** Integration with Rich console for rendering

#### Data Flow
```
Trainer State → DisplayManager → TrainingDisplay → Rich Components → Terminal Output
     ↓
Log Messages → Rich Log Panel → Real-time Display
     ↓
Metrics → Progress Bar → Visual Progress Indication
```

#### UI Component Hierarchy
```
Live Display
├── Layout (root)
    ├── Layout (main_log)
    │   └── Panel (log_panel)
    │       └── Group (log_messages)
    └── Layout (progress_display)
        └── Progress (progress_bar)
            └── Task (training_task)
```

---

### 6. Implementation Notes 💡

#### Design Decisions
1. **Two-Panel Layout:** Separate expandable log area and fixed progress display
2. **Comprehensive Metrics:** Single progress bar displays all key training metrics
3. **Real-time Updates:** Live display with configurable refresh rate
4. **Scrolling Logs:** Automatic log scrolling based on console size
5. **Optional Spinner:** Configurable spinner for visual activity indication

#### Rich Framework Integration
- **Console Management:** Integrates with Rich console system
- **Layout System:** Uses Rich Layout for structured UI organization
- **Progress API:** Leverages Rich Progress for advanced progress visualization
- **Panel System:** Uses Rich Panel for bordered content areas
- **Live Updates:** Utilizes Rich Live for real-time display updates

#### Performance Considerations
- **Efficient Updates:** Only update changed UI elements
- **Log Trimming:** Limit log display to visible area to reduce memory usage
- **Refresh Rate Control:** Configurable refresh rate to balance responsiveness and performance
- **Non-blocking Updates:** UI updates don't block training execution

---

### 7. Testing Strategy 🧪

#### Unit Tests
```python
def test_training_display_initialization():
    """Test TrainingDisplay initialization with valid parameters."""
    pass

def test_progress_bar_setup():
    """Test progress bar configuration and task creation."""
    pass

def test_layout_structure():
    """Test Rich layout creation and panel organization."""
    pass

def test_progress_updates():
    """Test progress bar updates with various metrics."""
    pass

def test_log_panel_updates():
    """Test log panel updates and scrolling behavior."""
    pass
```

#### Integration Tests
```python
def test_trainer_integration():
    """Test integration with trainer state and log messages."""
    pass

def test_display_manager_integration():
    """Test integration with DisplayManager coordination."""
    pass

def test_configuration_integration():
    """Test display configuration and option handling."""
    pass
```

#### UI Tests
```python
def test_console_rendering():
    """Test Rich console rendering and display output."""
    pass

def test_responsive_layout():
    """Test layout responsiveness to console size changes."""
    pass

def test_real_time_updates():
    """Test live display updates and refresh behavior."""
    pass
```

#### Testing Considerations
- Mock Rich components for isolated testing
- Test UI responsiveness to various console sizes
- Validate metric display formatting and accuracy
- Test log scrolling and message management

---

### 8. Performance Considerations ⚡

#### Efficiency Factors
- **Selective Updates:** Only update changed progress bar fields
- **Log Trimming:** Display only visible log messages to reduce memory usage
- **Efficient Rendering:** Rich framework optimizes terminal rendering
- **Configurable Refresh:** Balance update frequency with performance

#### Optimization Opportunities
- **Update Batching:** Batch multiple UI updates for efficiency
- **Lazy Rendering:** Defer expensive rendering operations
- **Memory Management:** Limit log message retention and display
- **Responsive Refresh:** Adjust refresh rate based on training activity

#### Resource Management
- **Memory Usage:** Log message management prevents unbounded growth
- **CPU Usage:** UI updates optimized to minimize training impact
- **Terminal Resources:** Efficient use of terminal rendering capabilities
- **Console Integration:** Proper console resource management

---

### 9. Security Considerations 🔒

#### Input Validation
- **Console Size:** Validate console dimensions for layout calculations
- **Log Messages:** Display log content safely without injection risks
- **Configuration Values:** Validate refresh rates and display options

#### Security Measures
- **Safe Rendering:** Rich framework handles safe terminal output
- **Content Validation:** Log message content validated before display
- **Resource Limits:** Bounded log message display prevents resource exhaustion

#### Potential Vulnerabilities
- **Terminal Injection:** Log message content could contain terminal escape sequences
- **Resource Exhaustion:** Unbounded log message accumulation
- **Console Control:** Terminal control sequences in displayed content

---

### 10. Error Handling 🚨

#### Exception Management
```python
# Safe console size access
visible_rows = max(0, self.rich_console.size.height - 6)

# Safe log message access
if trainer.rich_log_messages:
    display_messages = trainer.rich_log_messages[-visible_rows:]
else:
    log_panel.renderable = Text("")
```

#### Error Categories
- **Console Errors:** Terminal size and rendering issues
- **Configuration Errors:** Invalid display configuration values
- **State Errors:** Missing trainer state or log messages
- **Rich Framework Errors:** UI component creation and update failures

#### Recovery Strategies
- **Graceful Degradation:** Continue with basic display when advanced features fail
- **Default Values:** Use sensible defaults for missing configuration
- **Error Logging:** Log UI errors without disrupting training
- **Safe Fallbacks:** Provide fallback display modes for error conditions

---

### 11. Configuration 📝

#### Display Configuration
```python
display_config = {
    "enable_spinner": True,           # Show spinner in progress bar
    "refresh_per_second": 4,         # UI refresh rate
    "console_size_responsive": True,  # Adapt to console size changes
    "log_panel_enabled": True,       # Enable log panel display
    "progress_bar_enabled": True     # Enable progress bar display
}
```

#### Progress Bar Configuration
```python
progress_bar_config = {
    "show_elapsed_time": True,
    "show_remaining_time": True,
    "show_speed": True,
    "show_episode_metrics": True,
    "show_ppo_metrics": True,
    "show_win_rates": True,
    "color_scheme": {
        "episode_metrics": "bright_cyan",
        "ppo_metrics": "bright_yellow",
        "game_results": "bright_green",
        "win_rates": "bright_blue"
    }
}
```

#### Layout Configuration
```python
layout_config = {
    "log_panel_ratio": 1,            # Expandable log area
    "progress_panel_size": 4,        # Fixed progress bar height
    "log_panel_title": "[b]Live Training Log[/b]",
    "log_panel_border_style": "bright_green",
    "layout_responsive": True
}
```

---

### 12. Future Enhancements 🚀

#### Planned Improvements
1. **Customizable Layouts:** User-configurable UI layout and panel arrangement
2. **Advanced Metrics:** Additional training metrics and visualization options
3. **Export Functionality:** Save UI snapshots and training visualizations
4. **Theme Support:** Multiple color themes and display styles
5. **Interactive Features:** Mouse interaction and keyboard shortcuts

#### Extension Points
- **Custom Panels:** Support for additional UI panels and components
- **Metric Plugins:** Pluggable metric display components
- **Layout Templates:** Pre-defined layout configurations for different use cases
- **Export Formats:** Multiple export formats for training visualizations

#### API Evolution
- **Component Modularity:** More modular UI component architecture
- **Configuration API:** Enhanced configuration interface for display customization
- **Event System:** Event-driven UI updates for better responsiveness

---

### 13. Usage Examples 📋

#### Basic Display Setup
```python
# Initialize Rich console
console = Console()

# Create TrainingDisplay
display = TrainingDisplay(
    config=training_config,
    trainer=trainer_instance,
    rich_console=console
)

# Start live display
live_display = display.start()
with live_display:
    # Training loop with UI updates
    display.update_progress(trainer, speed=10.5, pending_updates={
        "ep_metrics": "Ep L:42 R:0.85",
        "ppo_metrics": "Loss: 0.023"
    })
    display.update_log_panel(trainer)
```

#### Progress Updates
```python
# Update progress with comprehensive metrics
pending_updates = {
    "ep_metrics": f"Ep L:{episode_length} R:{reward:.2f}",
    "ppo_metrics": f"Loss: {loss:.3f} KL: {kl_div:.3f}",
    "black_wins_cum": trainer.black_wins,
    "white_wins_cum": trainer.white_wins,
    "draws_cum": trainer.draws,
    "black_win_rate": trainer.black_wins / total_episodes,
    "white_win_rate": trainer.white_wins / total_episodes,
    "draw_rate": trainer.draws / total_episodes
}

display.update_progress(trainer, speed=current_speed, pending_updates=pending_updates)
```

#### Log Panel Management
```python
# Log messages automatically displayed in panel
trainer.rich_log_messages.append(Text("Training started", style="green"))
trainer.rich_log_messages.append(Text("Model loaded", style="blue"))
trainer.rich_log_messages.append(Text("Checkpoint saved", style="yellow"))

# Update log panel to show recent messages
display.update_log_panel(trainer)
```

#### Custom Configuration
```python
# Custom display configuration
config.training.enable_spinner = False
config.training.refresh_per_second = 2

# Create display with custom settings
display = TrainingDisplay(config, trainer, console)
live_display = display.start()
```

---

### 14. Maintenance Notes 🔧

#### Regular Maintenance
- **Rich Framework Updates:** Monitor Rich library updates and API changes
- **Console Compatibility:** Test display compatibility across different terminal types
- **Performance Monitoring:** Track UI update performance and refresh overhead
- **Memory Usage:** Monitor log message memory usage and implement cleanup if needed

#### Monitoring Points
- **Refresh Performance:** Monitor UI refresh rate impact on training performance
- **Memory Growth:** Track log message accumulation and memory usage
- **Console Support:** Verify display compatibility across different terminal environments
- **Update Efficiency:** Monitor UI update frequency and batch optimization opportunities

#### Documentation Dependencies
- **`training_display_manager.md`** - Display manager coordination and update orchestration
- **`training_trainer.md`** - Trainer integration and state access patterns
- **`rich_integration.md`** - Rich framework integration patterns and best practices
- **`ui_configuration.md`** - Display configuration options and customization guide
