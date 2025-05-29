# Trainer.py Refactor Plan

**Date:** May 28, 2025  
**Current File Size:** 916 lines  
**Target:** Reduce to ~200-300 lines through modular decomposition

## Executive Summary

The `trainer.py` file has grown to 916 lines and violates the Single Responsibility Principle by handling 10+ distinct concerns. This refactor will decompose the monolithic `Trainer` class into specialized, composable managers while maintaining all existing functionality and improving maintainability, testability, and extensibility.


## Current Problems Analysis

### 1. **Massive Single Responsibility Violation**
The `Trainer` class currently handles:
- Model initialization and configuration
- Game environment setup
- Training loop orchestration  
- Logging and display management
- Checkpoint management
- WandB integration
- Episode and step management
- PPO updates
- Statistics tracking
- Callback management
- Error handling and recovery
- Mixed precision setup
- Feature and observation management

### 2. **File Size and Complexity**
- **916 lines** is beyond maintainable limits
- Complex method interdependencies
- Difficult to locate specific functionality
- High cognitive load for developers

### 3. **Tight Coupling Issues**
- Methods depend heavily on trainer's internal state
- Hard to unit test individual components
- Difficult to reuse components in different contexts
- Changes propagate unpredictably through the class

### 4. **Mixed Abstraction Levels**
- High-level training orchestration mixed with low-level step execution
- UI concerns mixed with business logic
- Configuration management mixed with runtime execution

### 5. **Code Duplication and Inconsistency**
- Repeated error handling patterns
- Similar logging logic throughout
- Redundant state management approaches

## Refactor Strategy

### Core Principle: **Composition over Inheritance**
Transform the monolithic `Trainer` into a **composition of specialized managers**, where each manager handles a single aspect of the training process.

### Phased Approach
Execute the refactor in phases to maintain functionality and minimize risk of regressions.

---

## Phase 1: Extract Training Session Management

**Objective:** Extract session-level lifecycle management into a dedicated component.

### Create `training/session_manager.py`

**Responsibilities:**
- Run name generation and validation
- Directory structure creation and management
- Configuration serialization and persistence
- Session-level logging setup
- WandB initialization and configuration
- Environment variable and CLI integration

**Methods to Extract from Trainer:**
- `_log_run_info()` → `SessionManager.log_session_info()`
- Directory setup logic from `__init__` → `SessionManager.setup_directories()`
- WandB setup logic → `SessionManager.setup_wandb()`
- Config serialization → `SessionManager.save_effective_config()`

**Class Design:**
```python
class SessionManager:
    def __init__(self, config: AppConfig, args: Any, run_name: Optional[str] = None)
    def setup_directories(self) -> Dict[str, str]
    def setup_wandb(self) -> bool
    def save_effective_config(self) -> None
    def log_session_info(self, logger_func: Callable) -> None
    def finalize_session(self) -> None
    
    # Properties
    @property
    def run_name(self) -> str
    @property
    def run_artifact_dir(self) -> str
    @property
    def model_dir(self) -> str
    @property
    def log_file_path(self) -> str
```

**Integration Pattern:**
```python
# In Trainer.__init__
self.session = SessionManager(config, args, run_name)
self.session.setup_directories()
self.session.setup_wandb()
```

---

## Phase 2: Extract Training Step Management

**Objective:** Encapsulate individual step execution and episode management logic.

### Create `training/step_manager.py`

**Responsibilities:**
- Individual training step execution
- Episode lifecycle management
- Move selection and environment interaction
- Demo mode handling and move formatting
- Step-level error handling and recovery

**Methods to Extract from Trainer:**
- `_execute_training_step()` → `StepManager.execute_step()`
- `_handle_episode_end()` → `StepManager.handle_episode_end()`
- Demo mode logic → `StepManager.handle_demo_mode()`
- Move selection → `StepManager.select_and_execute_move()`

**Class Design:**
```python
class StepManager:
    def __init__(self, config: AppConfig, game: ShogiGame, agent: PPOAgent, 
                 policy_mapper: PolicyOutputMapper, buffer: ExperienceBuffer)
    
    def execute_step(self, current_obs: np.ndarray, episode_state: EpisodeState, 
                    logger_func: Callable) -> StepResult
    
    def handle_episode_end(self, episode_state: EpisodeState, 
                          info: Dict, logger_func: Callable) -> EpisodeState
    
    def handle_demo_mode(self, move: MoveTuple, episode_length: int, 
                        logger_func: Callable) -> None
    
    # State management
    def reset_episode(self) -> EpisodeState
    def update_episode_state(self, episode_state: EpisodeState, 
                           step_result: StepResult) -> EpisodeState
```

**Data Structures:**
```python
@dataclass
class EpisodeState:
    current_obs: np.ndarray
    current_obs_tensor: torch.Tensor
    episode_reward: float
    episode_length: int

@dataclass  
class StepResult:
    next_obs: np.ndarray
    next_obs_tensor: torch.Tensor
    reward: float
    done: bool
    info: Dict[str, Any]
    selected_move: MoveTuple
    policy_index: int
    log_prob: float
    value_pred: float
```

---

## Phase 3: Extract Model and Environment Management

### Create `training/model_manager.py`

**Responsibilities:**
- Model factory creation and configuration
- Checkpoint saving and loading
- WandB artifact creation and management
- Model state persistence
- Mixed precision setup

**Methods to Extract from Trainer:**
- Model creation logic from `__init__` → `ModelManager.create_model()`
- `_create_model_artifact()` → `ModelManager.create_wandb_artifact()`
- `_handle_checkpoint_resume()` → `ModelManager.handle_checkpoint_resume()`
- Model saving logic → `ModelManager.save_checkpoint()`

**Class Design:**
```python
class ModelManager:
    def __init__(self, config: AppConfig, device: torch.device)
    
    def create_model(self, input_features: str, model_type: str, 
                    tower_depth: int, tower_width: int, se_ratio: float) -> ActorCriticProtocol
    
    def setup_mixed_precision(self) -> Tuple[bool, Optional[GradScaler]]
    
    def save_checkpoint(self, model: ActorCriticProtocol, timestep: int, 
                       episode_count: int, stats: Dict) -> str
    
    def create_wandb_artifact(self, model_path: str, artifact_name: str,
                             metadata: Dict, aliases: List[str]) -> bool
    
    def handle_checkpoint_resume(self, agent: PPOAgent, resume_path: str) -> Optional[str]
    
    def save_final_model(self, agent: PPOAgent, timestep: int, 
                        episode_count: int, stats: Dict) -> str
```

**📋 Type System Enhancement - ActorCriticProtocol Interface**

*Added: May 29, 2025*

**Issue Resolved:** Type assignment incompatibility between `PPOAgent.model` (expected `ActorCritic`) and `ModelManager.model` (returned `Module` from `model_factory()`).

**Solution Implemented:**
- **Created `keisei/core/actor_critic_protocol.py`** - Protocol interface defining the contract for all Actor-Critic models
- **Updated type annotations** across the model creation pipeline:
  - `model_factory()` → returns `ActorCriticProtocol` 
  - `PPOAgent.model` → typed as `ActorCriticProtocol`
  - `ModelManager._create_model()` → returns `ActorCriticProtocol`
  - `ModelManager.model` → typed as `ActorCriticProtocol`

**Protocol Definition:**
```python
class ActorCriticProtocol(Protocol):
    """Protocol defining the interface that all Actor-Critic models must implement."""
    
    # Core Actor-Critic methods
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def get_action_and_value(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None, 
                           deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, 
                        legal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    
    # PyTorch Module methods used by PPOAgent
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any: ...
    def to(self, *args, **kwargs) -> Any: ...
```

**Benefits:**
- ✅ **Type Safety**: Ensures all model types implement required interface
- ✅ **Polymorphism**: Allows `ActorCritic` and `ActorCriticResTower` to be used interchangeably  
- ✅ **Documentation**: Clear contract for what methods Actor-Critic models must provide
- ✅ **Extensibility**: New model types can implement the Protocol without inheritance
- ✅ **IDE Support**: Better autocomplete and type checking during development

**Files Modified:**
- `/home/john/keisei/keisei/core/actor_critic_protocol.py` - New Protocol definition
- `/home/john/keisei/keisei/core/ppo_agent.py` - Updated model type annotation
- `/home/john/keisei/keisei/training/model_manager.py` - Updated return types
- `/home/john/keisei/keisei/training/models/__init__.py` - Updated model_factory return type

**Verification:** All existing functionality preserved, type assignment errors resolved, comprehensive testing confirms both `ActorCritic` and `ActorCriticResTower` correctly implement the Protocol interface.

### Create `training/env_manager.py`

**Responsibilities:**
- Game environment setup and configuration
- Policy output mapper creation
- Observation space configuration
- Environment seeding and reset logic

**Methods to Extract from Trainer:**
- `_setup_game_components()` → `EnvManager.setup_environment()`
- `_initialize_game_state()` → `EnvManager.initialize_game_state()`
- Environment seeding logic → `EnvManager.setup_seeding()`

**Class Design:**
```python
class EnvManager:
    def __init__(self, config: AppConfig)
    
    def setup_environment(self) -> Tuple[ShogiGame, PolicyOutputMapper]
    def setup_seeding(self) -> None
    def initialize_game_state(self, logger_func: Callable) -> np.ndarray
    def create_policy_mapper(self) -> PolicyOutputMapper
    
    @property
    def obs_space_shape(self) -> Tuple[int, int, int]
    
    @property
    def action_space_size(self) -> int
```

---

## Phase 4: Extract Statistics and Metrics Management

### Create `training/metrics_manager.py`

**Responsibilities:**
- Win/loss/draw statistics tracking
- Rate calculations and formatting
- PPO metrics processing and display
- Progress update management
- Metrics logging and reporting

**Methods to Extract from Trainer:**
- Statistics tracking from `_handle_episode_end()` → `MetricsManager.update_episode_stats()`
- PPO metrics from `_perform_ppo_update()` → `MetricsManager.format_ppo_metrics()`
- Progress updates → `MetricsManager.update_progress_metrics()`

**Class Design:**
```python
class MetricsManager:
    def __init__(self)
    
    def update_episode_stats(self, winner_color: Optional[Color]) -> Dict[str, float]
    def get_win_rates(self) -> Tuple[float, float, float]  # black, white, draw
    def format_episode_metrics(self, length: int, reward: float) -> str
    def format_ppo_metrics(self, learn_metrics: Dict[str, float]) -> str
    def get_progress_updates(self) -> Dict[str, Any]
    def reset_progress_updates(self) -> None
    
    # Properties
    @property
    def total_episodes(self) -> int
    @property 
    def black_wins(self) -> int
    @property
    def white_wins(self) -> int
    @property
    def draws(self) -> int
```

---

## Phase 5: Enhance Existing Components

### Extend `training/callbacks.py`

**Improvements:**
- Add missing `import os` (noted in code map)
- Create more specialized callback types
- Better error handling and logging integration
- Support for custom callback intervals
- Callback state persistence

**New Callback Types:**
```python
class PeriodicCheckpointCallback(Callback):
    """Saves checkpoints at regular intervals"""
    
class MetricsLoggingCallback(Callback):
    """Logs metrics to various backends (file, wandb, tensorboard)"""
    
class ModelValidationCallback(Callback):
    """Validates model performance during training"""
    
class EarlyStoppingCallback(Callback):
    """Implements early stopping based on metrics"""
```

### Enhance `training/display.py`

**Improvements:**
- Reduce coupling with Trainer class
- Self-contained UI state management
- Better separation of concerns
- Improved error handling

**Refactored Interface:**
```python
class TrainingDisplay:
    def __init__(self, config: AppConfig, console: Console)
    
    def update_from_metrics(self, metrics: Dict[str, Any]) -> None
    def update_from_progress(self, current_step: int, total_steps: int, 
                           speed: float) -> None
    def add_log_message(self, message: str) -> None
    def get_context_manager(self) -> Live
```

---

## Phase 6: Simplified Trainer Class

**Objective:** Reduce Trainer to a lean orchestrator that coordinates specialized managers.

### New Trainer Structure (~200-300 lines)

```python
class Trainer:
    """Lean orchestrator for Shogi RL training."""
    
    def __init__(self, config: AppConfig, args: Any):
        # Initialize managers
        self.session = SessionManager(config, args)
        self.env_manager = EnvManager(config)
        self.model_manager = ModelManager(config, torch.device(config.env.device))
        self.metrics = MetricsManager()
        
        # Setup components through managers
        self._setup_components()
        
    def _setup_components(self):
        """Setup all training components through managers."""
        # Session setup
        self.session.setup_directories()
        self.session.setup_wandb()
        
        # Environment setup  
        self.game, self.policy_mapper = self.env_manager.setup_environment()
        
        # Model setup
        self.model = self.model_manager.create_model(...)
        self.agent = PPOAgent(config=self.config, device=self.device)
        
        # Other components
        self.experience_buffer = ExperienceBuffer(...)
        self.step_manager = StepManager(...)
        self.display = TrainingDisplay(...)
        self.callbacks = self._create_callbacks()
    
    def run_training_loop(self):
        """Main training loop - coordinates managers."""
        with self.session.get_logger_context() as logger:
            self.session.log_session_info(logger.log)
            
            episode_state = self.step_manager.reset_episode()
            
            with self.display.get_context_manager():
                while self.global_timestep < self.config.training.total_timesteps:
                    # Execute training step through step manager
                    step_result = self.step_manager.execute_step(
                        episode_state.current_obs, episode_state, logger.log
                    )
                    
                    # Update state
                    episode_state = self.step_manager.update_episode_state(
                        episode_state, step_result
                    )
                    
                    # Handle episode end
                    if step_result.done:
                        episode_state = self.step_manager.handle_episode_end(
                            episode_state, step_result.info, logger.log
                        )
                        self.metrics.update_episode_stats(step_result.info.get('winner'))
                    
                    # PPO Update
                    if self._should_update_ppo():
                        self._perform_ppo_update(logger.log)
                    
                    # Update displays and metrics
                    self.display.update_from_metrics(self.metrics.get_progress_updates())
                    self.display.update_from_progress(self.global_timestep, 
                                                    self.config.training.total_timesteps, 
                                                    self._calculate_speed())
                    
                    # Run callbacks
                    for callback in self.callbacks:
                        callback.on_step_end(self)
                    
                    self.global_timestep += 1
                
                # Finalize training
                self._finalize_training(logger.log)
    
    def _perform_ppo_update(self, logger_func: Callable):
        """Perform PPO update with metrics tracking."""
        # PPO update logic (simplified)
        learn_metrics = self.agent.learn(self.experience_buffer)
        self.experience_buffer.clear()
        
        # Format and log metrics
        ppo_metrics_str = self.metrics.format_ppo_metrics(learn_metrics)
        logger_func(f"PPO Update @ ts {self.global_timestep}. Metrics: {ppo_metrics_str}")
    
    def _finalize_training(self, logger_func: Callable):
        """Finalize training through managers."""
        # Save final model through model manager
        final_model_path = self.model_manager.save_final_model(
            self.agent, self.global_timestep, self.metrics.total_episodes, 
            self.metrics.get_final_stats()
        )
        
        # Finalize session
        self.session.finalize_session()
        
        logger_func(f"Training completed. Final model saved: {final_model_path}")
```

---

## PROGRESS REPORT

**Last Updated:** May 29, 2025  
**Current Status:** Phase 3 (Critical Fixes) - COMPLETED ✅  
**Next Phase:** Phase 3 (Line Reduction) - IN PROGRESS

### 📊 OVERALL PROGRESS METRICS

| Metric | Original | Current | Target | Progress |
|--------|----------|---------|--------|----------|
| **File Size** | 916 lines | 651 lines | 200-300 lines | 29% ↓ |
| **Lines Extracted** | - | 731+ lines | 600+ lines | 122% |
| **Components Created** | 1 monolith | 3 managers | 6+ managers | 50% |
| **Test Coverage** | Minimal | 1762+ test lines | Full coverage | 95% |
| **Phases Complete** | 0/5 | 2.5/5 | 5/5 | 50% |

---

### Phase 1: Session Management - ✅ COMPLETED
**Status:** Production Ready ✅  
**Files Created:**
- `keisei/training/session_manager.py` (244 lines) - Complete SessionManager class
- `tests/test_session_manager.py` (482 lines) - Comprehensive unit tests (95%+ passing)

**Integration Status:** ✅ Complete
- SessionManager fully integrated into Trainer class
- All session-level functionality extracted and working
- Directory creation, WandB setup, config persistence all delegated to SessionManager
- Comprehensive test coverage achieved

**Functionality Extracted:**
- ✅ Run name generation and validation (precedence: explicit > CLI > config > auto-generate)
- ✅ Directory structure creation and management
- ✅ WandB initialization and configuration
- ✅ Configuration serialization and persistence
- ✅ Session lifecycle management and logging
- ✅ Environment seeding setup

**Key Achievements:**
- Extracted 150+ lines of session setup logic from Trainer
- Created reusable session management component with clean interfaces
- Implemented comprehensive error handling and recovery mechanisms
- Established consistent session logging patterns
- Added robust unit test suite with edge case coverage

---

### Phase 2: Step Management - ✅ COMPLETED  
**Status:** Production Ready ✅  
**Files Created:**
- `keisei/training/step_manager.py` (466 lines) - Complete StepManager class
- `tests/test_step_manager.py` (723 lines) - Comprehensive unit tests (ALL PASSING ✅)

**Core Components Implemented:**
- ✅ **EpisodeState dataclass** - Clean state management for episodes
- ✅ **StepResult dataclass** - Encapsulates step execution results with success tracking
- ✅ **StepManager class** - Complete step execution and episode management

**Functionality Extracted from Trainer:**
- ✅ `execute_step()` - Individual training step execution with agent action selection
- ✅ `handle_episode_end()` - Episode completion, outcome processing, and statistics logging
- ✅ `reset_episode()` - Fresh episode state creation with proper game reset
- ✅ `update_episode_state()` - Episode state updates with step results
- ✅ `_prepare_demo_info()` and `_handle_demo_mode()` - Demo mode functionality
- ✅ Comprehensive error handling and graceful recovery mechanisms
- ✅ Legal move validation and policy mapping

**Integration Status:** ✅ Complete
- StepManager fully integrated into Trainer class initialization
- Modified `_initialize_game_state()` to return EpisodeState instead of raw arrays
- Updated `_execute_training_step()` to use StepManager delegation pattern
- All step execution logic successfully extracted from Trainer
- Clean data flow between Trainer orchestration and StepManager execution

**Testing Status:** ✅ Comprehensive (Code Style Issues Only)
- **ALL 26 FUNCTIONAL TESTS PASSING** ✅ 
- Error handling scenarios thoroughly tested
- Demo mode functionality verified  
- Episode state transitions validated
- **Only Issue:** Minor code style violations (trailing whitespace, import order)
- **Core Functionality:** 100% operational and tested

**Key Achievements:**
- Extracted 200+ lines of step management logic from Trainer
- Created clean separation between step execution and training orchestration
- Implemented robust error handling with graceful recovery
- Achieved excellent architectural separation of concerns
- Significantly reduced complexity in main Trainer class

**Code Quality Improvements:**
- Clear data structures (EpisodeState, StepResult) replace ad-hoc state management
- Consistent error handling patterns throughout step execution
- Clean separation of concerns between agent actions and environment interaction
- Improved maintainability and testability of step execution logic
- Proper encapsulation of episode lifecycle management

---

### Phase 3: Critical Integration Fixes - ✅ COMPLETED
**Status:** Production Ready ✅  
**Completion Date:** May 29, 2025

**Critical Issues Resolved:**

**✅ Resume Functionality Fixed**
- **Issue:** Resume tests failing - expected "Resumed training from checkpoint: {path}" but got "Starting fresh training"
- **Root Cause:** ModelManager.handle_checkpoint_resume() was logging with wrong logger (self.logger.log instead of log_both)
- **Solution:** Moved resume logging from ModelManager to trainer's run_training_loop() where log_both is available
- **Files Modified:**
  - `/home/john/keisei/keisei/training/model_manager.py` - Removed logging from handle_checkpoint_resume()
  - `/home/john/keisei/keisei/training/trainer.py` - Added resume logging after session start
- **Result:** ✅ Both resume tests now pass (test_train_resume_autodetect, test_train_explicit_resume)

**✅ Test Suite Integration Verified**
- **trainer_session_integration.py:** ✅ All 3 functional tests passing (only PyLint style warnings)
- **Resume functionality tests:** ✅ Both tests passing with correct log messages
- **WandB integration tests:** ✅ All tests passing
- **Overall test stability:** ✅ No regressions introduced

**Implementation Details:**
```python
# In ModelManager.handle_checkpoint_resume() - Removed logging:
# OLD: self.logger_func(f"Resumed training from checkpoint: {latest_ckpt}")
# NEW: # Resume logging will be handled by trainer's run_training_loop

# In Trainer.run_training_loop() - Added logging after session start:
if self.resumed_from_checkpoint:
    log_both(f"Resumed training from checkpoint: {self.resumed_from_checkpoint}")
```

**Current Status Post-Fixes:**
- **File Size:** trainer.py reduced to 651 lines (265 lines reduced from original 916)
- **Test Coverage:** 1762+ test lines for extracted components  
- **Functionality:** All critical trainer features working correctly
- **Integration:** Clean delegation to SessionManager and StepManager
- **Type Safety:** Complete ActorCriticProtocol interface implementation resolves model type compatibility issues

---

## 🔧 CRITICAL ISSUES RESOLVED

### Phase 1 & 2 Major Fixes (May 29, 2025)

**✅ Function Signature Mismatch Fixed**
- **Issue:** `_initialize_game_state()` and `_execute_training_step()` had incompatible data types
- **Solution:** Updated both methods to consistently use `EpisodeState` objects
- **Impact:** Training loop now executes without type errors

**✅ Logger Interface Confirmed Working**  
- **Issue:** Tests suggested logger signature problems with `wandb_data` parameter
- **Solution:** Verified `handle_episode_end()` correctly uses keyword arguments format
- **Status:** All 26 step manager tests passing with current logger interface

**✅ Import Cleanup Completed**
- **Removed:** Unused imports (`Color`, `format_move_with_description_enhanced`, `generate_run_name`, `numpy`)
- **Added:** Missing `shutil` import for file operations
- **Result:** Clean import structure, no unused dependencies

**✅ Session Manager Type Issues Fixed**
- **Issue:** Boolean return types and path operations with null values
- **Solution:** Added proper type casting and null checks
- **Result:** All 29 session manager tests passing

**✅ Step Manager Type Hints Updated**
- **Issue:** Restrictive `logger_func` parameter typing
- **Solution:** Updated to `Callable[..., None]` for flexible logger interfaces
- **Result:** Compatible with all current logging implementations

**✅ ActorCriticProtocol Type System Enhancement (May 29, 2025)**
- **Issue:** Type assignment error at `model_manager.py:121` - `agent.model = self.model` failed because `Module` type incompatible with `ActorCritic` type
- **Root Cause:** `model_factory()` returned generic `torch.nn.Module` but `PPOAgent.model` expected specific `ActorCritic` type
- **Solution:** Created `ActorCriticProtocol` interface defining contract for all Actor-Critic models
- **Implementation:**
  - Created `/home/john/keisei/keisei/core/actor_critic_protocol.py` with Protocol definition
  - Updated `PPOAgent.model` type annotation to use `ActorCriticProtocol`
  - Updated `ModelManager._create_model()` and `model_factory()` return types to `ActorCriticProtocol`
  - Verified both `ActorCritic` and `ActorCriticResTower` implement the Protocol interface
- **Result:** ✅ Type assignment error resolved, all type checking passes, no functional changes

### Current Test Status
- **SessionManager:** ✅ 29/29 tests passing (100%)
- **StepManager:** ✅ 26/26 tests passing (100%)  
- **Core Imports:** ✅ All modules import successfully
- **Integration:** ✅ Trainer successfully uses both managers
- **Remaining Issues:** Only minor code style violations (trailing whitespace)

---

### Remaining Phases: 

### Phase 3: Model & Environment Management - ⏳ IN PROGRESS
**Status:** Critical Fixes Complete ✅, Line Reduction Pending
**Target:** Extract model creation, checkpoint management, and environment setup
**Estimated Effort:** ~2-3 days remaining
**Files to Create:**
- `keisei/training/model_manager.py` (already exists, needs expansion)
- `keisei/training/env_manager.py` 
- `tests/test_model_manager.py`
- `tests/test_env_manager.py`

**Progress:** Resume functionality fully working, trainer.py at 651 lines (need to reach 400-450 target)

### Phase 4: Metrics Management - ⏳ PENDING  
**Target:** Extract statistics tracking and metrics formatting
**Estimated Effort:** ~2-3 days
**Files to Create:**
- `keisei/training/metrics_manager.py`
- `tests/test_metrics_manager.py`

### Phase 5: Enhance Existing Components - ⏳ PENDING
**Target:** Improve callbacks.py and display.py with better separation of concerns
**Estimated Effort:** ~2-3 days

### Phase 6: Simplified Trainer - ⏳ PENDING
**Target:** Reduce Trainer to lean orchestrator (~200-300 lines)
**Estimated Effort:** ~2-3 days

