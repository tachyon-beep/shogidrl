# Getting Started Instructions for Completing the Evaluation System Refactor

**Project:** Keisei Shogi DRL Evaluation System Performance Optimization  
**Current Status:** ~75-80% Complete - Core architecture implemented, performance optimization needed  
**Next Phase:** Phase 6 - Performance Optimization Implementation

## 📋 Pre-Work: Essential Reading and Understanding

### Step 1: Read All Documentation (30-45 minutes)

**Primary Documents (MUST READ):**
1. `/home/john/keisei/docs/EVALUATION_REFACTOR_QUICK_REFERENCE.md` - Current status overview
2. `/home/john/keisei/docs/EVALUATION_SYSTEM_REFACTOR_STATUS_REPORT.md` - Comprehensive analysis  
3. `/home/john/keisei/docs/EVALUATION_IMPLEMENTATION_GUIDE.md` - Technical implementation details
4. `/home/john/keisei/EVALUATION_REFACTOR_IMPLEMENTATION_PLAN.md` - Original plan - REMOVED.
5. `/home/john/keisei/docs/EVALUATION_AUDIT.md` - Original problems identified

**Secondary Documents:**
6. `/home/john/keisei/HOW_TO_USE.md` - Current system usage
7. `/home/john/keisei/docs/2. CODE_MAP.md` - Overall codebase structure

### Step 2: Examine Current Implementation (45-60 minutes)

**Core Architecture Files:**
```bash
# Read these files to understand current implementation:
keisei/evaluation/core/base_evaluator.py      # Abstract interface
keisei/evaluation/core/evaluation_config.py   # Configuration system
keisei/evaluation/core/evaluation_context.py  # Context structures
keisei/evaluation/core/evaluation_result.py   # Result data structures
keisei/evaluation/manager.py                  # Main orchestrator
```

**Integration Points:**
```bash
# Understanding how evaluation integrates with training:
keisei/training/trainer.py                    # EvaluationManager integration
keisei/training/callbacks.py                  # EvaluationCallback implementation
keisei/evaluation/opponents/opponent_pool.py  # OpponentPool system
```

**Strategy Implementations:**
```bash
# Current working evaluators:
keisei/evaluation/strategies/single_opponent.py
keisei/evaluation/strategies/tournament.py
keisei/evaluation/strategies/ladder.py
keisei/evaluation/strategies/benchmark.py
```

### Step 3: Understand Test Coverage (20-30 minutes)

**Test Files to Review:**
```bash
tests/evaluation/test_evaluation_manager.py
tests/evaluation/test_opponent_pool.py
tests/evaluation/strategies/test_single_opponent_evaluator.py
tests/evaluation/strategies/test_tournament_evaluator.py
```

**Run Tests to Verify Current State:**
```bash
cd /home/john/keisei
pytest tests/evaluation/ -v
```

## 🎯 Phase 6 Implementation Plan: Performance Optimization

### Priority 1: In-Memory Model Communication (5-7 days)

**Problem:** Current system saves models to disk and reloads them for evaluation, causing I/O overhead.

**Solution:** Pass model weights directly in memory.

#### Task 6.1.1: Create Model Weight Management System

**File to Create:** `keisei/evaluation/core/model_manager.py`

**Implementation Steps:**
1. **Read existing code patterns:**
   - Examine how `PPOAgent` handles model weights in `keisei/core/ppo_agent.py`
   - Check how model checkpoints are currently saved in `keisei/training/model_manager.py`
   - Look at existing model loading in `keisei/utils/agent_loading.py`

2. **Create ModelWeightManager class:**
   ```python
   # Key responsibilities:
   # - Extract agent weights for evaluation
   # - Cache opponent weights from checkpoints 
   # - Create temporary agents from weights
   # - Manage memory usage with LRU cache
   ```

3. **Key methods to implement:**
   - `extract_agent_weights(agent) -> Dict[str, torch.Tensor]`
   - `cache_opponent_weights(opponent_id, checkpoint_path) -> Dict[str, torch.Tensor]`
   - `create_agent_from_weights(weights, agent_class, config)`
   - `clear_cache()`

#### Task 6.1.2: Update EvaluationManager for In-Memory Evaluation

**File to Modify:** `keisei/evaluation/manager.py`

**Before Starting:**
1. **Read current evaluate_current_agent method** to understand flow
2. **Check how trainer.agent is passed** in `keisei/training/callbacks.py`
3. **Examine OpponentPool.sample()** to understand opponent selection

**Implementation Steps:**
1. **Add ModelWeightManager to __init__:**
   ```python
   self.model_weight_manager = ModelWeightManager()
   self.enable_in_memory_eval = getattr(config, 'enable_in_memory_evaluation', True)
   ```

2. **Add new method evaluate_current_agent_in_memory:**
   ```python
   async def evaluate_current_agent_in_memory(self, agent, opponent_checkpoint=None) -> EvaluationResult:
       # Extract agent weights
       # Get opponent weights from cache or pool
       # Run in-memory evaluation
       # Fallback to file-based on failure
   ```

3. **Create _run_in_memory_evaluation method:**
   ```python
   # Delegate to evaluator with in-memory weights
   # Handle weight passing to SingleOpponentEvaluator
   ```

#### Task 6.1.3: Update SingleOpponentEvaluator for In-Memory Operation

**File to Modify:** `keisei/evaluation/strategies/single_opponent.py`

**Before Starting:**
1. **Read current evaluate method** to understand game execution flow
2. **Check how agents are currently created** from checkpoints
3. **Examine game loop in existing _play_single_game methods**

**Implementation Steps:**
1. **Add weight storage attributes:**
   ```python
   self.agent_weights: Optional[Dict[str, torch.Tensor]] = None
   self.opponent_weights: Optional[Dict[str, torch.Tensor]] = None
   ```

2. **Add evaluate_in_memory method:**
   ```python
   async def evaluate_in_memory(self, context: EvaluationContext) -> EvaluationResult:
       # Use pre-loaded weights instead of checkpoint loading
       # Create temporary agents from weights
       # Run game loop
   ```

3. **Add _create_temp_agent_from_weights helper:**
   ```python
   # Create PPOAgent instance from weights dictionary
   # Handle device placement and eval mode
   ```

#### Task 6.1.4: Update EvaluationCallback to Use In-Memory Evaluation

**File to Modify:** `keisei/training/callbacks.py`

**Before Starting:**
1. **Read current on_step_end method** in EvaluationCallback
2. **Understand how current_model.eval() is used**
3. **Check error handling patterns**

**Implementation Steps:**
1. **Replace evaluation call:**
   ```python
   # Change from:
   eval_results = trainer.evaluation_manager.evaluate_current_agent(trainer.agent)
   
   # To:
   if hasattr(trainer.evaluation_manager, 'evaluate_current_agent_in_memory'):
       eval_results = trainer.evaluation_manager.evaluate_current_agent_in_memory(trainer.agent)
   else:
       eval_results = trainer.evaluation_manager.evaluate_current_agent(trainer.agent)  # Fallback
   ```

#### Task 6.1.5: Update Configuration System

**File to Modify:** `keisei/evaluation/core/evaluation_config.py`

**Implementation Steps:**
1. **Add in-memory evaluation options:**
   ```python
   enable_in_memory_evaluation: bool = True
   model_weight_cache_size: int = 5
   temp_agent_device: str = "cpu"
   clear_cache_after_evaluation: bool = True
   ```

#### Task 6.1.6: Create Tests for In-Memory Evaluation

**Files to Create:**
- `tests/evaluation/test_model_manager.py`
- `tests/evaluation/test_in_memory_evaluation.py`

**Test Cases to Implement:**
1. **ModelWeightManager tests:**
   - Weight extraction from agent
   - Cache management (LRU behavior)
   - Agent creation from weights
   - Memory cleanup

2. **In-memory evaluation tests:**
   - Performance comparison (mock I/O operations)
   - Result correctness vs file-based
   - Error handling and fallback
   - Memory usage validation

### Priority 2: Parallel Game Execution (7-10 days)

**Problem:** All games run sequentially, underutilizing multi-core systems.

**Solution:** Implement multiprocessing for parallel game execution.

#### Task 6.2.1: Create Parallel Execution Framework

**File to Create:** `keisei/evaluation/core/parallel_executor.py`

**Before Starting:**
1. **Research multiprocessing constraints:**
   - Read about pickle limitations for PyTorch models
   - Understand process memory isolation
   - Check existing multiprocessing usage in codebase

2. **Understand game execution flow:**
   - Read `keisei/evaluation/loop.py` for current game loop
   - Check `keisei/shogi/shogi_game.py` for game state management
   - Examine `keisei/utils/policy_output_mapper.py` for action conversion

**Implementation Steps:**
1. **Create GameExecutionTask class:**
   ```python
   # Pickleable container for game execution parameters
   # Include weights, config, random seed
   ```

2. **Create execute_single_game_worker function:**
   ```python
   # Top-level function (must be pickleable)
   # Create agents from weights
   # Execute single game
   # Return GameResult data
   ```

3. **Create ParallelGameExecutor class:**
   ```python
   # Manage ProcessPoolExecutor
   # Distribute games across workers
   # Collect and aggregate results
   # Handle timeouts and errors
   ```

#### Task 6.2.2: Update SingleOpponentEvaluator for Parallel Execution

**File to Modify:** `keisei/evaluation/strategies/single_opponent.py`

**Implementation Steps:**
1. **Add parallel execution support:**
   ```python
   # Check if parallel execution is enabled and beneficial
   # Route to parallel or sequential execution
   ```

2. **Add _evaluate_parallel method:**
   ```python
   # Use ParallelGameExecutor
   # Handle result aggregation
   # Manage error recovery
   ```

3. **Keep _evaluate_sequential as fallback:**
   ```python
   # Existing implementation
   # Used when parallel execution disabled or for small game counts
   ```

#### Task 6.2.3: Update Configuration for Parallel Execution

**File to Modify:** `keisei/evaluation/core/evaluation_config.py`

**Implementation Steps:**
1. **Add parallel execution settings:**
   ```python
   enable_parallel_execution: bool = True
   max_concurrent_games: int = 4
   timeout_per_game: Optional[float] = 300.0
   process_restart_threshold: int = 100
   ```

### Priority 3: Background Tournament System (5-8 days)

**Problem:** No continuous tournament evaluation system.

**Solution:** Implement background tournament scheduler.

#### Task 6.3.1: Create Tournament Scheduler

**File to Create:** `keisei/evaluation/core/tournament_scheduler.py`

**Implementation Steps:**
1. **Create TournamentScheduler class:**
   ```python
   # Background thread for tournament processing
   # Match queue management
   # Opponent selection algorithms
   ```

2. **Add to EvaluationManager:**
   ```python
   # Integration with opponent pool
   # Automatic match scheduling for new agents
   ```

## 🧪 Testing Strategy

### Phase 6 Testing Plan

1. **Performance Benchmarks:**
   ```bash
   # Create tests to verify performance improvements
   tests/evaluation/test_performance_benchmarks.py
   ```

2. **Integration Tests:**
   ```bash
   # Test trainer integration
   tests/evaluation/test_trainer_integration.py
   ```

3. **Memory Management Tests:**
   ```bash
   # Test memory usage and cleanup
   tests/evaluation/test_memory_management.py
   ```

### Running Tests During Development

```bash
# Run specific test categories:
pytest tests/evaluation/test_model_manager.py -v
pytest tests/evaluation/test_in_memory_evaluation.py -v
pytest tests/evaluation/test_performance_benchmarks.py -v

# Run all evaluation tests:
pytest tests/evaluation/ -v

# Run with coverage:
pytest tests/evaluation/ --cov=keisei.evaluation --cov-report=html
```

## 🚀 Getting Started Checklist

### Before Starting Implementation:

- [ ] **Read all documentation** (Step 1-3 above)
- [ ] **Run existing tests** to verify current state:
  ```bash
  cd /home/john/keisei
  pytest tests/evaluation/ -v
  ```
- [ ] **Verify trainer integration** works:
  ```bash
  python -c "from keisei.training.trainer import Trainer; print('✅ Trainer imports successfully')"
  ```
- [ ] **Check evaluation manager setup**:
  ```bash
  python -c "from keisei.evaluation.manager import EvaluationManager; print('✅ EvaluationManager imports successfully')"
  ```

### Day 1 Tasks:

1. **Create model_manager.py** with ModelWeightManager class
2. **Write basic tests** for ModelWeightManager
3. **Verify weight extraction** works with existing PPOAgent

### Day 2-3 Tasks:

1. **Update EvaluationManager** with in-memory methods
2. **Update SingleOpponentEvaluator** for in-memory operation
3. **Create comprehensive tests** for in-memory evaluation

### Day 4-5 Tasks:

1. **Update EvaluationCallback** to use in-memory evaluation
2. **Performance testing** and optimization
3. **Memory usage validation**

## 🔍 Debug and Validation Commands

### Verify Current Architecture:
```bash
# Check file structure
find keisei/evaluation -name "*.py" | head -20

# Verify imports work
python -c "
from keisei.evaluation.manager import EvaluationManager
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator
from keisei.evaluation.opponents.opponent_pool import OpponentPool
print('✅ All core imports successful')
"
```

### Monitor Progress:
```bash
# Run tests to check what's working
pytest tests/evaluation/ -v --tb=short

# Check test coverage
pytest tests/evaluation/ --cov=keisei.evaluation --cov-report=term-missing
```

### Performance Monitoring:
```bash
# Monitor memory usage during development
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Current memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## 📚 Key Reference Points During Implementation

### When Working on ModelWeightManager:
- **Reference:** `keisei/core/ppo_agent.py` - How models are structured
- **Reference:** `keisei/training/model_manager.py` - How checkpoints are saved
- **Reference:** `keisei/utils/agent_loading.py` - Current model loading patterns

### When Working on Parallel Execution:
- **Reference:** `keisei/evaluation/loop.py` - Current game execution
- **Reference:** `keisei/shogi/shogi_game.py` - Game state management
- **Reference:** `tests/evaluation/test_evaluate_loop.py` - Game execution tests

### When Working on Integration:
- **Reference:** `keisei/training/callbacks.py` - Current evaluation callback
- **Reference:** `keisei/training/trainer.py` - EvaluationManager setup
- **Reference:** `tests/evaluation/test_evaluation_callback_integration.py` - Integration tests

## ⚡ Quick Start Commands

### Set Up Development Environment:
```bash
cd /home/john/keisei
source env/bin/activate  # If using virtual environment
pip install -e .  # Install in development mode
```

### Start Implementation:
```bash
# Create the first new file
touch keisei/evaluation/core/model_manager.py

# Create test file
touch tests/evaluation/test_model_manager.py

# Start with basic structure and imports
```

This comprehensive guide provides everything needed to pick up where the evaluation system refactor left off and complete the performance optimization phase.
