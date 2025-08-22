# Keisei Codebase Remediation Plan

**Start Date**: January 21, 2025  
**Target Completion**: February 18, 2025 (4 weeks)  
**Based on**: CODEBASE_QUALITY_ASSESSMENT.md

## Plan Overview

This remediation plan follows a "test as we go" approach with meaningful tests that minimize mocking. Each phase includes documentation updates and verification steps.

## Progress Tracking

- ⬜ Not Started
- 🟡 In Progress  
- ✅ Completed
- ❌ Blocked

## Phase 1: Critical Blockers (Days 1-3)

### Day 1: Fix Installation and Dependencies

⬜ **Task 1.1: Clean up requirements.txt**
```bash
# Actions:
1. Remove line: -e git+https://github.com/tachyon-beep/shogidrl.git@60aa1cdb...
2. Remove duplicate wcwidth entry (keep only one)
3. Pin numpy version: numpy>=1.24.0,<2.0
4. Update certifi to latest: certifi>=2025.7.14
5. Align psutil versions (use 7.0.0 in both files)
6. Remove six dependency
```

⬜ **Task 1.2: Test installation process**
```bash
# Create fresh virtual environment and test:
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
python -c "import keisei; print('Installation successful')"
```

⬜ **Task 1.3: Write installation test**
```python
# tests/test_installation.py
import subprocess
import sys
import tempfile
import os

def test_requirements_valid():
    """Test that requirements files are valid and installable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test requirements.txt
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"requirements.txt invalid: {result.stderr}"

def test_package_imports():
    """Test that all major modules can be imported."""
    import keisei
    import keisei.core
    import keisei.shogi
    import keisei.training
    import keisei.evaluation
    import keisei.utils
```

⬜ **Documentation Update**: Update CODEBASE_QUALITY_ASSESSMENT.md to mark dependency issues as resolved

### Day 2: Create Missing Core Documentation

⬜ **Task 2.1: Create DESIGN.md**
```markdown
# docs/1. DESIGN.md
- System architecture overview
- Manager-based design rationale
- Component relationships
- Data flow diagrams
- Design decisions and trade-offs
```

⬜ **Task 2.2: Create CODE_MAP.md**
```markdown
# docs/2. CODE_MAP.md
- Directory structure explanation
- Module responsibilities
- Key classes and their purposes
- Navigation guide for common tasks
- Import dependency graph
```

⬜ **Task 2.3: Fix README.md references**
```bash
# Update README to point to correct file locations
# Fix LICENSE file (change "John Doe" to actual author)
```

⬜ **Documentation Update**: Mark documentation blockers as resolved

### Day 3: Standardize Logging Infrastructure

⬜ **Task 3.1: Choose logging approach**
```python
# Decision: Standardize on Python's logging module
# Remove or deprecate UnifiedLogger
# Create keisei/utils/logging_config.py
```

⬜ **Task 3.2: Create logging configuration module**
```python
# keisei/utils/logging_config.py
import logging
import logging.config
from pathlib import Path

def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Configure logging for the entire application."""
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(log_dir / 'keisei.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'standard',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['file', 'console']
        }
    }
    logging.config.dictConfig(config)
```

⬜ **Task 3.3: Update all modules to use standard logging**
```bash
# Script to update imports
find keisei -name "*.py" -exec sed -i 's/from.*unified_logger.*/import logging/' {} \;
# Manual review and update of logging calls
```

⬜ **Task 3.4: Write logging tests**
```python
# tests/test_logging_config.py
def test_logging_setup():
    """Test that logging is configured correctly."""
    # Test without mocking - actually configure logging and verify
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_logging(Path(tmpdir), "DEBUG")
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Verify log file exists and contains message
        log_file = Path(tmpdir) / "keisei.log"
        assert log_file.exists()
        assert "Test message" in log_file.read_text()
```

## Phase 2: Core Component Testing (Days 4-10)

### Day 4-5: Test Trainer Class

⬜ **Task 4.1: Create comprehensive Trainer tests**
```python
# tests/test_trainer.py
# Focus on integration tests with minimal mocking
# Test actual training loop with small config

def test_trainer_initialization():
    """Test trainer can be initialized with config."""
    config = create_minimal_config()
    trainer = Trainer(config)
    assert trainer is not None
    assert trainer.config == config

def test_trainer_short_training_run():
    """Test actual training for a few steps."""
    config = create_minimal_config()
    config.training.total_timesteps = 100
    config.training.steps_per_epoch = 50
    
    trainer = Trainer(config)
    trainer.train()
    
    # Verify training completed
    assert trainer.training_loop_manager.timesteps_done >= 100
    # Verify model was updated
    assert trainer.model_manager.model is not None

def test_trainer_checkpoint_resume():
    """Test training can be resumed from checkpoint."""
    # Train for 100 steps, save checkpoint
    # Create new trainer, resume, verify continues from 100
```

⬜ **Task 4.2: Test callback system**
```python
# tests/test_callbacks.py
def test_callback_execution_order():
    """Test callbacks are executed in correct order."""
    # Create actual callback instances
    # Run them through real callback manager
    # Verify order without mocking

def test_evaluation_callback_integration():
    """Test evaluation callback with real evaluation."""
    # Use actual game and evaluation
    # Verify metrics are computed correctly
```

### Day 6-7: Test Display Components

⬜ **Task 5.1: Test display managers**
```python
# tests/test_display_manager.py
def test_display_manager_with_real_terminal():
    """Test display updates in actual terminal."""
    # Use pytest-console or similar to test real output
    # Verify progress bars update correctly
    # Test error handling for terminal resize
```

⬜ **Task 5.2: Test metrics formatting**
```python
# tests/test_metrics_manager.py
def test_metrics_calculation_accuracy():
    """Test metrics are calculated correctly with real data."""
    # Use actual game outcomes
    # Verify win rate, episode length calculations
    # No mocking of statistics
```

### Day 8-10: Test Parallel Components

⬜ **Task 6.1: Test parallel training infrastructure**
```python
# tests/test_parallel_training.py
def test_worker_communication():
    """Test actual worker process communication."""
    # Spawn real worker processes
    # Test data serialization/deserialization
    # Verify no data corruption

def test_model_synchronization():
    """Test model weights are synchronized correctly."""
    # Create multiple workers with same initial model
    # Update model in one worker
    # Verify synchronization across workers
```

## Phase 3: Performance Optimization (Days 11-17)

### Day 11-12: Optimize Shogi Engine

⬜ **Task 7.1: Implement king position caching**
```python
# keisei/shogi/shogi_game.py
class ShogiGame:
    def __init__(self):
        self._king_positions = {Player.SENTE: None, Player.GOTE: None}
        self._update_king_positions()
    
    def _update_king_positions(self):
        """Cache king positions for quick lookup."""
        # Implementation
```

⬜ **Task 7.2: Optimize legal move generation**
```python
# Create incremental move generation
# Avoid deep copies where possible
# Use move ordering heuristics
```

⬜ **Task 7.3: Write performance benchmarks**
```python
# tests/benchmarks/test_shogi_performance.py
def test_legal_move_generation_performance():
    """Benchmark legal move generation."""
    game = ShogiGame()
    # Set up complex position
    
    start = time.time()
    for _ in range(1000):
        moves = game.get_legal_moves()
    elapsed = time.time() - start
    
    # Assert performance threshold
    assert elapsed < 1.0, f"Move generation too slow: {elapsed}s for 1000 iterations"
```

### Day 13-14: Implement Zobrist Hashing

⬜ **Task 8.1: Replace tuple hashing with Zobrist**
```python
# keisei/shogi/zobrist.py
class ZobristHasher:
    """Efficient position hashing for repetition detection."""
    def __init__(self):
        self.piece_keys = self._generate_piece_keys()
        self.side_key = random.getrandbits(64)
    
    def hash_position(self, game: ShogiGame) -> int:
        """Generate Zobrist hash for current position."""
        # Implementation
```

⬜ **Task 8.2: Test hash collisions**
```python
def test_zobrist_hash_distribution():
    """Test hash function has good distribution."""
    # Generate many positions
    # Verify low collision rate
    # Test incremental updates work correctly
```

### Day 15-17: Clean Technical Debt

⬜ **Task 9.1: Remove deprecated directory**
```bash
# After verifying nothing references deprecated code
rm -rf deprecated/
git commit -m "Remove deprecated code after verification"
```

⬜ **Task 9.2: Extract magic numbers to constants**
```python
# keisei/constants.py
BOARD_SIZE = 9
BOARD_ROWS = 9
BOARD_COLS = 9
MAX_PIECES_IN_HAND = 18
DEFAULT_MAX_MOVES = 500
```

⬜ **Task 9.3: Simplify manager architecture**
```python
# Consolidate overlapping managers
# Create clear interfaces
# Document responsibilities
```

## Phase 4: Documentation and Error Handling (Days 18-21)

### Day 18-19: Improve Error Handling

⬜ **Task 10.1: Add logging to silent failures**
```python
# Fix all instances of silent exception handling
# Add at least WARNING level logs
# Include context in error messages
```

⬜ **Task 10.2: Add global exception handler**
```python
# keisei/utils/error_handler.py
def setup_global_error_handler():
    """Configure global exception handling."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Graceful shutdown logic
    
    sys.excepthook = handle_exception
```

### Day 20-21: Create Developer Documentation

⬜ **Task 11.1: Create developer setup guide**
```markdown
# docs/DEVELOPER_SETUP.md
- Development environment setup
- IDE configuration
- Running tests locally
- Debugging tips
```

⬜ **Task 11.2: Create troubleshooting guide**
```markdown
# docs/TROUBLESHOOTING.md
- Common installation issues
- Training failures and solutions
- Performance problems
- GPU/CUDA issues
```

⬜ **Task 11.3: Create contribution guide**
```markdown
# CONTRIBUTING.md
- Code style guidelines
- Testing requirements
- Pull request process
- Issue reporting guidelines
```

## Phase 5: Final Validation (Days 22-28)

### Day 22-24: Integration Testing

⬜ **Task 12.1: End-to-end training test**
```python
def test_full_training_pipeline():
    """Test complete training pipeline with all components."""
    # Run actual training for 1000 steps
    # Verify all components work together
    # Check metrics, checkpoints, evaluation
```

⬜ **Task 12.2: Performance validation**
```python
def test_training_performance_baseline():
    """Establish and test performance baselines."""
    # Measure steps per second
    # Memory usage
    # GPU utilization
```

### Day 25-26: Documentation Review

⬜ **Task 13.1: Review all documentation**
- Verify all links work
- Check code examples run
- Ensure consistency across docs

⬜ **Task 13.2: Update CODEBASE_QUALITY_ASSESSMENT.md**
- Mark all completed items
- Document any remaining issues
- Create maintenance recommendations

### Day 27-28: Final Cleanup

⬜ **Task 14.1: Run full CI pipeline**
```bash
./scripts/run_local_ci.sh
# Fix any issues found
```

⬜ **Task 14.2: Create release notes**
```markdown
# REMEDIATION_COMPLETE.md
- Summary of changes
- Remaining known issues
- Performance improvements
- Next steps recommendations
```

## Testing Philosophy

1. **Minimal Mocking**: Only mock external dependencies (filesystem, network)
2. **Integration Focus**: Test components working together
3. **Real Data**: Use actual game states and training data
4. **Performance Tests**: Include benchmarks to prevent regression
5. **Error Path Testing**: Test failure cases with real scenarios

## Documentation Updates

After each task completion:
1. Update this REMEDIATION_PLAN.md with ✅
2. Update CODEBASE_QUALITY_ASSESSMENT.md 
3. Add notes about any new issues discovered
4. Document any deviations from plan

## Success Criteria

- [ ] All blockers resolved
- [ ] Installation works on fresh system
- [ ] Core components have >80% test coverage
- [ ] Performance improved by >50% for move generation
- [ ] All silent failures eliminated
- [ ] Documentation complete and accurate
- [ ] CI pipeline passes without warnings

## Risk Mitigation

1. **Daily Backups**: Commit working changes daily
2. **Branch Strategy**: Work on feature branches
3. **Rollback Plan**: Tag stable points for quick rollback
4. **Testing**: Run tests after each change
5. **Documentation**: Update docs as you go, not at end

This plan provides concrete, executable tasks with clear verification steps and minimal test mocking as requested.