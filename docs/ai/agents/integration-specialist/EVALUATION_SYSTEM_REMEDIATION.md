# Keisei Evaluation System: Technical Remediation Specification

**Document Version**: 1.0  
**Date**: January 22, 2025  
**Prepared By**: Integration Specialist  
**Review Status**: Ready for Implementation

---

## Executive Summary

The Keisei evaluation system exhibits a critical architectural pattern: **well-tested individual components with broken integration layer**. While unit and integration tests pass successfully, the system suffers from fundamental coordination failures that prevent end-to-end operation in production environments.

**Core Issue**: The evaluation system was designed as a distributed architecture but lacks the integration infrastructure necessary for component coordination. This results in policy mapper mismatches, configuration incompatibilities, device placement failures, and incomplete implementations that render the system unusable for actual model evaluation.

**Remediation Scope**: 17 critical integration issues across 9 subsystems requiring coordinated fixes to policy propagation, configuration systems, async execution models, and missing implementations.

---

## Detailed Bug Inventory

### CRITICAL SEVERITY - System Blockers

#### BUG-001: Policy Mapper Propagation Failure
**Location**: `keisei/evaluation/strategies/tournament.py:67`, `keisei/utils/agent_loading.py:17-19`  
**Issue**: Each evaluation strategy creates its own `PolicyOutputMapper` instance instead of receiving the shared instance from the training context.

```python
# Current broken pattern (tournament.py:67)
class TournamentEvaluator(BaseEvaluator):
    def __init__(self, config: TournamentConfig):
        super().__init__(config)
        self.policy_mapper = PolicyOutputMapper()  # ❌ Creates new instance
```

**Impact**: Action space mismatches between training and evaluation, causing model action selection to fail with dimension errors.

**Root Cause**: No dependency injection mechanism for sharing PolicyOutputMapper instances across system boundaries.

**Evidence**: Training system uses one action space mapping while evaluation creates an independent mapping, leading to tensor dimension mismatches during action selection.

---

#### BUG-002: Configuration Schema Incompatibility  
**Location**: `keisei/evaluation/core/evaluation_config.py:172-178`, `keisei/utils/agent_loading.py:48-170`  
**Issue**: Multiple incompatible configuration schemas that cannot be serialized/deserialized consistently.

```python
# agent_loading.py creates incompatible AppConfig
config = AppConfig(
    # ... 120 lines of hardcoded config values
    env=EnvConfig(device=device_str, input_channels=input_channels, ...)
    # This config is incompatible with EvaluationConfig schema
)
```

**Impact**: Runtime configuration validation failures when evaluation contexts are passed between components.

**Root Cause**: No unified configuration interface; each subsystem expects different config object types.

---

#### BUG-003: Async/Sync Execution Model Conflicts
**Location**: `keisei/evaluation/core_manager.py:101-102`, `keisei/evaluation/core_manager.py:131`  
**Issue**: Unsafe use of `asyncio.run()` within potentially already-running event loops.

```python
# Broken pattern in EvaluationManager
def evaluate_checkpoint(self, agent_checkpoint: str) -> EvaluationResult:
    evaluator = EvaluatorFactory.create(self.config)
    return asyncio.run(evaluator.evaluate(agent_info, context))  # ❌ Unsafe
```

**Impact**: "RuntimeError: asyncio.run() cannot be called from a running event loop" in training contexts.

**Root Cause**: No coordination between training system event loop and evaluation async operations.

---

### HIGH SEVERITY - Integration Failures

#### BUG-004: Model Loading Device Placement Violations
**Location**: `keisei/utils/agent_loading.py:173-176`, `keisei/evaluation/core_manager.py:46-51`  
**Issue**: Inconsistent device placement for model loading across evaluation strategies.

```python
# agent_loading.py forces CPU device creation
device = torch.device(device_str)
temp_model = ActorCritic(input_channels, policy_mapper.get_total_actions()).to(device)
# But ModelWeightManager may expect different device
```

**Impact**: CUDA/CPU device mismatches causing evaluation failures and GPU memory leaks.

**Root Cause**: No centralized device management strategy across evaluation components.

---

#### BUG-005: Missing Tournament In-Memory Implementation
**Location**: `keisei/evaluation/strategies/tournament.py:312-336`  
**Issue**: Tournament in-memory evaluation returns empty results instead of actual implementation.

```python
async def evaluate_in_memory(self, agent_info: AgentInfo, ...) -> EvaluationResult:
    # For now, return empty results as in-memory tournament is not fully implemented
    evaluation_result = EvaluationResult(
        context=context, games=[], summary_stats=SummaryStats.from_games([]), ...
    )  # ❌ Empty placeholder implementation
```

**Impact**: In-memory evaluation mode completely non-functional for tournament strategy.

**Root Cause**: Incomplete implementation marked as "temporary" but never completed.

---

#### BUG-006: EvaluatorFactory Missing Dependency Injection
**Location**: `keisei/evaluation/core/base_evaluator.py:316-359`  
**Issue**: Factory creates evaluators without runtime context (device, policy mapper, etc.).

```python
@classmethod
def create(cls, config: EvaluationConfig) -> BaseEvaluator:
    evaluator_class = cls._evaluators[strategy_name]
    return evaluator_class(config)  # ❌ No dependency injection
```

**Impact**: Evaluators cannot access shared runtime context, leading to configuration mismatches.

**Root Cause**: Factory pattern implementation lacks dependency injection capabilities.

---

### MEDIUM SEVERITY - Protocol Violations  

#### BUG-007: Parallel Executor Integration Disconnection
**Location**: `keisei/evaluation/core/parallel_executor.py:162-185`  
**Issue**: Parallel executor attempts to use `evaluate_step_in_memory` method that doesn't exist on evaluators.

```python
if hasattr(task.game_executor.__self__, "evaluate_step_in_memory"):
    result = asyncio.run(
        task.game_executor.__self__.evaluate_step_in_memory(...)
    )  # ❌ Method doesn't exist
```

**Impact**: Parallel evaluation falls back to regular evaluation, negating performance benefits.

**Root Cause**: Parallel executor assumes protocol compliance that evaluators don't implement.

---

#### BUG-008: CUSTOM Strategy Not Implemented
**Location**: `keisei/evaluation/core/evaluation_config.py:23`, `keisei/evaluation/core/evaluation_config.py:177`  
**Issue**: CUSTOM strategy defined in enum but no implementation exists.

**Impact**: Configuration validation passes but runtime creation fails for CUSTOM strategy.

**Root Cause**: Incomplete strategy implementation suite.

---

#### BUG-009: Model Weight Manager Cache Inconsistency
**Location**: `keisei/evaluation/core_manager.py:175-215`  
**Issue**: Model weight caching without proper cache invalidation or consistency checks.

```python
# No mechanism to ensure cached weights match current model state
opponent_weights = self.model_weight_manager.cache_opponent_weights(
    opponent_path.stem, opponent_path
)  # ❌ No consistency validation
```

**Impact**: Stale cached weights used for evaluation, leading to incorrect results.

**Root Cause**: Cache management without proper invalidation strategy.

---

### MEDIUM SEVERITY - Missing Integrations

#### BUG-010: ELO System Architecture Undefined
**Location**: `keisei/evaluation/analytics/elo_tracker.py:17-192`  
**Issue**: ELO tracking system exists but no integration points defined for automatic updates.

**Impact**: ELO ratings not automatically updated during evaluations despite `update_elo=True` configuration.

**Root Cause**: No architectural decision on when and how ELO updates should be triggered.

---

#### BUG-011: Background Tournament Manager Missing Implementation
**Location**: `keisei/evaluation/enhanced_manager.py:72-83`  
**Issue**: Background tournament feature attempts import but implementation doesn't exist.

```python
try:
    from .core.background_tournament import BackgroundTournamentManager  # ❌ Missing
except ImportError as e:
    logger.warning(f"Background tournaments not available: {e}")
```

**Impact**: Enhanced evaluation manager silently disables background tournament functionality.

**Root Cause**: Feature architecture designed but implementation never created.

---

#### BUG-012: OpponentPool Integration Incomplete
**Location**: `keisei/evaluation/core_manager.py:42`, `keisei/evaluation/opponents/opponent_pool.py`  
**Issue**: OpponentPool imported but file doesn't exist.

**Impact**: EvaluationManager initialization fails when opponent pool functionality is needed.

**Root Cause**: Missing implementation for opponent pool management system.

---

### LOW SEVERITY - Interface Issues

#### BUG-013: Configuration Validation Bypass
**Location**: `keisei/evaluation/core/evaluation_config.py:227-248`  
**Issue**: Configuration factory falls back to base config with strategy_params instead of proper validation.

```python
except TypeError as e:
    logger.error("Invalid parameters for %s config: %s", strategy.value, e)
    # Fall back to base config with strategy params  # ❌ Bypasses validation
    return EvaluationConfig(strategy=strategy, **base_params)
```

**Impact**: Invalid configurations silently accepted, leading to runtime failures.

**Root Cause**: Error handling that bypasses proper validation instead of failing fast.

---

#### BUG-014: Game Result Metadata Inconsistency
**Location**: `keisei/evaluation/strategies/tournament.py:172-186`  
**Issue**: Game result metadata fields inconsistent across different execution paths.

**Impact**: Analytics and reporting systems receive inconsistent metadata structures.

**Root Cause**: No standardized metadata schema for game results.

---

#### BUG-015: Agent Loading Configuration Bloat
**Location**: `keisei/utils/agent_loading.py:48-170`  
**Issue**: 120+ lines of hardcoded configuration values for simple agent loading.

**Impact**: Maintenance burden and potential configuration drift from actual system configs.

**Root Cause**: No shared configuration service; each component recreates full config objects.

---

#### BUG-016: Error Handling Inconsistency
**Location**: Multiple files across evaluation system  
**Issue**: Inconsistent error handling patterns - some return None, some raise, some log and continue.

**Impact**: Unpredictable error behavior makes debugging difficult.

**Root Cause**: No standardized error handling protocol across evaluation components.

---

#### BUG-017: Resource Cleanup Missing
**Location**: `keisei/evaluation/core_manager.py:172-258`  
**Issue**: No cleanup mechanisms for failed evaluation attempts or memory pressure handling.

**Impact**: Resource leaks during evaluation failures, especially in GPU environments.

**Root Cause**: No resource lifecycle management in evaluation contexts.

---

## Phase-by-Phase Remediation Plan

### Phase 1: Critical Integration Infrastructure (Week 1-2)
**Priority**: HIGHEST - System Blockers  
**Risk Level**: HIGH (Breaking changes required)  
**Dependencies**: None

#### Task 1.1: Implement Dependency Injection Framework
**Files**: `keisei/evaluation/core/evaluation_context.py`, `keisei/evaluation/core/base_evaluator.py`

```python
# NEW: Evaluation context with runtime dependencies
@dataclass
class EvaluationRuntimeContext:
    """Runtime dependencies for evaluation system."""
    device: torch.device
    policy_mapper: PolicyOutputMapper
    event_loop: Optional[asyncio.AbstractEventLoop] = None
    resource_manager: Optional[ResourceManager] = None
    
class BaseEvaluator(ABC):
    def __init__(self, config: EvaluationConfig, runtime_context: EvaluationRuntimeContext):
        self.config = config
        self.runtime_context = runtime_context
        # Policy mapper now injected, not created locally
        self.policy_mapper = runtime_context.policy_mapper
```

**Testing Requirements**: Unit tests for dependency injection, integration tests for policy mapper consistency.

**Success Criteria**: All evaluators receive consistent runtime context without creating local instances.

---

#### Task 1.2: Unified Configuration Interface
**Files**: `keisei/evaluation/core/evaluation_config.py`, `keisei/config_schema.py`

```python
# MODIFIED: Unified configuration protocol
class EvaluationConfigProtocol(Protocol):
    """Protocol for evaluation configuration objects."""
    strategy: EvaluationStrategy
    num_games: int
    device: str
    
    def to_evaluation_context(self, runtime_deps: EvaluationRuntimeContext) -> EvaluationContext:
        """Convert to evaluation context with runtime dependencies."""
        ...

# MODIFIED: Configuration factory with validation
def create_evaluation_config(
    strategy: Union[str, EvaluationStrategy], 
    runtime_context: EvaluationRuntimeContext,
    **kwargs
) -> EvaluationConfigProtocol:
    """Factory with proper validation - no fallback to invalid configs."""
    # Strict validation, fail fast on invalid configs
    ...
```

**Testing Requirements**: Configuration serialization/deserialization tests, validation boundary testing.

**Success Criteria**: All configuration objects implement consistent interface, no silent validation bypasses.

---

#### Task 1.3: Async-Safe Evaluation Manager
**Files**: `keisei/evaluation/core_manager.py`

```python
# MODIFIED: Safe async/sync coordination
class EvaluationManager:
    def __init__(self, config, runtime_context: EvaluationRuntimeContext):
        self.config = config
        self.runtime_context = runtime_context
        
    def evaluate_checkpoint(self, agent_checkpoint: str) -> EvaluationResult:
        """Thread-safe synchronous wrapper."""
        if self.runtime_context.event_loop and self.runtime_context.event_loop.is_running():
            # Running in async context - use thread executor
            return self._run_in_thread(self._evaluate_checkpoint_async, agent_checkpoint)
        else:
            # Safe to create new event loop
            return asyncio.run(self._evaluate_checkpoint_async(agent_checkpoint))
            
    async def _evaluate_checkpoint_async(self, agent_checkpoint: str) -> EvaluationResult:
        """Actual async implementation."""
        ...
```

**Testing Requirements**: Event loop conflict testing, thread safety validation.

**Success Criteria**: No `asyncio.run()` errors in training contexts, proper async/sync coordination.

---

### Phase 2: Protocol Implementation Completion (Week 3)
**Priority**: HIGH - Missing Core Features  
**Risk Level**: MEDIUM (Additive implementations)  
**Dependencies**: Phase 1 completion

#### Task 2.1: Tournament In-Memory Evaluation Implementation
**Files**: `keisei/evaluation/strategies/tournament.py`

```python
async def evaluate_in_memory(
    self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None,
    *, agent_weights: Optional[Dict[str, torch.Tensor]] = None,
    opponent_weights: Optional[Dict[str, torch.Tensor]] = None,
    opponent_info: Optional[OpponentInfo] = None,
) -> EvaluationResult:
    """Full implementation of in-memory tournament evaluation."""
    
    # Load opponents using provided weights or fallback to file loading
    opponents = await self._load_tournament_opponents_in_memory(
        opponent_weights, opponent_info
    )
    
    # Create in-memory model instances
    agent_model = self._create_model_from_weights(agent_weights, agent_info)
    
    # Execute tournament games using in-memory models
    all_games = []
    for opponent in opponents:
        opponent_model = self._create_model_from_weights(
            opponent.metadata.get("weights"), opponent
        )
        games = await self._play_in_memory_games(
            agent_model, opponent_model, context
        )
        all_games.extend(games)
    
    return EvaluationResult(
        context=context,
        games=all_games,
        summary_stats=SummaryStats.from_games(all_games),
        analytics_data={"tournament_specific_analytics": self._calculate_tournament_standings(all_games, opponents, agent_info)},
    )
```

**Testing Requirements**: In-memory tournament integration tests, memory usage validation.

**Success Criteria**: Tournament strategy supports both file-based and in-memory evaluation modes.

---

#### Task 2.2: Parallel Executor Protocol Compliance
**Files**: `keisei/evaluation/core/parallel_executor.py`, `keisei/evaluation/core/base_evaluator.py`

```python
# ADDED: Protocol method to BaseEvaluator
class BaseEvaluator(ABC):
    async def evaluate_step_in_memory(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo, 
        context: EvaluationContext,
    ) -> GameResult:
        """In-memory game step evaluation - default implementation."""
        # Default fallback to regular evaluate_step
        return await self.evaluate_step(agent_info, opponent_info, context)

# MODIFIED: Parallel executor with protocol compliance check
class ParallelGameExecutor:
    def _execute_single_game(self, task: ParallelGameTask) -> Optional[GameResult]:
        try:
            evaluator = task.game_executor.__self__
            if (use_in_memory and 
                hasattr(evaluator, 'evaluate_step_in_memory') and
                callable(getattr(evaluator, 'evaluate_step_in_memory'))):
                # Protocol-compliant in-memory evaluation
                result = asyncio.run(evaluator.evaluate_step_in_memory(...))
            else:
                # Regular evaluation fallback
                result = asyncio.run(task.game_executor(...))
```

**Testing Requirements**: Parallel execution with in-memory evaluation, protocol compliance validation.

**Success Criteria**: Parallel executor properly uses in-memory evaluation when available.

---

#### Task 2.3: CUSTOM Strategy Implementation Scaffolding  
**Files**: `keisei/evaluation/strategies/custom.py` (NEW)

```python
class CustomEvaluator(BaseEvaluator):
    """Custom evaluation strategy with user-defined logic."""
    
    def __init__(self, config: EvaluationConfig, runtime_context: EvaluationRuntimeContext):
        super().__init__(config, runtime_context)
        # Load custom evaluation logic from strategy_params
        self.custom_logic = self._load_custom_logic(config.strategy_params)
        
    def _load_custom_logic(self, params: Dict[str, Any]):
        """Load custom evaluation logic from configuration."""
        # Support for custom Python modules or configuration-driven logic
        if "module_path" in params:
            return self._import_custom_module(params["module_path"])
        elif "logic_config" in params:
            return self._create_logic_from_config(params["logic_config"])
        else:
            raise ValueError("CUSTOM strategy requires either module_path or logic_config")
    
    async def evaluate_step(
        self, agent_info: AgentInfo, opponent_info: OpponentInfo, context: EvaluationContext
    ) -> GameResult:
        """Execute custom evaluation step."""
        return await self.custom_logic.evaluate_step(agent_info, opponent_info, context)

# Register custom evaluator
EvaluatorFactory.register(EvaluationStrategy.CUSTOM.value, CustomEvaluator)
```

**Testing Requirements**: Custom strategy loading tests, configuration validation.

**Success Criteria**: CUSTOM strategy can be configured and executed without runtime errors.

---

### Phase 3: Integration Layer Fixes (Week 4)
**Priority**: MEDIUM - System Polish  
**Risk Level**: LOW (Non-breaking improvements)  
**Dependencies**: Phase 1-2 completion

#### Task 3.1: Model Weight Manager Consistency
**Files**: `keisei/evaluation/core/model_manager.py` (NEW)

```python
class ModelWeightManager:
    """Centralized model weight management with consistency guarantees."""
    
    def __init__(self, device: torch.device, max_cache_size: int = 5):
        self.device = device
        self.cache = {}
        self.cache_metadata = {}  # Track checksums and timestamps
        self.max_cache_size = max_cache_size
    
    def cache_opponent_weights(
        self, opponent_id: str, checkpoint_path: Path
    ) -> Dict[str, torch.Tensor]:
        """Cache opponent weights with consistency validation."""
        
        # Check if cache is still valid
        if self._is_cache_valid(opponent_id, checkpoint_path):
            return self.cache[opponent_id]
            
        # Load and validate weights
        weights = self._load_and_validate_weights(checkpoint_path)
        
        # Update cache with metadata
        self._update_cache(opponent_id, weights, checkpoint_path)
        return weights
    
    def _is_cache_valid(self, opponent_id: str, checkpoint_path: Path) -> bool:
        """Validate cache consistency using file metadata."""
        if opponent_id not in self.cache:
            return False
            
        cached_metadata = self.cache_metadata[opponent_id]
        current_mtime = checkpoint_path.stat().st_mtime
        current_size = checkpoint_path.stat().st_size
        
        return (cached_metadata["mtime"] == current_mtime and 
                cached_metadata["size"] == current_size)
```

**Testing Requirements**: Cache consistency tests, memory pressure handling.

**Success Criteria**: Model weight caching with proper invalidation, no stale weights used.

---

#### Task 3.2: ELO System Integration Points
**Files**: `keisei/evaluation/core/elo_integration.py` (NEW)

```python
class ELOIntegrationManager:
    """Manages ELO rating updates during evaluation workflows."""
    
    def __init__(self, elo_tracker: EloTracker, config: EvaluationConfig):
        self.elo_tracker = elo_tracker
        self.should_update = config.update_elo
        
    async def process_evaluation_result(self, result: EvaluationResult) -> None:
        """Process evaluation result and update ELO ratings if configured."""
        if not self.should_update:
            return
            
        for game in result.games:
            await self._update_elo_from_game(game)
            
    async def _update_elo_from_game(self, game: GameResult) -> None:
        """Update ELO ratings based on single game result."""
        if not game.agent_info or not game.opponent_info:
            return
            
        agent_id = game.agent_info.name
        opponent_id = game.opponent_info.name
        
        # Convert game winner to score (0.0, 0.5, 1.0)
        agent_score = self._game_result_to_score(game.winner)
        
        # Update ratings
        new_agent_rating, new_opponent_rating = self.elo_tracker.update_rating(
            agent_id, opponent_id, agent_score
        )
        
        logger.debug(f"ELO updated: {agent_id}: {new_agent_rating}, {opponent_id}: {new_opponent_rating}")
```

**Testing Requirements**: ELO integration tests, rating persistence validation.

**Success Criteria**: ELO ratings automatically updated during evaluations when configured.

---

#### Task 3.3: Resource Cleanup and Error Handling
**Files**: `keisei/evaluation/core/resource_manager.py` (NEW)

```python
class EvaluationResourceManager:
    """Manages resources and cleanup for evaluation contexts."""
    
    def __init__(self, max_memory_mb: int = 2048, cleanup_threshold: float = 0.8):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.active_contexts = weakref.WeakSet()
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._start_monitoring()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources on context exit."""
        await self._cleanup_all_contexts()
        self._stop_monitoring()
        
    def register_context(self, context: EvaluationContext) -> None:
        """Register context for resource monitoring."""
        self.active_contexts.add(context)
        
    async def _cleanup_all_contexts(self) -> None:
        """Clean up all registered contexts."""
        for context in self.active_contexts:
            try:
                await self._cleanup_context(context)
            except Exception as e:
                logger.error(f"Error cleaning up context {context.session_id}: {e}")
                
    async def _cleanup_context(self, context: EvaluationContext) -> None:
        """Clean up specific evaluation context."""
        # Clear any cached models, close file handles, etc.
        if hasattr(context, '_cached_models'):
            for model in context._cached_models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()  # Move to CPU to free GPU memory
            context._cached_models.clear()
```

**Testing Requirements**: Resource cleanup tests, memory leak detection.

**Success Criteria**: Proper resource cleanup during evaluation failures, no memory leaks.

---

## Testing Strategy

### Integration Test Requirements

1. **End-to-End Policy Consistency Tests**
   ```python
   async def test_policy_mapper_consistency():
       """Test policy mapper consistency between training and evaluation."""
       # Create training context with policy mapper
       training_policy_mapper = PolicyOutputMapper()
       
       # Create evaluation context with same policy mapper
       runtime_context = EvaluationRuntimeContext(
           device=torch.device('cpu'),
           policy_mapper=training_policy_mapper
       )
       
       # Verify both contexts use same action space
       training_actions = training_policy_mapper.get_total_actions()
       eval_manager = EvaluationManager(config, runtime_context)
       
       # Evaluation should use same action space
       assert eval_manager.runtime_context.policy_mapper.get_total_actions() == training_actions
   ```

2. **Configuration Compatibility Tests**
   ```python
   def test_configuration_serialization():
       """Test configuration objects can be serialized/deserialized consistently."""
       original_config = TournamentConfig(
           num_games=100, 
           opponent_pool_config=[{"name": "test", "type": "random"}]
       )
       
       # Serialize and deserialize
       config_dict = original_config.to_dict()
       restored_config = EvaluationConfig.from_dict(config_dict)
       
       # Should be equivalent
       assert restored_config.num_games == original_config.num_games
       assert restored_config.strategy == original_config.strategy
   ```

3. **Async Event Loop Compatibility Tests**
   ```python
   async def test_evaluation_in_async_context():
       """Test evaluation manager works correctly in async contexts."""
       # This should not raise RuntimeError
       eval_manager = EvaluationManager(config, runtime_context)
       result = await eval_manager.evaluate_current_agent_async(agent)
       assert result is not None
   ```

### Regression Test Plans

1. **Performance Validation**
   - Ensure in-memory evaluation is faster than file-based evaluation
   - Validate parallel execution provides expected speedup
   - Memory usage should not exceed configured limits

2. **Compatibility Testing**  
   - All existing evaluation tests continue to pass
   - Configuration migrations work correctly
   - Backward compatibility for existing checkpoint files

### Mock Reduction Strategy

Current tests rely heavily on mocking, which can hide integration issues. The remediation includes:

1. **Real Component Integration Tests**: Use actual game engines, real model loading, actual file I/O
2. **Lightweight Test Fixtures**: Minimal models and configurations for faster test execution  
3. **Focused Mock Usage**: Mock only external dependencies (network, filesystem) not internal components

---

## Resource Planning

### Developer Skill Requirements

**Phase 1 (Integration Infrastructure)**: Senior developer with asyncio and dependency injection experience  
**Phase 2 (Protocol Implementation)**: Mid-level developer familiar with PyTorch and evaluation patterns  
**Phase 3 (System Polish)**: Junior-to-mid level developer for cleanup and testing tasks

### Time Estimates

| Phase | Tasks | Estimated Time | Risk Buffer | Total |
|-------|-------|----------------|-------------|-------|
| Phase 1 | Critical Integration (3 tasks) | 8 days | 2 days | 10 days |
| Phase 2 | Protocol Implementation (3 tasks) | 5 days | 2 days | 7 days |  
| Phase 3 | Integration Polish (3 tasks) | 4 days | 1 day | 5 days |
| **Total** | **9 tasks** | **17 days** | **5 days** | **22 days** |

### Infrastructure Needs

1. **Development Environment**: GPU-enabled testing environment for device placement validation
2. **CI/CD Updates**: Extended test timeouts for integration tests, memory usage monitoring
3. **Documentation**: Architecture decision records for integration patterns

### Risk Mitigation Strategies

1. **Incremental Rollout**: Each phase can be deployed independently with feature flags
2. **Rollback Planning**: Maintain current evaluation system as fallback during Phase 1
3. **Testing Isolation**: Phase 1 changes tested in isolated environment before integration
4. **Dependency Management**: Clear dependency ordering prevents task conflicts

---

## Implementation Guide

### Code Changes Summary

**New Files Created**:
- `keisei/evaluation/core/evaluation_context.py` - Runtime dependency injection
- `keisei/evaluation/core/resource_manager.py` - Resource cleanup management
- `keisei/evaluation/core/elo_integration.py` - ELO system integration  
- `keisei/evaluation/strategies/custom.py` - Custom strategy implementation

**Major Modifications**:
- `keisei/evaluation/core/base_evaluator.py` - Add dependency injection support
- `keisei/evaluation/core_manager.py` - Safe async/sync coordination  
- `keisei/evaluation/strategies/tournament.py` - Complete in-memory implementation
- `keisei/evaluation/core/parallel_executor.py` - Protocol compliance fixes

**Configuration Updates**:
- `keisei/config_schema.py` - Add evaluation runtime context fields
- `keisei/evaluation/core/evaluation_config.py` - Unified configuration interface

### Interface Specifications

#### EvaluationRuntimeContext Interface
```python
@dataclass
class EvaluationRuntimeContext:
    """Shared runtime dependencies for evaluation system."""
    device: torch.device
    policy_mapper: PolicyOutputMapper
    event_loop: Optional[asyncio.AbstractEventLoop] = None
    resource_manager: Optional[EvaluationResourceManager] = None
    model_weight_manager: Optional[ModelWeightManager] = None
    elo_integration: Optional[ELOIntegrationManager] = None
    
    def validate(self) -> None:
        """Validate runtime context consistency."""
        if not self.policy_mapper:
            raise ValueError("PolicyOutputMapper is required")
        if not self.device:
            raise ValueError("Device specification is required")
```

#### Unified Configuration Protocol
```python
class EvaluationConfigProtocol(Protocol):
    """Protocol for all evaluation configuration objects."""
    strategy: EvaluationStrategy
    num_games: int
    max_concurrent_games: int
    
    def to_evaluation_context(self, runtime_context: EvaluationRuntimeContext) -> EvaluationContext:
        """Convert config to evaluation context with runtime dependencies."""
        ...
        
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        ...
```

### Migration Guide

#### For Existing Code Using EvaluationManager
```python
# OLD: Direct instantiation without dependencies
evaluation_manager = EvaluationManager(config, "run_name")
evaluation_manager.setup(device, policy_mapper, model_dir, wandb_active)

# NEW: Dependency injection pattern  
runtime_context = EvaluationRuntimeContext(
    device=torch.device(device),
    policy_mapper=policy_mapper,
    resource_manager=EvaluationResourceManager()
)
evaluation_manager = EvaluationManager(config, "run_name", runtime_context)
```

#### For Custom Evaluation Strategies
```python
# OLD: Create own policy mapper
class CustomEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.policy_mapper = PolicyOutputMapper()  # ❌ Creates local instance

# NEW: Use injected dependencies
class CustomEvaluator(BaseEvaluator):
    def __init__(self, config, runtime_context):
        super().__init__(config, runtime_context)
        self.policy_mapper = runtime_context.policy_mapper  # ✅ Uses shared instance
```

---

## Troubleshooting Guide

### Common Integration Issues

#### Issue: "RuntimeError: asyncio.run() cannot be called from a running event loop"
**Cause**: Using synchronous evaluation wrapper within async training context  
**Solution**: Use `evaluate_current_agent_async()` instead of `evaluate_current_agent()`

#### Issue: "Action space mismatch during evaluation"  
**Cause**: Policy mapper not shared between training and evaluation  
**Solution**: Ensure `EvaluationRuntimeContext` receives same `PolicyOutputMapper` instance from training

#### Issue: "CUDA out of memory during evaluation"
**Cause**: Model weights not properly moved between devices  
**Solution**: Use `EvaluationResourceManager` for proper device management and cleanup

#### Issue: "Configuration validation failed"  
**Cause**: Incompatible configuration objects passed between components  
**Solution**: Use unified configuration factory with proper validation

### Performance Diagnostics

#### Memory Usage Monitoring
```python
# Add to evaluation context for debugging
runtime_context.enable_memory_monitoring = True
runtime_context.memory_check_interval = 10  # games
```

#### Evaluation Speed Analysis  
```python  
# Enable detailed timing in evaluation config
config.strategy_params["enable_performance_timing"] = True
config.strategy_params["log_game_durations"] = True
```

---

## Success Criteria and Validation

### Acceptance Criteria

1. **✅ Policy Mapper Consistency**: Training and evaluation use identical action space mappings
2. **✅ Configuration Compatibility**: All config objects serialize/deserialize without errors  
3. **✅ Async Safety**: No event loop conflicts in any usage context
4. **✅ Complete Implementation**: All evaluation strategies functional including in-memory modes
5. **✅ Resource Management**: No memory leaks, proper cleanup on failures
6. **✅ Performance Goals**: In-memory evaluation ≥50% faster than file-based evaluation

### Validation Testing

```python
# Integration validation test suite
class TestEvaluationSystemRemediation:
    def test_end_to_end_evaluation_workflow(self):
        """Test complete evaluation workflow from training context."""
        # Train model briefly
        trainer = Trainer(config)
        trainer.train_steps(100)
        
        # Extract runtime context from trainer  
        runtime_context = trainer.get_evaluation_runtime_context()
        
        # Evaluate using same context
        eval_manager = EvaluationManager(eval_config, "test", runtime_context)
        result = eval_manager.evaluate_current_agent(trainer.agent)
        
        # Validation
        assert result.summary_stats.total_games > 0
        assert len(result.errors) == 0
        assert result.context.agent_info.name == trainer.agent.name
        
    def test_configuration_serialization_roundtrip(self):
        """Test configuration objects maintain consistency through serialization."""
        configs = [
            TournamentConfig(num_games=50, opponent_pool_config=[]),
            SingleOpponentConfig(opponent_name="test"),
            LadderConfig(elo_config={"initial_rating": 1500})
        ]
        
        for original_config in configs:
            # Serialize and deserialize
            config_dict = original_config.to_dict()
            restored_config = EvaluationConfig.from_dict(config_dict)
            
            # Should maintain essential properties
            assert restored_config.strategy == original_config.strategy
            assert restored_config.num_games == original_config.num_games
```

---

## Conclusion

The Keisei evaluation system requires systematic integration remediation to address fundamental coordination failures between well-designed but poorly-integrated components. The three-phase approach addresses critical blockers first, then completes missing implementations, and finally polishes the integration layer.

**Key Success Factors**:
1. **Dependency Injection**: Shared runtime context eliminates component isolation issues
2. **Unified Configuration**: Consistent interfaces prevent validation and compatibility failures  
3. **Async-Safe Design**: Proper event loop coordination prevents runtime conflicts
4. **Complete Implementation**: No placeholder implementations remain
5. **Resource Management**: Proper cleanup prevents memory leaks and resource exhaustion

**Timeline**: 22 days with proper risk mitigation buffers  
**Risk Level**: High for Phase 1 (breaking changes), Medium-Low for subsequent phases  
**Success Probability**: High given systematic approach and clear interface specifications

The remediation transforms the evaluation system from a collection of well-tested but disconnected components into a properly integrated subsystem capable of production operation within the Keisei training pipeline.