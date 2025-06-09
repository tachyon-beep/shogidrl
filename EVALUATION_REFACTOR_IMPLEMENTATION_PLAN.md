# Keisei Shogi DRL Evaluation System Refactor - Implementation Plan

## Executive Summary

This document provides a detailed implementation plan for refactoring the Keisei Shogi DRL evaluation system. The refactor aims to create a more modular, maintainable, and comprehensive evaluation pipeline that supports multiple evaluation strategies, better integration with training, and enhanced analytics.

## Current Architecture Analysis

### Key Components Identified

1. **Core Evaluation Files:**
   - `keisei/evaluation/evaluate.py` - Main Evaluator class and execute_full_evaluation_run function
   - `keisei/evaluation/loop.py` - Core evaluation loop logic
   - `keisei/evaluation/elo_registry.py` - Elo rating persistence and calculations

2. **Training Integration:**
   - `keisei/training/trainer.py` - Main trainer class with evaluation integration
   - `keisei/training/callbacks.py` - EvaluationCallback for periodic evaluation
   - `keisei/training/previous_model_selector.py` - Model pool management

3. **Configuration:**
   - `keisei/config_schema.py` - EvaluationConfig schema

### Current Strengths
- Functional Evaluator class with W&B integration
- Basic Elo rating system
- Integration with training callbacks
- Policy output mapping support

### Current Limitations
- Monolithic evaluation logic in single files
- Limited evaluation strategies (only single opponent)
- No comprehensive analytics or reporting
- Minimal validation and error handling
- Hard-coded evaluation parameters
- Limited extensibility for new evaluation types

## Implementation Plan

### Phase 1: Core Architecture Refactor (Days 1-3)

#### 1.1 Create New Directory Structure

```
keisei/evaluation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base_evaluator.py       # Abstract base evaluator
│   ├── evaluation_context.py   # Evaluation context and metadata
│   ├── evaluation_result.py    # Result data structures
│   └── evaluation_config.py    # Evaluation-specific configuration
├── strategies/
│   ├── __init__.py
│   ├── single_opponent.py      # Current 1v1 evaluation
│   ├── tournament.py           # Round-robin tournament
│   ├── ladder.py              # ELO ladder system
│   └── benchmark.py           # Fixed opponent benchmarking
├── opponents/
│   ├── __init__.py
│   ├── opponent_manager.py     # Opponent loading and management
│   ├── opponent_pool.py        # Pool of opponents for evaluation
│   └── adaptive_opponent.py   # Difficulty-adaptive opponents
├── analytics/
│   ├── __init__.py
│   ├── performance_analyzer.py # Win rate, game length analysis
│   ├── elo_tracker.py         # Enhanced Elo tracking
│   ├── trend_analyzer.py      # Performance trend analysis
│   └── report_generator.py    # Comprehensive reporting
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Input validation utilities
│   ├── logging.py            # Evaluation-specific logging
│   └── metrics.py            # Evaluation metrics calculation
└── legacy/
    ├── __init__.py
    ├── evaluate.py           # Current evaluate.py (renamed)
    ├── loop.py              # Current loop.py (renamed)
    └── elo_registry.py      # Current elo_registry.py (renamed)
```

#### 1.2 Implementation Tasks

**Task 1.1.1: Create Core Infrastructure**
```python
# File: keisei/evaluation/core/evaluation_context.py
@dataclass
class EvaluationContext:
    """Context information for an evaluation session."""
    session_id: str
    timestamp: datetime
    agent_info: AgentInfo
    configuration: EvaluationConfig
    environment_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

# File: keisei/evaluation/core/evaluation_result.py
@dataclass
class GameResult:
    """Result of a single game."""
    game_id: str
    winner: Optional[int]  # 0=agent, 1=opponent, None=draw
    moves_count: int
    duration_seconds: float
    agent_info: AgentInfo
    opponent_info: OpponentInfo
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    context: EvaluationContext
    games: List[GameResult]
    summary_stats: SummaryStats
    analytics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
```

**Task 1.1.2: Create Base Evaluator Abstract Class**
```python
# File: keisei/evaluation/core/base_evaluator.py
class BaseEvaluator(ABC):
    """Abstract base class for all evaluation strategies."""
    
    def __init__(self, config: EvaluationConfig, context: EvaluationContext):
        self.config = config
        self.context = context
        self.logger = EvaluationLogger(context.session_id)
    
    @abstractmethod
    def evaluate(self, agent: PPOAgent) -> EvaluationResult:
        """Run evaluation and return comprehensive results."""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> List[str]:
        """Validate evaluator configuration. Return list of errors."""
        pass
    
    def setup(self) -> None:
        """Setup evaluator resources."""
        pass
    
    def teardown(self) -> None:
        """Cleanup evaluator resources."""
        pass
```

#### 1.3 Enhanced Configuration Schema

**Task 1.1.3: Extend Configuration Schema**
```python
# File: keisei/evaluation/core/evaluation_config.py
class EvaluationStrategy(str, Enum):
    SINGLE_OPPONENT = "single_opponent"
    TOURNAMENT = "tournament"
    LADDER = "ladder"
    BENCHMARK = "benchmark"

class OpponentConfig(BaseModel):
    type: str  # "random", "heuristic", "ppo", "pool"
    checkpoint_path: Optional[str] = None
    pool_size: Optional[int] = None
    difficulty_level: Optional[float] = None

class AnalyticsConfig(BaseModel):
    enable_performance_analysis: bool = True
    enable_trend_analysis: bool = True
    enable_move_analysis: bool = False
    generate_detailed_report: bool = True
    save_game_logs: bool = False

class EvaluationConfig(BaseModel):
    # Core evaluation settings
    strategy: EvaluationStrategy = EvaluationStrategy.SINGLE_OPPONENT
    num_games: int = Field(20, gt=0)
    max_moves_per_game: int = Field(500, gt=0)
    timeout_per_game: Optional[float] = Field(None, gt=0)
    
    # Opponent configuration
    opponent: OpponentConfig
    
    # Analytics and reporting
    analytics: AnalyticsConfig = AnalyticsConfig()
    
    # Integration settings
    wandb_logging: bool = False
    elo_tracking: bool = True
    save_results: bool = True
    
    # Advanced settings
    parallel_games: bool = False
    max_parallel_workers: int = Field(1, gt=0)
    validation_strict: bool = True
```

### Phase 2: Strategy Implementation (Days 4-6)

#### 2.1 Single Opponent Strategy (Refactored)

**Task 2.1.1: Implement SingleOpponentEvaluator**
```python
# File: keisei/evaluation/strategies/single_opponent.py
class SingleOpponentEvaluator(BaseEvaluator):
    """Evaluates agent against a single opponent."""
    
    def __init__(self, config: EvaluationConfig, context: EvaluationContext):
        super().__init__(config, context)
        self.opponent_manager = OpponentManager(config.opponent)
    
    def validate_configuration(self) -> List[str]:
        errors = []
        if self.config.strategy != EvaluationStrategy.SINGLE_OPPONENT:
            errors.append("Strategy mismatch for SingleOpponentEvaluator")
        if not self.config.opponent.type:
            errors.append("Opponent type must be specified")
        return errors
    
    def evaluate(self, agent: PPOAgent) -> EvaluationResult:
        self.setup()
        try:
            opponent = self.opponent_manager.load_opponent()
            games = []
            
            for game_idx in range(self.config.num_games):
                game_result = self._play_single_game(
                    agent, opponent, game_idx
                )
                games.append(game_result)
                
                # Progress reporting
                self._report_progress(game_idx + 1, self.config.num_games)
            
            return self._compile_results(games)
        finally:
            self.teardown()
    
    def _play_single_game(self, agent: PPOAgent, opponent: BaseOpponent, 
                         game_idx: int) -> GameResult:
        """Play a single game and return detailed results."""
        # Enhanced game loop with timeout handling, error recovery
        # Move validation, and comprehensive logging
        pass
```

#### 2.2 Tournament Strategy

**Task 2.2.1: Implement TournamentEvaluator**
```python
# File: keisei/evaluation/strategies/tournament.py
class TournamentEvaluator(BaseEvaluator):
    """Round-robin tournament evaluation against multiple opponents."""
    
    def evaluate(self, agent: PPOAgent) -> EvaluationResult:
        opponents = self._load_tournament_opponents()
        results = []
        
        for opponent in opponents:
            opponent_results = self._evaluate_against_opponent(agent, opponent)
            results.extend(opponent_results)
        
        return self._compile_tournament_results(results)
    
    def _load_tournament_opponents(self) -> List[BaseOpponent]:
        """Load all opponents for tournament."""
        pass
    
    def _calculate_tournament_standings(self, results: List[GameResult]) -> Dict:
        """Calculate tournament standings and statistics."""
        pass
```

#### 2.3 Ladder Strategy

**Task 2.3.1: Implement LadderEvaluator**
```python
# File: keisei/evaluation/strategies/ladder.py
class LadderEvaluator(BaseEvaluator):
    """ELO ladder system with adaptive opponent selection."""
    
    def evaluate(self, agent: PPOAgent) -> EvaluationResult:
        ladder = self._load_elo_ladder()
        current_rating = ladder.get_agent_rating(agent)
        
        # Select opponents based on current rating
        opponents = self._select_ladder_opponents(current_rating)
        
        results = []
        for opponent in opponents:
            games = self._play_ladder_match(agent, opponent)
            results.extend(games)
            
            # Update ratings after each match
            ladder.update_ratings(agent, opponent, games)
        
        return self._compile_ladder_results(results, ladder)
```

### Phase 3: Analytics and Reporting (Days 7-9)

#### 3.1 Performance Analytics

**Task 3.1.1: Implement PerformanceAnalyzer**
```python
# File: keisei/evaluation/analytics/performance_analyzer.py
class PerformanceAnalyzer:
    """Analyzes agent performance across multiple dimensions."""
    
    def analyze(self, games: List[GameResult]) -> Dict[str, Any]:
        return {
            'win_rate_analysis': self._analyze_win_rates(games),
            'game_length_analysis': self._analyze_game_lengths(games),
            'opening_performance': self._analyze_opening_moves(games),
            'endgame_performance': self._analyze_endgame_moves(games),
            'time_performance': self._analyze_move_times(games),
            'consistency_metrics': self._analyze_consistency(games)
        }
    
    def _analyze_win_rates(self, games: List[GameResult]) -> Dict:
        """Detailed win rate analysis with confidence intervals."""
        wins = sum(1 for g in games if g.winner == 0)
        total = len(games)
        win_rate = wins / total if total > 0 else 0.0
        
        # Calculate confidence intervals
        confidence_interval = self._calculate_confidence_interval(wins, total)
        
        return {
            'win_rate': win_rate,
            'wins': wins,
            'losses': sum(1 for g in games if g.winner == 1),
            'draws': sum(1 for g in games if g.winner is None),
            'total_games': total,
            'confidence_interval_95': confidence_interval,
            'statistical_significance': self._assess_significance(wins, total)
        }
```

#### 3.2 Enhanced ELO Tracking

**Task 3.2.1: Implement Enhanced EloTracker**
```python
# File: keisei/evaluation/analytics/elo_tracker.py
class EloTracker:
    """Enhanced ELO tracking with historical data and analytics."""
    
    def __init__(self, registry_path: Path, config: EloConfig):
        self.registry_path = registry_path
        self.config = config
        self.history: List[EloUpdate] = []
    
    def update_rating(self, agent_id: str, opponent_id: str, 
                     results: List[GameResult]) -> EloUpdate:
        """Update ELO ratings and track changes."""
        old_agent_rating = self.get_rating(agent_id)
        old_opponent_rating = self.get_rating(opponent_id)
        
        # Calculate new ratings
        new_agent_rating, new_opponent_rating = self._calculate_new_ratings(
            old_agent_rating, old_opponent_rating, results
        )
        
        # Create update record
        update = EloUpdate(
            timestamp=datetime.now(),
            agent_id=agent_id,
            opponent_id=opponent_id,
            old_agent_rating=old_agent_rating,
            new_agent_rating=new_agent_rating,
            old_opponent_rating=old_opponent_rating,
            new_opponent_rating=new_opponent_rating,
            games_played=len(results),
            result_summary=self._summarize_results(results)
        )
        
        self.history.append(update)
        self._save_ratings()
        
        return update
    
    def get_rating_history(self, agent_id: str) -> List[EloUpdate]:
        """Get historical ELO changes for an agent."""
        return [u for u in self.history if u.agent_id == agent_id]
    
    def analyze_rating_trends(self) -> Dict[str, Any]:
        """Analyze ELO rating trends across all agents."""
        pass
```

#### 3.3 Comprehensive Reporting

**Task 3.3.1: Implement ReportGenerator**
```python
# File: keisei/evaluation/analytics/report_generator.py
class ReportGenerator:
    """Generates comprehensive evaluation reports."""
    
    def generate_report(self, result: EvaluationResult) -> EvaluationReport:
        """Generate a comprehensive evaluation report."""
        return EvaluationReport(
            executive_summary=self._generate_executive_summary(result),
            performance_analysis=self._generate_performance_section(result),
            game_analysis=self._generate_game_analysis(result),
            comparison_analysis=self._generate_comparison_section(result),
            recommendations=self._generate_recommendations(result),
            appendices=self._generate_appendices(result)
        )
    
    def export_report(self, report: EvaluationReport, 
                     format: str = "html") -> Path:
        """Export report to various formats (HTML, PDF, JSON)."""
        pass
    
    def _generate_executive_summary(self, result: EvaluationResult) -> str:
        """Generate executive summary with key findings."""
        template = """
        # Evaluation Summary for {agent_name}
        
        **Overall Performance:** {overall_rating}
        **Win Rate:** {win_rate:.1%} ({wins}/{total} games)
        **Average Game Length:** {avg_length:.1f} moves
        **ELO Rating Change:** {elo_change:+.1f} points
        
        ## Key Findings:
        {key_findings}
        
        ## Recommendations:
        {recommendations}
        """
        
        return template.format(
            agent_name=result.context.agent_info.name,
            overall_rating=self._calculate_overall_rating(result),
            win_rate=result.summary_stats.win_rate,
            wins=result.summary_stats.wins,
            total=result.summary_stats.total_games,
            avg_length=result.summary_stats.avg_game_length,
            elo_change=result.analytics.get('elo_change', 0),
            key_findings=self._extract_key_findings(result),
            recommendations=self._generate_recommendations_text(result)
        )
```

### Phase 4: Integration and Migration (Days 10-12)

#### 4.1 Trainer Integration

**Task 4.1.1: Update Trainer Class**
```python
# File: keisei/training/trainer.py (modifications)
class Trainer(CompatibilityMixin):
    def __init__(self, config: AppConfig, args: Any):
        # ...existing code...
        
        # Initialize new evaluation system
        self.evaluation_manager = EvaluationManager(
            config.evaluation, 
            self.run_name
        )
    
    def _setup_evaluation_components(self):
        """Setup new evaluation system components."""
        self.evaluation_manager.setup(
            device=self.device,
            policy_mapper=self.policy_output_mapper,
            model_dir=self.model_dir,
            wandb_active=self.is_train_wandb_active
        )
```

**Task 4.1.2: Update EvaluationCallback**
```python
# File: keisei/training/callbacks.py (modifications)
class EvaluationCallback(Callback):
    def __init__(self, eval_cfg, interval: int):
        self.eval_cfg = eval_cfg
        self.interval = interval
        self.evaluation_manager = None  # Set by trainer
    
    def on_step_end(self, trainer: "Trainer"):
        if not self.eval_cfg.enable_periodic_evaluation:
            return
            
        if (trainer.global_timestep + 1) % self.interval == 0:
            # Use new evaluation system
            result = trainer.evaluation_manager.evaluate_current_agent(
                trainer.agent
            )
            
            # Update trainer state with results
            trainer.evaluation_elo_snapshot = result.elo_snapshot
            
            # Log results
            if trainer.log_both:
                trainer.log_both(
                    f"Evaluation completed: {result.summary_stats}",
                    also_to_wandb=True,
                    wandb_data=result.to_wandb_dict()
                )
```

#### 4.2 Legacy Compatibility

**Task 4.2.1: Create Compatibility Layer**
```python
# File: keisei/evaluation/legacy/compatibility.py
def execute_full_evaluation_run(*args, **kwargs) -> Optional[ResultsDict]:
    """Legacy-compatible wrapper for new evaluation system."""
    # Convert legacy parameters to new configuration
    config = _convert_legacy_params(*args, **kwargs)
    
    # Create evaluation context
    context = EvaluationContext(
        session_id=f"legacy_{int(time.time())}",
        timestamp=datetime.now(),
        agent_info=_extract_agent_info(kwargs),
        configuration=config,
        environment_info={}
    )
    
    # Run evaluation with new system
    evaluator = SingleOpponentEvaluator(config, context)
    result = evaluator.evaluate(_load_agent_from_path(kwargs['agent_checkpoint_path']))
    
    # Convert result back to legacy format
    return _convert_to_legacy_result(result)

def _convert_legacy_params(*args, **kwargs) -> EvaluationConfig:
    """Convert legacy parameters to new configuration format."""
    pass

def _convert_to_legacy_result(result: EvaluationResult) -> ResultsDict:
    """Convert new result format to legacy ResultsDict."""
    pass
```

#### 4.3 Configuration Migration

**Task 4.3.1: Update Configuration Schema**
```python
# File: keisei/config_schema.py (modifications)
class EvaluationConfig(BaseModel):
    # Legacy fields (deprecated but supported)
    enable_periodic_evaluation: bool = Field(True, deprecated=True)
    evaluation_interval_timesteps: int = Field(50000, deprecated=True)
    num_games: int = Field(20, deprecated=True)
    opponent_type: str = Field("random", deprecated=True)
    
    # New configuration structure
    strategy: EvaluationStrategy = EvaluationStrategy.SINGLE_OPPONENT
    opponents: List[OpponentConfig] = Field(default_factory=lambda: [
        OpponentConfig(type="random")
    ])
    analytics: AnalyticsConfig = AnalyticsConfig()
    reporting: ReportingConfig = ReportingConfig()
    
    # Migration helper
    def migrate_from_legacy(self) -> 'EvaluationConfig':
        """Migrate legacy configuration to new format."""
        pass
```

### Phase 5: Testing and Validation (Days 13-15)

#### 5.1 Unit Tests

**Task 5.1.1: Core Component Tests**
```python
# File: tests/evaluation/test_core.py
class TestEvaluationContext:
    def test_context_creation(self):
        """Test evaluation context creation and validation."""
        pass
    
    def test_context_serialization(self):
        """Test context serialization/deserialization."""
        pass

class TestEvaluationResult:
    def test_result_aggregation(self):
        """Test game result aggregation."""
        pass
    
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        pass
```

**Task 5.1.2: Strategy Tests**
```python
# File: tests/evaluation/test_strategies.py
class TestSingleOpponentEvaluator:
    def test_evaluation_flow(self):
        """Test complete evaluation flow."""
        pass
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        pass
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        pass

class TestTournamentEvaluator:
    def test_tournament_execution(self):
        """Test tournament evaluation execution."""
        pass
    
    def test_standings_calculation(self):
        """Test tournament standings calculation."""
        pass
```

#### 5.2 Integration Tests

**Task 5.2.1: Training Integration Tests**
```python
# File: tests/evaluation/test_training_integration.py
class TestTrainerIntegration:
    def test_periodic_evaluation(self):
        """Test periodic evaluation during training."""
        pass
    
    def test_callback_integration(self):
        """Test evaluation callback integration."""
        pass
    
    def test_wandb_logging(self):
        """Test W&B logging integration."""
        pass
```

#### 5.3 Performance Tests

**Task 5.3.1: Performance Benchmarks**
```python
# File: tests/evaluation/test_performance.py
class TestEvaluationPerformance:
    def test_single_game_performance(self):
        """Benchmark single game evaluation time."""
        pass
    
    def test_batch_evaluation_performance(self):
        """Benchmark batch evaluation performance."""
        pass
    
    def test_memory_usage(self):
        """Test memory usage during evaluation."""
        pass
```

### Phase 6: Documentation and Examples (Days 16-18)

#### 6.1 API Documentation

**Task 6.1.1: Create Comprehensive Documentation**
```markdown
# File: docs/evaluation/API_REFERENCE.md
# Evaluation System API Reference

## Core Classes

### BaseEvaluator
Abstract base class for all evaluation strategies.

### EvaluationContext
Container for evaluation session metadata and configuration.

### EvaluationResult
Comprehensive results from an evaluation session.

## Strategies

### SingleOpponentEvaluator
Evaluates agent against a single opponent.

### TournamentEvaluator
Round-robin tournament evaluation.

### LadderEvaluator
ELO ladder system evaluation.

## Analytics

### PerformanceAnalyzer
Analyzes agent performance metrics.

### EloTracker
Enhanced ELO rating tracking.

### ReportGenerator
Generates comprehensive evaluation reports.
```

#### 6.2 Usage Examples

**Task 6.2.1: Create Usage Examples**
```python
# File: examples/evaluation_examples.py
def example_single_opponent_evaluation():
    """Example: Single opponent evaluation."""
    config = EvaluationConfig(
        strategy=EvaluationStrategy.SINGLE_OPPONENT,
        num_games=50,
        opponent=OpponentConfig(type="random"),
        analytics=AnalyticsConfig(
            enable_performance_analysis=True,
            generate_detailed_report=True
        )
    )
    
    context = EvaluationContext(
        session_id="example_session",
        timestamp=datetime.now(),
        agent_info=AgentInfo(name="test_agent"),
        configuration=config,
        environment_info={}
    )
    
    evaluator = SingleOpponentEvaluator(config, context)
    result = evaluator.evaluate(agent)
    
    print(f"Win rate: {result.summary_stats.win_rate:.2%}")
    print(f"ELO change: {result.analytics['elo_change']:+.1f}")

def example_tournament_evaluation():
    """Example: Tournament evaluation."""
    pass

def example_custom_analytics():
    """Example: Custom analytics and reporting."""
    pass
```

#### 6.3 Migration Guide

**Task 6.3.1: Create Migration Guide**
```markdown
# File: docs/evaluation/MIGRATION_GUIDE.md
# Migration Guide: Legacy to New Evaluation System

## Overview
This guide helps migrate from the legacy evaluation system to the new modular system.

## Key Changes
1. **Modular Architecture**: Evaluation logic split into strategies, analytics, and reporting
2. **Enhanced Configuration**: More flexible and comprehensive configuration options
3. **Comprehensive Results**: Detailed analytics and reporting capabilities
4. **Better Integration**: Improved integration with training and W&B logging

## Migration Steps

### Step 1: Update Configuration
Old format:
```yaml
evaluation:
  enable_periodic_evaluation: true
  num_games: 20
  opponent_type: "random"
```

New format:
```yaml
evaluation:
  strategy: "single_opponent"
  opponents:
    - type: "random"
  analytics:
    enable_performance_analysis: true
    generate_detailed_report: true
```

### Step 2: Update Code
Old code:
```python
result = execute_full_evaluation_run(...)
```

New code:
```python
evaluator = EvaluationManager.from_config(config)
result = evaluator.evaluate(agent)
```

## Backward Compatibility
The legacy `execute_full_evaluation_run` function is still supported but deprecated.
```

## Risk Assessment and Mitigation

### High Risk Areas
1. **Training Integration**: Changes to evaluation callbacks could disrupt training
2. **Configuration Breaking Changes**: New configuration format may break existing configs
3. **Performance Regression**: New system might be slower than current implementation

### Mitigation Strategies
1. **Gradual Migration**: Implement alongside legacy system with feature flags
2. **Extensive Testing**: Comprehensive test suite covering all integration points
3. **Performance Monitoring**: Benchmark new system against legacy implementation
4. **Rollback Plan**: Ability to quickly revert to legacy system if issues arise

## Success Metrics

### Technical Metrics
- [ ] All legacy tests pass with compatibility layer
- [ ] New system performance within 10% of legacy system
- [ ] 100% test coverage for new components
- [ ] Zero regressions in training integration

### Quality Metrics
- [ ] Comprehensive documentation and examples
- [ ] Clear migration path for existing users
- [ ] Enhanced analytics provide actionable insights
- [ ] Improved maintainability and extensibility

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1 | 3 days | Core architecture and base classes |
| 2 | 3 days | Strategy implementations |
| 3 | 3 days | Analytics and reporting |
| 4 | 3 days | Integration and migration |
| 5 | 3 days | Testing and validation |
| 6 | 3 days | Documentation and examples |

**Total Duration**: 18 days

## Next Steps

1. **Review and Approve Plan**: Stakeholder review of implementation plan
2. **Environment Setup**: Prepare development environment and dependencies
3. **Phase 1 Kickoff**: Begin core architecture implementation
4. **Regular Check-ins**: Daily progress reviews and issue resolution
5. **Testing Integration**: Continuous testing throughout development
6. **Documentation Review**: Regular documentation updates and reviews

## Conclusion

This implementation plan provides a comprehensive approach to refactoring the Keisei Shogi DRL evaluation system. The modular architecture will improve maintainability, extensibility, and provide enhanced analytics capabilities while maintaining backward compatibility with existing training workflows.

The phased approach ensures minimal disruption to current development while providing clear milestones and deliverables. With proper execution, this refactor will significantly enhance the evaluation capabilities of the Keisei system and provide a solid foundation for future enhancements.
