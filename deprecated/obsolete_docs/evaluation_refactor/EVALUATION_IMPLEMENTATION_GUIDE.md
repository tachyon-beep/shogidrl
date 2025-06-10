# Evaluation System Implementation Guide - Remaining Tasks

**Date:** June 10, 2025  
**Author:** AI Assistant  
**Purpose:** Detailed technical guide for completing the evaluation system refactor

## Overview

This document provides specific implementation details, code examples, and step-by-step instructions for completing the remaining phases of the evaluation system refactor. It serves as a technical blueprint for continuing the work.

## Current Architecture Summary

### Working Components
```python
# Core architecture successfully implemented
keisei/evaluation/
├── core/
│   ├── base_evaluator.py      # ✅ Abstract base class with async interface
│   ├── evaluation_config.py   # ✅ Configuration management
│   ├── evaluation_context.py  # ✅ Context and metadata structures  
│   └── evaluation_result.py   # ✅ Result data structures
├── strategies/
│   ├── single_opponent.py     # ✅ Basic 1v1 evaluation
│   ├── tournament.py          # ✅ Round-robin tournaments
│   ├── ladder.py              # ✅ ELO ladder system
│   └── benchmark.py           # ✅ Fixed opponent benchmarking
├── opponents/
│   └── opponent_pool.py       # ✅ Agent pool management with ELO
├── analytics/
│   ├── performance_analyzer.py # ✅ Advanced performance metrics
│   ├── elo_tracker.py         # ✅ ELO rating management
│   └── report_generator.py    # ✅ Multi-format reporting
└── manager.py                 # ✅ Main orchestrator
```

### Integration Points
```python
# In trainer.py - Successfully integrated
self.evaluation_manager = EvaluationManager(
    new_eval_cfg, self.run_name,
    pool_size=config.evaluation.previous_model_pool_size,
    elo_registry_path=config.evaluation.elo_registry_path,
)

# In callbacks.py - Using new system
eval_results = trainer.evaluation_manager.evaluate_current_agent(trainer.agent)
opponent_ckpt = trainer.evaluation_manager.opponent_pool.sample()
```

## Phase 6: Performance Optimization Implementation

### 6.1 In-Memory Model Communication

**Current Problem:** System still saves models to disk and reloads them for evaluation, causing I/O overhead.

**Solution:** Pass model weights directly in memory between training and evaluation.

#### Step 1: Create Model Weight Management System

```python
# File: keisei/evaluation/core/model_manager.py
from typing import Dict, Optional, Any
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelWeightManager:
    """Manages model weights for in-memory evaluation."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._weight_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.max_cache_size = 5  # Configurable
    
    def extract_agent_weights(self, agent) -> Dict[str, torch.Tensor]:
        """Extract and clone current agent weights for evaluation."""
        if not hasattr(agent, 'model') or agent.model is None:
            raise ValueError("Agent must have a model attribute")
        
        # Clone weights to avoid interference with training
        weights = {}
        for name, param in agent.model.state_dict().items():
            weights[name] = param.clone().detach().cpu()
        
        return weights
    
    def cache_opponent_weights(self, opponent_id: str, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
        """Load and cache opponent weights from checkpoint."""
        if opponent_id in self._weight_cache:
            return self._weight_cache[opponent_id]
        
        if len(self._weight_cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self._weight_cache))
            del self._weight_cache[oldest_key]
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                weights = checkpoint['model_state_dict']
            else:
                weights = checkpoint
            
            self._weight_cache[opponent_id] = weights
            logger.debug(f"Cached weights for opponent {opponent_id}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to load weights from {checkpoint_path}: {e}")
            raise
    
    def create_agent_from_weights(
        self, 
        weights: Dict[str, torch.Tensor], 
        agent_class, 
        config: Any
    ):
        """Create an agent instance from weights."""
        # This needs to be implemented based on your PPOAgent structure
        agent = agent_class(config=config, device=self.device)
        agent.model.load_state_dict(weights)
        agent.model.eval()
        return agent
    
    def clear_cache(self):
        """Clear the weight cache."""
        self._weight_cache.clear()
```

#### Step 2: Update EvaluationManager for In-Memory Evaluation

```python
# File: keisei/evaluation/manager.py - Add these methods
from .core.model_manager import ModelWeightManager

class EvaluationManager:
    def __init__(self, config: EvaluationConfig, run_name: str, **kwargs):
        # ...existing init...
        self.model_weight_manager = ModelWeightManager()
        self.enable_in_memory_eval = getattr(config, 'enable_in_memory_evaluation', True)
    
    async def evaluate_current_agent_in_memory(
        self, 
        agent, 
        opponent_checkpoint: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate agent using in-memory weights without file I/O."""
        if not self.enable_in_memory_eval:
            return await self.evaluate_current_agent(agent)  # Fallback
        
        try:
            # Extract current agent weights
            agent_weights = self.model_weight_manager.extract_agent_weights(agent)
            agent_info = AgentInfo(
                name=getattr(agent, 'name', 'current_agent'),
                model_type=type(agent).__name__,
                training_timesteps=getattr(agent, 'training_timesteps', None)
            )
            
            # Get opponent
            if opponent_checkpoint:
                opponent_path = Path(opponent_checkpoint)
                opponent_weights = self.model_weight_manager.cache_opponent_weights(
                    opponent_path.stem, opponent_path
                )
                opponent_info = OpponentInfo(
                    name=opponent_path.stem,
                    type="ppo",
                    checkpoint_path=str(opponent_path)
                )
            else:
                # Sample from pool
                opponent_checkpoint = self.opponent_pool.sample()
                if not opponent_checkpoint:
                    raise ValueError("No opponents available in pool")
                
                opponent_weights = self.model_weight_manager.cache_opponent_weights(
                    opponent_checkpoint.stem, opponent_checkpoint
                )
                opponent_info = OpponentInfo(
                    name=opponent_checkpoint.stem,
                    type="ppo", 
                    checkpoint_path=str(opponent_checkpoint)
                )
            
            # Create evaluation context
            context = EvaluationContext(
                session_id=f"inmem_{int(time.time())}",
                timestamp=datetime.now(),
                agent_info=agent_info,
                configuration=self.config,
                environment_info={"evaluation_mode": "in_memory"}
            )
            
            # Run in-memory evaluation
            return await self._run_in_memory_evaluation(
                agent_weights, opponent_weights, agent_info, opponent_info, context
            )
            
        except Exception as e:
            logger.error(f"In-memory evaluation failed: {e}")
            # Fallback to file-based evaluation
            return await self.evaluate_current_agent(agent)
    
    async def _run_in_memory_evaluation(
        self,
        agent_weights: Dict[str, torch.Tensor],
        opponent_weights: Dict[str, torch.Tensor],
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext
    ) -> EvaluationResult:
        """Execute evaluation using in-memory model weights."""
        from .strategies.single_opponent import SingleOpponentEvaluator
        
        # Create evaluator with in-memory configuration
        evaluator = SingleOpponentEvaluator(self.config)
        
        # Set up in-memory agents
        evaluator.agent_weights = agent_weights
        evaluator.opponent_weights = opponent_weights
        evaluator.agent_info = agent_info
        evaluator.opponent_info = opponent_info
        
        # Run evaluation
        result = await evaluator.evaluate_in_memory(context)
        return result
```

#### Step 3: Update SingleOpponentEvaluator for In-Memory Operation

```python
# File: keisei/evaluation/strategies/single_opponent.py - Add these methods

class SingleOpponentEvaluator(BaseEvaluator):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.agent_weights: Optional[Dict[str, torch.Tensor]] = None
        self.opponent_weights: Optional[Dict[str, torch.Tensor]] = None
        self.agent_info: Optional[AgentInfo] = None
        self.opponent_info: Optional[OpponentInfo] = None
    
    async def evaluate_in_memory(self, context: EvaluationContext) -> EvaluationResult:
        """Run evaluation using pre-loaded in-memory weights."""
        if not all([self.agent_weights, self.opponent_weights, self.agent_info, self.opponent_info]):
            raise ValueError("In-memory evaluation requires pre-loaded weights and info")
        
        games = []
        
        for game_idx in range(self.config.num_games):
            try:
                game_result = await self._play_single_game_in_memory(
                    game_idx, context
                )
                games.append(game_result)
                
            except Exception as e:
                logger.error(f"Game {game_idx} failed: {e}")
                # Could add error game result or continue
        
        # Calculate summary statistics
        summary_stats = SummaryStats.from_games(games)
        
        # Create result
        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats
        )
        
        return result
    
    async def _play_single_game_in_memory(
        self, 
        game_idx: int, 
        context: EvaluationContext
    ) -> GameResult:
        """Play a single game using in-memory agents."""
        from keisei.shogi.shogi_game import ShogiGame
        from keisei.core.ppo_agent import PPOAgent
        
        # Create temporary agents from weights
        agent = self._create_temp_agent_from_weights(self.agent_weights)
        opponent = self._create_temp_agent_from_weights(self.opponent_weights)
        
        # Create game
        game = ShogiGame()
        
        # Play game (reuse existing game loop logic)
        start_time = time.time()
        moves_count = 0
        
        while not game.game_over and moves_count < self.config.max_moves_per_game:
            current_player = agent if game.current_player == Color.BLACK else opponent
            
            try:
                # Get observation and action
                obs = game.get_observation()
                action = current_player.get_action(obs, legal_moves=game.get_legal_moves())
                
                # Make move
                if action < game.action_space_size:
                    move = self.policy_output_mapper.action_to_move(action, game)
                    if move and game.is_legal_move(*move):
                        game.make_move(move)
                        moves_count += 1
                    else:
                        # Illegal move - end game
                        break
                else:
                    # Invalid action
                    break
                    
            except Exception as e:
                logger.error(f"Error during game {game_idx}: {e}")
                break
        
        duration = time.time() - start_time
        
        # Determine winner
        winner = None
        if game.game_over:
            if game.winner == Color.BLACK:
                winner = 0  # Agent win
            elif game.winner == Color.WHITE:
                winner = 1  # Opponent win
            # else: draw (winner stays None)
        
        return GameResult(
            game_id=f"{context.session_id}_game_{game_idx}",
            winner=winner,
            moves_count=moves_count,
            duration_seconds=duration,
            agent_info=self.agent_info,
            opponent_info=self.opponent_info,
            metadata={"evaluation_mode": "in_memory"}
        )
    
    def _create_temp_agent_from_weights(self, weights: Dict[str, torch.Tensor]):
        """Create temporary agent from weights for evaluation."""
        # This is a simplified version - you'll need to adapt based on your PPOAgent structure
        from keisei.core.ppo_agent import PPOAgent
        from keisei.config_schema import make_test_config
        
        config = make_test_config()  # Or appropriate config
        temp_agent = PPOAgent(config=config, device=torch.device('cpu'))
        temp_agent.model.load_state_dict(weights)
        temp_agent.model.eval()
        
        return temp_agent
```

#### Step 4: Update Configuration for In-Memory Evaluation

```python
# File: keisei/evaluation/core/evaluation_config.py - Add configuration options

@dataclass
class EvaluationConfig:
    # ...existing fields...
    
    # In-memory evaluation settings
    enable_in_memory_evaluation: bool = True
    model_weight_cache_size: int = 5
    temp_agent_device: str = "cpu"  # Device for temporary evaluation agents
    
    # Memory management
    max_memory_usage_mb: Optional[int] = None  # Optional memory limit
    clear_cache_after_evaluation: bool = True
```

#### Step 5: Update EvaluationCallback to Use In-Memory Evaluation

```python
# File: keisei/training/callbacks.py - Update EvaluationCallback

class EvaluationCallback(Callback):
    def on_step_end(self, trainer: "Trainer"):
        if not getattr(self.eval_cfg, "enable_periodic_evaluation", False):
            return
        if (trainer.global_timestep + 1) % self.interval == 0:
            # ...existing checks...
            
            current_model.eval()  # Set the agent's model to eval mode
            
            # Use in-memory evaluation if available
            if hasattr(trainer.evaluation_manager, 'evaluate_current_agent_in_memory'):
                eval_results = trainer.evaluation_manager.evaluate_current_agent_in_memory(
                    trainer.agent,
                    opponent_checkpoint=str(opponent_ckpt) if opponent_ckpt else None
                )
            else:
                # Fallback to existing method
                eval_results = trainer.evaluation_manager.evaluate_current_agent(trainer.agent)
            
            current_model.train()  # Set model back to train mode
            
            # ...rest of callback unchanged...
```

### 6.2 Parallel Game Execution

**Current Problem:** All games run sequentially, underutilizing multi-core systems.

**Solution:** Implement multiprocessing to run games in parallel.

#### Step 1: Create Parallel Execution Framework

```python
# File: keisei/evaluation/core/parallel_executor.py
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import logging
import time
import pickle

logger = logging.getLogger(__name__)

class GameExecutionTask:
    """Represents a single game execution task for multiprocessing."""
    
    def __init__(
        self,
        game_id: str,
        agent_weights: Dict[str, Any],
        opponent_weights: Dict[str, Any],
        config: Dict[str, Any],
        random_seed: Optional[int] = None
    ):
        self.game_id = game_id
        self.agent_weights = agent_weights
        self.opponent_weights = opponent_weights
        self.config = config
        self.random_seed = random_seed

def execute_single_game_worker(task: GameExecutionTask) -> Dict[str, Any]:
    """
    Worker function to execute a single game in a separate process.
    This function must be pickleable and self-contained.
    """
    import torch
    import random
    import numpy as np
    from keisei.shogi.shogi_game import ShogiGame
    from keisei.shogi.shogi_core_definitions import Color
    from keisei.core.ppo_agent import PPOAgent
    from keisei.config_schema import make_test_config
    
    # Set random seed if provided
    if task.random_seed is not None:
        random.seed(task.random_seed)
        np.random.seed(task.random_seed)
        torch.manual_seed(task.random_seed)
    
    try:
        # Create agents from weights
        config = make_test_config()  # You may need to pass actual config
        
        agent = PPOAgent(config=config, device=torch.device('cpu'))
        agent.model.load_state_dict(task.agent_weights)
        agent.model.eval()
        
        opponent = PPOAgent(config=config, device=torch.device('cpu'))
        opponent.model.load_state_dict(task.opponent_weights)
        opponent.model.eval()
        
        # Create game
        game = ShogiGame()
        
        # Play game
        start_time = time.time()
        moves_count = 0
        max_moves = task.config.get('max_moves_per_game', 500)
        
        while not game.game_over and moves_count < max_moves:
            current_player = agent if game.current_player == Color.BLACK else opponent
            
            try:
                # Get observation and legal moves
                obs = game.get_observation()
                legal_moves = game.get_legal_moves()
                
                # Get action from agent
                with torch.no_grad():
                    action = current_player.get_action(obs, legal_moves=legal_moves)
                
                # Convert action to move and execute
                # You'll need to implement this based on your PolicyOutputMapper
                move = convert_action_to_move(action, game)  # Placeholder
                
                if move and game.is_legal_move(*move):
                    game.make_move(move)
                    moves_count += 1
                else:
                    # Invalid move - terminate game
                    break
                    
            except Exception as e:
                logger.error(f"Error in game {task.game_id}: {e}")
                break
        
        duration = time.time() - start_time
        
        # Determine winner
        winner = None
        if game.game_over:
            if game.winner == Color.BLACK:
                winner = 0  # Agent (Black) wins
            elif game.winner == Color.WHITE:
                winner = 1  # Opponent (White) wins
            # else: draw
        
        return {
            'game_id': task.game_id,
            'winner': winner,
            'moves_count': moves_count,
            'duration_seconds': duration,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'game_id': task.game_id,
            'winner': None,
            'moves_count': 0,
            'duration_seconds': 0.0,
            'success': False,
            'error': str(e)
        }

def convert_action_to_move(action: int, game) -> Optional[tuple]:
    """Convert neural network action to game move. Implement based on your system."""
    # This is a placeholder - implement based on your PolicyOutputMapper
    # return policy_mapper.action_to_move(action, game)
    pass

class ParallelGameExecutor:
    """Executes multiple games in parallel using multiprocessing."""
    
    def __init__(
        self, 
        max_workers: Optional[int] = None,
        timeout_per_game: float = 300.0
    ):
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.timeout_per_game = timeout_per_game
        self.executor: Optional[ProcessPoolExecutor] = None
    
    async def execute_games_parallel(
        self,
        agent_weights: Dict[str, Any],
        opponent_weights: Dict[str, Any],
        num_games: int,
        config: Dict[str, Any],
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext
    ) -> List[GameResult]:
        """Execute multiple games in parallel and return results."""
        
        # Create tasks
        tasks = []
        for i in range(num_games):
            task = GameExecutionTask(
                game_id=f"{context.session_id}_game_{i}",
                agent_weights=agent_weights,
                opponent_weights=opponent_weights,
                config=config,
                random_seed=hash(f"{context.session_id}_{i}") % (2**32)
            )
            tasks.append(task)
        
        # Execute in parallel
        results = await self._execute_tasks_parallel(tasks)
        
        # Convert to GameResult objects
        game_results = []
        for result in results:
            if result['success']:
                game_result = GameResult(
                    game_id=result['game_id'],
                    winner=result['winner'],
                    moves_count=result['moves_count'],
                    duration_seconds=result['duration_seconds'],
                    agent_info=agent_info,
                    opponent_info=opponent_info,
                    metadata={'parallel_execution': True}
                )
                game_results.append(game_result)
            else:
                logger.error(f"Game {result['game_id']} failed: {result['error']}")
                # Optionally add a failed game result
        
        return game_results
    
    async def _execute_tasks_parallel(self, tasks: List[GameExecutionTask]) -> List[Dict[str, Any]]:
        """Execute game tasks in parallel using ProcessPoolExecutor."""
        
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                loop.run_in_executor(executor, execute_single_game_worker, task): task
                for task in tasks
            }
            
            results = []
            completed = 0
            total = len(tasks)
            
            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=self.timeout_per_game * total):
                try:
                    result = await future
                    results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:  # Log progress
                        logger.info(f"Completed {completed}/{total} games")
                        
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Game {task.game_id} failed with exception: {e}")
                    results.append({
                        'game_id': task.game_id,
                        'success': False,
                        'error': str(e),
                        'winner': None,
                        'moves_count': 0,
                        'duration_seconds': 0.0
                    })
            
            return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
```

#### Step 2: Update SingleOpponentEvaluator for Parallel Execution

```python
# File: keisei/evaluation/strategies/single_opponent.py - Add parallel methods

class SingleOpponentEvaluator(BaseEvaluator):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.parallel_executor = None
        if getattr(config, 'enable_parallel_execution', False) and config.max_concurrent_games > 1:
            from ..core.parallel_executor import ParallelGameExecutor
            self.parallel_executor = ParallelGameExecutor(
                max_workers=config.max_concurrent_games,
                timeout_per_game=getattr(config, 'timeout_per_game', 300.0)
            )
    
    async def evaluate_in_memory(self, context: EvaluationContext) -> EvaluationResult:
        """Run evaluation with optional parallel execution."""
        if not all([self.agent_weights, self.opponent_weights, self.agent_info, self.opponent_info]):
            raise ValueError("In-memory evaluation requires pre-loaded weights and info")
        
        # Use parallel execution if available and beneficial
        if (self.parallel_executor and 
            self.config.num_games >= self.config.max_concurrent_games and
            self.config.enable_parallel_execution):
            
            games = await self._evaluate_parallel(context)
        else:
            games = await self._evaluate_sequential(context)
        
        # Calculate summary statistics
        summary_stats = SummaryStats.from_games(games)
        
        # Create result
        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats
        )
        
        return result
    
    async def _evaluate_parallel(self, context: EvaluationContext) -> List[GameResult]:
        """Run evaluation using parallel game execution."""
        config_dict = {
            'max_moves_per_game': getattr(self.config, 'max_moves_per_game', 500),
            'strategy': self.config.strategy.value if hasattr(self.config.strategy, 'value') else str(self.config.strategy)
        }
        
        return await self.parallel_executor.execute_games_parallel(
            agent_weights=self.agent_weights,
            opponent_weights=self.opponent_weights,
            num_games=self.config.num_games,
            config=config_dict,
            agent_info=self.agent_info,
            opponent_info=self.opponent_info,
            context=context
        )
    
    async def _evaluate_sequential(self, context: EvaluationContext) -> List[GameResult]:
        """Run evaluation using sequential game execution (existing method)."""
        games = []
        
        for game_idx in range(self.config.num_games):
            try:
                game_result = await self._play_single_game_in_memory(
                    game_idx, context
                )
                games.append(game_result)
                
            except Exception as e:
                logger.error(f"Game {game_idx} failed: {e}")
        
        return games
```

#### Step 3: Update Configuration for Parallel Execution

```python
# File: keisei/evaluation/core/evaluation_config.py - Add parallel execution options

@dataclass
class EvaluationConfig:
    # ...existing fields...
    
    # Parallel execution settings
    enable_parallel_execution: bool = True
    max_concurrent_games: int = 4
    timeout_per_game: Optional[float] = 300.0  # 5 minutes per game
    
    # Process management
    process_restart_threshold: int = 100  # Restart workers after N games
    max_memory_per_worker_mb: Optional[int] = 512
    
    # Performance tuning
    batch_size_per_worker: int = 1  # Games per worker process
    worker_initialization_timeout: float = 30.0
```

## Testing Implementation

### Performance Benchmarking Tests

```python
# File: tests/evaluation/test_performance_optimization.py
import time
import pytest
import asyncio
from unittest.mock import MagicMock

from keisei.evaluation.core.evaluation_config import EvaluationConfig, EvaluationStrategy
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator
from keisei.evaluation.core.evaluation_context import EvaluationContext, AgentInfo, OpponentInfo

class TestInMemoryEvaluation:
    """Test in-memory evaluation performance and correctness."""
    
    @pytest.mark.asyncio
    async def test_in_memory_vs_file_based_performance(self):
        """Test that in-memory evaluation is faster than file-based."""
        config = EvaluationConfig(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=10,
            enable_in_memory_evaluation=True,
            enable_parallel_execution=False  # Test sequential first
        )
        
        evaluator = SingleOpponentEvaluator(config)
        
        # Setup mock weights and info
        evaluator.agent_weights = self._create_mock_weights()
        evaluator.opponent_weights = self._create_mock_weights()
        evaluator.agent_info = AgentInfo(name="test_agent")
        evaluator.opponent_info = OpponentInfo(name="test_opponent", type="ppo")
        
        context = EvaluationContext(
            session_id="test_session",
            timestamp=datetime.now(),
            agent_info=evaluator.agent_info,
            configuration=config,
            environment_info={}
        )
        
        # Time in-memory evaluation
        start_time = time.time()
        result_in_memory = await evaluator.evaluate_in_memory(context)
        in_memory_duration = time.time() - start_time
        
        # Time file-based evaluation (would need to implement comparison)
        # This is conceptual - you'd need actual file-based evaluation for comparison
        
        # Verify results are valid
        assert result_in_memory.summary_stats.total_games == 10
        assert in_memory_duration < 60  # Should complete in reasonable time
        
        print(f"In-memory evaluation took {in_memory_duration:.2f} seconds")
    
    def _create_mock_weights(self) -> Dict[str, torch.Tensor]:
        """Create mock model weights for testing."""
        return {
            'layer1.weight': torch.randn(64, 32),
            'layer1.bias': torch.randn(64),
            'layer2.weight': torch.randn(32, 64),
            'layer2.bias': torch.randn(32)
        }

class TestParallelEvaluation:
    """Test parallel game execution performance and correctness."""
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self):
        """Test that parallel execution provides expected speedup."""
        # Sequential config
        config_sequential = EvaluationConfig(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=20,
            enable_parallel_execution=False,
            max_concurrent_games=1
        )
        
        # Parallel config
        config_parallel = EvaluationConfig(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=20,
            enable_parallel_execution=True,
            max_concurrent_games=4
        )
        
        # Test sequential
        evaluator_seq = SingleOpponentEvaluator(config_sequential)
        self._setup_evaluator(evaluator_seq)
        
        context = self._create_test_context(config_sequential)
        
        start_time = time.time()
        result_seq = await evaluator_seq.evaluate_in_memory(context)
        sequential_duration = time.time() - start_time
        
        # Test parallel
        evaluator_par = SingleOpponentEvaluator(config_parallel)
        self._setup_evaluator(evaluator_par)
        
        context = self._create_test_context(config_parallel)
        
        start_time = time.time()
        result_par = await evaluator_par.evaluate_in_memory(context)
        parallel_duration = time.time() - start_time
        
        # Verify results
        assert result_seq.summary_stats.total_games == 20
        assert result_par.summary_stats.total_games == 20
        
        # Parallel should be faster (allowing for overhead)
        speedup = sequential_duration / parallel_duration
        print(f"Speedup: {speedup:.2f}x (Sequential: {sequential_duration:.2f}s, Parallel: {parallel_duration:.2f}s)")
        
        # Should achieve at least 1.5x speedup with 4 workers
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"
    
    def _setup_evaluator(self, evaluator):
        """Setup evaluator with mock data."""
        evaluator.agent_weights = self._create_mock_weights()
        evaluator.opponent_weights = self._create_mock_weights() 
        evaluator.agent_info = AgentInfo(name="test_agent")
        evaluator.opponent_info = OpponentInfo(name="test_opponent", type="ppo")
    
    def _create_test_context(self, config):
        """Create test evaluation context."""
        return EvaluationContext(
            session_id=f"test_{int(time.time())}",
            timestamp=datetime.now(),
            agent_info=AgentInfo(name="test_agent"),
            configuration=config,
            environment_info={}
        )
    
    def _create_mock_weights(self):
        """Create mock model weights."""
        return {
            'fc1.weight': torch.randn(128, 64),
            'fc1.bias': torch.randn(128),
            'fc2.weight': torch.randn(64, 128),
            'fc2.bias': torch.randn(64)
        }

class TestMemoryManagement:
    """Test memory usage and cleanup."""
    
    def test_weight_cache_management(self):
        """Test that weight cache properly manages memory."""
        from keisei.evaluation.core.model_manager import ModelWeightManager
        
        manager = ModelWeightManager()
        manager.max_cache_size = 3
        
        # Add weights to cache
        for i in range(5):
            weights = {f'layer_{i}.weight': torch.randn(10, 10)}
            manager._weight_cache[f'model_{i}'] = weights
        
        # Should only keep max_cache_size entries
        assert len(manager._weight_cache) <= manager.max_cache_size
    
    def test_memory_cleanup_after_evaluation(self):
        """Test that temporary objects are cleaned up."""
        # This test would verify that evaluation doesn't leak memory
        # Implementation depends on your specific memory tracking needs
        pass
```

### Integration Tests

```python
# File: tests/evaluation/test_integration_performance.py
import pytest
from unittest.mock import MagicMock, patch

class TestTrainerIntegration:
    """Test integration of performance optimizations with trainer."""
    
    def test_trainer_uses_in_memory_evaluation(self):
        """Test that trainer properly uses in-memory evaluation."""
        # Mock trainer setup
        trainer = MagicMock()
        trainer.agent = MagicMock()
        trainer.evaluation_manager = MagicMock()
        
        # Mock evaluation manager to support in-memory evaluation
        trainer.evaluation_manager.evaluate_current_agent_in_memory = MagicMock()
        
        from keisei.training.callbacks import EvaluationCallback
        
        callback = EvaluationCallback(MagicMock(), interval=1000)
        
        # Simulate callback execution
        with patch.object(callback, '_should_run_evaluation', return_value=True):
            callback.on_step_end(trainer)
        
        # Verify in-memory evaluation was called
        trainer.evaluation_manager.evaluate_current_agent_in_memory.assert_called_once()
    
    def test_evaluation_manager_setup_with_performance_features(self):
        """Test evaluation manager setup with performance optimizations."""
        from keisei.evaluation.manager import EvaluationManager
        from keisei.evaluation.core.evaluation_config import EvaluationConfig, EvaluationStrategy
        
        config = EvaluationConfig(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            enable_in_memory_evaluation=True,
            enable_parallel_execution=True,
            max_concurrent_games=4
        )
        
        manager = EvaluationManager(config, "test_run")
        manager.setup(
            device="cpu",
            policy_mapper=MagicMock(),
            model_dir="/tmp",
            wandb_active=False
        )
        
        # Verify performance features are enabled
        assert manager.enable_in_memory_eval
        assert manager.model_weight_manager is not None
```

## Next Steps Summary

1. **Immediate Priority - Phase 6.1 (In-Memory Evaluation):**
   - Implement `ModelWeightManager` class
   - Update `EvaluationManager` with in-memory methods
   - Modify `SingleOpponentEvaluator` for weight-based evaluation
   - Update callbacks to use new system

2. **Follow-up - Phase 6.2 (Parallel Execution):**
   - Implement `ParallelGameExecutor` 
   - Create worker function for multiprocessing
   - Update evaluators for parallel capability
   - Add configuration options

3. **Testing and Validation:**
   - Create performance benchmark tests
   - Implement memory usage monitoring
   - Add integration tests with trainer

4. **Monitoring and Rollback:**
   - Add feature flags for safe deployment
   - Implement performance metrics collection
   - Maintain legacy fallback capabilities

This implementation guide provides the detailed technical foundation needed to complete the evaluation system refactor and achieve the performance goals outlined in the original audit.
