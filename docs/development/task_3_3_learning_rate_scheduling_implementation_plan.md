# Task 3.3 Learning Rate Scheduling Implementation Plan

## Overview

This document outlines the detailed implementation plan for **Task 3.3: Implement Learning Rate Scheduling** from the MLCORE Implementation Plan. This task involves adding configurable learning rate scheduling to the PPO optimizer to improve training convergence and final performance.

## Problem Analysis

### Current State
1. **Fixed Learning Rate**: PPOAgent currently uses a fixed learning rate (`3e-4`) throughout training
2. **No Scheduler Support**: No infrastructure for learning rate scheduling exists
3. **Limited Optimization**: Training could benefit from adaptive learning rate strategies
4. **Configuration Gap**: No configuration options for learning rate scheduling

### Requirements from MLCORE Plan
- Add configuration options to `TrainingConfig` for LR scheduling
- Initialize scheduler in `PPOAgent.__init__` based on configuration
- Call `scheduler.step()` in `PPOAgent.learn()` after `optimizer.step()`
- Ensure scheduler state is saved/loaded with checkpoints
- Add comprehensive tests for LR scheduling

## Implementation Plan

### Phase 1: Configuration Schema Enhancement

**File**: `/home/john/keisei/keisei/config_schema.py`

**Changes to TrainingConfig**:
```python
class TrainingConfig(BaseModel):
    # ...existing fields...
    
    # Learning Rate Scheduling Configuration
    lr_schedule_type: Optional[str] = Field(
        None, 
        description="Type of learning rate scheduler: 'linear', 'cosine', 'exponential', 'step', or None to disable"
    )
    lr_schedule_kwargs: Optional[dict] = Field(
        None,
        description="Additional keyword arguments for the learning rate scheduler"
    )
    lr_schedule_step_on: str = Field(
        "epoch",
        description="When to step the scheduler: 'epoch' (per PPO epoch) or 'update' (per minibatch update)"
    )
    
    @validator("lr_schedule_type")
    def validate_lr_schedule_type(cls, v):
        if v is not None and v not in ["linear", "cosine", "exponential", "step"]:
            raise ValueError("lr_schedule_type must be one of: 'linear', 'cosine', 'exponential', 'step', or None")
        return v
    
    @validator("lr_schedule_step_on")
    def validate_lr_schedule_step_on(cls, v):
        if v not in ["epoch", "update"]:
            raise ValueError("lr_schedule_step_on must be 'epoch' or 'update'")
        return v
```

### Phase 2: Scheduler Factory Implementation

**File**: `/home/john/keisei/keisei/core/scheduler_factory.py` (new file)

**Purpose**: Create a factory pattern for learning rate schedulers to support different types.

```python
"""
Learning rate scheduler factory for PPO training.
"""

from typing import Optional, Dict, Any
import torch
from torch.optim.lr_scheduler import (
    LambdaLR, 
    CosineAnnealingLR, 
    ExponentialLR, 
    StepLR,
    _LRScheduler
)

class SchedulerFactory:
    """Factory for creating PyTorch learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        schedule_type: Optional[str],
        total_steps: int,
        schedule_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[_LRScheduler]:
        """
        Create a learning rate scheduler based on configuration.
        
        Args:
            optimizer: PyTorch optimizer
            schedule_type: Type of scheduler ('linear', 'cosine', 'exponential', 'step')
            total_steps: Total number of training steps for scheduling
            schedule_kwargs: Additional arguments for the scheduler
            
        Returns:
            Configured scheduler or None if schedule_type is None
        """
        if schedule_type is None:
            return None
            
        schedule_kwargs = schedule_kwargs or {}
        
        if schedule_type == "linear":
            return SchedulerFactory._create_linear_scheduler(optimizer, total_steps, schedule_kwargs)
        elif schedule_type == "cosine":
            return SchedulerFactory._create_cosine_scheduler(optimizer, total_steps, schedule_kwargs)
        elif schedule_type == "exponential":
            return SchedulerFactory._create_exponential_scheduler(optimizer, schedule_kwargs)
        elif schedule_type == "step":
            return SchedulerFactory._create_step_scheduler(optimizer, schedule_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {schedule_type}")
    
    @staticmethod
    def _create_linear_scheduler(optimizer, total_steps, kwargs):
        """Create linear decay scheduler."""
        final_lr_fraction = kwargs.get("final_lr_fraction", 0.1)
        
        def linear_decay(step):
            return max(final_lr_fraction, 1.0 - step / total_steps)
        
        return LambdaLR(optimizer, lr_lambda=linear_decay)
    
    @staticmethod
    def _create_cosine_scheduler(optimizer, total_steps, kwargs):
        """Create cosine annealing scheduler."""
        eta_min_fraction = kwargs.get("eta_min_fraction", 0.0)
        initial_lr = optimizer.param_groups[0]['lr']
        eta_min = initial_lr * eta_min_fraction
        
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    
    @staticmethod
    def _create_exponential_scheduler(optimizer, kwargs):
        """Create exponential decay scheduler."""
        gamma = kwargs.get("gamma", 0.995)
        return ExponentialLR(optimizer, gamma=gamma)
    
    @staticmethod
    def _create_step_scheduler(optimizer, kwargs):
        """Create step decay scheduler."""
        step_size = kwargs.get("step_size", 1000)
        gamma = kwargs.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
```

### Phase 3: PPOAgent Integration

**File**: `/home/john/keisei/keisei/core/ppo_agent.py`

**Changes to `__init__` method**:
```python
def __init__(
    self,
    model: ActorCriticProtocol,
    config: AppConfig,
    device: torch.device,
    name: str = "PPOAgent",
):
    # ...existing initialization code...
    
    # Learning rate scheduler setup
    self.lr_schedule_type = getattr(config.training, "lr_schedule_type", None)
    self.lr_schedule_step_on = getattr(config.training, "lr_schedule_step_on", "epoch")
    
    # Calculate total steps for scheduler
    total_steps = self._calculate_total_scheduler_steps(config)
    
    # Create scheduler if configured
    from keisei.core.scheduler_factory import SchedulerFactory
    self.scheduler = SchedulerFactory.create_scheduler(
        optimizer=self.optimizer,
        schedule_type=self.lr_schedule_type,
        total_steps=total_steps,
        schedule_kwargs=getattr(config.training, "lr_schedule_kwargs", None)
    )
    
def _calculate_total_scheduler_steps(self, config: AppConfig) -> int:
    """Calculate total number of scheduler steps based on configuration."""
    if self.lr_schedule_step_on == "epoch":
        # Step per PPO epoch: total_timesteps / steps_per_epoch * ppo_epochs
        return (config.training.total_timesteps // config.training.steps_per_epoch) * config.training.ppo_epochs
    else:  # "update"
        # Step per minibatch update
        steps_per_learn = config.training.steps_per_epoch // config.training.minibatch_size * config.training.ppo_epochs
        return (config.training.total_timesteps // config.training.steps_per_epoch) * steps_per_learn
```

**Changes to `learn` method**:
```python
def learn(self, experience_buffer: ExperienceBuffer) -> Dict[str, float]:
    # ...existing learning code...
    
    # PPO training loops with scheduler stepping
    for epoch in range(self.ppo_epochs):
        # ...existing epoch code...
        
        for start_idx in range(0, num_samples, self.minibatch_size):
            # ...existing minibatch training code...
            
            self.optimizer.step()
            
            # Step scheduler if configured to step on updates
            if self.scheduler and self.lr_schedule_step_on == "update":
                self.scheduler.step()
        
        # Step scheduler if configured to step on epochs
        if self.scheduler and self.lr_schedule_step_on == "epoch":
            self.scheduler.step()
    
    # Update current learning rate in metrics
    current_lr = self.optimizer.param_groups[0]["lr"]
    
    metrics: Dict[str, float] = {
        # ...existing metrics...
        "ppo/learning_rate": current_lr,
    }
    return metrics
```

**Changes to `save_model` method**:
```python
def save_model(
    self,
    file_path: str,
    global_timestep: int = 0,
    total_episodes_completed: int = 0,
    stats_to_save: Optional[Dict[str, int]] = None,
) -> None:
    """Saves the model, optimizer, scheduler, and training state to a file."""
    save_dict = {
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "global_timestep": global_timestep,
        "total_episodes_completed": total_episodes_completed,
    }
    
    # Save scheduler state if scheduler exists
    if self.scheduler:
        save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        save_dict["lr_schedule_type"] = self.lr_schedule_type
        save_dict["lr_schedule_step_on"] = self.lr_schedule_step_on
    
    if stats_to_save:
        save_dict.update(stats_to_save)
    
    torch.save(save_dict, file_path)
    print(f"PPOAgent model, optimizer, scheduler, and state saved to {file_path}")
```

**Changes to `load_model` method**:
```python
def load_model(self, file_path: str) -> Dict[str, Any]:
    """Loads the model, optimizer, scheduler, and training state from a file."""
    # ...existing loading code...
    
    try:
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if it exists and scheduler is configured
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Return checkpoint data including scheduler info
        return {
            "global_timestep": checkpoint.get("global_timestep", 0),
            "total_episodes_completed": checkpoint.get("total_episodes_completed", 0),
            "black_wins": checkpoint.get("black_wins", 0),
            "white_wins": checkpoint.get("white_wins", 0),
            "draws": checkpoint.get("draws", 0),
            "lr_schedule_type": checkpoint.get("lr_schedule_type", None),
            "lr_schedule_step_on": checkpoint.get("lr_schedule_step_on", "epoch"),
        }
    # ...existing error handling...
```

### Phase 4: Configuration Examples

**Example configurations for different scheduler types**:

**Linear Decay (recommended for PPO)**:
```yaml
training:
  learning_rate: 3e-4
  lr_schedule_type: "linear"
  lr_schedule_step_on: "epoch"
  lr_schedule_kwargs:
    final_lr_fraction: 0.1  # End at 10% of initial LR
```

**Cosine Annealing**:
```yaml
training:
  learning_rate: 3e-4
  lr_schedule_type: "cosine"
  lr_schedule_step_on: "epoch"
  lr_schedule_kwargs:
    eta_min_fraction: 0.05  # Minimum LR as fraction of initial
```

**Exponential Decay**:
```yaml
training:
  learning_rate: 3e-4
  lr_schedule_type: "exponential"
  lr_schedule_step_on: "update"
  lr_schedule_kwargs:
    gamma: 0.9995  # Decay factor per step
```

**Step Decay**:
```yaml
training:
  learning_rate: 3e-4
  lr_schedule_type: "step"
  lr_schedule_step_on: "epoch"
  lr_schedule_kwargs:
    step_size: 50   # Decay every 50 epochs
    gamma: 0.5      # Multiply by 0.5
```

### Phase 5: Testing Strategy

**File**: `/home/john/keisei/tests/test_lr_scheduling.py` (new file)

**Test Coverage**:
1. **Configuration Validation Tests**:
   - Valid scheduler types
   - Invalid scheduler types
   - Valid step_on options
   - Invalid step_on options

2. **Scheduler Factory Tests**:
   - Creation of each scheduler type
   - Proper parameter passing
   - Error handling for invalid types

3. **PPOAgent Integration Tests**:
   - Scheduler initialization with different configs
   - Scheduler stepping during training
   - Learning rate changes over time
   - No scheduler when disabled

4. **Checkpoint Tests**:
   - Saving and loading scheduler state
   - Resuming training with correct LR
   - Backward compatibility with old checkpoints

5. **End-to-End Tests**:
   - Training with different schedulers
   - Metrics tracking of LR changes
   - Performance impact validation

### Phase 6: Documentation Updates

**Files to update**:
1. `docs/component_audit/core_ppo_agent.md` - Add scheduler documentation
2. `default_config.yaml` - Add scheduler configuration examples
3. `HOW_TO_USE.md` - Document learning rate scheduling usage

## Implementation Risks & Mitigation

### Risk 1: Scheduler Step Timing
**Issue**: Incorrect scheduler stepping could lead to unexpected LR changes
**Mitigation**: 
- Clear documentation of step_on options
- Comprehensive testing of both epoch and update stepping
- Validation in configuration

### Risk 2: Checkpoint Compatibility
**Issue**: Old checkpoints without scheduler state might fail to load
**Mitigation**:
- Graceful handling of missing scheduler state in checkpoints
- Backward compatibility tests
- Default behavior when scheduler state is missing

### Risk 3: Performance Impact
**Issue**: Scheduler overhead might slow training
**Mitigation**:
- Minimal scheduler overhead (PyTorch built-ins are efficient)
- Option to disable scheduling entirely
- Performance benchmarking

### Risk 4: Configuration Complexity
**Issue**: Too many configuration options might confuse users
**Mitigation**:
- Sensible defaults
- Clear documentation with examples
- Validation of configuration combinations

## Success Criteria

1. ✅ **Configuration Schema**: TrainingConfig supports all planned scheduler options
2. ✅ **Scheduler Factory**: Factory pattern successfully creates all scheduler types
3. ✅ **PPOAgent Integration**: Scheduler properly initialized and stepped during training
4. ✅ **State Management**: Scheduler state saved/loaded with checkpoints
5. ✅ **Testing Coverage**: Comprehensive tests for all components and scenarios
6. ✅ **Documentation**: Complete documentation of learning rate scheduling features
7. ✅ **Backward Compatibility**: Old checkpoints and configs continue to work
8. ✅ **Performance**: No significant performance regression from scheduler addition

## Implementation Timeline

**Estimated Total Effort**: 2-3 days

**Day 1**: 
- Phase 1: Configuration schema enhancement
- Phase 2: Scheduler factory implementation
- Initial unit tests

**Day 2**:
- Phase 3: PPOAgent integration
- Phase 5: Comprehensive testing
- Checkpoint compatibility

**Day 3**:
- Phase 6: Documentation updates
- Integration testing
- Performance validation
- Final review and cleanup

## Next Steps

1. **Start Implementation**: Begin with Phase 1 configuration schema changes
2. **Incremental Testing**: Test each phase before moving to the next
3. **Integration Validation**: Ensure end-to-end functionality works
4. **Performance Benchmarking**: Verify no significant overhead introduced
5. **Documentation Review**: Ensure all changes are properly documented

This implementation plan provides a comprehensive approach to adding learning rate scheduling to the PPO agent while maintaining system stability and backward compatibility.