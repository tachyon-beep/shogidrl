# Keisei Evaluation System Integration Testing Plan

## Overview
This plan outlines comprehensive integration testing for the remediated Keisei evaluation system. The system has passed basic validation tests and now requires thorough integration verification with the broader Keisei ecosystem.

## Phase 1: Training Pipeline Integration Tests

### 1.1 Callback Integration Testing
- **Test Target**: `EvaluationCallback` in `/home/john/keisei/keisei/training/callbacks.py`
- **Integration Points**: 
  - Trainer → EvaluationCallback → EvaluationManager
  - Checkpoint-based evaluation triggers
  - Model state management (eval/train mode)
  - Results propagation to training metrics
- **Test Strategy**: Mock training scenarios with different evaluation intervals
- **PLAN_UNCERTAINTY**: Need to verify how callback handles agent model state transitions

### 1.2 Training-Evaluation Flow Testing  
- **Test Target**: Main training loop integration in `/home/john/keisei/keisei/training/trainer.py`
- **Integration Points**:
  - Shared resource access (device, policy_mapper, model_dir)
  - Training checkpoint → evaluation checkpoint flow
  - WandB logging integration
  - Memory management during evaluation
- **Test Strategy**: Simulate training steps that trigger evaluation
- **PLAN_UNCERTAINTY**: Resource contention between training and evaluation threads

### 1.3 Configuration System Integration
- **Test Target**: `AppConfig` and `EvaluationConfig` integration in `/home/john/keisei/keisei/config_schema.py`
- **Integration Points**:
  - Central config → evaluation config transformation
  - CLI parameter overrides flowing to evaluation
  - YAML configuration loading and validation
  - Environment variable integration
- **Test Strategy**: Test config edge cases and override scenarios

## Phase 2: Model and Agent Integration Tests

### 2.1 PPOAgent Integration Testing
- **Test Target**: Agent protocol compliance with evaluation system
- **Integration Points**:
  - Actor-critic model interface
  - Model state extraction for evaluation
  - In-memory vs checkpoint-based evaluation
  - Device management for temporary agents
- **Test Strategy**: Create test PPOAgent instances and verify evaluation works
- **PLAN_UNCERTAINTY**: Memory management for multiple agent instances

### 2.2 Checkpoint Loading Integration
- **Test Target**: Model reconstruction from checkpoints
- **Integration Points**:
  - ModelManager checkpoint format compatibility
  - Weight extraction and caching
  - Model architecture reconstruction
  - Mixed precision and device handling
- **Test Strategy**: Test with various checkpoint formats and model architectures

### 2.3 Policy Mapper Integration  
- **Test Target**: Action space mapping consistency
- **Integration Points**:
  - Shared policy mapper between training and evaluation
  - Action space validation
  - Move translation accuracy
  - Performance under load
- **Test Strategy**: Verify action mappings are consistent across contexts

## Phase 3: Game Engine Integration Tests

### 3.1 ShogiGame Integration
- **Test Target**: Game execution within evaluation framework
- **Integration Points**:
  - Game state management
  - Move validation and execution
  - Termination condition detection
  - Performance under parallel execution
- **Test Strategy**: Run extended game sequences to test stability

### 3.2 Move Validation Integration
- **Test Target**: Legal move checking and game rules
- **Integration Points**:
  - Agent moves → game validation
  - Error handling for illegal moves
  - Game state consistency
  - Drop and promotion rule enforcement
- **Test Strategy**: Test edge cases and complex game positions

### 3.3 Color/Player Management
- **Test Target**: Sente/gote alternation and player management
- **Integration Points**:
  - Player color assignment
  - Turn management
  - Game history tracking
  - Result attribution
- **Test Strategy**: Test color balance and alternation patterns

## Phase 4: Async and Concurrency Integration Tests

### 4.1 Event Loop Management
- **Test Target**: Async evaluation within training loops
- **Integration Points**:
  - Async/sync boundary management
  - Event loop lifecycle
  - Exception propagation
  - Resource cleanup
- **Test Strategy**: Test async evaluation from sync training context
- **PLAN_UNCERTAINTY**: Complex async/sync interactions need careful testing

### 4.2 Parallel Execution Testing
- **Test Target**: Concurrent game execution
- **Integration Points**:
  - Thread safety
  - Resource sharing
  - Error isolation
  - Performance scaling
- **Test Strategy**: Load testing with multiple concurrent evaluations

### 4.3 Resource Management Testing
- **Test Target**: GPU/CPU resource sharing
- **Integration Points**:
  - Device allocation
  - Memory management
  - Resource contention handling
  - Cleanup procedures
- **Test Strategy**: Monitor resource usage under various load conditions

## Phase 5: Data Flow Integration Tests

### 5.1 Metrics Collection Integration
- **Test Target**: Evaluation metrics flow to training system
- **Integration Points**:
  - EvaluationResult → TrainingManager metrics
  - WandB logging integration
  - Real-time metrics updates
  - Historical data persistence
- **Test Strategy**: Verify metrics propagate correctly and consistently

### 5.2 ELO System Integration
- **Test Target**: ELO rating integration with opponent pool
- **Integration Points**:
  - Rating updates after evaluation
  - Opponent selection based on ratings
  - Rating persistence
  - Historical rating tracking
- **Test Strategy**: Test ELO calculations and persistence

### 5.3 Results Persistence Integration
- **Test Target**: Evaluation result storage and retrieval
- **Integration Points**:
  - File system persistence
  - Result format consistency
  - Retrieval performance
  - Data integrity
- **Test Strategy**: Test various result storage scenarios

## Success Criteria

### Functional Requirements
- All integration points pass basic functionality tests
- No data corruption or loss during integration operations
- Error handling maintains system stability
- Performance meets baseline requirements

### Performance Requirements  
- Evaluation integration adds <10% overhead to training
- Memory usage remains within acceptable bounds
- Concurrent operations scale linearly up to configured limits
- Resource cleanup prevents memory leaks

### Reliability Requirements
- Error conditions are handled gracefully
- System recovers from integration failures
- No data races or concurrency issues
- Integration failures don't affect training stability

## Risk Assessment

### High Risk Areas
- Async/sync boundary interactions
- Resource contention between training and evaluation
- Memory management with multiple agent instances
- Complex error propagation chains

### Medium Risk Areas
- Configuration parameter interactions
- Performance under high load
- ELO system data consistency
- WandB integration edge cases

### Low Risk Areas
- Basic game engine integration
- Simple configuration loading
- File I/O operations
- Basic metrics collection

## Testing Methodology

### Test Categories
1. **Unit Integration Tests**: Test individual integration points in isolation
2. **Component Integration Tests**: Test multiple related components together
3. **System Integration Tests**: Test complete evaluation workflows
4. **Load Tests**: Test performance under realistic load conditions
5. **Error Recovery Tests**: Test system behavior under failure conditions

### Test Data Requirements
- Sample training checkpoints
- Various game positions and states
- Different configuration combinations
- Performance baseline measurements

### Test Environment Setup
- Isolated test environment
- Mock training infrastructure
- Controlled resource allocation
- Monitoring and logging infrastructure

This plan provides comprehensive coverage of all integration points while identifying key risk areas that require special attention during testing.