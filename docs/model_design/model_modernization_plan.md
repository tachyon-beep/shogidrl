# Model Modernization Implementation Plan (T-1 to T-9)

## Overview
This plan details the steps to modernize the Keisei DRL Shogi model pipeline, enabling flexible observation planes, a new ResNet-based ActorCritic model with SE blocks, mixed-precision, DDP support, robust checkpoint migration, and strong test/benchmark infrastructure.

---

## Task Breakdown & Progress

### T-1: FeatureSpec Registry & Core46 Builder
- [x] `keisei/shogi/features.py` created.
- [x] `FeatureSpec` registry implemented.
- [x] Core 46-plane builder implemented and tested.

### T-2: Optional Planes
- [x] Optional feature planes (`check`, `repetition`, `prom_zone`, `last2ply`, `hand_onehot`) implemented in `features.py`.
- [x] Feature selection is configurable.
- [x] Unit tests cover all feature combinations and edge cases.

### T-3: ResNet Tower Model
- [x] `keisei/training/models/resnet_tower.py` created.
- [x] `ActorCriticResTower` implemented with configurable depth/width, SE block toggle, and late-flatten/1x1 heads.
- [x] Policy/value heads slimmed to avoid parameter explosion.
- [x] Unit tests for forward pass, SE toggle, and fp16.

### T-4: Checkpoint Shim
- [x] `utils/checkpoint.py` created.
- [x] Logic for zero-padding or truncating the first conv layer for input channel changes implemented (no legacy/old checkpoint support required; only current model format is supported).
- [x] Migration unit tests implemented and passing.
- [!] Note: No backwards compatibility with pre-modernization checkpoints is provided, as this is a pre-alpha codebase.

### T-5: Trainer/Runner Config Integration
- [ ] Update `keisei/training/runner.py` and Trainer to use new config fields (`input_features`, `tower_depth`, `tower_width`, `se_ratio`).
- [ ] Autopopulate `obs_shape` from feature builder.
- [ ] Add CLI flags: `--model=resnet`, `--mixed_precision`, `--ddp`.

### T-6: Mixed-Precision Training
- [ ] Add mixed-precision context (`torch.cuda.amp.autocast`, `GradScaler`) to Trainer.
- [ ] Add CLI/config flag to enable/disable.

### T-7: DDP Self-Play (Optional)
- [ ] Create `scripts/selfplay_ddp.py` for two-process DDP self-play.
- [ ] Wrap model/optimizer with `DistributedDataParallel`.
- [ ] Add launch script and documentation.

### T-8: Unit Tests
- [x] Parametric tests for feature sets and model forward pass.
- [x] SE block tested in isolation.
- [x] All new/updated tests pass.

### T-9: Benchmark Script
- [ ] Create `scripts/benchmark.py` to measure games/sec, VRAM, and Elo vs depth-2 bot.
- [ ] Output results in a simple table.

---

## Acceptance Criteria (CI Gates)
- [x] All shapes and forward passes are correct for every feature subset.
- [ ] Checkpoint migration works (old weights load into new model).
- [ ] Mixed-precision and DDP runs do not crash.
- [ ] After 5k self-play games, new model outperforms old by ≥60% win rate.
- [x] All new/updated tests pass.

---

## Implementation Order
1. T-1, T-2: Feature builder and registry. **[DONE]**
2. T-3: ResNet model and SE block. **[DONE]**
3. T-4: Checkpoint migration. **[DONE]**
4. T-5: Trainer/config integration. **[PENDING]**
5. T-6: Mixed-precision. **[PENDING]**
6. T-7: DDP (optional, can be last). **[PENDING]**
7. T-8: Unit tests. **[DONE]**
8. T-9: Benchmarking. **[PENDING]**

---

## Notes
- Keep all new code modular and well-documented.
- Use feature flags and config-driven design for maximum flexibility.
- Prioritize test coverage for all new features and migration logic.

---

**Progress as of 2025-05-27:**
- Feature extraction, registry, and all optional planes are complete and tested.
- ResNet model (with SE, late flatten, slim heads) is implemented and tested.
- Unit tests for all new features and model logic are passing.
- Next: implement checkpoint migration logic, trainer/config integration, mixed-precision, DDP, and benchmarking.

---

## Next Step: Trainer/Runner Config Integration (T-5) — Detailed Plan

### Objective
Integrate the new feature builder, model, and configuration options into the training pipeline, making the system fully modular and configurable from YAML/CLI.

### Subtasks & File Touchpoints

1. **Config Schema Updates**
   - File: `keisei/config_schema.py`
   - Add fields to `AppConfig`/`TrainingConfig` for:
     - `input_features` (str, e.g. "core46", "core46+all")
     - `tower_depth`, `tower_width`, `se_ratio` (int/float, for ResNet)
     - `model_type` (str, e.g. "resnet")
     - `mixed_precision` (bool)
     - `ddp` (bool)

2. **Trainer/Runner Integration**
   - Files: `keisei/training/trainer.py`, `keisei/training/runner.py`
   - Read new config fields and pass them to model/feature builder.
   - Use the feature registry to build observations: `features.FEATURE_SPECS[config.input_features].build(game)`
   - Instantiate the model with config-driven parameters.
   - Autopopulate `obs_shape` from the feature builder.
   - Add CLI flags for `--model`, `--input_features`, `--tower_depth`, `--tower_width`, `--se_ratio`, `--mixed_precision`, `--ddp`.
   - Ensure Trainer uses the correct model and feature builder based on config.

3. **Model Factory**
   - File: `keisei/training/models/__init__.py` (or in `trainer.py`)
   - Add a factory function to instantiate the correct model class based on `model_type` and config.

4. **Tests/Validation**
   - Files: `tests/` (existing or new)
   - Add/extend tests to check that Trainer and Runner correctly instantiate the model and feature builder from config/CLI.
   - Test that invalid config values raise clear errors.

### Implementation Notes
- All new config fields should have sensible defaults to preserve backward compatibility.
- The Trainer should log the selected feature set, model type, and all relevant hyperparameters at startup.
- CLI flags should override config file values as before.
- The integration should be modular to allow easy addition of new feature sets or model types in the future.
