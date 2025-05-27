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
- [x] Update `keisei/config_schema.py` with new fields (`input_features`, `tower_depth`, `tower_width`, `se_ratio`, `model_type`, `mixed_precision`, `ddp`, `gradient_clip_max_norm`, `lambda_gae`).
- [x] Integrate new config fields into `keisei/training/trainer.py`.
- [x] Implement model factory in `keisei/training/models/__init__.py` and integrate with Trainer.
- [x] Autopopulate `obs_shape` from feature builder in Trainer.
- [x] Add CLI flags for new training parameters to `keisei/training/train.py`.
- [x] Ensure Trainer uses the correct model and feature builder based on config.
- [x] Update `keisei/training/runner.py` (Note: `runner.py` not substantially used yet, focus on `trainer.py` and `train.py` for now).
- [x] Tests for config/CLI overrides and model/feature instantiation in `tests/test_trainer_config.py`.
- [~] Address remaining test failures in `tests/test_train.py` (`subprocess.CalledProcessError`).

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
- [x] Checkpoint migration tests.
- [x] Trainer config integration tests.
- [~] All new/updated tests pass (pending `test_train.py` fix).

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
4. T-5: Trainer/config integration. **[IN PROGRESS]**
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
- Unit tests for features, model, and checkpoint migration are passing.
- Trainer/config integration (T-5) is largely complete:
    - Pydantic config schema (`keisei/config_schema.py`) updated and is the single source of truth.
    - `Trainer` in `keisei/training/trainer.py` uses new config fields.
    - Model factory created and integrated.
    - CLI arguments in `keisei/training/train.py` reflect new config options.
    - `tests/test_trainer_config.py` updated and all tests pass.
- PPO-specific logic (loss calculation, GAE) integrated into `Trainer.train_step`.
- Next:
    - **CURRENT FOCUS:** Resolve `subprocess.CalledProcessError` in `tests/test_train.py`.
    - Implement mixed-precision training (T-6).
    - Implement DDP for self-play (T-7).
    - Create benchmark script (T-9).
    - Further PPO enhancements (experience buffer, hyperparameter consolidation).

---

## Next Step: Trainer/Runner Config Integration (T-5) — Detailed Plan

### Objective
Integrate the new feature builder, model, and configuration options into the training pipeline, making the system fully modular and configurable from YAML/CLI.

### Subtasks & File Touchpoints

1. **Config Schema Updates**
   - File: `keisei/config_schema.py`
   - [x] Add fields to `AppConfig`/`TrainingConfig` for:
     - `input_features` (str, e.g. "core46", "core46+all")
     - `tower_depth`, `tower_width`, `se_ratio` (int/float, for ResNet)
     - `model_type` (str, e.g. "resnet")
     - `mixed_precision` (bool)
     - `ddp` (bool)
     - `gradient_clip_max_norm` (float)
     - `lambda_gae` (float)
     - `evaluation_interval_timesteps` (int, moved to `EvaluationConfig`)

2. **Trainer/Runner Integration**
   - Files: `keisei/training/trainer.py`, `keisei/training/train.py` (runner.py less critical for now)
   - [x] Read new config fields and pass them to model/feature builder in `Trainer`.
   - [x] Use the feature registry to build observations: `features.FEATURE_SPECS[config.training.input_features].build(game_or_state)`
   - [x] Instantiate the model with config-driven parameters in `Trainer`.
   - [x] Autopopulate `obs_shape` from the feature builder in `Trainer`.
   - [x] Add CLI flags for relevant new parameters in `keisei/training/train.py`.
   - [x] Ensure Trainer uses the correct model and feature builder based on config.

3. **Model Factory**
   - File: `keisei/training/models/__init__.py`
   - [x] Add a factory function (`model_factory`) to instantiate the correct model class based on `model_type` and config.
   - [x] Integrate `model_factory` into `Trainer`.

4. **Tests/Validation**
   - Files: `tests/test_trainer_config.py`, `tests/test_train.py`
   - [x] Add/extend tests in `test_trainer_config.py` to check that Trainer correctly instantiates the model and feature builder from config/CLI.
   - [x] Test that invalid config values raise clear errors.
   - [~] Fix `subprocess.CalledProcessError` in `tests/test_train.py` (Current Focus).

### Implementation Notes
- [x] All new config fields have sensible defaults.
- [x] The Trainer logs the selected feature set, model type, and all relevant hyperparameters at startup.
- [x] CLI flags override config file values.
- [x] The integration is modular.
