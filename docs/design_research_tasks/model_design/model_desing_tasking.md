## 1 — What stays the same

| Sub-system                                       | Status                          |
| ------------------------------------------------ | ------------------------------- |
| **Environment** (`ShogiGame`, legality, reward)  | **Untouched**                   |
| **PPO loop** (`PPOAgent.step()`, GAE, optimiser) | **99 % unchanged**              |
| **Logging & run-dir layout**                     | **Same**                        |
| **CI pipeline**                                  | Only new unit tests need adding |

The only moving parts are the **observation builder** and **model factory**; everything else plugs back in.

---

## 2 — Delta map (old → new)

| Area                   | From                              | To                                                             | Notes                                 |
| ---------------------- | --------------------------------- | -------------------------------------------------------------- | ------------------------------------- |
| **Observation tensor** | `torch.Size([46,9,9])` hard-coded | `torch.Size([C,9,9])` where `46 ≤ C ≤ 77`                      | Feature-flag list in config decides C |
| **Model file**         | `ActorCriticSingleConv`           | `ActorCriticResTower` (stem 256, N residual blocks, SE toggle) | Late-flatten 1×1 heads                |
| **Config**             | flat YAML                         | +`input_features`, `tower_depth`, `tower_width`, `se_ratio`    | Defaults replicate old behaviour      |
| **Checkpoint loader**  | strict shape match                | zero-pad first-layer weights when C grows                      | Backwards compatible                  |
| **Trainer flags**      | none                              | `--model=resnet` (default), `--mixed_precision`, `--ddp`       | CLI only – class signatures unchanged |
| **Unit tests**         | fixed shapes                      | parametric over feature sets                                   | use `pytest.mark.parametrize`         |

---

## 3 — Concrete tasks & file touch-points

| ID      | Task                                                                                      | Files                         | Effort |
| ------- | ----------------------------------------------------------------------------------------- | ----------------------------- | ------ |
| **T-1** | Add `FeatureSpec` registry and implement current 46 planes as builder `"core46"`          | `keisei/shogi/features.py`    | ½ day  |
| **T-2** | Implement optional planes: `check`, `repetition`, `prom_zone`, `last2ply`, `hand_onehot`  | same                          | 1 day  |
| **T-3** | Replace `ActorCritic` with `resnet_tower.py` (+SE block in `layers/se.py`)                | `keisei/training/models/`     | 1 day  |
| **T-4** | Shim for loading old checkpoints (`utils/checkpoint.py`)                                  | 30 min                        |        |
| **T-5** | Update trainer to read `input_features` & `tower_*` from config; autopopulate `obs_shape` | `keisei/training/runner.py`   | 1 day  |
| **T-6** | Add mixed-precision context + GradScaler                                                  | trainer                       | 1 hour |
| **T-7** | Add two-proc DDP self-play (optional but nice)                                            | new `scripts/selfplay_ddp.py` | 1 day  |
| **T-8** | Extend unit tests for variable-C tensors & SE block                                       | `tests/`                      | ½ day  |
| **T-9** | Benchmark script: games/s, VRAM, Elo vs depth-2 bot                                       | `scripts/benchmark.py`        | ½ day  |

Total: **≈ 4–5 dev-days**, matching the earlier estimate.

---

## 4 — Acceptance checklist (CI gates)

1. **Shapes:** `pytest -k "observation_shape"` passes for every enabled feature subset.
2. **Forward pass:** `ActorCriticResTower(torch.rand(2,C,9,9)).policy.shape == (2,13527)`.
3. **Checkpoint migration:** can load `v0_singleconv` weights with no exceptions.
4. **Training sanity:** 1 epoch on CPU reaches non-NaN loss; mixed-precision run does not blow up.
5. **Performance smoke test:** after 5 k self-play games the new net’s win-rate vs old ≥ 60 % (quick proxy).

---

## 5 — Timeline (assuming one developer)

| Weekday | Deliverable                                       |
| ------- | ------------------------------------------------- |
| **Mon** | T-1, T-2 merged; tests green with `core46`        |
| **Tue** | T-3 done; forward-pass tests pass                 |
| **Wed** | T-4, T-5, T-6; first mixed-precision run launches |
| **Thu** | T-8, quick benchmark; threshold gate 5 k games    |
| **Fri** | Buffer / DDP (T-7) or tidy-ups                    |

Plenty of slack for review & tweaks.

---

## 6 — How this lines up with the earlier roadmap

* **Phase 0-4** from my previous message maps 1-to-1 onto T-1 … T-8 above.
* Same insistence on *late flatten*, *SE blocks*, *feature flags*, *mixed-precision*, *checkpoint shim*.
* Deviations? Only that we’ve rolled DDP into the same sprint instead of “later” — easy win while the hood’s up.

Alignment score: **very high** (≈ 95 %); this is the operational cut-down of the conceptual roadmap you okayed.

---

### Confidence assessment (WEP)

*Feasibility of shipping the list in 5 working days on existing codebase*: **high** (80 – 85 %).
*Risk of nasty unforeseen architectural refactor*: **low-moderate** (25 %; checkpoint shim is the only hairy bit).

Ping me if you need code skeletons, SE block snippet, or just moral support.  Happy building!
