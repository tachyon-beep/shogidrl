# How to Use the DRL Shogi Client

This guide explains how to run the DRL Shogi Client for training a Reinforcement Learning agent to play Shogi.

## 1. Prerequisites

*   Python 3.10+
*   PyTorch (ensure CUDA is set up if using GPU, see `config.py`)
*   Other dependencies listed in `requirements.txt`

## 2. Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd keisei
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Linux/macOS
    # env\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 3. Configuration

*   All main training parameters are now set in a YAML or JSON config file (see `default_config.yaml`).
*   Key parameters to review before a long training run:
    *   `training.total_timesteps`: Total number of interactions with the environment.
    *   `training.steps_per_epoch`: Number of steps collected before each learning phase.
    *   `training.learning_rate`: Learning rate for the optimizer.
    *   `env.device`: Set to "cuda" for GPU training or "cpu".
    *   `logging.model_dir`: Directory where models will be saved (default: `models/`).
    *   `logging.log_file`: Path to the training log file (default: `logs/shogi_training_log.txt`).
    *   `display.display_moves`: If true, shows each move and waits between turns for easier observation.
    *   `display.turn_tick`: Delay in seconds between moves when displaying moves (default: 0.5).
*   The `keisei/shogi/shogi_engine.py` and related files define the game environment.
*   The `keisei/neural_network.py` defines the `ActorCritic` model.
*   The `keisei/ppo_agent.py` implements the PPO algorithm.

## 4. Running Training

1.  **Activate the virtual environment (if not already active):**
    ```bash
    source env/bin/activate
    ```

2.  **Start the training process:**
    ```bash
    python train.py --config default_config.yaml
    ```
    * To enable move display, set `display.display_moves: true` in your config file, or override via CLI:
      ```bash
      python train.py --config default_config.yaml --override display.display_moves=true
      ```

3.  **Monitoring Training:**
    *   Training progress, including episode rewards, losses, and evaluation results, will be printed to the console.
    *   If move display is enabled, each move will be shown with a delay for easier observation.
    *   Detailed logs are saved to the file specified by `logging.log_file` (default: `logs/shogi_training_log.txt`).
    *   Model checkpoints will be saved periodically to the directory specified by `logging.model_dir` (default: `models/`). The filename will indicate the episode number, e.g., `ppo_shogi_agent_episode_X.pth`.

## 5. Evaluation

*   The `train.py` script includes a basic evaluation phase at the end of the training run (or after a set number of episodes in the current test setup).
*   For more robust evaluation against specific checkpoints or baselines, use the `evaluate.py` script as shown below.

### Running Evaluation from the Command Line

To evaluate a trained agent against a baseline (random, heuristic, or PPO):

```bash
python -m keisei.evaluation.evaluate \
  --agent_checkpoint path/to/agent.ckpt \
  --opponent_type random \
  --num_games 20 \
  --device cpu \
  --log_file eval_log.txt \
  --wandb_log_eval
```

- `--opponent_type` can be `random`, `heuristic`, or `ppo` (if `ppo`, provide `--opponent_checkpoint`).
- Add `--wandb_log_eval` to enable Weights & Biases logging.
- See `python -m keisei.evaluation.evaluate --help` for all options.

### Programmatic Evaluation (Python API)

```python
from keisei.evaluation.evaluate import Evaluator, PolicyOutputMapper

policy_mapper = PolicyOutputMapper()
evaluator = Evaluator(
    agent_checkpoint_path="path/to/agent.ckpt",
    opponent_type="random",
    num_games=10,
    device_str="cpu",
    log_file_path_eval="eval_log.txt",
    policy_mapper=policy_mapper,
)
results = evaluator.evaluate()
print(results)
```

### Troubleshooting & Error Handling

- If the agent checkpoint file is missing, a clear `FileNotFoundError` will be raised.
- Invalid opponent types will raise a `ValueError`.
- W&B logging failures are caught and will not crash evaluation.
- For more details, see the docstrings in `keisei/evaluation/evaluate.py` and the tests in `tests/test_evaluate.py`.

---

## 6. Stopping and Resuming Training

*   **Stopping:** You can stop training by pressing `Ctrl+C` in the terminal where `train.py` is running. The most recently saved model checkpoint can be used to resume later.
*   **Resuming:** To resume training from a checkpoint:
    1.  Modify `PPOAgent` in `ppo_agent.py` to include a `load_model(path)` method if it doesn't already have one that loads both model and optimizer states.
    2.  Modify `train.py` to check for a checkpoint path (e.g., passed as a command-line argument or set in `config.py`) and call `agent.load_model()` before starting the training loop.
    3.  You might also need to adjust `TOTAL_TIMESTEPS` or the starting episode/timestep number to reflect the resumed training.

    *(Note: Full resume functionality might require further implementation based on the current state of `PPOAgent` and `train.py`)*

## 7. Understanding the Output

*   **Console Output:** Shows real-time progress.
*   **Log File (`logs/shogi_training_log.txt`):** Contains a history of training metrics.
    *   `Episode X finished after Y steps. Reward: Z`
    *   `Evaluation: ... Wins: A | Draws: B | Losses: C`
    *   `Saved model checkpoint to models/ppo_shogi_agent_episode_X.pth`
*   **Model Checkpoints (`models/`):** Saved PyTorch models (`.pth` files).

## 8. Enhanced TUI Dashboard

The training script includes an optional Rich-based dashboard with an ASCII board,
metric trends and Elo ratings. These features are controlled by the `display`
section of the configuration. To enable the dashboard, use the defaults in
`default_config.yaml` or provide an override file such as
`examples/enhanced_display_config.yaml`:

```bash
python train.py --config examples/enhanced_display_config.yaml
```

See `docs/development/tui_display_quick_reference.md` for a description of each
option.

This guide provides the basic steps to get the training process started. For more advanced operations, hyperparameter tuning, and troubleshooting, refer to the `OPS_PLAN.md` and the source code documentation.
