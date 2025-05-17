# DRL Shogi Client

A Deep Reinforcement Learning (DRL) client for Shogi, designed to learn the game from scratch using self-play and Proximal Policy Optimization (PPO).

## Project Overview
- **No hardcoded opening books or human-designed evaluation functions.**
- **AI learns solely through self-play and RL.**
- **Implements a full Shogi environment, PPO agent, experience buffer, and neural network.**

## Key Features
- Full Shogi rules, including drops, promotions, and repetition.
- Modular codebase: `shogi_engine.py`, `neural_network.py`, `ppo_agent.py`, `experience_buffer.py`, `utils.py`, `train.py`, `config.py`.
- PyTorch-based neural network and PPO implementation.
- Logging, model saving, and reproducible training.

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd keisei
   ```
2. **Install dependencies:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. **Run training:**
   ```bash
   python train.py
   ```

## Project Structure
```
shogi_drl/
├── shogi_engine.py       # Game rules, state
├── neural_network.py     # ActorCritic NN model
├── ppo_agent.py          # PPO algorithm, action selection, learning
├── experience_buffer.py  # Replay buffer
├── utils.py              # PolicyOutputMapper, other helpers
├── train.py              # Main training script
├── config.py             # Hyperparameters and configuration
├── README.md
├── requirements.txt
├── models/               # To save trained models
└── logs/                 # To save training logs
```

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy
- (Optional) TensorBoard

## License
MIT
