# Software Documentation Template for Subsystems - Evaluation Loop

## 📘 loop.py as of 2025-05-31

**Project Name:** `Keisei - Deep Reinforcement Learning Shogi Client`
**Folder Path:** `/keisei/evaluation/`
**Documentation Version:** `1.0`
**Date:** `2025-05-31`
**Responsible Author or Agent ID:** `GitHub Copilot`

---

### 1. Overview 📜

* **Purpose of this Folder/Module Set:**
  Implements the core evaluation loop that orchestrates multi-game tournaments between trained PPO agents and various opponents. This module handles the game-by-game execution, result tracking, and statistical computation for agent performance assessment.

* **Key Responsibilities:**
  - Execute multiple game matches between agent and opponent
  - Handle alternating player roles and legal move validation
  - Track game outcomes (wins, losses, draws) and statistics
  - Compute win rates, game lengths, and performance metrics
  - Integrate with logging systems for detailed evaluation tracking
  - Support different opponent types (PPO agents, heuristic players, random players)

* **Domain Context:**
  Multi-game evaluation framework for Shogi reinforcement learning agents, providing tournament-style assessment with statistical analysis of agent performance against various baselines and opponents.

* **High-Level Architecture / Interaction Summary:**
  The evaluation loop serves as the execution engine for agent assessment, receiving configured agents and opponents from the evaluation orchestrator and running systematic game matches. It interfaces with the Shogi game engine for rule enforcement and move validation, while providing structured results to logging and analysis systems.

---

### 2. Modules 📦

* **Module Name:** `loop.py`

  * **Purpose:** Implement the core evaluation loop for systematic agent performance assessment.
  * **Design Patterns Used:** Tournament pattern for game orchestration, Strategy pattern for different opponent types, Observer pattern for result logging.
  * **Key Functions/Classes Provided:** 
    - `run_evaluation_loop()` - Main evaluation execution function
    - `ResultsDict` - TypedDict for structured result data
  * **Configuration Surface:**
    - `num_games`: Number of games to play in evaluation
    - `max_moves_per_game`: Maximum moves per game to prevent infinite games
    - Opponent configuration passed through from evaluation orchestrator
  * **Dependencies:**
    * **Internal:**
      - `keisei.core.ppo_agent.PPOAgent`: Agent being evaluated
      - `keisei.shogi.shogi_game.ShogiGame`: Game engine for rule enforcement
      - `keisei.shogi.shogi_core_definitions.MoveTuple`: Move representation
      - `keisei.utils.BaseOpponent`: Base class for non-PPO opponents
      - `keisei.utils.EvaluationLogger`: Logging interface
      - `keisei.utils.PolicyOutputMapper`: Action space mapping
    * **External:**
      - `torch`: PyTorch for device management and tensor operations
      - `typing`: Type annotations (TypedDict, Union, Optional)
  * **External API Contracts:**
    - Returns standardized ResultsDict with evaluation metrics
    - Compatible with various opponent implementations
  * **Side Effects / Lifecycle Considerations:**
    - Modifies game state through multiple game iterations
    - Logs game progress and results through provided logger
    - Accumulates statistics across multiple games
  * **Usage Examples:**
    ```python
    from keisei.evaluation.loop import run_evaluation_loop, ResultsDict
    
    results = run_evaluation_loop(
        agent_to_eval=ppo_agent,
        opponent=baseline_opponent,
        num_games=100,
        logger=evaluation_logger,
        max_moves_per_game=300
    )
    print(f"Win rate: {results['win_rate']:.2%}")
    ```

---

### 3. Classes 🏛️

* **Class Name:** `ResultsDict`

  * **Defined In Module:** `loop.py`
  * **Purpose:** Provide structured type definition for evaluation results with statistical metrics.
  * **Design Role:** Data transfer object ensuring consistent result format across evaluation system.
  * **Inheritance:**
    * **Extends:** `typing.TypedDict`
    * **Subclasses (internal only):** None
  * **Key Attributes/Properties:**
    - `games_played: int` – Total number of completed games
    - `agent_wins: int` – Number of games won by the evaluated agent
    - `opponent_wins: int` – Number of games won by the opponent
    - `draws: int` – Number of drawn games
    - `game_results: list[str]` – Detailed list of individual game outcomes
    - `win_rate: float` – Percentage of games won by agent
    - `loss_rate: float` – Percentage of games lost by agent
    - `draw_rate: float` – Percentage of drawn games
    - `avg_game_length: float` – Average number of moves per game
  * **Key Methods:**
    - None (TypedDict only defines structure)
  * **Interconnections:**
    * **Internal Class/Module Calls:** Used by evaluation orchestrator and logging systems
    * **External Systems:** Integration with W&B logging and analysis tools
  * **Lifecycle & State:**
    - Initialized with zero values at evaluation start
    - Incrementally updated during game loop execution
    - Finalized with computed statistics after all games
  * **Threading/Concurrency:**
    - Immutable after creation; thread-safe for read operations
  * **Usage Example:**
    ```python
    results: ResultsDict = {
        "games_played": 100,
        "agent_wins": 65,
        "opponent_wins": 30,
        "draws": 5,
        "game_results": ["agent_win", "opponent_win", ...],
        "win_rate": 0.65,
        "loss_rate": 0.30,
        "draw_rate": 0.05,
        "avg_game_length": 85.4
    }
    ```

---

### 4. Functions/Methods ⚙️

* **Function/Method Name:** `run_evaluation_loop`

  * **Defined In:** `loop.py`
  * **Belongs To:** Module-level function
  * **Purpose:** Execute a systematic evaluation of a PPO agent against an opponent over multiple games.
  * **Parameters:**
    - `agent_to_eval: PPOAgent` – The PPO agent being evaluated
    - `opponent: Union[PPOAgent, BaseOpponent]` – Opponent to play against
    - `num_games: int` – Number of games to play in the evaluation
    - `logger: EvaluationLogger` – Logger for tracking evaluation progress
    - `max_moves_per_game: int` – Maximum moves per game to prevent infinite games
  * **Returns:**
    - `ResultsDict` – Comprehensive evaluation results with statistics
  * **Raises/Exceptions:**
    - Handles game engine exceptions gracefully with logging
    - Continues evaluation even if individual games fail
  * **Side Effects:**
    - Logs game progress and outcomes through provided logger
    - Modifies agent evaluation mode (sets to non-training)
    - Creates and destroys multiple ShogiGame instances
  * **Calls To:**
    - `ShogiGame()` - Game instance creation
    - `agent.select_action()` - Agent move selection
    - `opponent.select_move()` or `opponent.select_action()` - Opponent move selection
    - `game.make_move()` - Move execution
    - `game.get_legal_moves()` - Legal move validation
    - `PolicyOutputMapper.get_legal_mask()` - Action masking
  * **Preconditions:** 
    - Valid agent and opponent instances
    - Functional logger instance
    - Positive number of games and maximum moves
  * **Postconditions:** 
    - Returns complete evaluation statistics
    - All games logged with outcomes
    - Agents returned to original state
  * **Algorithmic Note:**
    - Alternates between agent (Black/Sente) and opponent (White/Gote) moves
    - Validates all moves against legal move sets
    - Handles timeouts through maximum move limits
    - Computes statistics incrementally for memory efficiency
  * **Usage Example:**
    ```python
    results = run_evaluation_loop(
        agent_to_eval=trained_agent,
        opponent=random_opponent,
        num_games=50,
        logger=eval_logger,
        max_moves_per_game=300
    )
    ```

---

### 5. Shared or Complex Data Structures 📊

* **Structure Name:** `Game Flow Control`
  * **Type:** `Game state management and move validation`
  * **Purpose:** Orchestrate proper game execution with legal move enforcement
  * **Format:** Procedural game loop with state checks
  * **Fields:**
    - Legal move generation and validation
    - Player alternation logic (Sente/Gote)
    - Move execution with error handling
    - Game termination detection
  * **Validation Constraints:**
    - All moves must be in legal move set
    - Game state must be valid after each move
    - Maximum move limit enforced
  * **Used In:** Main evaluation loop, game outcome determination

* **Structure Name:** `Statistical Accumulation`
  * **Type:** `Running statistics computation`
  * **Purpose:** Track and compute evaluation metrics across multiple games
  * **Format:** Incremental counters and computed ratios
  * **Fields:**
    - Game outcome counters (wins, losses, draws)
    - Move count tracking for average game length
    - Rate computations (win rate, loss rate, draw rate)
  * **Validation Constraints:**
    - Rates must sum to 1.0 (within floating point precision)
    - Counts must be non-negative integers
    - Average game length must be positive
  * **Used In:** Result compilation, performance reporting

---

### 6. Inter-Module Relationships & Data Flow 🔄

* **Dependency Graph (Internal):**
  ```
  loop.py
    ├── uses → core/ppo_agent.py (agent evaluation)
    ├── uses → shogi/shogi_game.py (game engine)
    ├── uses → utils/BaseOpponent (opponent interface)
    ├── uses → utils/EvaluationLogger (logging)
    ├── uses → utils/PolicyOutputMapper (action mapping)
    └── used by → evaluation/evaluate.py (orchestration)
  ```

* **Cross-Folder Imports:**
  - Core PPO agent functionality from `/core/`
  - Shogi game engine from `/shogi/`
  - Utility classes from `/utils/`
  - Called by evaluation orchestrator

* **Data Flow Summary:**
  - Agent/Opponent → game moves → game engine → game outcomes
  - Game outcomes → statistical accumulation → ResultsDict
  - Progress updates → evaluation logger → log files/console
  - Final results → evaluation orchestrator → W&B/reporting systems

---

### 7. Non-Functional Aspects 🛠️

* **Performance:**
  - Scales linearly with number of games
  - Memory usage constant per game (games not stored)
  - Move validation overhead minimal with legal move caching

* **Security:**
  - No external input validation required (internal component)
  - Move validation through game engine prevents invalid states

* **Error Handling & Logging:**
  - Graceful handling of illegal moves with game termination
  - Comprehensive logging of game progress and outcomes
  - Exception handling for game engine errors

* **Scalability Concerns:**
  - Designed for single-threaded evaluation loops
  - Can be parallelized at evaluation orchestrator level
  - Memory usage independent of number of games

* **Testing & Instrumentation:**
  - Unit tests for statistical computation accuracy
  - Integration tests with mock agents and opponents
  - Performance profiling for large-scale evaluations

---

### 8. Configuration & Environment ♻️

* **Environment Variables:**
  - None directly; inherits device configuration from agents

* **CLI Interfaces / Entry Points:**
  - Not applicable (internal component called by evaluation orchestrator)

* **Config File Schema:**
  - Evaluation parameters configured in main evaluation module:
    ```yaml
    evaluation:
      num_games: 100
      max_moves_per_game: 300
      log_detailed_games: true
    ```

---

### 9. Glossary 📖

* **Sente:** Black player (first player) in Shogi, typically the agent being evaluated
* **Gote:** White player (second player) in Shogi, typically the opponent
* **Tournament:** Systematic evaluation involving multiple games with statistical analysis
* **Win Rate:** Percentage of games won by the evaluated agent
* **Draw Rate:** Percentage of games ending without a clear winner
* **Game Length:** Number of moves played in a single game

---

### 10. Known Issues, TODOs, Future Work 🧭

* **Known Issues:**
  - No support for parallel game execution
  - Limited error recovery for game engine failures
  - Move timeout not implemented (relies on max_moves_per_game)

* **TODOs / Deferred Features:**
  - Implement parallel game execution for faster evaluation
  - Add move timing and timeout handling
  - Support for tournament brackets with multiple opponents
  - Detailed move-by-move analysis and logging
  - Statistical significance testing for results

* **Suggested Refactors:**
  - Extract game execution logic to separate function
  - Add support for different game configurations
  - Implement result serialization for offline analysis
  - Add progress reporting with estimated completion times

---

## Notes for AI/Agent Developers 🧠

1. **Player Alternation:** The loop correctly handles Shogi player alternation with agent as Sente (Black) and opponent as Gote (White)
2. **Legal Move Validation:** All moves are validated against legal move sets before execution to prevent invalid game states
3. **Statistical Accuracy:** Results computation handles edge cases like zero games played and maintains floating-point precision
4. **Extensibility:** The opponent interface allows easy integration of new opponent types without modifying the core loop
