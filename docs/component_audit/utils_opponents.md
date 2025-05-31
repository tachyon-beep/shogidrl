# Opponents Module

## Module Overview

The `utils/opponents.py` module provides concrete implementations of opponent classes for evaluation and testing in the Keisei Shogi system. This module implements the BaseOpponent interface with two distinct strategies: random move selection for baseline evaluation and simple heuristic-based play for more challenging evaluation scenarios. These opponents serve as benchmarks for measuring agent performance and provide consistent evaluation targets throughout training.

## Dependencies

### External Dependencies
- `random`: Random number generation for move selection
- `typing`: Type hints for lists and other types

### Internal Dependencies
- `keisei.shogi.shogi_core_definitions`: MoveTuple, PieceType enum definitions
- `keisei.shogi.shogi_game`: ShogiGame class for game state interaction
- `keisei.utils.utils`: BaseOpponent abstract base class

## Class Documentation

### SimpleRandomOpponent

A basic opponent implementation that selects moves randomly from all legal options, providing a baseline for evaluation.

**Inheritance:**
- Extends `BaseOpponent` abstract base class

**Attributes:**
- `name: str` - Opponent identifier (default: "SimpleRandomOpponent")

**Key Methods:**

#### `__init__(name: str = "SimpleRandomOpponent")`
Initializes the random opponent with an optional custom name.

#### `select_move(game_instance: ShogiGame) -> MoveTuple`
**Purpose:** Selects a random legal move from the current game state.

**Algorithm:**
1. **Legal Move Retrieval**: Gets all legal moves from game instance
2. **Availability Check**: Validates that legal moves exist
3. **Random Selection**: Uses `random.choice()` for uniform random selection
4. **Error Handling**: Raises ValueError if no legal moves available

**Performance Characteristics:**
- **Uniform Distribution**: All legal moves have equal probability
- **No Game Knowledge**: No consideration of board position or piece values
- **Fast Execution**: O(1) selection after legal move generation
- **Consistent Baseline**: Provides repeatable baseline performance

### SimpleHeuristicOpponent

An enhanced opponent that uses simple heuristics to make more strategic move decisions, providing a more challenging evaluation target.

**Inheritance:**
- Extends `BaseOpponent` abstract base class

**Attributes:**
- `name: str` - Opponent identifier (default: "SimpleHeuristicOpponent")

**Key Methods:**

#### `__init__(name: str = "SimpleHeuristicOpponent")`
Initializes the heuristic opponent with an optional custom name.

#### `select_move(game_instance: ShogiGame) -> MoveTuple`
**Purpose:** Selects a move using simple heuristic evaluation with move prioritization.

**Heuristic Strategy:**

**1. Move Classification:**
```python
capturing_moves: List[MoveTuple] = []
non_promoting_pawn_moves: List[MoveTuple] = []
other_moves: List[MoveTuple] = []
```

**2. Capture Detection:**
- Identifies moves that capture opponent pieces
- Checks destination square for enemy pieces
- Prioritizes material gain

**3. Pawn Move Analysis:**
- Identifies pawn moves without promotion
- Considers pawn advancement as secondary priority
- Excludes promoted pawn moves from this category

**4. Move Prioritization (in order):**
```python
if capturing_moves:
    return random.choice(capturing_moves)       # Highest priority
if non_promoting_pawn_moves:
    return random.choice(non_promoting_pawn_moves)  # Medium priority
if other_moves:
    return random.choice(other_moves)           # Lower priority
return random.choice(legal_moves)               # Fallback
```

**Heuristic Logic:**
- **Capture Priority**: Material advantage through captures
- **Pawn Development**: Advance pawns for positional advantage
- **Random Selection**: Within each category, maintains unpredictability
- **Fallback Safety**: Ensures valid move selection in all scenarios

## Data Structures

### Move Classification Categories
```python
move_categories = {
    "capturing_moves": List[MoveTuple],          # Moves that capture pieces
    "non_promoting_pawn_moves": List[MoveTuple], # Pawn advances without promotion
    "other_moves": List[MoveTuple]               # All other legal moves
}
```

### Opponent Interface
```python
class OpponentProtocol:
    name: str
    
    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        """Select a move given the current game state."""
        pass
```

## Inter-Module Relationships

### Core Integration
- **BaseOpponent**: Implements the abstract opponent interface
- **ShogiGame**: Interacts with game state for move generation and board analysis
- **Move Definitions**: Uses MoveTuple types for move representation

### Evaluation Integration
- **Evaluation System**: Primary consumers for opponent instances
- **Agent Testing**: Provides benchmarks for agent performance measurement
- **Training Validation**: Used for periodic evaluation during training

### Game Logic Integration
- **Legal Move Generation**: Relies on ShogiGame for legal move lists
- **Board State Analysis**: Accesses board state for capture detection
- **Piece Identification**: Uses piece type information for heuristic decisions

## Implementation Notes

### Random Opponent Design
- **Simplicity**: Minimal implementation for baseline evaluation
- **Consistency**: Deterministic behavior given the same random seed
- **Performance**: Efficient O(1) move selection
- **Baseline Value**: Provides lower bound for agent performance

### Heuristic Opponent Design
- **Strategic Thinking**: Basic tactical awareness through capture prioritization
- **Balanced Approach**: Combines strategic goals with randomness
- **Extensibility**: Clear structure for adding additional heuristics
- **Educational Value**: Demonstrates basic Shogi strategic principles

### Move Analysis Implementation
```python
# Capture detection logic
destination_piece = game_instance.board[to_r][to_c]
if (destination_piece is not None and 
    destination_piece.color != game_instance.current_player):
    is_capture = True

# Pawn move identification
source_piece = game_instance.board[from_r][from_c]
if (source_piece and 
    source_piece.type == PieceType.PAWN and 
    not promote):
    is_pawn_move_no_promo = True
```

## Testing Strategy

### Unit Tests
```python
def test_random_opponent_move_selection():
    """Test random opponent move selection."""
    # Test with various game states
    # Verify move legality
    # Test error handling for no legal moves
    pass

def test_heuristic_opponent_prioritization():
    """Test heuristic opponent move prioritization."""
    # Test capture prioritization
    # Test pawn move preferences
    # Test fallback behavior
    pass

def test_opponent_interface_compliance():
    """Test that opponents implement BaseOpponent correctly."""
    # Test interface compliance
    # Test name property
    # Test select_move signature
    pass
```

### Integration Tests
```python
def test_opponents_with_real_games():
    """Test opponents in actual game scenarios."""
    # Test full game completion
    # Test performance against each other
    # Verify game state integrity
    pass

def test_evaluation_integration():
    """Test opponents in evaluation framework."""
    # Test with evaluation system
    # Verify results collection
    # Test performance metrics
    pass
```

### Performance Tests
```python
def test_move_selection_performance():
    """Test opponent move selection speed."""
    # Benchmark random opponent
    # Benchmark heuristic opponent
    # Compare selection times
    pass
```

## Performance Considerations

### Random Opponent Performance
- **O(1) Selection**: Constant time move selection after legal move generation
- **Minimal Memory**: No state or history maintenance
- **Fast Execution**: Suitable for high-volume evaluation

### Heuristic Opponent Performance
- **O(n) Analysis**: Linear scan of legal moves for classification
- **Bounded Complexity**: Simple heuristics with predictable cost
- **Reasonable Speed**: Acceptable for evaluation scenarios

### Optimization Opportunities
- **Move Caching**: Cache legal move analysis within a turn
- **Precomputed Heuristics**: Pre-compute common heuristic patterns
- **Parallel Evaluation**: Support for parallel opponent evaluation

## Security Considerations

### Input Validation
- **Game State Validation**: Relies on ShogiGame for valid game states
- **Move Legality**: Only selects from legal moves provided by game
- **Error Handling**: Proper handling of edge cases and invalid states

### Opponent Isolation
- **Stateless Design**: No persistent state that could be corrupted
- **Pure Functions**: Move selection based only on current game state
- **No Side Effects**: Opponents don't modify game state

## Error Handling

### Exception Management
```python
# Random opponent error handling
if not legal_moves:
    raise ValueError(
        "No legal moves available for SimpleRandomOpponent, game should be over."
    )

# Heuristic opponent error handling
if not legal_moves:
    raise ValueError(
        "No legal moves available for SimpleHeuristicOpponent, game should be over."
    )
```

### Error Recovery
- **Graceful Degradation**: Fallback to random selection when heuristics fail
- **Clear Error Messages**: Descriptive error messages for debugging
- **State Preservation**: No state corruption on error conditions

## Configuration

### Opponent Customization
```python
# Custom opponent names
random_opponent = SimpleRandomOpponent(name="Baseline_Random")
heuristic_opponent = SimpleHeuristicOpponent(name="Tactical_AI")

# No other configuration required - opponents are self-contained
```

### Evaluation Configuration
- Opponents are configured through the evaluation system
- No internal configuration parameters
- Behavior is deterministic given game state and random seed

## Future Enhancements

### Planned Improvements
- **Advanced Heuristics**: More sophisticated evaluation functions
- **Opening Books**: Integration with opening move databases
- **Endgame Tables**: Specialized endgame evaluation
- **Strength Levels**: Configurable difficulty levels

### Extensibility Points
- **Custom Heuristics**: Plugin architecture for custom evaluation functions
- **Learning Opponents**: Opponents that adapt based on game history
- **Statistical Analysis**: Detailed move statistics and pattern recognition
- **Multi-Agent Support**: Coordination between multiple opponent types

### Advanced Opponent Types
- **Monte Carlo Opponents**: MCTS-based move selection
- **Rule-Based Opponents**: Complex rule systems for strategic play
- **Hybrid Opponents**: Combination of multiple strategies
- **Adaptive Opponents**: Opponents that adjust to player strength

## Usage Examples

### Basic Opponent Creation
```python
from keisei.utils.opponents import SimpleRandomOpponent, SimpleHeuristicOpponent

# Create basic random opponent
random_opponent = SimpleRandomOpponent()

# Create heuristic opponent with custom name
strategic_opponent = SimpleHeuristicOpponent("Strategic_AI_v1")
```

### Evaluation Setup
```python
# Set up opponents for evaluation
opponents = {
    "random": SimpleRandomOpponent("Random_Baseline"),
    "heuristic": SimpleHeuristicOpponent("Heuristic_Challenge")
}

# Use in evaluation loop
for name, opponent in opponents.items():
    results = evaluate_agent_vs_opponent(agent, opponent, num_games=100)
    print(f"Results vs {name}: {results}")
```

### Game Simulation
```python
# Simulate game between opponents
game = ShogiGame()
player1 = SimpleRandomOpponent("Random_1")
player2 = SimpleHeuristicOpponent("Heuristic_1")

while not game.is_game_over():
    current_player = player1 if game.current_player == Color.BLACK else player2
    move = current_player.select_move(game)
    game.make_move(move)

print(f"Game result: {game.get_result()}")
```

### Performance Benchmarking
```python
import time

# Benchmark opponent performance
def benchmark_opponent(opponent, game_states, iterations=1000):
    start_time = time.time()
    
    for _ in range(iterations):
        for game_state in game_states:
            move = opponent.select_move(game_state)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / (iterations * len(game_states))
    print(f"{opponent.name}: {avg_time:.4f}s per move")

# Run benchmarks
benchmark_opponent(SimpleRandomOpponent(), test_positions)
benchmark_opponent(SimpleHeuristicOpponent(), test_positions)
```

## Maintenance Notes

### Code Quality
- Maintain clear separation between random and heuristic logic
- Document any changes to heuristic evaluation criteria
- Follow consistent error handling patterns
- Keep move analysis logic efficient and readable

### Dependencies
- Monitor BaseOpponent interface changes
- Keep ShogiGame integration up to date
- Update move type definitions with core changes
- Maintain compatibility with evaluation system

### Testing Requirements
- Comprehensive unit test coverage for both opponent types
- Integration tests with real game scenarios
- Performance benchmarks for move selection speed
- Regression tests for heuristic behavior changes
