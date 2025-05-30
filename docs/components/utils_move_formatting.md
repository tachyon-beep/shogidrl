# Move Formatting Module

## Module Overview

The `utils/move_formatting.py` module provides utilities for formatting Shogi moves into human-readable descriptions with both USI notation and English explanations. This module serves the crucial role of translating internal move representations into meaningful text for logging, demo mode, debugging, and user interfaces. It includes comprehensive piece name mappings with Japanese names and English translations, making the system accessible to both traditional Shogi players and international audiences.

## Dependencies

### Internal Dependencies
- `keisei.shogi.shogi_core_definitions`: PieceType enum and core type definitions

### External Dependencies
None - this is a pure utility module with no external dependencies.

## Function Documentation

### format_move_with_description

Formats a Shogi move with USI notation and English description, with optional game context for enhanced piece identification.

**Signature:**
```python
def format_move_with_description(
    selected_shogi_move, 
    policy_output_mapper, 
    game=None
) -> str
```

**Parameters:**
- `selected_shogi_move` - MoveTuple (BoardMoveTuple or DropMoveTuple)
- `policy_output_mapper` - PolicyOutputMapper instance for USI conversion
- `game` - Optional ShogiGame instance for piece information

**Returns:**
- `str` - Formatted string like "7g7f (pawn move to 7f)" or "P*5e (pawn drop to 5e)"

**Functionality:**
1. **Input Validation**: Handles None moves gracefully
2. **USI Conversion**: Uses policy mapper to get USI notation
3. **Move Type Detection**: Distinguishes between board moves and drops
4. **Piece Identification**: Uses game context when available for accurate piece names
5. **Description Generation**: Creates human-readable move descriptions

### format_move_with_description_enhanced

Enhanced version of move formatting that accepts piece information as a parameter, optimized for demo mode logging where piece information is pre-fetched.

**Signature:**
```python
def format_move_with_description_enhanced(
    selected_shogi_move, 
    policy_output_mapper, 
    piece_info=None
) -> str
```

**Parameters:**
- `selected_shogi_move` - MoveTuple (BoardMoveTuple or DropMoveTuple)
- `policy_output_mapper` - PolicyOutputMapper instance for USI conversion
- `piece_info` - Pre-fetched piece object from game.get_piece()

**Returns:**
- `str` - Formatted string like "7g7f - Fuhyō (Pawn) moving from 7g to 7f."

**Enhancements:**
- **Performance**: Avoids redundant piece lookups during demo mode
- **Accuracy**: Uses pre-validated piece information
- **Consistency**: Ensures piece information matches the actual move

### _get_piece_name (Internal)

Converts PieceType enum to Japanese name with English translation, including promotion transformations.

**Signature:**
```python
def _get_piece_name(piece_type, is_promoting=False) -> str
```

**Parameters:**
- `piece_type` - PieceType enum value
- `is_promoting` - Boolean indicating if the piece is promoting during this move

**Returns:**
- `str` - Piece name like "Fuhyō (Pawn)" or "Fuhyō (Pawn) → Tokin (Promoted Pawn)"

**Piece Name Mappings:**
```python
piece_names = {
    PieceType.PAWN: "Fuhyō (Pawn)",
    PieceType.LANCE: "Kyōsha (Lance)",
    PieceType.KNIGHT: "Keima (Knight)",
    PieceType.SILVER: "Ginsho (Silver General)",
    PieceType.GOLD: "Kinshō (Gold General)",
    PieceType.BISHOP: "Kakugyō (Bishop)",
    PieceType.ROOK: "Hisha (Rook)",
    PieceType.KING: "Ōshō (King)",
    PieceType.PROMOTED_PAWN: "Tokin (Promoted Pawn)",
    PieceType.PROMOTED_LANCE: "Narikyo (Promoted Lance)",
    PieceType.PROMOTED_KNIGHT: "Narikei (Promoted Knight)",
    PieceType.PROMOTED_SILVER: "Narigin (Promoted Silver)",
    PieceType.PROMOTED_BISHOP: "Ryūma (Dragon Horse)",
    PieceType.PROMOTED_ROOK: "Ryūō (Dragon King)"
}
```

### _coords_to_square_name (Internal)

Converts 0-indexed coordinates to Shogi square notation.

**Signature:**
```python
def _coords_to_square_name(row, col) -> str
```

**Parameters:**
- `row` - 0-indexed row coordinate (0-8)
- `col` - 0-indexed column coordinate (0-8)

**Returns:**
- `str` - Square name like "7f" (file + rank)

**Coordinate System:**
- **Files**: 9-1 from left to right (9-col conversion)
- **Ranks**: a-i from top to bottom (chr(ord("a") + row))

## Data Structures

### Move Type Classifications

**Board Move Format:**
```python
BoardMoveTuple = (from_r, from_c, to_r, to_c, promote_flag)
# Example: (6, 6, 5, 6, False) -> "7g7f - piece moving from 7g to 7f"
```

**Drop Move Format:**
```python
DropMoveTuple = (None, None, to_r, to_c, piece_type)
# Example: (None, None, 4, 4, PieceType.PAWN) -> "P*5e - Fuhyō (Pawn) drop to 5e"
```

### Promotion Transformations
```python
promotion_mappings = {
    PieceType.PAWN: "Fuhyō (Pawn) → Tokin (Promoted Pawn)",
    PieceType.LANCE: "Kyōsha (Lance) → Narikyo (Promoted Lance)",
    PieceType.KNIGHT: "Keima (Knight) → Narikei (Promoted Knight)",
    PieceType.SILVER: "Ginsho (Silver General) → Narigin (Promoted Silver)",
    PieceType.BISHOP: "Kakugyō (Bishop) → Ryūma (Dragon Horse)",
    PieceType.ROOK: "Hisha (Rook) → Ryūō (Dragon King)"
}
```

## Inter-Module Relationships

### Core Integration
- **Shogi Definitions**: Uses PieceType enum for piece identification
- **Policy Mapping**: Integrates with PolicyOutputMapper for USI conversion
- **Game Logic**: Optional integration with ShogiGame for piece context

### User Interface Integration
- **Demo Mode**: Primary consumer for enhanced move formatting
- **Training Display**: Uses basic move formatting for progress logs
- **Evaluation Logging**: Formats moves for evaluation game records
- **Debugging Tools**: Provides readable move representations for debugging

### Internationalization
- **Japanese Names**: Authentic Japanese piece names for traditional players
- **English Translations**: Clear English descriptions for international users
- **USI Compatibility**: Standard USI notation for interoperability

## Implementation Notes

### Error Handling Strategy
```python
try:
    # Attempt formatting with full context
    return f"{usi_notation} - {description}."
except Exception as e:
    # Fallback to string representation
    return f"{str(selected_shogi_move)} (format error: {e})"
```

### Performance Optimizations
- **Piece Info Caching**: Enhanced version avoids redundant game queries
- **Simple Lookups**: Direct dictionary lookups for piece names
- **Minimal String Operations**: Efficient string formatting and concatenation

### Cultural Sensitivity
- **Authentic Names**: Uses proper Japanese romanization
- **Respectful Translations**: Maintains traditional terminology
- **Balanced Approach**: Provides both Japanese and English information

## Testing Strategy

### Unit Tests
```python
def test_format_board_move():
    """Test formatting of board moves."""
    # Test normal moves
    # Test promotion moves
    # Test edge cases (corners, etc.)
    pass

def test_format_drop_move():
    """Test formatting of drop moves."""
    # Test all droppable piece types
    # Test various board positions
    pass

def test_piece_name_mappings():
    """Test piece name generation."""
    # Test all piece types
    # Test promotion transformations
    # Test unknown piece types
    pass

def test_coordinate_conversion():
    """Test coordinate to square name conversion."""
    # Test all board positions
    # Test boundary conditions
    pass
```

### Integration Tests
```python
def test_with_policy_mapper():
    """Test integration with PolicyOutputMapper."""
    # Test USI conversion integration
    # Test error handling for invalid moves
    pass

def test_with_game_context():
    """Test formatting with game context."""
    # Test piece identification from game state
    # Test error handling for invalid game state
    pass
```

### Localization Tests
```python
def test_japanese_names():
    """Test Japanese piece name accuracy."""
    # Verify authentic Japanese romanization
    # Test special characters and diacritics
    pass

def test_english_translations():
    """Test English translation clarity."""
    # Verify clear and accurate translations
    # Test consistency across piece types
    pass
```

## Performance Considerations

### Formatting Efficiency
- **Direct Lookups**: Dictionary-based piece name lookups
- **Minimal Computation**: Simple string operations and formatting
- **Caching Strategy**: Enhanced version eliminates redundant piece queries

### Memory Usage
- **Static Data**: Piece name mappings stored as constants
- **No State**: Functions are stateless for thread safety
- **Minimal Allocation**: Efficient string formatting without excessive allocation

## Security Considerations

### Input Validation
- **Move Validation**: Graceful handling of malformed moves
- **Type Safety**: Proper type checking for piece types and coordinates
- **Error Isolation**: Exception handling prevents crashes from invalid input

### Data Integrity
- **Immutable Mappings**: Piece name mappings are immutable
- **Consistent Output**: Deterministic formatting for identical inputs
- **Safe Fallbacks**: Error conditions result in safe string representations

## Error Handling

### Exception Categories
```python
# Formatting errors - graceful degradation
except Exception as e:
    return f"{str(selected_shogi_move)} (format error: {e})"

# Coordinate errors - handled in _coords_to_square_name
if not (0 <= r <= 8 and 0 <= c <= 8):
    # Return error indication or raise exception

# Piece type errors - handled in _get_piece_name  
return piece_names.get(piece_type, str(piece_type))
```

### Recovery Strategies
- **Graceful Degradation**: Always returns some string representation
- **Error Context**: Includes original move data in error fallbacks
- **Diagnostic Information**: Error messages aid in debugging

## Configuration

### Customization Points
- **Piece Names**: Piece name mappings can be customized for different languages
- **Format Strings**: Output format can be modified for different use cases
- **Coordinate System**: Square naming convention can be adjusted

### Language Support
- **Japanese**: Traditional Japanese piece names with proper romanization
- **English**: Clear English translations and descriptions
- **Extensible**: Structure supports additional language mappings

## Future Enhancements

### Planned Features
- **Multi-Language Support**: Additional language mappings beyond Japanese/English
- **Format Customization**: Configurable output formats for different contexts
- **Enhanced Context**: Integration with more game state information
- **Abbreviation Support**: Short form notation for compact displays

### Internationalization
- **Unicode Support**: Full Unicode support for non-ASCII characters
- **Cultural Variants**: Support for different cultural naming conventions
- **Localization**: Complete localization framework for all text

## Usage Examples

### Basic Move Formatting
```python
from keisei.utils.move_formatting import format_move_with_description

# Format a board move
move = (6, 6, 5, 6, False)  # 7g to 7f
formatted = format_move_with_description(move, policy_mapper)
# Result: "7g7f - piece moving from 7g to 7f."

# Format a drop move  
drop = (None, None, 4, 4, PieceType.PAWN)  # Pawn drop to 5e
formatted = format_move_with_description(drop, policy_mapper)
# Result: "P*5e - Fuhyō (Pawn) drop to 5e."
```

### Enhanced Demo Mode Formatting
```python
from keisei.utils.move_formatting import format_move_with_description_enhanced

# Pre-fetch piece information for performance
piece_info = game.get_piece(from_r, from_c)

# Format with enhanced context
formatted = format_move_with_description_enhanced(
    move, 
    policy_mapper, 
    piece_info
)
# Result: "7g7f - Fuhyō (Pawn) moving from 7g to 7f."
```

### Promotion Move Formatting
```python
# Promotion move
promo_move = (6, 0, 0, 0, True)  # Lance promotion

# With game context
formatted = format_move_with_description(promo_move, policy_mapper, game)
# Result: "9g9a+ - Kyōsha (Lance) → Narikyo (Promoted Lance) moving from 9g to 9a."
```

## Maintenance Notes

### Code Quality
- Maintain consistent formatting patterns across all functions
- Keep piece name mappings up to date with any PieceType changes
- Document any changes to coordinate system or USI conventions
- Follow consistent error handling patterns

### Dependencies
- Monitor changes to PieceType enum in shogi_core_definitions
- Keep USI conversion compatibility with PolicyOutputMapper
- Update type hints with Python version upgrades
- Maintain compatibility with ShogiGame interface changes

### Testing Requirements
- Comprehensive unit test coverage for all formatting functions
- Integration tests with real game scenarios
- Localization tests for Japanese/English accuracy
- Performance tests for formatting efficiency
