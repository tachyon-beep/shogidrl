# Test Mapping Document

This document maps the existing tests across different test files to help identify duplication, deprecated files, and gaps in test coverage.

## Shogi Game Tests

| Functionality                        | test_shogi_game_core_logic.py           | test_shogi_game_mock_comprehensive.py (DEPRECATED) | test_make_undo_move.py | Other Files                          | Deprecated/Notes                |
|--------------------------------------|------------------------------------------|---------------------------------------------------|-----------------------|--------------------------------------|-------------------------------|
| Game initialization                  | ✓ (implicit in fixtures)                 | ✓ test_game_initialization                        |                       |                                      | DEPRECATED                     |
| Game reset                           | ✓ (partial, in fixtures)                 | ✓ test_game_reset                                 |                       |                                      | DEPRECATED                     |
| Get/set piece                        | ✓ (implicit in fixtures)                 | ✓ test_get_set_piece                              |                       |                                      | DEPRECATED                     |
| Is on board                          | ❌                                        | ✓ test_is_on_board                                |                       | test_shogi_engine_integration.py     | DEPRECATED                     |
| Make move (basic)                    | ❌                                        | ✓ test_make_move_basic                            |                       |                                      | DEPRECATED                     |
| Make move (capture)                  | ❌                                        | ✓ test_make_move_capture                          |                       |                                      | DEPRECATED                     |
| Make move (promotion)                | ❌                                        | ✓ test_make_move_promotion                        |                       |                                      | DEPRECATED                     |
| Make move (piece drop)               | ❌                                        | ✓ test_make_move_piece_drop                       |                       |                                      | DEPRECATED                     |
| Undo move (basic)                    | ✓ test_undo_move_simple_board_move        | ✓ test_undo_basic_move                            | ✓                     |                                      | DEPRECATED (partial)           |
| Undo move (capture)                  | ✓ test_undo_move_capture                 | ✓ test_undo_capture_move                          | ✓                     |                                      | DEPRECATED (partial)           |
| Undo move (promotion)                | ✓ test_undo_move_promotion_no_capture    | ✓ test_undo_promotion_move                        |                       |                                      | DEPRECATED (partial)           |
| Undo move (drop)                     | ✓ test_undo_move_drop                    | ✓ test_undo_drop_move                             |                       |                                      | DEPRECATED (partial)           |
| Observation handling                 | ✓ Multiple observation tests             | (see deprecated/test_observation_constants.py)     |                       | test_observation_constants.py (DEPRECATED) | DEPRECATED/merged         |
| SFEN handling                        | ✓ Multiple SFEN tests                    |                                                   |                       |                                      |                                 |
| Move limit                           | ❌                                        | ✓ test_move_limit                                 |                       |                                      | DEPRECATED                     |

## Shogi Game I/O Tests

| Functionality                        | test_shogi_game_observation_and_io.py (Active) | Deprecated/Other Files                  | Notes                                 |
|--------------------------------------|------------------------------------------|-----------------------------------------|---------------------------------------|
| Neural network observation           | ✓ Multiple comprehensive tests           | deprecated/test_observation_constants.py| All new coverage in test_shogi_game_observation_and_io.py |
| Text representation                  | ✓ Multiple tests for different states    |                                         |                                       |
| KIF export                           | ✓ test_game_to_kif_writes_valid_kif_file_after_moves |                                         |                                       |
| SFEN move parsing                    | ✓ test_sfen_to_move_tuple_parses_standard_and_drop_moves |                                         |                                       |
| Helper functions                     | ✓ test_parse_sfen_square_parses_various_squares, etc. |                                         |                                       |

## Shogi Rules Logic Tests

| Functionality                        | test_shogi_rules_and_validation.py       | Other Files                          | Deprecated/Notes                |
|--------------------------------------|------------------------------------------|--------------------------------------|-------------------------------|
| Piece drop validation                | ✓ Multiple tests                         |                                      |                                 |
| Move generation                      | ✓ Multiple tests                         | test_shogi_engine_integration.py (partial) |                                 |
| Check detection                      | ✓ (implicit in other tests)              |                                      |                                 |
| Uchi-fu-zume (pawn drop mate)        | ✓ Multiple detailed tests                |                                      |                                 |

## Deprecated Tests

The following test files have been moved to `tests/deprecated/` and are no longer maintained:
- test_shogi_game_mock_comprehensive.py
- test_shogi_game_mocked.py
- test_shogi_game_updated_with_mocks.py
- test_shogi_updated.py
- test_shogi_with_mocks_example.py
- test_observation_constants.py

These files may contain redundant or outdated tests. All active coverage for neural network observation, text representation, and I/O is now in `test_shogi_game_io.py`.

## Duplicated Tests To Address

1. **Undo Move Tests**: Redundancy across test_shogi_game.py, test_shogi_game_mock_comprehensive.py (deprecated), and test_make_undo_move.py. Only test_shogi_game.py and test_make_undo_move.py are active.
2. **Observation Tests**: Some overlap between test_shogi_game.py and test_observation_constants.py (deprecated).

## Coverage Gaps To Address

1. **Game I/O**: Previously a gap, now addressed with new test_shogi_game_io.py.
2. **Core Game Logic**: Despite many tests, coverage is only 1% for shogi_game.py, suggesting incomplete mock integration or missing edge cases. Further targeted tests may be needed.

## Summary
- All deprecated and legacy test files are now isolated in `tests/deprecated/`.
- All active coverage for I/O and observation is in `test_shogi_game_io.py`.
- No new major gaps have been created; coverage has improved and is more maintainable.
- Some duplication remains in undo/observation tests, but is now clearly documented.
