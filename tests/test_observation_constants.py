"""
Test for the observation plane constants defined in shogi_core_definitions.py
"""

# DEPRECATED: Observation plane constant and initial state tests are now covered in
# 'test_shogi_game_updated_with_mocks.py'. This file is retained for reference only.

import numpy as np

from keisei.shogi.shogi_core_definitions import (
    OBS_CURR_PLAYER_HAND_START,
    OBS_CURR_PLAYER_INDICATOR,
    OBS_CURR_PLAYER_PROMOTED_START,
    OBS_CURR_PLAYER_UNPROMOTED_START,
    OBS_MOVE_COUNT,
    OBS_OPP_PLAYER_HAND_START,
    OBS_OPP_PLAYER_PROMOTED_START,
    OBS_OPP_PLAYER_UNPROMOTED_START,
    OBS_PROMOTED_ORDER,
    OBS_RESERVED_1,
    OBS_RESERVED_2,
)
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_game_io import generate_neural_network_observation

INPUT_CHANNELS = 46  # Use the default from config_schema for tests


def test_observation_plane_constants_match_implementation():
    """Test that the observation plane constants match the values used in the implementation."""
    game = ShogiGame()

    # Get observation
    obs = generate_neural_network_observation(game)

    # Test the shape based on constants - we have config.INPUT_CHANNELS planes total (through OBS_RESERVED_2 which is index 45)
    expected_num_planes = OBS_RESERVED_2 + 1
    assert (
        obs.shape[0] == expected_num_planes
    ), f"Expected {expected_num_planes} planes, got {obs.shape[0]}"

    # Testing dimensions
    assert obs.shape[1] == 9, "Observation should have 9 rows"
    assert obs.shape[2] == 9, "Observation should have 9 columns"

    # Verify constants match the actual structure
    # 1. Verify board piece channels start at the correct positions (by looking at initial board setup)
    # Black pawns are at row 6, all columns in the initial board setup
    # They should be on the current player's unpromoted plane (since it's Black's turn initially)
    for col in range(9):
        assert (
            obs[OBS_CURR_PLAYER_UNPROMOTED_START, 6, col] > 0.9
        ), f"Black pawn not found at (6,{col})"

    # Similarly, white pawns should be on the opponent's unpromoted plane
    for col in range(9):
        assert (
            obs[OBS_OPP_PLAYER_UNPROMOTED_START, 2, col] > 0.9
        ), f"White pawn not found at (2,{col})"

    # 2. There should be no promoted pieces initially
    num_promoted_types = len(OBS_PROMOTED_ORDER)
    for r in range(9):
        for c in range(9):
            assert np.all(
                obs[
                    OBS_CURR_PLAYER_PROMOTED_START : OBS_CURR_PLAYER_PROMOTED_START
                    + num_promoted_types,
                    r,
                    c,
                ]
                < 0.1
            ), f"Found promoted piece for current player at ({r},{c})"
            assert np.all(
                obs[
                    OBS_OPP_PLAYER_PROMOTED_START : OBS_OPP_PLAYER_PROMOTED_START
                    + num_promoted_types,
                    r,
                    c,
                ]
                < 0.1
            ), f"Found promoted piece for opponent at ({r},{c})"

    # 3. Hands should be empty initially
    for i in range(7):
        assert np.all(
            obs[OBS_CURR_PLAYER_HAND_START + i] < 0.1
        ), f"Current player hand channel {i} not empty"
        assert np.all(
            obs[OBS_OPP_PLAYER_HAND_START + i] < 0.1
        ), f"Opponent hand channel {i} not empty"

    # 4. Current player indicator should show it's Black's turn (1.0 for Black)
    assert np.all(
        obs[OBS_CURR_PLAYER_INDICATOR] > 0.9
    ), "Current player indicator should be 1.0 for Black"

    # 5. Move count should be 0 initially (normalized by max_moves_per_game)
    assert np.all(
        obs[OBS_MOVE_COUNT] < 0.01
    ), "Move count should be close to 0 initially"

    # 6. Reserved channels should all be zero
    assert np.all(obs[OBS_RESERVED_1] < 0.01), "Reserved channel 1 should be all zeros"
    assert np.all(obs[OBS_RESERVED_2] < 0.01), "Reserved channel 2 should be all zeros"
