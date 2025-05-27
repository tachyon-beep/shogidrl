"""
test_features.py: Unit tests for keisei/shogi/features.py
"""
import numpy as np
import types
import pytest
from keisei.shogi import features

class DummyGame:
    # Minimal stub for testing feature extraction
    def __init__(self):
        self.board = [[None for _ in range(9)] for _ in range(9)]
        self.OBS_CURR_PLAYER_UNPROMOTED_START = 0
        self.OBS_CURR_PLAYER_PROMOTED_START = 8
        self.OBS_OPP_PLAYER_UNPROMOTED_START = 14
        self.OBS_OPP_PLAYER_PROMOTED_START = 22
        self.OBS_CURR_PLAYER_HAND_START = 28
        self.OBS_OPP_PLAYER_HAND_START = 35
        self.OBS_CURR_PLAYER_INDICATOR = 42
        self.OBS_MOVE_COUNT = 43
        self.OBS_RESERVED_1 = 44
        self.OBS_RESERVED_2 = 45
        self.OBS_UNPROMOTED_ORDER = ["P","L","N","S","G","B","R","K"]
        self.OBS_PROMOTED_ORDER = ["+P","+L","+N","+S","+B","+R"]
        self.hands = {0: {pt: 0 for pt in self.OBS_UNPROMOTED_ORDER}, 1: {pt: 0 for pt in self.OBS_UNPROMOTED_ORDER}}
        self.current_player = 0
        self.Color = types.SimpleNamespace(BLACK=0, WHITE=1)
        self.move_count = 0
        self.repetition_count = 0
        self.is_sennichite = lambda: False
        self.move_history = []
    def opponent(self):
        return 1 - self.current_player
    def is_in_check(self, color) -> bool:
        # Always False by default; override in test subclasses
        return False


def test_core46_shape_and_zeros():
    game = DummyGame()
    obs = features.build_core46(game)
    assert obs.shape == (46, 9, 9)
    # All planes except current player indicator should be close to zero
    for i in range(46):
        if i == game.OBS_CURR_PLAYER_INDICATOR:
            assert np.allclose(obs[i], 1.0)
        else:
            assert np.allclose(obs[i], 0.0)

def test_core46all_shape_and_zeros():
    game = DummyGame()
    obs = features.build_core46_all(game)
    assert obs.shape == (51, 9, 9)
    # Check core planes (0-45): all except indicator should be zero
    for i in range(46):
        if i == game.OBS_CURR_PLAYER_INDICATOR:
            assert np.allclose(obs[i], 1.0)
        else:
            assert np.allclose(obs[i], 0.0)
    # Check extra planes: check, repetition, last2ply, hand_onehot should be zero
    idx_check = 46 + features.EXTRA_PLANES["check"]
    idx_repetition = 46 + features.EXTRA_PLANES["repetition"]
    idx_last2ply = 46 + features.EXTRA_PLANES["last2ply"]
    idx_hand_onehot = 46 + features.EXTRA_PLANES["hand_onehot"]
    assert np.allclose(obs[idx_check], 0.0)
    assert np.allclose(obs[idx_repetition], 0.0)
    assert np.allclose(obs[idx_last2ply], 0.0)
    assert np.allclose(obs[idx_hand_onehot], 0.0)
    # Promotion zone plane: for Black, rows 0-2 should be 1, rest 0
    idx_prom_zone = 46 + features.EXTRA_PLANES["prom_zone"]
    assert np.allclose(obs[idx_prom_zone, 0:3, :], 1.0)
    assert np.allclose(obs[idx_prom_zone, 3:, :], 0.0)

def test_check_plane():
    class DummyGameCheck(DummyGame):
        def is_in_check(self, color) -> bool:
            return True
    game = DummyGameCheck()
    obs = features.build_core46_all(game)
    idx = 46 + features.EXTRA_PLANES["check"]
    assert np.allclose(obs[idx], 1.0)

def test_repetition_plane():
    game = DummyGame()
    game.repetition_count = 4
    obs = features.build_core46_all(game)
    idx = 46 + features.EXTRA_PLANES["repetition"]
    assert np.allclose(obs[idx], 1.0)

def test_prom_zone_plane():
    game = DummyGame()
    game.current_player = 0  # Black
    obs = features.build_core46_all(game)
    idx = 46 + features.EXTRA_PLANES["prom_zone"]
    assert np.allclose(obs[idx, 0:3, :], 1.0)
    assert np.allclose(obs[idx, 3:, :], 0.0)

def test_last2ply_plane():
    game = DummyGame()
    class Move:
        def __init__(self, to_square):
            self.to_square = to_square
    game.move_history = [Move((1,2)), Move((3,4))]
    obs = features.build_core46_all(game)
    idx = 46 + features.EXTRA_PLANES["last2ply"]
    assert np.isclose(obs[idx, 1, 2], 1.0)
    assert np.isclose(obs[idx, 3, 4], 1.0)

def test_hand_onehot_plane():
    game = DummyGame()
    game.hands[0]["P"] = 1
    obs = features.build_core46_all(game)
    idx = 46 + features.EXTRA_PLANES["hand_onehot"]
    assert np.isclose(obs[idx, 0, 0], 1.0)

def test_registry_and_spec():
    assert "core46" in features.FEATURE_REGISTRY
    assert "core46+all" in features.FEATURE_REGISTRY
    assert "core46" in features.FEATURE_SPECS
    assert features.FEATURE_SPECS["core46"].num_planes == 46
    assert features.FEATURE_SPECS["core46+all"].num_planes == 51
