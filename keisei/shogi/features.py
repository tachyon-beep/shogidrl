"""
features.py: FeatureSpec registry and core46 observation builder for Keisei Shogi.
"""
from typing import Callable, Dict, List
import numpy as np

# Registry for feature builders
FEATURE_REGISTRY: Dict[str, Callable] = {}

def register_feature(name: str):
    def decorator(fn: Callable):
        FEATURE_REGISTRY[name] = fn
        return fn
    return decorator

class FeatureSpec:
    """
    Describes a set of feature planes for Shogi observation tensors.
    """
    def __init__(self, name: str, builder: Callable, num_planes: int):
        self.name = name
        self.builder = builder
        self.num_planes = num_planes

    def build(self, game) -> np.ndarray:
        return self.builder(game)

# --- Constants for extra planes ---
EXTRA_PLANES = {
    "check": 0,
    "repetition": 1,
    "prom_zone": 2,
    "last2ply": 3,
    "hand_onehot": 4,
}

# --- Core46 Feature Builder ---

@register_feature("core46")
def build_core46(game) -> np.ndarray:
    """
    Build the standard 46-plane observation tensor for the given game state.
    Args:
        game: ShogiGame instance
    Returns:
        obs: np.ndarray of shape (46, 9, 9)
    """
    # This implementation mirrors generate_neural_network_observation in shogi_game_io.py
    obs = np.zeros((46, 9, 9), dtype=np.float32)
    # --- Board pieces: current player POV ---
    for r in range(9):
        for c in range(9):
            piece = game.board[r][c]
            if piece is None:
                continue
            # Determine if piece belongs to current player or opponent
            is_curr = piece.color == game.current_player
            # Unpromoted
            if not piece.is_promoted():
                if is_curr:
                    idx = game.OBS_CURR_PLAYER_UNPROMOTED_START + game.OBS_UNPROMOTED_ORDER.index(piece.piece_type)
                else:
                    idx = game.OBS_OPP_PLAYER_UNPROMOTED_START + game.OBS_UNPROMOTED_ORDER.index(piece.piece_type)
            else:
                # Promoted
                if is_curr:
                    idx = game.OBS_CURR_PLAYER_PROMOTED_START + game.OBS_PROMOTED_ORDER.index(piece.piece_type)
                else:
                    idx = game.OBS_OPP_PLAYER_PROMOTED_START + game.OBS_PROMOTED_ORDER.index(piece.piece_type)
            obs[idx, r, c] = 1.0
    # --- Hand pieces ---
    for i, pt in enumerate(game.OBS_UNPROMOTED_ORDER):
        obs[game.OBS_CURR_PLAYER_HAND_START + i, :, :] = game.hands[game.current_player].get(pt, 0)
        opp = game.current_player.opponent() if hasattr(game.current_player, 'opponent') else (1 - game.current_player)
        obs[game.OBS_OPP_PLAYER_HAND_START + i, :, :] = game.hands[opp].get(pt, 0)
    # --- Meta planes ---
    obs[game.OBS_CURR_PLAYER_INDICATOR, :, :] = 1.0 if game.current_player == game.Color.BLACK else 0.0
    obs[game.OBS_MOVE_COUNT, :, :] = game.move_count / 512.0  # Normalize by max moves
    obs[game.OBS_RESERVED_1, :, :] = 0.0  # Always zero-filled
    obs[game.OBS_RESERVED_2, :, :] = 0.0  # Always zero-filled
    return obs

# --- Optional Feature Planes (T-2, refactored) ---

def add_check_plane(obs: np.ndarray, game, base_planes=46) -> None:
    idx = base_planes + EXTRA_PLANES["check"]
    if hasattr(game, 'is_in_check'):
        obs[idx, :, :] = 1.0 if game.is_in_check(game.current_player) else 0.0
    else:
        obs[idx, :, :] = 0.0

def add_repetition_plane(obs: np.ndarray, game, base_planes=46) -> None:
    idx = base_planes + EXTRA_PLANES["repetition"]
    # If game has a repetition count or sennichite detection, use it; else zeros
    if hasattr(game, 'repetition_count'):
        obs[idx, :, :] = min(game.repetition_count / 4.0, 1.0)  # Normalize to [0,1]
    elif hasattr(game, 'is_sennichite') and game.is_sennichite():
        obs[idx, :, :] = 1.0
    else:
        obs[idx, :, :] = 0.0

def add_prom_zone_plane(obs: np.ndarray, game, base_planes=46) -> None:
    idx = base_planes + EXTRA_PLANES["prom_zone"]
    # Mark promotion zone squares for current player (1.0 in zone, 0.0 elsewhere)
    zone_rows = [0, 1, 2] if game.current_player == game.Color.BLACK else [6, 7, 8]
    for r in zone_rows:
        obs[idx, r, :] = 1.0

def add_last2ply_plane(obs: np.ndarray, game, base_planes=46) -> None:
    idx = base_planes + EXTRA_PLANES["last2ply"]
    # Mark the destination squares of the last two moves (if available)
    if hasattr(game, 'move_history') and len(game.move_history) >= 1:
        for move in game.move_history[-2:]:
            if hasattr(move, 'to_square'):
                r, c = move.to_square
                obs[idx, r, c] = 1.0


def add_hand_onehot_plane(obs: np.ndarray, game, base_planes=46) -> None:
    idx = base_planes + EXTRA_PLANES["hand_onehot"]
    # Mark only [0,0] as 1 if any hand piece present for current player
    for pt in game.OBS_UNPROMOTED_ORDER:
        if game.hands[game.current_player].get(pt, 0) > 0:
            obs[idx, 0, 0] = 1.0
            break

@register_feature("core46+all")
def build_core46_all(game) -> np.ndarray:
    base_planes = 46
    obs = build_core46(game)
    obs = np.concatenate([obs, np.zeros((5, 9, 9), dtype=np.float32)], axis=0)
    add_check_plane(obs, game, base_planes)
    add_repetition_plane(obs, game, base_planes)
    add_prom_zone_plane(obs, game, base_planes)
    add_last2ply_plane(obs, game, base_planes)
    add_hand_onehot_plane(obs, game, base_planes)
    return obs

# Register FeatureSpec for all feature sets
CORE46_SPEC = FeatureSpec("core46", build_core46, 46)
CORE46_ALL_SPEC = FeatureSpec("core46+all", build_core46_all, 51)

# Dummy specs for testing
DUMMY_FEATS_SPEC = FeatureSpec("dummyfeats", build_core46, 46)
TEST_FEATS_SPEC = FeatureSpec("testfeats", build_core46, 46)
RESUME_FEATS_SPEC = FeatureSpec("resumefeats", build_core46, 46) # Add resumefeats

FEATURE_SPECS = {
    "core46": CORE46_SPEC,
    "core46+all": CORE46_ALL_SPEC,
    "dummyfeats": DUMMY_FEATS_SPEC,
    "testfeats": TEST_FEATS_SPEC,
    "resumefeats": RESUME_FEATS_SPEC, # Add resumefeats
}
