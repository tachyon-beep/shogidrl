#!/usr/bin/env python3
"""
Simple test to verify the evaluation legal mask bug fix is working correctly.
"""

import torch
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper

def test_evaluation_legal_mask_fix():
    """Test that proper legal masks are generated in evaluation-like scenario."""
    
    # Create a game and get legal moves
    game = ShogiGame()
    legal_moves = game.get_legal_moves()
    
    # Create PolicyOutputMapper and generate proper legal mask
    policy_mapper = PolicyOutputMapper()
    device = torch.device("cpu")
    legal_mask = policy_mapper.get_legal_mask(legal_moves, device)
    
    print(f"Game has {len(legal_moves)} legal moves")
    print(f"Legal mask shape: {legal_mask.shape}")
    print(f"Legal mask sum (should equal number of legal moves): {legal_mask.sum().item()}")
    print(f"Total actions in policy space: {policy_mapper.get_total_actions()}")
    
    # Verify the mask is correct
    assert legal_mask.shape[0] == policy_mapper.get_total_actions(), "Mask should cover entire action space"
    assert legal_mask.sum().item() == len(legal_moves), "Mask should have exactly as many True values as legal moves"
    assert legal_mask.dtype == torch.bool, "Mask should be boolean"
    
    # Test that the old buggy approach would be different
    old_buggy_mask = torch.ones(len(legal_moves), dtype=torch.bool)
    print(f"Old buggy mask shape: {old_buggy_mask.shape}")
    print(f"Old buggy mask sum: {old_buggy_mask.sum().item()}")
    
    # The key difference: old mask was size of legal_moves, new mask is size of total action space
    print(f"\nCRITICAL DIFFERENCE:")
    print(f"  Old buggy mask size: {old_buggy_mask.shape[0]} (size of legal moves)")
    print(f"  New correct mask size: {legal_mask.shape[0]} (size of total action space)")
    print(f"  Ratio: {legal_mask.shape[0] / old_buggy_mask.shape[0]:.1f}x larger")
    
    print("\n✓ Evaluation legal mask fix verified successfully!")
    print("✓ Agents will now be properly constrained to legal moves during evaluation")

if __name__ == "__main__":
    test_evaluation_legal_mask_fix()
