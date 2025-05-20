"""
Unit tests for ExperienceBuffer in experience_buffer.py
"""

import torch

from keisei.experience_buffer import ExperienceBuffer


def test_experience_buffer_add_and_len():
    """Test ExperienceBuffer add and __len__ methods."""
    buf = ExperienceBuffer(
        buffer_size=3, gamma=0.99, lambda_gae=0.95
    )  # Added gamma and lambda_gae
    assert len(buf) == 0
    # Create a dummy legal_mask. Its content doesn't matter for this test, only its presence.
    # Assuming num_actions_total is known or can be inferred for a realistic mask.
    # For simplicity, using a fixed size, e.g., 10, if PolicyOutputMapper isn't readily available here.
    # A more robust test might initialize PolicyOutputMapper to get the exact number of actions.
    # However, for ExperienceBuffer unit test, a consistently shaped tensor is sufficient.
    dummy_legal_mask = torch.zeros(10, dtype=torch.bool)  # Example size

    buf.add(
        torch.zeros(1), 1, 0.5, 0.0, 0.0, False, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=False, dummy_legal_mask
    assert len(buf) == 1
    buf.add(
        torch.ones(1), 2, 1.0, 0.0, 0.0, False, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=False, dummy_legal_mask
    assert len(buf) == 2
    buf.add(
        torch.ones(1), 3, -1.0, 0.0, 0.0, True, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=True, dummy_legal_mask
    assert len(buf) == 3
    # Should not add beyond buffer_size
    buf.add(
        torch.ones(1), 4, 2.0, 0.0, 0.0, False, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=False, dummy_legal_mask
    assert len(buf) == 3
    assert buf.actions == [1, 2, 3]
    assert buf.rewards == [0.5, 1.0, -1.0]
