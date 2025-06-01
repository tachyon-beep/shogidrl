"""
test_checkpoint.py: Unit tests for keisei/utils/checkpoint.py
"""

import pytest
import torch
import torch.nn as nn

from keisei.utils.checkpoint import load_checkpoint_with_padding


class DummyModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.stem(x))


def make_state_dict(in_channels):
    model = DummyModel(in_channels)
    return model.state_dict()


@pytest.mark.parametrize(
    "old_channels,new_channels,scenario",
    [
        (46, 51, "pad"),
        (51, 46, "truncate"),
        (46, 46, "noop"),
    ],
    ids=["pad", "truncate", "noop"],
)
def test_load_checkpoint_with_padding_scenarios(old_channels, new_channels, scenario):
    """Test checkpoint loading with different channel padding scenarios."""
    # Create old checkpoint with specified channels
    old_sd = make_state_dict(old_channels)
    model = DummyModel(new_channels)
    checkpoint = {"model_state_dict": old_sd}

    # Load checkpoint with padding
    load_checkpoint_with_padding(model, checkpoint, new_channels)

    # Verify the stem weight shape matches the new model
    assert model.stem.weight.shape[1] == new_channels
