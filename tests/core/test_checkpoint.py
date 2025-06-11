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
    # Clone the original stem weights and bn parameters for later comparison
    original_stem_weight = old_sd["stem.weight"].clone()
    original_bn_weight = old_sd["bn.weight"].clone()
    original_bn_bias = old_sd["bn.bias"].clone()
    original_bn_running_mean = (
        old_sd["bn.running_mean"].clone()
        if old_sd["bn.running_mean"] is not None
        else None
    )
    original_bn_running_var = (
        old_sd["bn.running_var"].clone()
        if old_sd["bn.running_var"] is not None
        else None
    )

    model = DummyModel(new_channels)
    checkpoint = {"model_state_dict": old_sd}

    # Load checkpoint with padding
    load_checkpoint_with_padding(model, checkpoint, new_channels)

    # Verify the stem weight shape matches the new model
    assert model.stem.weight.shape[1] == new_channels

    # Verify BN parameters are loaded correctly and are unchanged by padding logic
    assert torch.equal(model.bn.weight, original_bn_weight)
    assert torch.equal(model.bn.bias, original_bn_bias)
    if original_bn_running_mean is not None and model.bn.running_mean is not None:
        assert torch.equal(model.bn.running_mean, original_bn_running_mean)
    if original_bn_running_var is not None and model.bn.running_var is not None:
        assert torch.equal(model.bn.running_var, original_bn_running_var)

    # Verify stem weight values
    if scenario == "pad":
        # First old_channels should match original weights
        assert torch.equal(
            model.stem.weight[:, :old_channels, :, :], original_stem_weight
        )
        # Padded channels should be zero (or whatever padding value is expected)
        # Assuming padding with zeros for this test
        assert torch.all(model.stem.weight[:, old_channels:, :, :] == 0)
    elif scenario == "truncate":
        # Loaded weights should match the truncated part of original weights
        assert torch.equal(
            model.stem.weight, original_stem_weight[:, :new_channels, :, :]
        )
    elif scenario == "noop":
        # All weights should match original weights
        assert torch.equal(model.stem.weight, original_stem_weight)
