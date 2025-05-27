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


def test_load_checkpoint_with_padding_pad():
    # Old checkpoint has fewer input channels
    old_sd = make_state_dict(46)
    model = DummyModel(51)
    checkpoint = {"model_state_dict": old_sd}
    load_checkpoint_with_padding(model, checkpoint, 51)
    # Check that the stem weight shape matches the new model
    assert model.stem.weight.shape[1] == 51


def test_load_checkpoint_with_padding_truncate():
    # Old checkpoint has more input channels
    old_sd = make_state_dict(51)
    model = DummyModel(46)
    checkpoint = {"model_state_dict": old_sd}
    load_checkpoint_with_padding(model, checkpoint, 46)
    assert model.stem.weight.shape[1] == 46


def test_load_checkpoint_with_padding_noop():
    # Old checkpoint has same input channels
    old_sd = make_state_dict(46)
    model = DummyModel(46)
    checkpoint = {"model_state_dict": old_sd}
    load_checkpoint_with_padding(model, checkpoint, 46)
    assert model.stem.weight.shape[1] == 46
