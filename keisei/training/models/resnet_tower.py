"""
resnet_tower.py: ActorCriticResTower model for Keisei Shogi with SE block support.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from keisei.core.base_actor_critic import BaseActorCriticModel


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: Optional[float] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio) if se_ratio else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se:
            out = self.se(out)  # pylint: disable=not-callable
        out += x
        return F.relu(out)

class ActorCriticResTower(BaseActorCriticModel):
    def __init__(self, input_channels: int, num_actions_total: int, tower_depth: int = 9, tower_width: int = 256, se_ratio: Optional[float] = None):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, tower_width, 3, padding=1)
        self.bn_stem = nn.BatchNorm2d(tower_width)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(tower_width, se_ratio) for _ in range(tower_depth)
        ])
        # Slim policy head: 2 planes, then flatten, then linear
        self.policy_head = nn.Sequential(
            nn.Conv2d(tower_width, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions_total)
        )
        # Slim value head: 2 planes, then flatten, then linear
        self.value_head = nn.Sequential(
            nn.Conv2d(tower_width, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 1)
        )
    def forward(self, x):
        x = F.relu(self.bn_stem(self.stem(x)))
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value
