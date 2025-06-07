#!/usr/bin/env python3
"""
Temporary script to inspect actual parameter names in ActorCriticResTower model.
This will help us understand the exact naming convention for the ResNet layers.
"""

import torch
from keisei.training.models.resnet_tower import ActorCriticResTower

# Create a test model with same structure as in config
model = ActorCriticResTower(
    input_channels=46,
    num_actions_total=3781,  # A reasonable action space size for Shogi
    tower_depth=6,
    tower_width=128,
    se_ratio=0.25,
)

print("ActorCriticResTower Model Parameters:")
print("=====================================")

# Print all parameter names to understand the structure
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

print("\n\nRelevant parameter name patterns for filtering:")
print("==============================================")

# Identify unique patterns for filtering
patterns = set()
for name, _ in model.named_parameters():
    parts = name.split('.')
    if len(parts) >= 2:
        patterns.add(f"{parts[0]}.{parts[1] if len(parts) > 1 else ''}")

for pattern in sorted(patterns):
    print(f"- {pattern}")

print("\n\nSuggested log_layer_keyword_filters for ResNet tower with individual blocks:")
print("===========================================================================")

# Generate suggestions based on actual structure
filters = ["stem", "policy_head", "value_head"]

# Add individual res_blocks
num_blocks = 6  # tower_depth from test model
for i in range(num_blocks):
    filters.append(f"res_blocks.{i}")

for filter_name in filters:
    print(f'    - "{filter_name}"')
