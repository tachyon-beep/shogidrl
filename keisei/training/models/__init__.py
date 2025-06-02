from keisei.core.actor_critic_protocol import ActorCriticProtocol

from .resnet_tower import ActorCriticResTower


def model_factory(
    model_type, obs_shape, num_actions, tower_depth, tower_width, se_ratio, **kwargs
) -> ActorCriticProtocol:
    if model_type == "resnet":
        return ActorCriticResTower(
            input_channels=obs_shape[0],
            num_actions_total=num_actions,
            tower_depth=tower_depth,
            tower_width=tower_width,
            se_ratio=se_ratio,
            **kwargs,
        )
    # Add dummy/test models for testing
    elif model_type in ["dummy", "testmodel", "resumemodel"]:
        # Use a simple version of ActorCriticResTower or a mock for testing
        # For now, let's use ActorCriticResTower with minimal fixed params
        # Ensure these params are sensible for a minimal test model
        return ActorCriticResTower(
            input_channels=obs_shape[0],  # Should come from feature_spec.num_planes
            num_actions_total=num_actions,
            tower_depth=1,  # Minimal depth
            tower_width=16,  # Minimal width
            se_ratio=None,  # No SE block for simplicity
            **kwargs,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
