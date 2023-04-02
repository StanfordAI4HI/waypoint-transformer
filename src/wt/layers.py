"""
Definitions of neural network layers.

Source: RvS GitHub repository (https://github.com/scottemmons/rvs)
"""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class DropoutActivation(nn.Module):
    """Combines dropout and activation into a single module.

    This is useful for adding dropout to a Stable Baselines3 policy, which takes an
    activation function as input.
    """

    activation_fn = nn.ReLU
    p = 0.1

    def __init__(self):
        """Instantiate the dropout and activation layers."""
        super(DropoutActivation, self).__init__()
        self.activation = DropoutActivation.activation_fn()
        self.dropout = nn.Dropout(p=DropoutActivation.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass: activation function first, then dropout."""
        return self.dropout(self.activation(x))

class IdentityExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space):
        super(IdentityExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))

    def forward(self, observations):
        return observations

