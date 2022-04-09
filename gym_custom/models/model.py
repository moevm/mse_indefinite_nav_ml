import logging
import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gym_custom.models.layers import Conv, Linear


class CustomModel(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        layers = [Conv(3, 2, nn.ReLU(), nn.MaxPool2d(kernel_size=2), kernel_size=5, padding=2),
                  Conv(2, 12, nn.ReLU(), nn.MaxPool2d(kernel_size=2), kernel_size=5, padding=2),
                  Conv(12, 24, nn.ReLU(), nn.MaxPool2d(kernel_size=2), padding=2, kernel_size=5),
                  Conv(24, 36, nn.ReLU(), nn.MaxPool2d(kernel_size=2), padding=2, kernel_size=5),
                  Conv(36, 48, nn.ReLU(), nn.MaxPool2d(kernel_size=2), padding=2, kernel_size=5)]

        self._conv_layers = nn.Sequential(*layers)
        layers = [Linear(192, 16, nn.ReLU()),
                  Linear(16, num_outputs, nn.ReLU())]
        self._linear_layers = nn.Sequential(*layers)
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._features = self._hidden_layers(obs)
        obs = self._features
        obs = obs.view(obs.size(0), obs.size(1) * obs.size(2) * obs.size(3))
        return self._linear_layers(obs), state


ModelCatalog.register_custom_model("custom_torch_model", CustomModel)
