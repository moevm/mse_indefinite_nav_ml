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
        conf = model_config['custom_model_config']
        layers = []
        for i in conf['conv']:
            layers.append(Conv(i["in_channels"], i["out_channels"], i["activations"], i["pool"],
                               kernel_size=i["kernel_size"], padding=i["padding"], stride=i["stride"]))

        self._hidden_layers = nn.Sequential(*layers)
        layers.clear()
        for i in conf['linear']:
            layers.append(Linear(i["in"], i["out"], i['pool']))
        self._linear_layers = nn.Sequential(*layers)
        self._features = None
        layers.clear()
        for i in conf["value"]:
            layers.append(Linear(i["in"], i["out"], i["pool"]))
        self._value_layer = nn.Sequential(*layers)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict["obs_flat"].float()

        self._features = self._hidden_layers(obs)
        obs = self._features
        obs = obs.view(obs.size(0), -1)
        return self._linear_layers(obs), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        x = self._features.view(self._features.size(0), -1)
        value = self._value_layer(x)
        return value.squeeze(1)


ModelCatalog.register_custom_model("custom_torch_model", CustomModel)
