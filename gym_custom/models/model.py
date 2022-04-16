import logging
import numpy as np
import gym
import torch
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.direction = None
        self.num_of_directions = 4

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict["obs_flat"]
        if type(obs['view']) != torch.Tensor:
            obs['view'] = torch.from_numpy(obs['view']).to(self.device)
            obs['direction'] = torch.from_numpy(obs['direction']).to(self.device)
        self.direction = obs['direction']
        obs = obs["view"]
        self._features = self._hidden_layers(obs)
        obs = self._features
        obs = obs.view(obs.size(0), -1)
        return self._linear_layers(obs), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        one_hot_direction = torch.nn.functional.one_hot(self.direction, self.num_of_directions).squeeze(1)
        x = self._features.view(self._features.size(0), -1)
        x = torch.cat((x, one_hot_direction), dim=1)
        value = self._value_layer(x)
        return value.squeeze(1)


ModelCatalog.register_custom_model("custom_torch_model", CustomModel)
