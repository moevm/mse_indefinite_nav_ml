from typing import Dict, Optional
from ray.rllib import RolloutWorker, Policy, BaseEnv, SampleBatch
from ray.rllib.utils.typing import PolicyID, TensorType
from ray.rllib.evaluation.episode import Episode
from ray.rllib.agents import DefaultCallbacks
import copy
import numpy as np
from PIL import Image
from ray.rllib.agents import callbacks
from pprint import pprint
import torch

class CustomCallback(DefaultCallbacks):
    #def __init__(self, period):
    #    self._stepcount = 0
    #    self._period = period
    
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: Episode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:

        env = worker.env_creator(copy.deepcopy(worker.env_context))
        model = worker.get_policy().model
        print(type(model))
        pprint(model.__dir__)
        #exit(0)
        env.reset()
        obs, reward, done, info = env.step((0, 0))
        print("on_episode_end")
        print("on_episode_end")
        print("on_episode_end")
        print("on_episode_end")
        print("on_episode_end")
        for i in range(10):
            while not done:
                print(worker.io_context.log_dir)
                input_dict = {"obs": torch.Tensor(obs).to("cpu")}
                state = []
                action = model.forward(input_dict, state, [])
                obs, reward, done, info = env.step(action)
                img = image.fromarray(np.uint8(obs)).convert("RGB")
                img.save(f"{worker.io_context.log_dir}/test.png")
