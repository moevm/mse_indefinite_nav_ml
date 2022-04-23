from typing import Dict, Optional
from types import FunctionType
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
        policy = policies['default_policy']
        env.reset()
        obs, reward, done, info = env.step((0, 0))
        for i in range(10):
            while not done:
                input_dict = {"obs": [obs]}
                action = policy.compute_actions_from_input_dict(input_dict)
                action = action[2]['action_dist_inputs']
                action = (action[0][0], action[0][1])
                print(action)
                obs, reward, done, info = env.step(action)
                img = Image.fromarray(np.uint8(env.render())).convert("RGB")
                img.save(f"{worker.io_context.log_dir}/test.png")
                exit(1)
