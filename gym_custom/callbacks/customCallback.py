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
import cv2
import gym
#from tensorboardX import SummaryWriter


class CustomCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.iteration = 0
        self.log_rate = 3
        #self.writer = SummaryWriter()
                
    def on_train_result(self, *, trainer: "Trainer", result: dict, **kwargs) -> None:
        if self.iteration % self.log_rate == 0: 
            env = trainer.env_creator(trainer.config['env_config'])
            #env = gym.wrappers.Monitor(env, f'{trainer.logdir}/project{self.iteration}')
            policy = trainer.get_policy()
            video = []
            self.iteration += 1
            for i in range(5):
                done = False
                obs = env.reset()
                while not done:
                    input_dict = {"obs": [obs]}
                    action = policy.compute_actions_from_input_dict(input_dict)
                    action = action[2]['action_dist_inputs']
                    action = (action[0][0], action[0][1])
                    obs, reward, done, info = env.step(action)
                    img = Image.fromarray(np.uint8(env.render())).convert("RGB")
                    img = img.rotate(180, expand=True)
                    video.append(np.array(img.copy()))
            video = np.array(video)
            size, height, width, channels = video.shape
            
            video = torch.Tensor([video])
            video = video.reshape((size, channels, 1, height, width))
            result['video'] = video
            #self.writer.add_video(f'{trainer.logdir}/video{self.iteration}', video, fps=30)
            
        self.iteration += 1
