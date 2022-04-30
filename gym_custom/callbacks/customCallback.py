from typing import Dict, Optional
from types import FunctionType
from ray.rllib import RolloutWorker, Policy, BaseEnv, SampleBatch
from ray.rllib.utils.typing import PolicyID, TensorType
from ray.rllib.evaluation.episode import Episode
from ray.rllib.agents import DefaultCallbacks
from ray.tune.utils.callback import TrialProgressCallback
import copy
import numpy as np
import cv2 as cv
from PIL import Image
from ray.rllib.agents import callbacks
from pprint import pprint
import torch
import cv2
import gym

WINDOW_HEIGHT, WINDOW_WIDTH = 600, 800


class FixedTrialProgressCallback(TrialProgressCallback):
    def log_result(self, trial: "Trial", result: Dict, error: bool = False):
        tmp_result = result.copy()
        tmp_result.update(video=None)  # Delete recorded video to prevent it from logging
        super(FixedTrialProgressCallback, self).log_result(trial, tmp_result, error)


class AgentTestRecordCallback(DefaultCallbacks):
    def __init__(self, log_rate, duration, tow_down):
        super().__init__()
        self.iteration = 0
        self.log_rate = log_rate
        self.num_of_frames = duration * 20  # sec * fps
        self.tow_down = tow_down

        # TODO: add user defined width and height of video
        # self.width
        # self.height

        self._skip_frames = 3  # need to increase video speed

    def _render_frame(self, env):
        img = env.unwrapped._render_img(  # TODO: draw chosen curve
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            env.multi_fbo_human,
            env.final_fbo_human,
            env.img_array_human,
            top_down=self.tow_down,
            segment=False,
        )
        return img
                
    def on_train_result(self, *, trainer: "Trainer", result: dict, **kwargs) -> None:
        if self.iteration == 0 or self.iteration % self.log_rate != 0:
            self.iteration += 1
            return

        print('Started progress recording on {} iter'.format(self.iteration))

        env = trainer.env_creator(trainer.config['env_config'])
        policy = trainer.get_policy()
        video = np.zeros((1, self.num_of_frames, 3, WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        obs = env.reset()

        done = False
        i, j = 0, 0
        while i < self.num_of_frames:
            if j % self._skip_frames == 0:
                render_im = self._render_frame(env)
                render_im = render_im.transpose(2, 0, 1)  # chnl last -> chnl first
                video[0, i, :] = render_im

                i += 1

            if done:
                obs = env.reset()
                done = False
                continue

            input_dict = {"obs": [obs]}
            action = policy.compute_actions_from_input_dict(input_dict)
            action = action[2]['action_dist_inputs']
            action = (action[0][0], action[0][1])
            obs, reward, done, info = env.step(action)

            j += 1

        result['video'] = video

        self.iteration += 1


def get_record_progress_callback(log_rate=10, duration=20, top_down: bool = True):
    """

    Args:
        log_rate: if iteration % log_rate == 0 -> record video
        duration: duration of recorded video in seconds
        top_down: top down video or bot view

    Returns:
        callback creator
    """
    def foo():
        return AgentTestRecordCallback(log_rate, duration, top_down)

    return foo
