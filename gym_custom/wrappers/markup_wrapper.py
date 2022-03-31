import cv2
import gym
import numpy as np
from gym import spaces

import gym_custom.utils.markup_utils.vis as vis

class MarkupWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MarkupWrapper, self).__init__(env)

    def observation(self, observation):
        lane_lines = vis.detect_lane(observation)
        lane_lines_image = vis.display_lines(observation, lane_lines)
        return lane_lines_image