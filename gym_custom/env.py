
import pyglet
pyglet.window.Window()
from pyglet.window import key

import logging
import sys
import random
import numpy as np

from gym_duckietown.simulator import Simulator, DEFAULT_ROBOT_SPEED, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
from .wrappers.general_wrappers import InconvenientSpawnFixingWrapper
from .wrappers.observe_wrappers import ResizeWrapper, NormalizeWrapper, ClipImageWrapper, MotionBlurWrapper, SegmentationWrapper, RandomFrameRepeatingWrapper, ObservationBufferWrapper, RGB2GrayscaleWrapper, LastPictureObsWrapper, ReshapeWrapper
from .wrappers.reward_wrappers import DtRewardTargetOrientation, DtRewardVelocity, DtRewardCollisionAvoidance, DtRewardPosingLaneWrapper, DtRewardPosAngle, DtRewardBezieWrapper
from .wrappers.action_wpappers import Heading2WheelVelsWrapper, ActionSmoothingWrapper
from .wrappers.envWrapper import ActionDelayWrapper, ForwardObstacleSpawnnigWrapper, ObstacleSpawningWrapper, PrepareLearningWrapper, TileWrapper
from .wrappers.aido_wrapper import AIDOWrapper
from ray.tune import register_env


logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, seed):
        self._env = None
        np.random.seed(seed)
        random.seed(seed)

    def create_env(self, env_config, wrap: bool = True, env_id=0) -> Simulator:
        self._env = Simulator(
            seed=env_config["seed"],
            map_name=env_config["map_name"],
            max_steps=env_config["max_steps"],
            camera_width=env_config["camera_width"],
            camera_height=env_config["camera_height"],
            accept_start_angle_deg=env_config["accept_start_angle_deg"],
            full_transparency=env_config["full_transparency"],
            distortion=env_config["distortion"],
            domain_rand=env_config["domain_rand"],
        )
        if wrap:
            self._env = env_config['wrapping'](self._env)
        return self._env




env = Environment(random.randint(0, 100000))
register_env('Duckietown', env.create_env)
