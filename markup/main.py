from PIL import Image
import argparse
import sys
import cv2
import os
import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv


if __name__ == "__main__":
    env = DuckietownEnv(
    seed=5123123,  # random seed
        map_name="udem1",
        max_steps=100,  # we don't want the gym to reset itself
        camera_width=640,
        camera_height=480,
        full_transparency=True,
        distortion=True,
        domain_rand=False
    )
    for i in range(1000):
        env.render()
        action=(1.0,1.0) # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        cv2.imwrite(f"frames/frame_{i}.jpg", observation)
        if done:
            observation = env.reset()
    env.close()
