import os

import cv2
from PIL import Image
import argparse
import sys
import os.path as osp

import numpy as np
import pyglet
from pyglet.window import key


sys.path.append(osp.abspath('.'))
from gym_duckietown.envs import DuckietownEnv
from gym_custom.wrappers.envWrapper import TileWrapper

STEP = 0

from gym_custom.utils.config import Config
from gym_custom.wrappers.debug_simulator import Simulator2

conf = Config.fromfile('./configs/conf.py')
conf['env_config']['draw_curve'] = True
env = Simulator2(**conf['env_config'])
env = TileWrapper(env)
# env = DtRewardBezieWrapper(env)

env.reset()
env.render()

if not osp.exists('./frames'):
    os.makedirs('./frames')


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    global STEP
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.15, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.15, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    print(action)
    obs, reward, done, info = env.step(action)
    # cv2.imwrite(f"frames/frame_{STEP}.jpg", obs)
    STEP += 1
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    print('direction_name: {}, direction_idx: {}, curve_idx: {}, tile_kind: {}'.format(info['direction_name'],
                                                                                       info['direction'],
                                                                                       info['curve_index'],
                                                                                       env.env.unwrapped._get_tile(
                                                                                           *info["Simulator"][
                                                                                               "tile_coords"])['kind']))

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)

        im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
