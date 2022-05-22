import sys
import os.path as osp
import ray
import argparse
import random

from ray import tune
from ray.tune import register_env
from ray.rllib.agents.ppo import PPOTrainer

sys.path.append(osp.abspath('.'))
import gym_custom.models.model
from gym_custom.utils.api import update_conf, get_default_rllib_conf, add_env_conf
from gym_custom.utils.config import Config
from gym_custom.env import Environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to ray trainer")
    parser.add_argument('--conf_path', type=str, default='./configs/config.py')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint_58/checkpoint-58')
    args = parser.parse_args()

    config = Config.fromfile(args.conf_path)
    env_config = config['run_env_config']
    checkpoint = config['checkpoint_path']
    env = Environment(random.randint(0, 100000))
    register_env('Duckietown', env.create_env)
    ray.init(**config['ray_run_init_config'])

    rllib_config = get_default_rllib_conf()
    rllib_config.update(config['ray_run_sys_conf'])

    conf = update_conf(rllib_config)
    conf = add_env_conf(conf, config['run_env_config'])
    trainer = PPOTrainer(config=conf)
    trainer.restore(args.checkpoint)
    env = env.create_env(env_config)
    for i in range(10):
        obs = env.reset()

        done = False
        c = 0
        while not done:
            action = trainer.compute_action(obs, explore=False)
            print(f"{c}. {action}")
            c += 1
            obs, reward, done, info = env.step(action)
            env.render()



