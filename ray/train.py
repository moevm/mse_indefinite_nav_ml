from pyglet.window import key
import sys
import ray
from ray import tune
from env import Environment
from pprint import pprint
from ray.tune import register_env
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo import PPOTrainer
from configs.api import update_conf, get_default_rllib_conf
import random


if __name__ == "__main__":
    ENV_NAME = 'Duckietown'
    ray_init_config = {
        "num_cpus": 8,
        "num_gpus": 0,
        "object_store_memory": 78643200,  # 8 * 1024 * 1024
    }
    config = {
        "seed": random.randint(0, 100000),
        "map_name": "loop_empty",
        "max_steps": 5000,
        "camera_width": 640,
        "camera_height": 480,
        "accept_start_angle_deg": 40,
        "full_transparency": True,
        "distortion": True,
        "domain_rand": False
    }
    env = Environment(config, random.randint(0, 100000))

    ray.init(**ray_init_config)

    register_env(ENV_NAME, env.create_env)

    rllib_config = get_default_rllib_conf()
    rllib_config.update({
        "env": ENV_NAME,
        "num_gpus": 1,
        "num_workers": 8,  # 32,
        "gpu_per_worker": 1,
        "env_per_worker": 2,
        "framework": "torch",
        "lr": 0.0001,
    })

    conf = update_conf(rllib_config)

    tune.run(
        PPOTrainer,
        stop={"timesteps_total": 2000000},
        checkpoint_at_end=True,
        config=conf,
        trial_name_creator=lambda trial: trial.trainable_name,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean"
    )




