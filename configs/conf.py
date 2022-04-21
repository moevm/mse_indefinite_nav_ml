import multiprocessing
import torch
import random
from gym_custom.utils.rllib_callback import get_callbacks

ENV_NAME = 'Duckietown'
ray_init_config = {
    "num_cpus": 6,
    "num_gpus": 0,
    "ignore_reinit_error": True,
}

ray_sys_conf = {
    "env": ENV_NAME,
    "num_gpus": 0,
    "num_workers": 5,
    "gpus_per_worker": 0,
    "env_per_worker": 1,
    "framework": "torch",
    "callbacks": get_callbacks(),
    "lr": 0.0001,
}

env_config = {
    "seed": random.randint(0, 100000),
    "map_name": "loop_empty",
    "max_steps": 5000,
    "camera_width": 640,
    "camera_height": 480,
    "accept_start_angle_deg": 40,
    "full_transparency": True,
    "distortion": True,
    "domain_rand": False,
}
