import multiprocessing
import torch
import random

ENV_NAME = 'Duckietown'
ray_init_config = {
    "num_cpus": 4,
    "num_gpus": 0,
    "ignore_reinit_error": True,
}

ray_sys_conf = {
    "env": ENV_NAME,
    "num_gpus": 0,
    "num_workers": 1,
    "gpus_per_worker": 0,
    "env_per_worker": 1,
    "framework": "torch",
    "lr": 0.0001,
}

env_config = {
    "seed": random.randint(0, 100000),
    "map_name": "crossroads",
    "max_steps": 5000,
    "camera_width": 640,
    "camera_height": 480,
    "accept_start_angle_deg": 40,
    "full_transparency": True,
    "distortion": True,
    "domain_rand": False
}

checkpoint_path = "../../PPO/PPO/checkpoint45/checkpoint-45"
