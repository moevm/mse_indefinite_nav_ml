import multiprocessing
import torch
import random
from gym_custom.callbacks.customCallback import get_record_progress_callback


ENV_NAME = 'Duckietown'
ray_init_config = {
    "num_cpus": 2,
    "num_gpus": torch.cuda.device_count(),
    "ignore_reinit_error": True,
}

ray_sys_conf = {
    "env": ENV_NAME,
    "num_gpus": torch.cuda.device_count(),
    "num_workers": 1,
    "gpus_per_worker": torch.cuda.device_count(),
    "env_per_worker": 1,
    "framework": "torch",
    "lr": 0.0001,
    "callbacks": get_record_progress_callback(log_rate=10, duration=10, top_down=True),
    "train_batch_size": 500
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
    "domain_rand": False
}
