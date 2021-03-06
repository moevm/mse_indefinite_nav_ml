import multiprocessing
import torch
import random
from gym_custom.callbacks.customCallback import get_record_progress_callback
from gym_custom.wrappers.observe_wrappers import ResizeWrapper, NormalizeWrapper, ClipImageWrapper, MotionBlurWrapper, \
    ReshapeWrapper, RandomFrameRepeatingWrapper, ObservationBufferWrapper
from gym_custom.wrappers.action_wpappers import Heading2WheelVelsWrapper
from gym_custom.wrappers.envWrapper import PrepareLearningWrapper, TileWrapper

ENV_NAME = 'Duckietown'

ray_init_config = {
    "num_cpus": 5,
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
    '_disable_preprocessor_api': True,
    "model": {
        "custom_model": "custom_torch_model",
        "custom_model_config": {
            "conv": [
                {
                    "in_channels": 3,
                    "out_channels": 12,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 0
                },
                {
                    "in_channels": 12,
                    "out_channels": 24,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 0
                },
                {
                    "in_channels": 24,
                    "out_channels": 36,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.AdaptiveMaxPool2d((1, 1)),
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 0
                },

            ],
            "linear": [
                {
                    "in": 36 + 4,  # out of prev + 4 types of direction
                    "out": 16,
                    "pool": torch.nn.ReLU(),
                },
                {
                    "in": 16,
                    "out": 2,
                    "pool": torch.nn.ReLU(),
                },
            ],
            "value": [
                {
                    "in": 36 + 4,
                    "out": 16,
                    "pool": torch.nn.ReLU(),
                },
                {
                    "in": 16,
                    "out": 1,
                    "pool": torch.nn.ReLU(),
                },
            ]
        },
    },
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


def train_env_wrapping(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = RandomFrameRepeatingWrapper(env)
    env = ObservationBufferWrapper(env)
    env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    env = ReshapeWrapper(env)
    env = Heading2WheelVelsWrapper(env)
    env = TileWrapper(env)
    env = PrepareLearningWrapper(env)
    return env


def run_env_wrapping(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    env = ReshapeWrapper(env)
    env = Heading2WheelVelsWrapper(env)
    env = TileWrapper(env)
    env = PrepareLearningWrapper(env)
    return env


# config for train
ray_train_init_config = ray_init_config.copy()
ray_train_init_config["num_cpus"] = multiprocessing.cpu_count() - 4

ray_train_sys_conf = ray_sys_conf.copy()
ray_train_sys_conf["num_workers"] = multiprocessing.cpu_count() - 5
ray_train_sys_conf["lr"] = 0.0001
ray_train_sys_conf["callbacks"] = get_record_progress_callback(log_rate=10, duration=10, top_down=True)
ray_train_sys_conf["train_batch_size"] = 1000

train_env_config = env_config.copy()
train_env_config["wrapping"] = train_env_wrapping

# config for run
ray_run_init_config = ray_init_config.copy()
ray_run_sys_conf = ray_sys_conf.copy()
run_env_config = env_config.copy()
run_env_config["wrapping"] = run_env_wrapping
