import multiprocessing
import torch
import random

ENV_NAME = 'Duckietown'
ray_init_config = {
    "num_cpus": multiprocessing.cpu_count() - 1,
    "num_gpus": torch.cuda.device_count(),
    "ignore_reinit_error": True,
}

ray_sys_conf = {
    "env": ENV_NAME,
    "num_gpus": torch.cuda.device_count(),
    "num_workers": multiprocessing.cpu_count() - 2,
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
                    "out_channels": 2,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 5,
                    "padding": 2,
                    "stride": 1
                },
                {
                    "in_channels": 2,
                    "out_channels": 12,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 5,
                    "padding": 2,
                    "stride": 1
                },
                {
                    "in_channels": 12,
                    "out_channels": 24,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 5,
                    "padding": 2,
                    "stride": 1
                },
                {
                    "in_channels": 24,
                    "out_channels": 36,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 5,
                    "padding": 2,
                    "stride": 1
                },
                {
                    "in_channels": 36,
                    "out_channels": 48,
                    "activations": torch.nn.ReLU(),
                    "pool": torch.nn.MaxPool2d(kernel_size=2),
                    "kernel_size": 5,
                    "padding": 2,
                    "stride": 1
                },

            ],
            "linear": [
                {
                    "in": 196,
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
                    "in": 196,
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
    "domain_rand": False
}
