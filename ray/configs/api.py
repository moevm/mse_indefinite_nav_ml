import os
import yaml
import glob
import logging
import numpy as np
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import ppo
from pprint import pprint

logger = logging.getLogger(__name__)


def load_config(path: str = "./main.yaml"):
    # create base config
    config: dict = dict()
    with open(path) as file:
        config = yaml.load(file, yaml.Loader)
    # add sub-configs
    return config


def get_rllib_config(path):
    conf = load_config(path)["rllib_config"]
    conf["env_config"] = env_config(path)
    return conf


def env_config(path):
    conf = load_config(path)
    return conf["env_config"]


def get_default_rllib_conf():
    return ppo.DEFAULT_CONFIG.copy()


def update_conf(conf):
    ray_conf = ppo.DEFAULT_CONFIG.copy()
    for key in conf:
        if key in ray_conf:
            ray_conf[key] = conf[key]
    return ray_conf


if __name__ == '__main__':
    pprint(load_config())
