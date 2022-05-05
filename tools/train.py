import sys
import os.path as osp
import ray
import argparse
import random
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
sys.path.append(osp.abspath('.'))
import gym_custom.models.model
from gym_custom.utils.api import update_conf, get_default_rllib_conf, add_env_conf
from gym_custom.utils.config import Config
from gym_custom.callbacks.customCallback import FixedTrialProgressCallback
import gym_custom.env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to ray trainer")
    parser.add_argument('--conf_path', type=str, default='./configs/conf.py')
    args = parser.parse_args()

    config = Config.fromfile(args.conf_path)
    ray.init(**config['ray_init_config'])

    rllib_config = get_default_rllib_conf()
    rllib_config.update(config['ray_sys_conf'])

    conf = update_conf(rllib_config)
    conf = add_env_conf(conf, config['env_config'])

    metric = conf.get('metric', None)
    tune.run(
        PPOTrainer,
        stop={"timesteps_total": 2000000},
        checkpoint_at_end=True,
        config=conf,
        trial_name_creator=lambda trial: trial.trainable_name,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean",
        metric=metric,
        callbacks=[FixedTrialProgressCallback(metric=metric)]
    )




