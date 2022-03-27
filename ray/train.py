import sys
import ray
import argparse
from ray import tune
import random
from ray.tune import register_env
from ray.rllib.agents.ppo import PPOTrainer
from utils.api import update_conf, get_default_rllib_conf, add_env_conf
from utils.config import Config
from env import Environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to ray trainer")
    parser.add_argument('--conf_path', type=str, default='./configs/conf.py')
    args = parser.parse_args()

    config = Config.fromfile(args.conf_path)
    env = Environment(random.randint(0, 100000))
    register_env('Duckietown', env.create_env)
    ray.init(**config['ray_init_config'])

    rllib_config = get_default_rllib_conf()
    rllib_config.update(config['ray_sys_conf'])

    conf = update_conf(rllib_config)
    conf = add_env_conf(conf, config['env_config'])

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




