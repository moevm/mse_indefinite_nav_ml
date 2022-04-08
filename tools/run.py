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
    parser.add_argument('--conf_path', type=str, default='./configs/run_conf.py')
    args = parser.parse_args()

    config = Config.fromfile(args.conf_path)
    env_config = config['env_config']
    checkpoint = config['checkpoint_path']
    env = Environment(random.randint(0, 100000))
    register_env('Duckietown', env.create_env)
    ray.init(**config['ray_init_config'])

    rllib_config = get_default_rllib_conf()
    rllib_config.update(config['ray_sys_conf'])

    conf = update_conf(rllib_config)
    conf = add_env_conf(conf, config['env_config'])
    trainer = PPOTrainer(config=conf)
    trainer.restore(checkpoint)
    env = env.create_env(env_config)
    for i in range(10):
        obs = env.reset()
        done = False
        c = 0
        while not done:
            action = trainer.compute_action(obs, explore=False)
            #print(f"{c}. {action}")
            c += 1
            obs, reward, done, info = env.step(action)
            ntl = env.next_tile(info["Simulator"]["tile_coords"], info["Simulator"]["cur_pos"], info["Simulator"]["cur_angle"])
            if ntl:
                print(c, ntl)
            env.render()


