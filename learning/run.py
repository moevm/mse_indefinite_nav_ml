from pyglet.window import key
import sys
import ray
from ray import tune
from env import launch_env
from pprint import pprint
from ray.tune import register_env
from ray.rllib.agents.ppo import ppo, PPOTrainer
from config.api import update_conf, get_default_rllib_conf, add_env_conf
from env import Environment




def registy():
    checkpoint_path =""#ray_result/checkpoint***" 

    ENV_NAME = 'Duckietown'

    ray_init_config = {
        "num_cpus": 4,
        "num_gpus": 0,
        "object_store_memory": 78643200,

        "local_mode": True
    }

    ray.init(**ray_init_config)
    register_env(ENV_NAME, launch_env)

    rllib_config = get_default_rllib_conf()
    rllib_config.update({
            "env": ENV_NAME,
            "num_gpus": 0,
            "num_workers": 1,
            "callbacks": get_callbacks(),
            "lr":0.0001,
        })
    pprint(rllib_config)

    trainer = PPOTrainer(config=rllib_config)
    trainer.restore(checkpoint_path)
    return trainer

trainer = registy()

env = Environment(random.randint(0, 100000))
for i in range(10):
    obs = env.reset()
    done = False
    c = 0
    while not done:
        action = trainer.compute_action(obs, explore=False)
        print(f"{c}. {action}")
        c += 1
        obs, reward, done, info = env.step(action)
        env.render()

