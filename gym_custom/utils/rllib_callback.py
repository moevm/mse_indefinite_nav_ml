from ray.rllib.agents import MultiCallbacks
from gym_custom.callbacks.customCallback import CustomCallback

def get_callbacks():
    return MultiCallbacks([CustomCallback])