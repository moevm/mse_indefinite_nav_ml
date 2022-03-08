import gym
import logging

logger = logging.getLogger(__name__)


class InconvenientSpawnFixingWrapper(gym.Wrapper):
    """
    Fixes the "Exception: Could not find a valid starting pose after 5000 attempts" in duckietown-gym-daffy 5.0.13
    The robot is first placed in a random drivable tile, than a configuration is sampled on this tile. If the later
    step fails, it is repeated (up to 5000 times). If a tile has too many obstacles on it, it might not have any
    convenient (collision risking) configurations, so another tile should be selected. Instead of selecting a new tile,
    Duckietown gym just raises the above exception.
    This wrapper calls reset() again and again if a new tile has to be sampled.
    .. note::
        ``gym_duckietown.simulator.Simulator.reset()`` is called in ``gym_duckietown.simulator.Simulator.__init__(...)``.
        **Simulator instantiation should also be wrapped in a similar while loop!!!**
    """

    def reset(self, **kwargs):
        spawn_successful = False
        spawn_attempts = 1
        while not spawn_successful:
            try:
                logger.debug(self.unwrapped.user_tile_start)
                logger.debug("+"*1000)
                print(self.unwrapped.seed_value)
                ret = self.env.reset(**kwargs)
                spawn_successful = True
            except Exception as e:
                self.unwrapped.seed_value += 1  # Otherwise it selects the same tile in the next attempt
                logger.debug(self.unwrapped.user_tile_start)
                self.unwrapped.seed(self.unwrapped.seed_value)
                logger.error("{}; Retrying with new seed: {}".format(e, self.unwrapped.seed_value))
                spawn_attempts += 1
        logger.debug("Reset and Spawn successful after {} attempts".format(spawn_attempts))
        return ret
